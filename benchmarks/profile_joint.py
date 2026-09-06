"""Bounded, synthetic B=1 joint-training benchmark; never evaluates model quality.

Run with the repository environment, for example::

    uv run --no-sync python benchmarks/profile_joint.py --output artifacts/profile.json

Loads a full checkpoint but trains only language LoRA and score_head. Joint model
losses (including chunked CE) belong to the main implementation, not this sidecar.
--steps counts measured optimizer/microbatch steps, in addition to warmup steps.
"""

import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time


def bounded_int(low, high):
    def parse(value):
        number = int(value)
        if not low <= number <= high:
            raise argparse.ArgumentTypeError(f"must be in [{low}, {high}]")
        return number
    return parse


def memory_fraction(value):
    number = float(value)
    if not math.isfinite(number) or not 0 < number <= 1:
        raise argparse.ArgumentTypeError("must be finite and in (0, 1]")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--attention", choices=("sdpa", "eager"), default="sdpa")
    parser.add_argument("--steps", type=bounded_int(1, 50), default=5)
    parser.add_argument("--warmup-steps", type=bounded_int(0, 10), default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=bounded_int(32, 1024), default=256)
    parser.add_argument("--text-tokens", type=bounded_int(1, 2048), default=256)
    parser.add_argument("--reasoning-tokens", type=bounded_int(1, 512), default=64)
    parser.add_argument("--checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--memory-fraction", type=memory_fraction, default=0.75)
    return parser.parse_args(argv)


def synthetic_text(tokenizer, target, phrase):
    # One bounded encode/decode pass; actual post-template lengths are reported.
    ids = tokenizer.encode(phrase * (target + 1), add_special_tokens=False)[:target]
    return tokenizer.decode(ids, skip_special_tokens=True)


def hardware_details(torch, psutil):
    try:
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                              capture_output=True, text=True, timeout=5, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        chip = platform.processor()
    return {"platform": platform.platform(), "macos": platform.mac_ver()[0],
            "machine": platform.machine(), "chip": chip, "python": platform.python_version(),
            "torch": str(torch.__version__), "cpu_logical_count": psutil.cpu_count(),
            "physical_memory_bytes": psutil.virtual_memory().total,
            "mps_built": torch.backends.mps.is_built(),
            "mps_available": torch.backends.mps.is_available(),
            "mps_recommended_max_memory_bytes": torch.mps.recommended_max_memory(),
            "environment": {key: os.environ.get(key) for key in (
                "PYTORCH_ENABLE_MPS_FALLBACK", "PYTORCH_MPS_HIGH_WATERMARK_RATIO",
                "PYTORCH_MPS_LOW_WATERMARK_RATIO", "HF_HOME", "HF_HUB_CACHE")}}


def run(args, report):
    # argparse (including --help and bounds) runs before any expensive imports.
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    os.environ.setdefault("HF_HOME", str(root / "artifacts" / "hf"))
    cache_dir = os.environ.get("HF_HUB_CACHE", str(Path(os.environ["HF_HOME"]) / "hub"))

    import psutil
    import torch
    import peft
    import transformers
    from PIL import Image, ImageDraw
    from transformers import AutoConfig, AutoProcessor, set_seed
    from multimodal_judge.joint_data import JointScoreCollator
    from multimodal_judge.joint_model import JointQwen3VLForConditionalGeneration
    from multimodal_judge.training import language_attention_pattern

    if not torch.backends.mps.is_available():
        raise RuntimeError("This benchmark requires MPS; no CPU/CUDA fallback is selected")
    torch.mps.set_per_process_memory_fraction(args.memory_fraction)
    set_seed(42)
    report.update(hardware=hardware_details(torch, psutil), cache_dir=cache_dir,
                  versions={"transformers": transformers.__version__, "peft": peft.__version__,
                            "psutil": psutil.__version__})
    report["source_sha256"] = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (Path(__file__).resolve(), root / "src/multimodal_judge/joint_model.py",
                     root / "src/multimodal_judge/joint_data.py",
                     root / "src/multimodal_judge/training.py")}
    process = psutil.Process()
    samples = report["memory_samples"]
    baseline_swap = psutil.swap_memory().used

    def sample(stage, step=None):
        torch.mps.synchronize()
        swap = psutil.swap_memory()
        samples.append({"stage": stage, "step": step,
                        "elapsed_seconds": time.perf_counter() - started,
                        "mps_current_allocated_bytes": torch.mps.current_allocated_memory(),
                        "mps_driver_allocated_bytes": torch.mps.driver_allocated_memory(),
                        "process_rss_bytes": process.memory_info().rss,
                        "system_swap_used_bytes": swap.used,
                        "system_swap_delta_from_start_bytes": swap.used - baseline_swap})

    started = time.perf_counter()
    sample("before_load")
    config = AutoConfig.from_pretrained(args.model, cache_dir=cache_dir, trust_remote_code=False)
    # Do not override judge_config: the joint model owns its defaults/schema.
    processor = AutoProcessor.from_pretrained(
        args.model, cache_dir=cache_dir, trust_remote_code=False,
        min_pixels=args.image_size ** 2, max_pixels=args.image_size ** 2)
    base = JointQwen3VLForConditionalGeneration.from_pretrained(
        args.model, config=config, cache_dir=cache_dir, trust_remote_code=False,
        torch_dtype=getattr(torch, args.dtype), attn_implementation=args.attention)
    sample("checkpoint_loaded_cpu")
    base.score_head.float()
    model = peft.get_peft_model(base, peft.LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=language_attention_pattern(["q_proj", "v_proj"]),
        modules_to_save=["score_head"]), autocast_adapter_dtype=True)
    model.score_head.float()
    model.config.use_cache = False
    if args.checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model.to("mps")
    model.train()
    trainable = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    if not trainable or any("visual" in name.split(".") for name, _ in trainable):
        raise RuntimeError("Expected trainable language adapters/head and a fully frozen vision tower")
    if any(param.dtype != torch.float32 for _, param in trainable):
        raise RuntimeError("All trainable adapters and score_head must be float32")
    report["parameters"] = {
        "total": sum(param.numel() for param in model.parameters()),
        "trainable": sum(param.numel() for _, param in trainable),
        "trainable_dtypes": sorted({str(param.dtype) for _, param in trainable}),
        "vision_frozen": True,
        "lora": {"r": 8, "alpha": 16, "dropout": 0.05, "targets": ["q_proj", "v_proj"]},
    }
    report["judge_config"] = model.config.to_dict().get("judge_config")
    report["checkpoint_commit"] = getattr(config, "_commit_hash", None)
    sample("load")

    image = Image.new("RGB", (args.image_size, args.image_size), (192, 192, 192))
    ImageDraw.Draw(image).rectangle((args.image_size // 4, args.image_size // 4,
                                   args.image_size * 3 // 4, args.image_size * 3 // 4),
                                  fill=(60, 100, 180))
    text = synthetic_text(processor.tokenizer, args.text_tokens,
                          " A blue square is centered on a plain gray background.")
    reasoning = synthetic_text(processor.tokenizer, args.reasoning_tokens,
                               " The image contains a blue square and a gray background.")
    record = {"image": image, "text": text, "score": 5, "reasoning": reasoning}
    collator = JointScoreCollator(processor, max_length=8192,
                                 max_reasoning_tokens=args.reasoning_tokens)
    batch = collator([record])
    ids = batch["input_ids"]
    image_id = getattr(config, "image_token_id", None)
    labels = batch.get("labels")
    report["tokens"] = {
        "input_text_retokenized": len(processor.tokenizer.encode(text, add_special_tokens=False)),
        "rationale_retokenized": len(processor.tokenizer.encode(reasoning, add_special_tokens=False)),
        "processor_sequence_length": int(ids.shape[-1]),
        "processor_nonpadding_tokens": int(batch["attention_mask"].sum()),
        "processor_image_tokens": int((ids == image_id).sum()) if image_id is not None else None,
        "supervised_tokens_including_template_suffix": int((labels != -100).sum())
        if labels is not None else None,
        "processor_rationale_body_tokens": int((labels != -100).sum()) - 1
        if labels is not None else None,
        "supervised_eos_tokens": 1,
        "image_grid_thw": batch["image_grid_thw"].tolist() if "image_grid_thw" in batch else None,
        "pixel_values_shape": list(batch["pixel_values"].shape) if "pixel_values" in batch else None,
        "note": "Requested lengths are pre-template targets; actual counts above may differ.",
    }
    if labels is None or not (labels != -100).any():
        raise RuntimeError("Joint collator must supply supervised rationale labels")
    batch = {key: value.to("mps") if isinstance(value, torch.Tensor) else value
             for key, value in batch.items()}
    optimizer = torch.optim.AdamW([param for _, param in trainable], lr=2e-4,
                                 betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01,
                                 foreach=False)
    groups = {"score_head": [(n, p) for n, p in trainable if "score_head" in n],
              "lora_B": [(n, p) for n, p in trainable if "lora_B" in n]}
    if not all(groups.values()):
        raise RuntimeError("Both score_head and LoRA B parameters must be trainable")
    initial = {name: param.detach().cpu().clone() for group in groups.values() for name, param in group}
    report["gradient_evidence"] = {key: {"nonzero_gradient_seen": False} for key in groups}
    sample("ready")

    for index in range(args.warmup_steps + args.steps):
        optimizer.zero_grad(set_to_none=True)
        torch.mps.synchronize()
        tick = time.perf_counter()
        outputs = model(**batch, use_cache=False, return_dict=True)
        torch.mps.synchronize()
        forward_seconds = time.perf_counter() - tick
        sample("forward", index)
        losses = {key: float(value.detach().float().cpu()) for key, value in outputs.items()
                  if "loss" in key and isinstance(value, torch.Tensor) and value.numel() == 1}
        if not {"loss", "score_loss", "rationale_loss"}.issubset(losses):
            raise RuntimeError("Joint model must return total, score and rationale losses")
        report["tokens"]["model_supervised_rationale_tokens_including_eos"] = int(
            outputs.rationale_tokens.detach().cpu())
        finite_loss = bool(losses) and all(math.isfinite(value) for value in losses.values())
        if not finite_loss or outputs.loss is None:
            raise FloatingPointError(f"Nonfinite or missing joint loss at step {index}")
        tick = time.perf_counter()
        outputs.loss.backward()
        torch.mps.synchronize()
        backward_seconds = time.perf_counter() - tick
        sample("backward", index)
        # No outputs or loss tensors survive a microbatch (including large logits).
        del outputs
        for name, param in trainable:
            if param.grad is not None and not bool(torch.isfinite(param.grad).all().item()):
                raise FloatingPointError(f"Nonfinite gradient in {name} at step {index}")
        for key, group in groups.items():
            nonzero = any(param.grad is not None and bool((param.grad != 0).any().item())
                          for _, param in group)
            report["gradient_evidence"][key]["nonzero_gradient_seen"] |= nonzero
        torch.mps.synchronize()
        tick = time.perf_counter()
        optimizer.step()
        torch.mps.synchronize()
        optimizer_seconds = time.perf_counter() - tick
        sample("optimizer", index)
        optimizer.zero_grad(set_to_none=True)
        report["steps"].append({"index": index, "warmup": index < args.warmup_steps,
                                "forward_seconds": forward_seconds,
                                "backward_seconds": backward_seconds,
                                "optimizer_seconds": optimizer_seconds,
                                "microbatch_seconds": forward_seconds + backward_seconds
                                + optimizer_seconds, "losses": losses,
                                "losses_finite": finite_loss, "gradients_finite": True})
        sample("released", index)

    report["update_evidence"] = {}
    for key, group in groups.items():
        changes = []
        for name, param in group:
            current = param.detach().cpu()
            if not bool(torch.isfinite(current).all()):
                raise FloatingPointError(f"Nonfinite updated parameter: {name}")
            changes.append(float((current - initial[name]).abs().max()))
        report["update_evidence"][key] = {
            "tensors_checked": len(changes), "tensors_changed": sum(value > 0 for value in changes),
            "max_absolute_change": max(changes), "parameters_finite": True}
    steady = [step["microbatch_seconds"] for step in report["steps"] if not step["warmup"]]
    mean = statistics.mean(steady)
    report["timing"] = {"mean_steady_microbatch_seconds": mean,
                        "median_steady_microbatch_seconds": statistics.median(steady),
                        "min_steady_microbatch_seconds": min(steady),
                        "max_steady_microbatch_seconds": max(steady),
                        "estimated_10000_pair_epoch_hours": 10000 * mean / 3600,
                        "scope": "Synchronized forward + backward + AdamW; B=1, no accumulation. "
                        "Warmup, loading, preprocessing, transfer, profiling/validation overhead, "
                        "eval and save excluded. Different lengths change cost; no quality claim."}
    if not all(item["tensors_changed"] for item in report["update_evidence"].values()):
        raise RuntimeError("Expected both score_head and LoRA B to change")
    report["status"] = "ok"
    return model, processor, record


def main(argv=None):
    args = parse_args(argv)
    report = {"status": "error", "arguments": {**vars(args), "output": str(args.output)},
              "seed": 42, "batch_size": 1, "optimizer": "AdamW",
              "reproducibility": "Fixed seed and synthetic inputs; bitwise MPS determinism "
              "is not guaranteed. Source hashes, checkpoint commit and versions are recorded.",
              "data": "Repeated deterministic synthetic image/text/score/rationale; no user data",
              "memory_samples": [], "steps": []}
    exit_code = 0
    try:
        # Preserve stdout as a single machine-readable compact JSON result.
        with contextlib.redirect_stdout(sys.stderr):
            run(args, report)
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        exit_code = 1
    finally:
        samples = report["memory_samples"]
        report["memory_summary"] = {
            "interpretation": "Maxima of synchronized stage samples, NOT true peaks. "
            "Swap is system-wide sampled usage/delta, not process-attributed or a true peak.",
            "max_sampled": {key: max(sample[key] for sample in samples) for key in (
                "mps_current_allocated_bytes", "mps_driver_allocated_bytes", "process_rss_bytes",
                "system_swap_used_bytes", "system_swap_delta_from_start_bytes")} if samples else {},
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps(report, separators=(",", ":"), allow_nan=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
