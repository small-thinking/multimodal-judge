"""Load a saved joint adapter/head with its pinned base and prompt configuration."""

import json
from pathlib import Path


def load_joint_checkpoint(checkpoint, device="auto", dtype=None):
    import torch
    from transformers import AutoConfig, AutoProcessor
    from peft import PeftModel

    from .cli import select_device
    from .joint_model import JointQwen3VLForConditionalGeneration
    from .joint_training import _check_mps_bfloat16

    directory = Path(checkpoint).resolve()
    manifest = json.loads((directory / "joint_manifest.json").read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported joint checkpoint manifest version")
    resolved = manifest["resolved_config"]
    device = select_device(device, torch)
    dtype = dtype or resolved["model"]["dtype"]
    if dtype not in ("float32", "bfloat16"):
        raise ValueError("Joint inference supports float32 or bfloat16")
    if device == "mps":
        torch.mps.set_per_process_memory_fraction(resolved["runtime"]["mps_memory_fraction"])
        if dtype == "bfloat16":
            _check_mps_bfloat16()
    options = {"trust_remote_code": False}
    if manifest.get("base_model_revision"):
        options["revision"] = manifest["base_model_revision"]
    base_path = manifest["base_model_name_or_path"]
    if (directory / "adapter_config.json").is_file():
        config = AutoConfig.from_pretrained(base_path, **options)
        if config.model_type != "qwen3_vl":
            raise ValueError("Joint checkpoint requires a Qwen3-VL base")
        config.judge_config = manifest["judge_config"]
        base = JointQwen3VLForConditionalGeneration.from_pretrained(
            base_path, config=config, dtype=getattr(torch, dtype),
            attn_implementation=resolved["model"]["attn_implementation"], **options,
        )
        # Load the saved head into FP32: casting afterward cannot undo BF16 rounding.
        base.score_head.float()
        model = PeftModel.from_pretrained(base, directory, is_trainable=False)
    else:
        model = JointQwen3VLForConditionalGeneration.from_pretrained(
            directory, dtype=getattr(torch, dtype), trust_remote_code=False,
            attn_implementation=resolved["model"]["attn_implementation"],
        )
    model.score_head.float()
    model.to(device).eval()
    processor = AutoProcessor.from_pretrained(directory, trust_remote_code=False)
    return model, processor, device, resolved
