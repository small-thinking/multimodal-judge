"""Environment checks and configurable VLM score training."""

import argparse
import json
from pathlib import Path

import yaml


def select_device(requested, torch):
    available = {"cpu": True, "cuda": torch.cuda.is_available(),
                 "mps": torch.backends.mps.is_available()}
    if requested == "auto":
        return next(device for device in ("cuda", "mps", "cpu") if available[device])
    if requested not in available or not available[requested]:
        raise ValueError(f"Requested device {requested!r} is unavailable: {available}")
    return requested


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["smoke", "inspect-data", "train-joint", "judge",
                                                     "evaluate", "evaluation-center", "log-evaluation",
                                                     "import-reasoning-reviews", "prepare-reasoning-reviews"])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--model", help="Override model.name_or_path")
    parser.add_argument("--data-dir", type=Path, help="Directory containing split JSONL files")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--score-head", choices=["regression", "classification"],
                        help="Training score head; inference uses the saved architecture")
    parser.add_argument("--score-weight", type=float,
                        help="Training multiplier for the scoring loss (default 1)")
    parser.add_argument("--wandb-mode", choices=["offline", "online", "disabled"])
    parser.add_argument("--resume-from-checkpoint", help="Trainer checkpoint directory")
    parser.add_argument("--checkpoint", type=Path, help="Saved joint adapter/head directory")
    parser.add_argument("--image", type=Path, help="Local image for joint inference")
    parser.add_argument("--text", help="Input text for joint inference")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--dtype", choices=["float32", "bfloat16"])
    parser.add_argument("--split", choices=["test", "validation"], default="test")
    parser.add_argument("--include-base", action="store_true")
    parser.add_argument("--base-system-prompt", type=Path, help="Override the base evaluation prompt file")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--wandb-project", default="multimodal-judge")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--training-run-url")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--port", type=int, default=8877)
    parser.add_argument("--reasoning-rubric", type=Path)
    parser.add_argument("--reviews", type=Path)
    parser.add_argument("--reviewer")
    parser.add_argument("--enable-llm-judge", action="store_true")
    parser.add_argument("--judge-model", default="grok-4.6")
    parser.add_argument("--judge-effort", choices=["none", "low", "medium", "high", "xhigh"], default="low")
    parser.add_argument("--judge-cache-dir", type=Path, default=Path("artifacts/evaluation/judge-cache"))
    args = parser.parse_args()
    if args.command != 'train-joint' and (
        args.score_head is not None or args.score_weight is not None
    ):
        parser.error('--score-head and --score-weight are only supported for train-joint')
    if args.command == "prepare-reasoning-reviews":
        if not all((args.report, args.reasoning_rubric, args.output_dir)):
            parser.error("prepare-reasoning-reviews requires --report, --reasoning-rubric, --output-dir")
        from .reasoning_evaluation import prepare_reasoning_report

        report = prepare_reasoning_report(args.report, args.reasoning_rubric, args.output_dir)
        if args.enable_llm_judge:
            from .llm_judge import judge_reasoning
            from .evaluation import log_evaluation

            judge_reasoning(report, args.output_dir, args.judge_cache_dir,
                            args.judge_model, args.judge_effort)
            log_evaluation(args.output_dir / "report.json", args.wandb_mode or "offline",
                           args.wandb_project, args.wandb_entity)
        return
    if args.command == "import-reasoning-reviews":
        if not all((args.report, args.reviews, args.output_dir, args.reviewer)):
            parser.error("import-reasoning-reviews requires --report, --reviews, --output-dir, --reviewer")
        from .reasoning_evaluation import import_reasoning_reviews
        from .evaluation import log_evaluation

        import_reasoning_reviews(args.report, args.reviews, args.output_dir, args.reviewer)
        log_evaluation(args.output_dir / "report.json", args.wandb_mode or "offline",
                       args.wandb_project, args.wandb_entity)
        return
    if args.command == "evaluation-center":
        from .evaluation_center import serve_center

        serve_center(args.root, args.port)
        return
    if args.command == "log-evaluation":
        if args.report is None:
            parser.error("log-evaluation requires --report")
        from .evaluation import log_evaluation

        log_evaluation(args.report, args.wandb_mode or "online",
                       args.wandb_project, args.wandb_entity)
        return
    if args.command == "evaluate":
        if args.checkpoint is None or args.data_dir is None or args.output_dir is None:
            parser.error("evaluate requires --checkpoint, --data-dir and --output-dir")
        from .evaluation import run_evaluation

        result = run_evaluation(args.checkpoint, args.data_dir, args.output_dir,
            split=args.split, device=args.device or "auto", include_base=args.include_base,
            max_samples=args.max_samples, max_new_tokens=args.max_new_tokens,
            wandb_mode=args.wandb_mode or "offline", wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity, training_run_url=args.training_run_url,
            reasoning_rubric=args.reasoning_rubric, enable_llm_judge=args.enable_llm_judge,
            judge_model=args.judge_model, judge_effort=args.judge_effort,
            judge_cache_dir=args.judge_cache_dir, base_system_prompt=args.base_system_prompt)
        print(json.dumps(result["metrics"], indent=2))
        return
    if args.command == "judge":
        if args.checkpoint is None or args.image is None or args.text is None:
            parser.error("judge requires --checkpoint, --image and --text")
        from PIL import Image
        from .joint_inference import load_joint_checkpoint
        from .joint_training import predict_joint

        with Image.open(args.image) as source:
            image = source.convert("RGB")
        model, processor, device, config = load_joint_checkpoint(
            args.checkpoint, args.device or "auto", args.dtype)
        result = predict_joint(model, processor, image, args.text,
                               config["data"]["max_length"],
                               config["training"]["max_new_tokens"], device,
                               system_prompt=config.get("prompt", {}).get("system", ""))
        print(json.dumps({"reasoning": result["reasoning"], "rating": result["score"]},
                         ensure_ascii=False, allow_nan=False))
        return
    default_configs = {"smoke": "configs/local.yaml", "inspect-data": "configs/data.yaml",
                       "train-joint": "configs/train-joint.yaml"}
    config_path = args.config or Path(default_configs[args.command])
    config = yaml.safe_load(config_path.read_text())
    if not isinstance(config, dict):
        parser.error("Configuration must be a YAML mapping")
    for section, key, value in [
        ("model", "name_or_path", args.model),
        ("data", "directory", str(args.data_dir) if args.data_dir is not None else None),
        ("training", "output_dir", str(args.output_dir) if args.output_dir is not None else None),
        ("training", "max_steps", args.max_steps),
        ("objective", "head_type", args.score_head),
        ("objective", "score_weight", args.score_weight),
        ("wandb", "mode", args.wandb_mode),
        ("runtime", "device", args.device),
        ("model", "dtype", args.dtype),
    ]:
        if value is not None:
            config.setdefault(section, {})[key] = value
    if args.command == "inspect-data":
        from .training_data import inspect_data

        data = config["data"]
        print(json.dumps(inspect_data(
            data["directory"], data.get("train_file", "train.jsonl"),
            data.get("validation_file", "validation.jsonl"),
        ), indent=2))
        return
    if args.command == "train-joint":
        from .joint_training import run_joint_training
        from .run_naming import make_run_name
        from .training_config import resolve_config

        if args.output_dir is None and args.resume_from_checkpoint is None:
            resolved = resolve_config(config)
            name = make_run_name('train', resolved['model']['name_or_path'],
                                 resolved['data']['directory'])
            config.setdefault('training', {})['output_dir'] = str(
                Path(resolved['training']['output_dir']).parent / name)
            if not config.get('wandb', {}).get('name'):
                config.setdefault('wandb', {})['name'] = name

        metrics = run_joint_training(config, resume_from_checkpoint=args.resume_from_checkpoint)
        print(json.dumps(metrics, indent=2))
        return
    import torch

    device = select_device(config["runtime"]["device"], torch)
    torch.manual_seed(42)
    model = torch.nn.Linear(4, 1).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    x = torch.randn(8, 4, device=device)
    target = torch.rand(8, 1, device=device)
    before = model.weight.detach().clone()
    loss = torch.nn.functional.huber_loss(model(x), target)
    loss.backward()
    grad_norm = model.weight.grad.norm().item()
    optimizer.step()
    if not torch.isfinite(loss) or grad_norm <= 0 or torch.equal(before, model.weight):
        raise RuntimeError("Forward/backward/optimizer smoke failed")
    print(json.dumps({"test": "dummy_linear_forward_backward", "device": device,
                      "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
                      "loss": loss.item(), "gradient_norm": grad_norm, "status": "passed"}))
