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
    parser.add_argument("command", choices=["smoke", "inspect-data", "train-joint", "judge"])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--model", help="Override model.name_or_path")
    parser.add_argument("--data-dir", type=Path, help="Directory containing split JSONL files")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--wandb-mode", choices=["offline", "online", "disabled"])
    parser.add_argument("--resume-from-checkpoint", help="Trainer checkpoint directory")
    parser.add_argument("--checkpoint", type=Path, help="Saved joint adapter/head directory")
    parser.add_argument("--image", type=Path, help="Local image for joint inference")
    parser.add_argument("--text", help="Input text for joint inference")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--dtype", choices=["float32", "bfloat16"])
    args = parser.parse_args()
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
        print(json.dumps({**result, "score_scale": [0, 9]}, ensure_ascii=False))
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
