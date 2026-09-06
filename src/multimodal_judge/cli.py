"""Runtime checks and lazy dataset inspection."""

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
    parser.add_argument("command", choices=["smoke", "inspect-data"])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"])
    args = parser.parse_args()
    config_path = args.config or Path(
        "configs/data.yaml" if args.command == "inspect-data" else "configs/local.yaml")
    config = yaml.safe_load(config_path.read_text())
    if not isinstance(config, dict):
        parser.error("Configuration must be a YAML mapping")
    if args.device is not None:
        config.setdefault("runtime", {})["device"] = args.device
    if args.command == "inspect-data":
        from .training_data import inspect_data

        data = config["data"]
        directory = args.data_dir if args.data_dir is not None else data["directory"]
        print(json.dumps(inspect_data(
            directory, data.get("train_file", "train.jsonl"),
            data.get("validation_file", "validation.jsonl")), indent=2))
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
