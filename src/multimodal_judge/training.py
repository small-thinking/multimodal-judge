"""Discrete 0..9 score-only SFT baseline, not the continuous M2 regression head.

Local, single-process VLM fine-tuning with aggregate-only experiment logging.

Generation errors are excluded from MAE/RMSE and count as incorrect for exact
accuracy. ``valid_count`` and ``invalid_rate`` make that denominator explicit.
No teacher-forced token accuracy is reported.
"""

import copy
import importlib
import json
import math
import re
import time
from pathlib import Path


_DEFAULTS = {
    "model": {"name_or_path": "Qwen/Qwen3-VL-2B-Instruct", "min_pixels": 4096,
              "max_pixels": 65536, "dtype": "float32", "attn_implementation": "eager"},
    "data": {"directory": "data/training_data/v2", "train_file": "train.jsonl",
             "validation_file": "validation.jsonl", "max_train_samples": None,
             "max_eval_samples": None, "max_length": 2048},
    "lora": {"enabled": True, "r": 8, "alpha": 16, "dropout": 0.05,
             "target_modules": ["q_proj", "v_proj"]},
    "training": {"output_dir": "artifacts/training/qwen3-vl-2b", "num_train_epochs": 1,
                 "max_steps": -1, "per_device_train_batch_size": 1,
                 "per_device_eval_batch_size": 1, "gradient_accumulation_steps": 4,
                 "learning_rate": 0.0002, "logging_steps": 1, "eval_steps": 10,
                 "save_steps": 10, "save_total_limit": 2, "gradient_checkpointing": True,
                 "dataloader_num_workers": 0, "seed": 42, "generate_eval": True,
                 "max_new_tokens": 8},
    "runtime": {"device": "auto"},
    "wandb": {"project": "multimodal-judge", "entity": None, "mode": "offline", "name": None},
}


def _number(value, name, minimum=0, integer=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < minimum
            or (integer and not isinstance(value, int))):
        raise ValueError(f"{name} must be {'an integer' if integer else 'finite'} >= {minimum}")


def _resolve_config(config, defaults=None):
    defaults = _DEFAULTS if defaults is None else defaults
    if not isinstance(config, dict):
        raise ValueError("config must be a dictionary")
    unknown = config.keys() - defaults.keys()
    if unknown:
        raise ValueError(f"Unknown config sections: {sorted(unknown)}")
    resolved = copy.deepcopy(defaults)
    for section, values in config.items():
        if not isinstance(values, dict) or values.keys() - defaults[section].keys():
            raise ValueError(f"Invalid or unknown settings in {section}")
        resolved[section].update(copy.deepcopy(values))
    for section, keys in {
        "model": ["name_or_path", "attn_implementation"],
        "data": ["directory", "train_file", "validation_file"],
        "training": ["output_dir"], "wandb": ["project"],
    }.items():
        for key in keys:
            value = resolved[section][key]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{section}.{key} must be a nonempty string")
    for key in ("entity", "name"):
        value = resolved["wandb"][key]
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"wandb.{key} must be null or a nonempty string")
    for section, keys in {
        "model": ["min_pixels", "max_pixels"], "data": ["max_length"],
        "lora": ["r"],
        "training": ["per_device_train_batch_size", "per_device_eval_batch_size",
                     "gradient_accumulation_steps", "logging_steps", "eval_steps",
                     "save_steps", "save_total_limit", "max_new_tokens"],
    }.items():
        for key in keys:
            _number(resolved[section][key], f"{section}.{key}", 1, integer=True)
    for key in ("max_train_samples", "max_eval_samples"):
        if resolved["data"][key] is not None:
            _number(resolved["data"][key], f"data.{key}", 0, integer=True)
    for key in ("dataloader_num_workers", "seed"):
        _number(resolved["training"][key], f"training.{key}", 0, integer=True)
    for section, key in (("training", "learning_rate"), ("training", "num_train_epochs"),
                         ("lora", "alpha")):
        _number(resolved[section][key], f"{section}.{key}")
        if resolved[section][key] == 0:
            raise ValueError(f"{section}.{key} must be positive")
    _number(resolved["training"]["max_steps"], "training.max_steps", -1, integer=True)
    if resolved["training"]["max_steps"] == 0:
        raise ValueError("training.max_steps must be -1 or positive")
    for section, key in (("lora", "enabled"), ("training", "gradient_checkpointing"),
                         ("training", "generate_eval")):
        if not isinstance(resolved[section][key], bool):
            raise ValueError(f"{section}.{key} must be boolean")
    _number(resolved["lora"]["dropout"], "lora.dropout")
    if resolved["lora"]["dropout"] >= 1:
        raise ValueError("lora.dropout must be < 1")
    targets = resolved["lora"]["target_modules"]
    if (not isinstance(targets, list) or not targets
            or any(t not in ("q_proj", "k_proj", "v_proj", "o_proj") for t in targets)):
        raise ValueError("lora.target_modules must contain language attention projections")
    if resolved["model"]["min_pixels"] > resolved["model"]["max_pixels"]:
        raise ValueError("model.min_pixels must not exceed max_pixels")
    for section, key, choices in (
        ("model", "dtype", ("float32", "float16", "bfloat16")),
        ("model", "attn_implementation", ("eager", "sdpa", "flash_attention_2")),
        ("runtime", "device", ("auto", "cpu", "mps", "cuda")),
        ("wandb", "mode", ("offline", "online", "disabled")),
    ):
        if resolved[section][key] not in choices:
            raise ValueError(f"{section}.{key} must be one of {choices}")
    return resolved


def language_attention_pattern(target_modules):
    """Match Qwen3-VL language decoder attention, never the visual encoder."""
    projections = "|".join(re.escape(name) for name in target_modules)
    return rf"(?!.*(?:^|\.)visual\.)(?:.*\.)?language_model\.layers\.\d+\.self_attn\.(?:{projections})"


def parse_score(text):
    """Accept exactly one ASCII digit (the shared score-only 0..9 contract)."""
    if not isinstance(text, str):
        return None
    value = text.strip()
    return int(value) if re.fullmatch(r"[0-9]", value) else None


def score_metrics(predictions, references):
    """Compute aggregate free-generation metrics; invalid outputs are not imputed."""
    predictions, references = list(predictions), list(references)
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have the same length")
    errors = []
    correct = 0
    for prediction, reference in zip(predictions, references):
        if isinstance(reference, bool) or not isinstance(reference, int) or not 0 <= reference <= 9:
            raise ValueError("Reference scores must be integers from 0 to 9")
        score = parse_score(prediction)
        if score is not None:
            errors.append(score - reference)
            correct += score == reference
    count, valid = len(references), len(errors)
    return {"mae": sum(abs(e) for e in errors) / valid if valid else None,
            "rmse": math.sqrt(sum(e * e for e in errors) / valid) if valid else None,
            "exact_accuracy": correct / count if count else None,
            "invalid_rate": (count - valid) / count if count else None,
            "count": count, "valid_count": valid}


def _generation_metrics(model, dataset, processor, device, max_new_tokens, max_length):
    import torch
    from tqdm.auto import tqdm

    from .training_data import score_messages

    predictions, references = [], []
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for index in tqdm(range(len(dataset)), desc="Validation generation"):
                record = dataset[index]
                prompt = processor.apply_chat_template(
                    score_messages(record["text"]), tokenize=False, add_generation_prompt=True)
                batch = processor(text=[prompt], images=[record["image"]], return_tensors="pt")
                prompt_length = batch["input_ids"].shape[-1]
                if prompt_length > max_length:
                    raise ValueError("Generation prompt exceeds data.max_length")
                batch = {key: value.to(device) if hasattr(value, "to") else value
                         for key, value in batch.items()}
                generated = model.generate(**batch, do_sample=False, num_beams=1,
                                           max_new_tokens=max_new_tokens, use_cache=True)
                prediction = processor.batch_decode(
                    generated[:, prompt_length:], skip_special_tokens=True)[0]
                predictions.append(prediction)
                references.append(record["score"])
    finally:
        model.train(was_training)
    return {f"generation_{key}": value
            for key, value in score_metrics(predictions, references).items()}


def _numeric_metrics(metrics):
    return {key: value for key, value in metrics.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value)}


def run_training(config: dict, resume_from_checkpoint: str | None = None):
    """Train and save locally, returning aggregate metrics (single process only)."""
    resolved = _resolve_config(config)
    model_config, data, training = (resolved[key] for key in ("model", "data", "training"))
    paths = [(Path(data["directory"]) / data[key]).resolve()
             for key in ("train_file", "validation_file")]
    if paths[0] == paths[1]:
        raise ValueError("Training and validation files must be distinct")
    for path in paths:
        if re.search(r"(^|[_.-])test([_.-]|$)", path.stem.lower()):
            raise ValueError("The test split must never be used for training or validation")
        if path == paths[0] and not path.is_file():
            raise ValueError(f"Dataset file does not exist: {path}")
    if resume_from_checkpoint is not None:
        if not isinstance(resume_from_checkpoint, str):
            raise ValueError("resume_from_checkpoint must be a checkpoint directory string")
        checkpoint = Path(resume_from_checkpoint).resolve()
        if not (checkpoint / "trainer_state.json").is_file():
            raise ValueError("resume_from_checkpoint must contain trainer_state.json")
        if not any((checkpoint / name).is_file() for name in (
            "model.safetensors", "pytorch_model.bin", "adapter_model.safetensors",
            "adapter_model.bin", "model.safetensors.index.json", "pytorch_model.bin.index.json",
        )):
            raise ValueError("resume_from_checkpoint has no model or adapter weights")
        resume_from_checkpoint = str(checkpoint)

    output = Path(training["output_dir"]).resolve()
    if output.exists() and not output.is_dir():
        raise ValueError("training.output_dir must be a directory")
    if resume_from_checkpoint is None and output.exists() and any(output.iterdir()):
        raise ValueError("training.output_dir is nonempty; choose a new directory or resume")

    import torch
    from transformers import (
        AutoModelForImageTextToText, AutoProcessor, Trainer, TrainerCallback,
        TrainingArguments, set_seed,
    )

    from .cli import select_device
    from .training_data import JsonlScoreDataset, ScoreCollator, inspect_data

    device = select_device(resolved["runtime"]["device"], torch)
    resolved["runtime"]["device"] = device
    if device != "cuda" and model_config["dtype"] == "float16":
        raise ValueError("float16 training requires CUDA; use float32 on CPU/MPS")
    if device == "mps" and model_config["dtype"] != "float32":
        raise ValueError("Use float32 for MPS training")
    if model_config["attn_implementation"] == "flash_attention_2":
        if device != "cuda" or model_config["dtype"] == "float32":
            raise ValueError("flash_attention_2 requires CUDA and float16/bfloat16")
        importlib.import_module("flash_attn")
    if (device == "cuda" and model_config["dtype"] == "bfloat16"
            and not torch.cuda.is_bf16_supported()):
        raise ValueError("Selected CUDA device does not support bfloat16")
    inspection = inspect_data(data["directory"], data["train_file"], data["validation_file"])
    if any(count for overlap in inspection["overlaps"].values() for count in overlap.values()):
        raise ValueError("Dataset split overlap detected; resolve identities/lineage before training")
    train_dataset = JsonlScoreDataset(paths[0], max_samples=data["max_train_samples"])
    eval_dataset = (JsonlScoreDataset(paths[1], max_samples=data["max_eval_samples"])
                    if paths[1].is_file() else [])
    if not len(train_dataset):
        raise ValueError("Training dataset is empty")
    do_eval = bool(len(eval_dataset))
    wandb = None
    if resolved["wandb"]["mode"] != "disabled":
        try:
            wandb = importlib.import_module("wandb")
        except ImportError as exc:
            raise ImportError("Install wandb or set wandb.mode='disabled'") from exc
    peft = importlib.import_module("peft") if resolved["lora"]["enabled"] else None
    args_values = {key: value for key, value in training.items()
                   if key not in ("generate_eval", "max_new_tokens")}
    args = TrainingArguments(
        **args_values, use_cpu=device == "cpu", report_to=[], disable_tqdm=False,
        remove_unused_columns=False, prediction_loss_only=True,
        eval_strategy="steps" if do_eval else "no", do_eval=do_eval,
        save_strategy="steps", logging_strategy="steps", dataloader_pin_memory=device == "cuda",
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=device == "cuda" and model_config["dtype"] == "float16",
        bf16=device == "cuda" and model_config["dtype"] == "bfloat16",
    )
    if args.device.type != device:
        raise ValueError(f"Trainer selected {args.device.type}, requested {device}")
    if args.world_size != 1:
        raise ValueError("This runner supports one process only")
    output.mkdir(parents=True, exist_ok=True)
    resolved["training"]["output_dir"] = str(output)
    (output / "resolved_config.json").write_text(
        json.dumps(resolved, indent=2, allow_nan=False) + "\n")
    (output / "data_inspection.json").write_text(
        json.dumps(inspection, indent=2, allow_nan=False) + "\n")
    run = None
    exit_code = 1
    started = time.perf_counter()
    try:
        if wandb is not None:
            # No Transformers W&B integration, watch(), artifacts, tables, or examples.
            run = wandb.init(**resolved["wandb"], config=resolved, dir=str(output),
                             settings=wandb.Settings(disable_code=True, disable_git=True,
                                                     console="off"))
            run.define_metric("optimizer_step")
            run.define_metric("*", step_metric="optimizer_step")

        class MetricsCallback(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                if run is not None and logs:
                    run.log({**_numeric_metrics(logs), "optimizer_step": state.global_step})

        set_seed(training["seed"])
        processor = AutoProcessor.from_pretrained(
            model_config["name_or_path"], min_pixels=model_config["min_pixels"],
            max_pixels=model_config["max_pixels"], trust_remote_code=False)
        model = AutoModelForImageTextToText.from_pretrained(
            model_config["name_or_path"], torch_dtype=getattr(torch, model_config["dtype"]),
            attn_implementation=model_config["attn_implementation"], trust_remote_code=False)
        if peft is not None:
            lora = resolved["lora"]
            pattern = language_attention_pattern(lora["target_modules"])
            if not any(re.fullmatch(pattern, name) for name, _ in model.named_modules()):
                raise ValueError("No language attention modules matched the LoRA configuration")
            model = peft.get_peft_model(model, peft.LoraConfig(
                r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
                target_modules=pattern, task_type="CAUSAL_LM", bias="none"))
        parameters = list(model.parameters())
        parameter_metrics = {
            "total_parameters": sum(parameter.numel() for parameter in parameters),
            "trainable_parameters": sum(parameter.numel() for parameter in parameters
                                        if parameter.requires_grad),
            "effective_batch_size": (training["per_device_train_batch_size"]
                                     * training["gradient_accumulation_steps"]
                                     * max(1, args.n_gpu) * args.world_size),
        }
        model.config.use_cache = False
        collator = ScoreCollator(processor, max_length=data["max_length"])
        trainer = Trainer(model=model, args=args, train_dataset=train_dataset,
                          eval_dataset=eval_dataset if do_eval else None,
                          data_collator=collator, processing_class=processor,
                          callbacks=[MetricsCallback()])
        result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        metrics = {**_numeric_metrics(result.metrics), **parameter_metrics}
        if do_eval:
            metrics.update(_numeric_metrics(trainer.evaluate()))
        if do_eval and training["generate_eval"]:
            metrics.update(_generation_metrics(model, eval_dataset, processor, args.device,
                                               training["max_new_tokens"], data["max_length"]))
        trainer.save_model(str(output))
        processor.save_pretrained(str(output))
        trainer.save_state()
        metrics.update(runtime_seconds=time.perf_counter() - started,
                       train_samples=len(train_dataset), eval_samples=len(eval_dataset))
        if run is not None:
            run.log({**_numeric_metrics(metrics), "optimizer_step": trainer.state.global_step})
        (output / "metrics.json").write_text(
            json.dumps(metrics, indent=2, allow_nan=False) + "\n")
        exit_code = 0
        return metrics
    finally:
        if run is not None:
            run.finish(exit_code=exit_code)
