"""Joint training defaults, validation, and lightweight runner helpers."""

import copy
import math
import re


_DEFAULTS = {
    "prompt": {"system": ""},
    "model": {"name_or_path": "Qwen/Qwen3-VL-2B-Instruct", "min_pixels": 4096,
              "max_pixels": 65536, "dtype": "bfloat16", "attn_implementation": "sdpa"},
    "data": {"directory": "data/training_data/v2", "train_file": "train.jsonl",
             "validation_file": "validation.jsonl", "max_train_samples": None,
             "max_eval_samples": None, "max_length": 1024, "max_reasoning_tokens": 128},
    "lora": {"enabled": True, "r": 8, "alpha": 16, "dropout": 0.05,
             "target_modules": ["q_proj", "v_proj"]},
    "training": {"output_dir": "artifacts/training/qwen3-vl-2b-joint", "num_train_epochs": 2,
                 "max_steps": -1, "per_device_train_batch_size": 1,
                 "per_device_eval_batch_size": 1, "gradient_accumulation_steps": 8,
                 "learning_rate": 0.0002, "logging_steps": 1, "eval_steps": 100,
                 "save_steps": 100, "save_total_limit": 2, "gradient_checkpointing": True,
                 "dataloader_num_workers": 0, "seed": 42, "generate_eval": False,
                 "max_new_tokens": 128},
    "runtime": {"device": "auto", "mps_memory_fraction": 0.75},
    "objective": {"head_type": "regression", "score_weight": 1.0,
                  "rationale_weight": 0.1, "huber_delta": 0.1,
                  "score_min": 0.0, "score_max": 9.0, "ce_chunk_size": 32},
    "wandb": {"project": "multimodal-judge", "entity": None, "mode": "offline", "name": None},
}


def validate_number(value, name, minimum=0, integer=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < minimum
            or (integer and not isinstance(value, int))):
        raise ValueError(f"{name} must be {'an integer' if integer else 'finite'} >= {minimum}")


def resolve_config(config):
    """Resolve independent joint settings and reject invalid values before model loading."""
    defaults = _DEFAULTS
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
    if not isinstance(resolved["prompt"]["system"], str):
        raise ValueError("prompt.system must be a string")
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
            validate_number(resolved[section][key], f"{section}.{key}", 1, integer=True)
    for key in ("max_train_samples", "max_eval_samples"):
        if resolved["data"][key] is not None:
            validate_number(resolved["data"][key], f"data.{key}", 0, integer=True)
    for key in ("dataloader_num_workers", "seed"):
        validate_number(resolved["training"][key], f"training.{key}", 0, integer=True)
    for section, key in (("training", "learning_rate"), ("training", "num_train_epochs"),
                         ("lora", "alpha")):
        validate_number(resolved[section][key], f"{section}.{key}")
        if resolved[section][key] == 0:
            raise ValueError(f"{section}.{key} must be positive")
    validate_number(resolved["training"]["max_steps"], "training.max_steps", -1, integer=True)
    if resolved["training"]["max_steps"] == 0:
        raise ValueError("training.max_steps must be -1 or positive")
    for section, key in (("lora", "enabled"), ("training", "gradient_checkpointing"),
                         ("training", "generate_eval")):
        if not isinstance(resolved[section][key], bool):
            raise ValueError(f"{section}.{key} must be boolean")
    validate_number(resolved["lora"]["dropout"], "lora.dropout")
    if resolved["lora"]["dropout"] >= 1:
        raise ValueError("lora.dropout must be < 1")
    targets = resolved["lora"]["target_modules"]
    if (not isinstance(targets, list) or not targets
            or any(t not in ("q_proj", "k_proj", "v_proj", "o_proj") for t in targets)):
        raise ValueError("lora.target_modules must contain language attention projections")
    if resolved["model"]["min_pixels"] > resolved["model"]["max_pixels"]:
        raise ValueError("model.min_pixels must not exceed max_pixels")
    for section, key, choices in (
        ("objective", "head_type", ("regression", "classification")),
        ("model", "dtype", ("float32", "float16", "bfloat16")),
        ("model", "attn_implementation", ("eager", "sdpa", "flash_attention_2")),
        ("runtime", "device", ("auto", "cpu", "mps", "cuda")),
        ("wandb", "mode", ("offline", "online", "disabled")),
    ):
        if resolved[section][key] not in choices:
            raise ValueError(f"{section}.{key} must be one of {choices}")
    validate_number(resolved['data']['max_reasoning_tokens'],
                   'data.max_reasoning_tokens', 1, integer=True)
    fraction = resolved['runtime']['mps_memory_fraction']
    validate_number(fraction, 'runtime.mps_memory_fraction')
    if not 0 < fraction <= 1:
        raise ValueError('runtime.mps_memory_fraction must be in (0, 1]')
    objective = resolved['objective']
    for key in ('score_weight', 'rationale_weight', 'huber_delta'):
        validate_number(objective[key], f'objective.{key}')
    if objective['huber_delta'] == 0:
        raise ValueError('objective.huber_delta must be positive')
    for key, expected in (('score_min', 0.0), ('score_max', 9.0)):
        validate_number(objective[key], f'objective.{key}')
        if objective[key] != expected:
            raise ValueError('Joint scores use the raw 0..9 contract')
    validate_number(objective['ce_chunk_size'], 'objective.ce_chunk_size', 1, integer=True)
    return resolved


def language_attention_pattern(target_modules):
    """Match Qwen3-VL language decoder attention, never the visual encoder."""
    projections = "|".join(re.escape(name) for name in target_modules)
    return rf"(?!.*(?:^|\.)visual\.)(?:.*\.)?language_model\.layers\.\d+\.self_attn\.(?:{projections})"


def numeric_metrics(metrics):
    return {key: value for key, value in metrics.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value)}
