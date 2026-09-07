"""Train a pointwise judge with continuous scores and optional rationales."""

import copy
import importlib
import json
import math
import re
import time
from pathlib import Path

import torch
from transformers import Trainer, TrainerCallback

from .training_config import (
    language_attention_pattern, numeric_metrics, validate_number,
    resolve_config as _resolve_config,
)


def joint_regression_metrics(prediction):
    """Raw continuous-score metrics; reject nonfinite outputs instead of hiding them."""
    import numpy as np

    scores = np.asarray(prediction.predictions, dtype=np.float64).reshape(-1)
    references = np.asarray(prediction.label_ids, dtype=np.float64).reshape(-1)
    if scores.shape != references.shape or not np.isfinite(scores).all() \
            or not np.isfinite(references).all():
        raise ValueError('Regression predictions/references must be finite and equally sized')
    if not len(scores):
        return {'count': 0}
    errors = scores - references
    return dict(mae=float(np.abs(errors).mean()), rmse=float(np.sqrt((errors ** 2).mean())),
                count=len(scores))


class _LossTotals:
    def __init__(self):
        self.samples = self.rationale_samples = self.rationale_tokens = 0
        self.score_sum = self.rationale_sum = 0.0

    def add(self, outputs, batch_size):
        def scalar(name):
            value = outputs[name]
            return float(value.detach().item()) if hasattr(value, 'detach') else float(value)
        count = scalar('rationale_samples')
        self.samples += batch_size
        self.rationale_samples += count
        self.rationale_tokens += scalar('rationale_tokens')
        self.score_sum += scalar('score_loss') * batch_size
        self.rationale_sum += scalar('rationale_loss') * count

    def metrics(self, prefix=''):
        if not self.samples:
            return {}
        return {prefix + key: value for key, value in dict(
            score_loss=self.score_sum / self.samples,
            rationale_loss=self.rationale_sum / max(1, self.rationale_samples),
            rationale_samples=self.rationale_samples, rationale_tokens=self.rationale_tokens,
            rationale_coverage=self.rationale_samples / self.samples,
            score_samples=self.samples).items()}


def sampled_memory_metrics(device):
    """Sample current MPS allocation; these are not peak-memory measurements."""

    if str(device).split(':')[0] != 'mps':
        return {}
    return {'mps_sampled_tensor_bytes': torch.mps.current_allocated_memory(),
            'mps_sampled_driver_bytes': torch.mps.driver_allocated_memory()}


def training_chart_metrics(logs):
    names = {'loss': 'loss', 'score_loss': 'score_loss', 'rationale_loss': 'reasoning_loss',
             'learning_rate': 'learning_rate', 'grad_norm': 'grad_norm'}
    result = {}
    for key, value in numeric_metrics(logs).items():
        if key.startswith('eval_'):
            name = key.removeprefix('eval_')
            if name in ('loss', 'mae', 'rmse', 'accuracy', 'within_one'):
                result['validation/' + name] = value
        elif key in names:
            result['train/' + names[key]] = value
    return result


class MetricsCallback(TrainerCallback):
    def __init__(self, run, device):
        self.run = run
        self.device = device
        self.started = time.perf_counter()
        self.start_step = 0

    def on_train_begin(self, args, state, control, **kwargs):
        self.started = time.perf_counter()
        self.start_step = state.global_step

    def on_log(self, args, state, control, logs=None, **kwargs):
        values = {**numeric_metrics(logs or {}), **sampled_memory_metrics(self.device),
                  'elapsed_seconds': time.perf_counter() - self.started,
                  'elapsed_optimizer_steps': state.global_step - self.start_step,
                  'optimizer_step': state.global_step}
        if logs is not None:
            logs.update(values)
        if self.run is not None:
            # W&B's internal step must advance even when optimizer_step is unchanged.
            charts = training_chart_metrics(logs or {})
            if charts:
                self.run.log({**charts, "optimizer_step": state.global_step})


class JointTrainer(Trainer):
    def __init__(self, *args, joint_manifest=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.joint_manifest = copy.deepcopy(joint_manifest)
        self.model_accepts_loss_kwargs = False
        self._train_totals = _LossTotals()
        self._eval_totals = _LossTotals()

    def _save(self, output_dir=None, state_dict=None):
        super()._save(output_dir, state_dict)
        # PEFT saves its adapter config, but not the native joint objective.
        destination = Path(output_dir or self.args.output_dir)
        self.model.config.save_pretrained(destination)
        if self.joint_manifest is not None:
            (destination / 'joint_manifest.json').write_text(
                json.dumps(self.joint_manifest, indent=2, allow_nan=False) + '\n')
            (destination / 'resolved_config.json').write_text(
                json.dumps(self.joint_manifest['resolved_config'], indent=2,
                           allow_nan=False) + '\n')

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        totals = self._train_totals if model.training else self._eval_totals
        totals.add(outputs, inputs['scores'].shape[0])
        return (outputs.loss, outputs) if return_outputs else outputs.loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad(), self.compute_loss_context_manager():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        if prediction_loss_only:
            return loss.detach().mean(), None, None
        return loss.detach().mean(), outputs.logits.detach(), inputs['scores'].detach()

    def evaluation_loop(self, *args, metric_key_prefix='eval', **kwargs):
        self._eval_totals = _LossTotals()
        output = super().evaluation_loop(*args, metric_key_prefix=metric_key_prefix, **kwargs)
        output.metrics.update(self._eval_totals.metrics(metric_key_prefix + '_'))
        return output

    def log(self, logs, *args, **kwargs):
        if 'loss' in logs:
            logs = {**logs, **self._train_totals.metrics()}
            self._train_totals = _LossTotals()
        return super().log(logs, *args, **kwargs)


def _check_mps_bfloat16():
    """Check native BF16 support before loading the full model."""
    import torch.nn.functional as F

    try:
        with torch.enable_grad():
            x = torch.ones((1, 2, 4), device='mps', dtype=torch.bfloat16, requires_grad=True)
            weight = torch.eye(4, device='mps', dtype=torch.bfloat16, requires_grad=True)
            y = F.linear(x, weight)
            z = F.scaled_dot_product_attention(y, y, y)
            z.float().square().mean().backward()
            torch.mps.synchronize()
            if not torch.isfinite(z).all().item() or not torch.isfinite(x.grad).all().item():
                raise RuntimeError('Nonfinite native BF16 probe')
    except (RuntimeError, TypeError, NotImplementedError) as exc:
        raise ValueError('Native MPS bfloat16 forward/backward probe failed; use float32') from exc


def predict_joint(model, processor, image, text, max_length=1024, max_new_tokens=128,
                  device='cpu', system_prompt='', return_metadata=False):
    """Score the input, then generate a rationale conditioned on that prediction."""
    from .joint_data import joint_messages, reasoning_prefix

    for name, value in (('max_length', max_length), ('max_new_tokens', max_new_tokens)):
        validate_number(value, name, 1, integer=True)

    def encode(prompt):
        batch = processor(text=[prompt], images=[image], return_tensors='pt', truncation=False)
        if batch['input_ids'].shape[-1] > max_length:
            raise ValueError('Generation prompt exceeds max_length')
        return {key: value.to(device) if hasattr(value, 'to') else value
                for key, value in batch.items()}

    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            prompt = processor.apply_chat_template(joint_messages(text, system_prompt), tokenize=False,
                                                   add_generation_prompt=True)
            batch = encode(prompt)
            positions = torch.tensor([batch['input_ids'].shape[-1] - 1],
                                     device=device, dtype=torch.long)
            outputs = model(**batch, score_positions=positions)
            score = float(outputs.logits.reshape(-1)[0].float().item())
            if not math.isfinite(score):
                raise ValueError('Model produced a nonfinite score')
            del outputs, batch
            batch = encode(reasoning_prefix(processor, text, score, system_prompt))
            length = batch['input_ids'].shape[-1]
            generated = model.generate(**batch, do_sample=False, num_beams=1,
                                       max_new_tokens=max_new_tokens, use_cache=True)
            reasoning = processor.batch_decode(generated[:, length:], skip_special_tokens=True)[0]
            result = {'score': score, 'reasoning': reasoning}
            if return_metadata:
                tokens = generated[0, length:].tolist()
                eos = model.generation_config.eos_token_id
                eos = [eos] if isinstance(eos, int) else (eos or [])
                result.update(tokens=len(tokens), hit_token_limit=len(tokens) >= max_new_tokens
                              and (not tokens or tokens[-1] not in eos))
            return result
    finally:
        model.train(was_training)


def _validate_paths(resolved, resume):
    data = resolved['data']
    paths = [(Path(data['directory']) / data[key]).resolve()
             for key in ('train_file', 'validation_file')]
    if paths[0] == paths[1]:
        raise ValueError('Training and validation files must be distinct')
    for path in paths:
        if re.search(r'(^|[_.-])test([_.-]|$)', path.stem.lower()):
            raise ValueError('The test split must never be used for training or validation')
    if not paths[0].is_file():
        raise ValueError('Training dataset file does not exist')
    output = Path(resolved['training']['output_dir']).resolve()
    if output.exists() and not output.is_dir():
        raise ValueError('training.output_dir must be a directory')
    if resume is None:
        if output.exists() and any(output.iterdir()):
            raise ValueError('training.output_dir is nonempty; choose a new directory or resume')
    else:
        if not isinstance(resume, str):
            raise ValueError('resume_from_checkpoint must be a checkpoint directory string')
        checkpoint = Path(resume).resolve()
        for name in ('trainer_state.json', 'optimizer.pt', 'scheduler.pt', 'rng_state.pth'):
            if not (checkpoint / name).is_file():
                raise ValueError(f'Real resume requires {name}')
        if not any((checkpoint / name).is_file() for name in (
            'model.safetensors', 'pytorch_model.bin', 'adapter_model.safetensors',
            'adapter_model.bin', 'model.safetensors.index.json', 'pytorch_model.bin.index.json')):
            raise ValueError('Checkpoint has no model or adapter weights')
        if output.exists() and any(output.iterdir()) and checkpoint.parent != output:
            raise ValueError('Nonempty output may only resume its own checkpoint')
        adapter_config = checkpoint / 'adapter_config.json'
        if resolved['lora']['enabled']:
            if not adapter_config.is_file() or 'score_head' not in (
                    json.loads(adapter_config.read_text()).get('modules_to_save') or []):
                raise ValueError('Joint resume requires an adapter checkpoint saving score_head')
        saved_config = checkpoint / 'resolved_config.json'
        if not saved_config.is_file():
            saved_config = checkpoint.parent / 'resolved_config.json'
        if saved_config.is_file():
            previous = json.loads(saved_config.read_text())
            if previous.get("prompt", {"system": ""}) != resolved["prompt"]:
                raise ValueError("Resume prompt configuration differs from saved run")
            for section in ('model', 'objective', 'lora'):
                if previous.get(section) != resolved[section]:
                    raise ValueError(f'Resume {section} configuration differs from saved run')
        resume = str(checkpoint)
    return paths, output, resume


def run_joint_training(config: dict, resume_from_checkpoint: str | None = None):
    """Train and save locally, returning aggregate metrics (single process only)."""
    resolved = _resolve_config(config)
    if not resolved['wandb']['name']:
        from .run_naming import make_run_name
        resolved['wandb']['name'] = make_run_name(
            'train', resolved['model']['name_or_path'], resolved['data']['directory'])
    model_config, data, training = (resolved[key] for key in ("model", "data", "training"))
    paths, output, resume_from_checkpoint = _validate_paths(resolved, resume_from_checkpoint)

    from transformers import (
        AutoConfig, AutoProcessor,
        TrainingArguments, set_seed,
    )

    from .cli import select_device
    from .training_data import inspect_data
    from .joint_data import JointScoreDataset, JointScoreCollator
    from .joint_model import JointQwen3VLForConditionalGeneration

    device = select_device(resolved["runtime"]["device"], torch)
    resolved["runtime"]["device"] = device
    if device != "cuda" and model_config["dtype"] == "float16":
        raise ValueError("float16 training requires CUDA; use float32 on CPU/MPS")
    if device == "mps":
        torch.mps.set_per_process_memory_fraction(resolved["runtime"]["mps_memory_fraction"])
        if model_config["dtype"] == "bfloat16":
            _check_mps_bfloat16()
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
    train_dataset = JointScoreDataset(paths[0], max_samples=data["max_train_samples"])
    eval_dataset = (JointScoreDataset(paths[1], max_samples=data["max_eval_samples"])
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
        remove_unused_columns=False, prediction_loss_only=False, label_names=["scores"],
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
            # Log scalars only; leave data and model artifacts local.
            run = wandb.init(**resolved["wandb"], job_type='training', config=resolved, dir=str(output),
                             settings=wandb.Settings(disable_code=True, disable_git=True,
                                                     console="off"))
            run.define_metric("optimizer_step", hidden=True)
            run.define_metric("*", step_metric="optimizer_step")

        set_seed(training["seed"])
        load_options = {"trust_remote_code": False}
        if resume_from_checkpoint is not None:
            manifest_path = Path(resume_from_checkpoint) / "joint_manifest.json"
            if manifest_path.is_file():
                saved_manifest = json.loads(manifest_path.read_text())
                if saved_manifest["base_model_name_or_path"] != model_config["name_or_path"]:
                    raise ValueError("Resume base model does not match the manifest")
                if saved_manifest.get("base_model_revision"):
                    load_options["revision"] = saved_manifest["base_model_revision"]
        hf_config = AutoConfig.from_pretrained(model_config["name_or_path"], **load_options)
        if hf_config.model_type != "qwen3_vl":
            raise ValueError("Joint training requires model_type=qwen3_vl")
        revision = getattr(hf_config, "_commit_hash", None) or load_options.get("revision")
        if revision:
            load_options["revision"] = revision
        hf_config.judge_config = copy.deepcopy(resolved["objective"])
        manifest = {"schema_version": 1, "base_model_name_or_path": model_config["name_or_path"],
                    "base_model_revision": revision, "judge_config": copy.deepcopy(hf_config.judge_config),
                    "resolved_config": copy.deepcopy(resolved),
                    "training_wandb_url": run.url if run is not None else None}
        processor = AutoProcessor.from_pretrained(
            model_config["name_or_path"], min_pixels=model_config["min_pixels"],
            max_pixels=model_config["max_pixels"], **load_options)
        model = JointQwen3VLForConditionalGeneration.from_pretrained(
            model_config["name_or_path"], torch_dtype=getattr(torch, model_config["dtype"]),
            attn_implementation=model_config["attn_implementation"],
            config=hf_config, **load_options)
        if peft is not None:
            lora = resolved["lora"]
            pattern = language_attention_pattern(lora["target_modules"])
            if not any(re.fullmatch(pattern, name) for name, _ in model.named_modules()):
                raise ValueError("No language attention modules matched the LoRA configuration")
            model = peft.get_peft_model(model, peft.LoraConfig(
                r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
                target_modules=pattern, task_type="CAUSAL_LM", bias="none",
                modules_to_save=["score_head"]))
        # Cast the whole PEFT head wrapper, including its saved trainable copy.
        model.score_head.float()
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
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
        collator = JointScoreCollator(processor, max_length=data["max_length"],
                                     max_reasoning_tokens=data["max_reasoning_tokens"],
                                     system_prompt=resolved["prompt"]["system"])
        trainer = JointTrainer(model=model, args=args, joint_manifest=manifest,
                          train_dataset=train_dataset,
                          eval_dataset=eval_dataset if do_eval else None,
                          data_collator=collator, processing_class=processor,
                          compute_metrics=joint_regression_metrics,
                          callbacks=[MetricsCallback(run, device)])
        result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        metrics = {**numeric_metrics(result.metrics), **parameter_metrics}
        if do_eval:
            metrics.update(numeric_metrics(trainer.evaluate()))
        if do_eval and training["generate_eval"]:
            from types import SimpleNamespace
            predictions, references = [], []
            for record in eval_dataset:
                prediction = predict_joint(model, processor, record["image"], record["text"],
                                           data["max_length"], training["max_new_tokens"], args.device,
                                           system_prompt=resolved["prompt"]["system"])
                predictions.append(prediction["score"])
                references.append(record["score"])
                # Reasoning is deliberately discarded, never serialized or sent to W&B.
            metrics.update({"generation_" + key: value for key, value in joint_regression_metrics(
                SimpleNamespace(predictions=predictions, label_ids=references)).items()})
        trainer.save_model(str(output))
        processor.save_pretrained(str(output))
        trainer.save_state()
        metrics.update(runtime_seconds=time.perf_counter() - started,
                       optimizer_step=trainer.state.global_step,
                       **sampled_memory_metrics(device),
                       train_samples=len(train_dataset), eval_samples=len(eval_dataset))
        if run is not None:
            run.summary.update({"validation/" + key.removeprefix("eval_"): value
                                for key, value in numeric_metrics(metrics).items()
                                if key in ("eval_loss", "eval_mae", "eval_rmse", "eval_accuracy",
                                           "eval_within_one")})
        (output / "metrics.json").write_text(
            json.dumps(metrics, indent=2, allow_nan=False) + "\n")
        exit_code = 0
        return metrics
    finally:
        if run is not None:
            run.finish(exit_code=exit_code)
