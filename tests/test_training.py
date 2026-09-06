"""Runner contracts without downloading or training a pretrained model."""

import copy
import json
import re
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from multimodal_judge import training


@pytest.mark.parametrize("text,expected", [("0", 0), (" 9\n", 9), ("5", 5),
    ("10", None), ("-1", None), ("5.0", None), ('{"score":5}', None),
    ("score: 5", None), ("5 explanation", None), ("", None), ("０", None),
    ("NaN", None), (True, None)])
def test_parse_score(text, expected):
    assert training.parse_score(text) == expected


def test_score_metrics_denominators():
    result = training.score_metrics(["2", "5", "bad", "10"], [0, 5, 3, 9])
    assert result == {"mae": 1, "rmse": pytest.approx(2 ** 0.5),
                      "exact_accuracy": 0.25, "invalid_rate": 0.5,
                      "count": 4, "valid_count": 2}
    assert training.score_metrics(["bad"], [1])["mae"] is None
    assert training.score_metrics([], [])["invalid_rate"] is None
    with pytest.raises(ValueError):
        training.score_metrics(["1"], [])
    for reference in (True, -1, 10, 1.5):
        with pytest.raises(ValueError):
            training.score_metrics(["1"], [reference])


def test_language_attention_regex():
    pattern = training.language_attention_pattern(["q_proj", "v_proj"])
    assert re.fullmatch(pattern, "model.language_model.layers.0.self_attn.q_proj")
    assert re.fullmatch(pattern, "base_model.model.model.language_model.layers.12.self_attn.v_proj")
    for name in ("model.visual.blocks.0.attn.q_proj", "model.layers.0.self_attn.q_proj",
                 "model.language_model.layers.0.mlp.q_proj",
                 "model.language_model.layers.0.self_attn.k_proj",
                 "visual.language_model.layers.0.self_attn.q_proj"):
        assert not re.fullmatch(pattern, name)


@pytest.fixture
def runner(tmp_path, monkeypatch):
    import multimodal_judge.training_data as data_module

    (tmp_path / "train.jsonl").write_text("{}\n")
    (tmp_path / "validation.jsonl").write_text("{}\n")
    config = {"data": {"directory": str(tmp_path)}, "runtime": {"device": "cpu"},
              "training": {"output_dir": str(tmp_path / "output"), "generate_eval": False},
              "wandb": {"mode": "disabled"}}
    datasets = []

    def dataset(path, max_samples=None):
        datasets.append((path, max_samples))
        return [{"score": 5}] if max_samples != 0 else []

    inspection = {"splits": {}, "overlaps": {"train_validation": {"id": 0}}}
    inspect = Mock(return_value=inspection)
    monkeypatch.setattr(data_module, "inspect_data", inspect)
    monkeypatch.setattr(data_module, "JsonlScoreDataset", dataset)
    monkeypatch.setattr(data_module, "ScoreCollator", Mock())
    processor = Mock()
    model = Mock()
    model.parameters.return_value = [torch.nn.Parameter(torch.ones(3)),
                                     torch.nn.Parameter(torch.ones(2), requires_grad=False)]
    model.named_modules.return_value = [("model.language_model.layers.0.self_attn.q_proj", Mock())]
    trainer = Mock()
    trainer.train.return_value.metrics = {"train_loss": 1.2, "train_runtime": 0.3}
    trainer.evaluate.return_value = {"eval_loss": 1.1}
    trainer.state.global_step = 3
    args_capture = {}

    def arguments(**kwargs):
        args_capture.update(kwargs)
        return SimpleNamespace(device=torch.device("cpu"), world_size=1, n_gpu=0, **kwargs)

    transformers = SimpleNamespace(
        AutoModelForImageTextToText=SimpleNamespace(from_pretrained=Mock(return_value=model)),
        AutoProcessor=SimpleNamespace(from_pretrained=Mock(return_value=processor)),
        Trainer=Mock(return_value=trainer), TrainerCallback=object,
        TrainingArguments=Mock(side_effect=arguments), set_seed=Mock())
    peft = SimpleNamespace(get_peft_model=Mock(return_value=model), LoraConfig=Mock())
    wandb_run = Mock()
    wandb = SimpleNamespace(init=Mock(return_value=wandb_run), Settings=Mock())
    for name, module in (("transformers", transformers), ("peft", peft), ("wandb", wandb)):
        monkeypatch.setitem(sys.modules, name, module)
    return SimpleNamespace(config=config, datasets=datasets, inspect=inspect, model=model,
                           processor=processor, trainer=trainer, args=args_capture,
                           transformers=transformers, peft=peft, wandb=wandb, run=wandb_run,
                           output=tmp_path / "output")


def test_runner_cpu_eval_lora_and_local_outputs(runner, monkeypatch):
    # Explicit CPU must stay CPU even when CUDA is available.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    original = copy.deepcopy(runner.config)
    metrics = training.run_training(runner.config)
    assert runner.config == original
    assert metrics["total_parameters"] == 5
    assert metrics["trainable_parameters"] == 3
    assert metrics["effective_batch_size"] == 4
    assert metrics["eval_loss"] == 1.1
    assert metrics["runtime_seconds"] >= 0
    assert runner.args["use_cpu"] is True
    assert runner.args["prediction_loss_only"] is True
    assert runner.args["eval_strategy"] == "steps"
    assert runner.args["report_to"] == []
    assert runner.args["disable_tqdm"] is False
    assert runner.args["remove_unused_columns"] is False
    assert [path.name for path, _ in runner.datasets] == ["train.jsonl", "validation.jsonl"]
    runner.inspect.assert_called_once()
    runner.trainer.save_state.assert_called_once()
    runner.trainer.save_model.assert_called_once_with(str(runner.output))
    runner.processor.save_pretrained.assert_called_once_with(str(runner.output))
    assert json.loads((runner.output / "metrics.json").read_text()) == metrics
    assert json.loads((runner.output / "resolved_config.json").read_text())["runtime"]["device"] == "cpu"
    assert (runner.output / "data_inspection.json").exists()
    assert runner.peft.LoraConfig.call_args.kwargs["task_type"] == "CAUSAL_LM"
    runner.wandb.init.assert_not_called()


@pytest.mark.parametrize("missing", [False, True])
def test_empty_or_absent_validation_disables_eval(runner, missing):
    if missing:
        (runner.output.parent / "validation.jsonl").unlink()
    else:
        runner.config["data"]["max_eval_samples"] = 0
    metrics = training.run_training(runner.config)
    assert metrics["eval_samples"] == 0
    assert "eval_loss" not in metrics
    assert runner.args["eval_strategy"] == "no"
    assert runner.args["do_eval"] is False
    runner.trainer.evaluate.assert_not_called()


@pytest.mark.parametrize("section,key,value", [
    ("training", "max_steps", 0), ("training", "learning_rate", float("nan")),
    ("training", "gradient_accumulation_steps", 0), ("training", "generate_eval", "yes"),
    ("training", "max_new_tokens", True), ("data", "max_train_samples", -1),
    ("model", "dtype", "garbage"), ("model", "min_pixels", 100000),
    ("wandb", "mode", "oops"), ("wandb", "raw_examples", []),
    ("lora", "target_modules", ["visual"]), ("lora", "dropout", 1),
    ("runtime", "device", "tpu"),
])
def test_invalid_config_precedes_download(runner, section, key, value):
    runner.config.setdefault(section, {})[key] = value
    with pytest.raises(ValueError):
        training.run_training(runner.config)
    runner.transformers.AutoProcessor.from_pretrained.assert_not_called()
    runner.transformers.AutoModelForImageTextToText.from_pretrained.assert_not_called()


def test_overlap_preflight_precedes_download(runner):
    runner.inspect.return_value["overlaps"]["train_validation"]["id"] = 1
    with pytest.raises(ValueError, match="overlap"):
        training.run_training(runner.config)
    runner.transformers.AutoProcessor.from_pretrained.assert_not_called()


def test_empty_train_precedes_download(runner):
    runner.config["data"]["max_train_samples"] = 0
    with pytest.raises(ValueError, match="empty"):
        training.run_training(runner.config)
    runner.transformers.AutoProcessor.from_pretrained.assert_not_called()


def test_test_split_rejected(runner):
    runner.config["data"]["train_file"] = "test.jsonl"
    with pytest.raises(ValueError, match="test split"):
        training.run_training(runner.config)
    runner.inspect.assert_not_called()


def test_device_mismatch_precedes_download(runner):
    runner.transformers.TrainingArguments.side_effect = None
    runner.transformers.TrainingArguments.return_value = SimpleNamespace(
        device=torch.device("cuda"), world_size=1)
    with pytest.raises(ValueError, match="Trainer selected"):
        training.run_training(runner.config)
    runner.transformers.AutoProcessor.from_pretrained.assert_not_called()


def test_existing_output_requires_resume(runner):
    runner.output.mkdir()
    marker = runner.output / "keep.txt"
    marker.write_text("keep")
    with pytest.raises(ValueError, match="nonempty"):
        training.run_training(runner.config)
    assert marker.read_text() == "keep"
    runner.transformers.AutoProcessor.from_pretrained.assert_not_called()


def test_resume_checkpoint_forwarded(runner):
    checkpoint = runner.output / "checkpoint-3"
    checkpoint.mkdir(parents=True)
    (checkpoint / "trainer_state.json").write_text("{}")
    (checkpoint / "adapter_model.safetensors").write_bytes(b"mock")
    training.run_training(runner.config, str(checkpoint))
    runner.trainer.train.assert_called_once_with(resume_from_checkpoint=str(checkpoint))


def test_invalid_resume_precedes_download(runner):
    with pytest.raises(ValueError, match="trainer_state"):
        training.run_training(runner.config, str(runner.output))
    runner.transformers.AutoProcessor.from_pretrained.assert_not_called()


@pytest.mark.parametrize("mode", ["offline", "online"])
def test_wandb_safe_logs_and_success_cleanup(runner, mode):
    runner.config["wandb"]["mode"] = mode
    training.run_training(runner.config)
    assert runner.wandb.init.call_args.kwargs["mode"] == mode
    runner.wandb.Settings.assert_called_once_with(disable_code=True, disable_git=True, console="off")
    callback = runner.transformers.Trainer.call_args.kwargs["callbacks"][0]
    callback.on_log(None, SimpleNamespace(global_step=3), None,
                    logs={"loss": 1, "grad_norm": 2, "learning_rate": 0.01, "epoch": 0.5,
                          "example": "private", "bad": float("nan")})
    assert runner.run.log.call_args.args[0] == {
        "loss": 1, "grad_norm": 2, "learning_rate": 0.01, "epoch": 0.5, "optimizer_step": 3}
    assert all("step" not in call.kwargs for call in runner.run.log.call_args_list)
    runner.run.define_metric.assert_any_call("optimizer_step")
    runner.run.define_metric.assert_any_call("*", step_metric="optimizer_step")
    runner.run.finish.assert_called_once_with(exit_code=0)


@pytest.mark.parametrize("failure", ["load", "train", "save"])
def test_wandb_closes_on_errors(runner, failure):
    runner.config["wandb"]["mode"] = "offline"
    target = {"load": runner.transformers.AutoProcessor.from_pretrained,
              "train": runner.trainer.train, "save": runner.trainer.save_model}[failure]
    target.side_effect = RuntimeError("failure")
    with pytest.raises(RuntimeError, match="failure"):
        training.run_training(runner.config)
    runner.run.finish.assert_called_once_with(exit_code=1)


def test_generation_uses_shared_prompt_no_answer_and_greedy(monkeypatch):
    from multimodal_judge import training_data

    processor = Mock()
    processor.apply_chat_template.return_value = "prompt"
    processor.return_value = {"input_ids": torch.tensor([[11, 12]]),
                              "attention_mask": torch.ones(1, 2),
                              "pixel_values": torch.ones(4, 8)}
    processor.batch_decode.side_effect = [["5"], ["10"]]
    model = Mock(training=True)
    model.generate.return_value = torch.tensor([[11, 12, 5]])
    helper = Mock(wraps=training_data.score_messages)
    monkeypatch.setattr(training_data, "score_messages", helper)
    records = [{"text": "synthetic", "image": object(), "score": 5}] * 2
    metrics = training._generation_metrics(model, records, processor, "cpu", 8, 2048)
    assert metrics["generation_exact_accuracy"] == 0.5
    assert metrics["generation_invalid_rate"] == 0.5
    assert helper.call_count == 2
    messages = processor.apply_chat_template.call_args.args[0]
    assert [message["role"] for message in messages] == ["user"]
    assert "0 to 9" in messages[0]["content"][1]["text"]
    assert model.generate.call_args.kwargs["do_sample"] is False
    assert model.generate.call_args.kwargs["num_beams"] == 1
    assert "labels" not in model.generate.call_args.kwargs
    assert processor.batch_decode.call_args.args[0].tolist() == [[5]]
    model.train.assert_called_once_with(True)


def test_generation_restores_training_state_on_error():
    model = Mock(training=True)
    processor = Mock(side_effect=RuntimeError("generation failed"))
    with pytest.raises(RuntimeError):
        training._generation_metrics(model, [{"text": "synthetic", "image": object(), "score": 1}],
                                     processor, "cpu", 8, 2048)
    model.train.assert_called_once_with(True)
