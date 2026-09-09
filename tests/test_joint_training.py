"""Offline runner contracts and tiny native Trainer/PEFT checkpoint verification."""

import copy
import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
import yaml

from multimodal_judge import joint_training as joint
from multimodal_judge import training_config


def test_config_overrides_and_defaults_are_independent():
    config = {"prompt": {"system": "synthetic"}, "training": {"num_train_epochs": 10},
              "objective": {"regression_loss": "mse"}}
    before = copy.deepcopy(config)
    defaults = joint._resolve_config({})
    assert defaults['objective']['head_type'] == 'regression'
    assert defaults['objective']['score_weight'] == 1.0
    assert defaults['runtime']['cpu_mkldnn'] is True
    resolved = joint._resolve_config(config)
    assert resolved["prompt"] == config["prompt"]
    assert resolved["training"]["num_train_epochs"] == 10
    assert resolved["objective"]["regression_loss"] == "mse"
    assert config == before
    resolved['lora']['target_modules'].append('k_proj')
    resolved['objective']['rationale_weight'] = 1
    assert config == before
    assert joint._resolve_config({}) == defaults


@pytest.mark.parametrize('section,key,value', [
    ('objective', 'rationale_weight', -1), ('objective', 'huber_delta', 0),
    ('objective', 'score_min', 1), ('objective', 'score_max', 10),
    ('objective', 'ce_chunk_size', True), ('objective', 'rationale_weight', float('nan')),
    ('objective', 'score_weight', -1), ('objective', 'score_weight', float('inf')),
    ('objective', 'score_weight', float('nan')), ('objective', 'score_weight', True),
    ('objective', 'head_type', 'ordinal'), ('objective', 'head_type', None),
    ('data', 'max_reasoning_tokens', 0), ('runtime', 'mps_memory_fraction', 1.1),
    ('runtime', 'mps_memory_fraction', 0), ('training', 'max_steps', 0),
    ('runtime', 'cpu_mkldnn', 'false'), ('runtime', 'cpu_mkldnn', 0),
    ('runtime', 'cpu_mkldnn', None),
    ('prompt', 'system', None), ('prompt', 'system', 42),
    ('model', 'trust_remote_code', True), ('lora', 'modules_to_save', []),
])
def test_invalid_config(section, key, value):
    with pytest.raises(ValueError):
        joint._resolve_config({section: {key: value}})


@pytest.mark.parametrize('enabled', [True, False])
def test_cpu_mkldnn_accepts_explicit_boolean(enabled):
    resolved = joint._resolve_config({'runtime': {'cpu_mkldnn': enabled}})
    assert resolved['runtime']['cpu_mkldnn'] is enabled


def test_continuous_metrics_no_rounding_or_clipping():
    result = joint.joint_regression_metrics(SimpleNamespace(
        predictions=np.array([[1.5], [5.0], [8.5]]), label_ids=np.array([0., 5., 9.])))
    assert result == {'mse': pytest.approx(2.5 / 3),
                      'mae': pytest.approx(2 / 3), 'rmse': pytest.approx((2.5 / 3) ** .5),
                      'count': 3}
    with pytest.raises(ValueError):
        joint.joint_regression_metrics(SimpleNamespace(predictions=[float('nan')], label_ids=[1]))
    with pytest.raises(ValueError):
        joint.joint_regression_metrics(SimpleNamespace(predictions=[1, 2], label_ids=[1]))


def test_loss_aggregation_uses_covered_samples_not_tokens():
    totals = joint._LossTotals()
    totals.add(dict(score_loss=torch.tensor(.2), rationale_loss=torch.tensor(2.),
                    rationale_samples=1, rationale_tokens=8), 2)
    totals.add(dict(score_loss=torch.tensor(.8), rationale_loss=torch.tensor(5.),
                    rationale_samples=2, rationale_tokens=2), 2)
    totals.add(dict(score_loss=torch.tensor(.5), rationale_loss=torch.tensor(0.),
                    rationale_samples=0, rationale_tokens=0), 2)
    result = totals.metrics()
    assert result['score_loss'] == pytest.approx(.5)
    assert result['rationale_loss'] == 4
    assert result['rationale_tokens'] == 10
    assert result['rationale_coverage'] == .5


def test_callback_numeric_only_global_step_and_sampled_memory(monkeypatch):
    monkeypatch.setattr(torch.mps, 'current_allocated_memory', lambda: 123)
    monkeypatch.setattr(torch.mps, 'driver_allocated_memory', lambda: 456)
    run = Mock()
    callback = joint.MetricsCallback(run, 'mps')
    callback.on_train_begin(None, SimpleNamespace(global_step=7), None)
    for _ in range(2):
        callback.on_log(None, SimpleNamespace(global_step=9), None,
                        logs={'loss': .1, 'reasoning': 'private', 'bad': float('nan'), 'bool': True})
    for call in run.log.call_args_list:
        values = call.args[0]
        assert values == {'train/loss': .1, 'optimizer_step': 9}
        assert not call.kwargs


def test_raw_score_metrics_aggregate_errors_before_square_root():
    from multimodal_judge.joint_model import JointJudgeOutput

    totals = joint._LossTotals()
    for prediction, target in [([1., 5.], [0., 5.]), ([8.], [5.])]:
        scores = torch.tensor(target)
        logits = torch.tensor(prediction).reshape(-1, 1).requires_grad_()
        normalized_mse = ((logits.reshape(-1) - scores) / 9).square().mean()
        totals.add(JointJudgeOutput(logits=logits, score_loss=normalized_mse,
            rationale_loss=torch.tensor(0.), rationale_samples=torch.tensor(0.),
            rationale_tokens=torch.tensor(0.)), len(target), scores)
    metrics = totals.metrics()
    assert metrics['mse'] == pytest.approx(10 / 3)
    assert metrics['rmse'] == pytest.approx((10 / 3) ** .5)
    assert metrics['mae'] == pytest.approx(4 / 3)
    assert metrics['mse'] == pytest.approx(81 * metrics['score_loss'])
    assert logits.grad is None
    charts = joint.training_chart_metrics({**metrics, 'eval_mse': 4.})
    assert charts['train/mse'] == pytest.approx(10 / 3)
    assert charts['validation/mse'] == 4.


def test_inference_uses_predicted_score_and_restores_mode(monkeypatch):
    from multimodal_judge import joint_data
    helper = Mock(wraps=joint_data.reasoning_prefix)
    monkeypatch.setattr(joint_data, 'reasoning_prefix', helper)
    processor = Mock()
    processor.apply_chat_template.return_value = 'prompt'
    processor.return_value = {'input_ids': torch.tensor([[1, 2, 3]]),
                              'attention_mask': torch.ones(1, 3, dtype=torch.long)}
    processor.batch_decode.return_value = ['local rationale']
    model = Mock(training=True)
    model.return_value = SimpleNamespace(logits=torch.tensor([[6.75]]))
    model.generate.return_value = torch.tensor([[1, 2, 3, 8, 9]])
    result = joint.predict_joint(model, processor, object(), 'synthetic',
                                 system_prompt='Use the rubric.')
    assert result == {'score': 6.75, 'reasoning': 'local rationale'}
    assert helper.call_args.args[2:] == (6.75, 'Use the rubric.')
    assert model.call_args.kwargs['score_positions'].tolist() == [2]
    assert 'scores' not in model.call_args.kwargs and 'labels' not in model.call_args.kwargs
    assert 'score_positions' not in model.generate.call_args.kwargs
    assert model.generate.call_args.kwargs['max_new_tokens'] == 128
    generation = model.generate.call_args.kwargs
    assert generation['do_sample'] is True
    assert generation['num_beams'] == 1
    assert generation['temperature'] == 0.7
    assert generation['top_p'] == 0.8
    assert generation['top_k'] == 0
    assert processor.batch_decode.call_args.args[0].tolist() == [[8, 9]]
    model.train.assert_called_once_with(True)
    processor.side_effect = ValueError('encode failed')
    with pytest.raises(ValueError, match='encode failed'):
        joint.predict_joint(model, processor, object(), 'synthetic')
    assert model.train.call_count == 2


def test_generation_rejects_overlong_prompt():
    processor = Mock(return_value={'input_ids': torch.ones(1, 3, dtype=torch.long)})
    model = Mock(training=False)
    with pytest.raises(ValueError, match='max_length'):
        joint.predict_joint(model, processor, object(), 'synthetic', max_length=2)
    model.generate.assert_not_called()
    model.train.assert_called_once_with(False)


@pytest.fixture
def config(tmp_path):
    (tmp_path / 'train.jsonl').write_text('{}\n')
    return {'data': {'directory': str(tmp_path)}, 'runtime': {'device': 'cpu'},
            'training': {'output_dir': str(tmp_path / 'output'), 'max_steps': 1,
                         'gradient_accumulation_steps': 1, 'gradient_checkpointing': False,
                         'save_steps': 1, 'eval_steps': 1}, 'wandb': {'mode': 'disabled'}}


def test_refuse_clobber_and_incomplete_resume(config):
    resolved = joint._resolve_config(config)
    output = Path(resolved['training']['output_dir'])
    output.mkdir()
    (output / 'keep').write_text('untouched')
    with pytest.raises(ValueError, match='nonempty'):
        joint._validate_paths(resolved, None)
    with pytest.raises(ValueError, match='trainer_state'):
        joint._validate_paths(resolved, str(output))
    (output / 'trainer_state.json').write_text('{}')
    with pytest.raises(ValueError, match='optimizer.pt'):
        joint._validate_paths(resolved, str(output))
    assert (output / 'keep').read_text() == 'untouched'


def _tiny_config():
    from transformers import Qwen3VLConfig
    return Qwen3VLConfig(
        text_config=dict(vocab_size=32, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                         head_dim=8, rope_scaling={'rope_type': 'default', 'mrope_section': [2, 1, 1]}),
        vision_config=dict(depth=1, hidden_size=16, intermediate_size=32, num_heads=2,
                           out_hidden_size=16, num_position_embeddings=16,
                           deepstack_visual_indexes=[0]),
        image_token_id=28, video_token_id=29, vision_start_token_id=30, vision_end_token_id=31,
        pad_token_id=0, eos_token_id=2)


@pytest.fixture
def local_runner(config, monkeypatch):
    import transformers
    from multimodal_judge import joint_data, joint_model, training_data

    processor = Mock()
    processor_loader = Mock(return_value=processor)
    config_loader = Mock(side_effect=lambda *a, **kw: _tiny_config())
    models = []

    def load(*args, **kwargs):
        model = joint_model.JointQwen3VLForConditionalGeneration(kwargs['config'])
        models.append(model)
        return model.to(kwargs['torch_dtype'])

    model_loader = Mock(side_effect=load)
    monkeypatch.setattr(transformers.AutoProcessor, 'from_pretrained', processor_loader)
    monkeypatch.setattr(transformers.AutoConfig, 'from_pretrained', config_loader)
    monkeypatch.setattr(joint_model.JointQwen3VLForConditionalGeneration,
                        'from_pretrained', model_loader)
    inspect = Mock(return_value={'splits': {}, 'overlaps': {'train_validation': {'id': 0}}})
    monkeypatch.setattr(training_data, 'inspect_data', inspect)
    monkeypatch.setattr(joint_data, 'JointScoreDataset', lambda *a, **kw: [0, 1])

    def collate(examples):
        b = len(examples)
        return dict(input_ids=torch.tensor([[1, 3, 4, 5, 2]] * b),
                    attention_mask=torch.ones(b, 5, dtype=torch.long),
                    score_positions=torch.ones(b, dtype=torch.long),
                    scores=torch.tensor([6.] * b),
                    labels=torch.tensor([[-100, -100, -100, 5, 2]] * b))

    monkeypatch.setattr(joint_data, 'JointScoreCollator', Mock(return_value=collate))
    return SimpleNamespace(config=config, processor=processor, processor_loader=processor_loader,
                           config_loader=config_loader, model_loader=model_loader, models=models,
                           inspect=inspect, collate=collate)


@pytest.mark.parametrize('head_type', ['regression', 'classification'])
def test_native_trainer_head_adapter_save_and_real_resume(local_runner, head_type):
    from safetensors.torch import load_file
    runner = local_runner
    config = runner.config
    config['objective'] = {'head_type': head_type}
    config['prompt'] = {'system': 'Use the saved rubric.'}
    (Path(config['data']['directory']) / 'validation.jsonl').write_text('{}\n')
    result = joint.run_joint_training(config)
    assert result['optimizer_step'] == 1
    assert result['eval_count'] == 2
    assert result['eval_rationale_coverage'] == 1
    assert result['eval_rationale_tokens'] == 4
    output = Path(config['training']['output_dir'])
    checkpoint = output / 'checkpoint-1'
    saved = load_file(str(checkpoint / 'adapter_model.safetensors'))
    head = {k: v for k, v in saved.items() if 'score_head' in k}
    assert head and all(value.dtype == torch.float32 for value in head.values())
    width = 10 if head_type == 'classification' else 1
    assert all(value.shape[0] == width for value in head.values())
    assert any('lora_B' in k and v.abs().sum() > 0 for k, v in saved.items())
    assert not any('visual' in k for k in saved)
    assert json.loads((checkpoint / 'adapter_config.json').read_text())['modules_to_save'] == ['score_head']
    assert json.loads((checkpoint / 'config.json').read_text())['judge_config']['rationale_weight'] == .1
    assert json.loads((checkpoint / 'config.json').read_text())['judge_config']['head_type'] == head_type
    assert json.loads((output / 'resolved_config.json').read_text())['objective']['rationale_weight'] == .1
    for loader in (runner.processor_loader, runner.config_loader, runner.model_loader):
        assert loader.call_args.kwargs['trust_remote_code'] is False
    # Native Trainer loads model, optimizer, scheduler, RNG and advances the saved step.
    assert json.loads((output / 'joint_manifest.json').read_text())['resolved_config']['prompt'] == config['prompt']
    config['prompt']['system'] = 'A changed rubric'
    with pytest.raises(ValueError, match='Resume prompt'):
        joint.run_joint_training(config, str(checkpoint))
    config['prompt']['system'] = 'Use the saved rubric.'
    config['training']['max_steps'] = 2
    result = joint.run_joint_training(config, str(checkpoint))
    assert result['optimizer_step'] == 2
    state = json.loads((output / 'checkpoint-2' / 'trainer_state.json').read_text())
    assert state['global_step'] == 2
    optimizer = torch.load(output / 'checkpoint-2' / 'optimizer.pt', weights_only=True)
    assert max(float(value['step']) for value in optimizer['state'].values()) == 2
    config['objective'] = {'rationale_weight': .2}
    with pytest.raises(ValueError, match='objective'):
        joint.run_joint_training(config, str(checkpoint))


@pytest.mark.parametrize('head_type', ['regression', 'classification'])
def test_bfloat16_base_fp32_entire_wrapped_head(local_runner, head_type):
    # CPU tiny native stack checks dtype persistence without claiming MPS support.
    local_runner.config['model'] = {'dtype': 'bfloat16'}
    local_runner.config['objective'] = {'head_type': head_type}
    joint.run_joint_training(local_runner.config)
    model = local_runner.models[0]
    assert all(p.dtype == torch.float32 for p in model.score_head.parameters())
    assert model.model.language_model.embed_tokens.weight.dtype == torch.bfloat16
    assert all(p.dtype == torch.float32 for p in model.parameters() if p.requires_grad)


def test_overlap_fails_before_any_model_download(local_runner):
    local_runner.inspect.return_value['overlaps']['train_validation']['id'] = 1
    with pytest.raises(ValueError, match='overlap'):
        joint.run_joint_training(local_runner.config)
    local_runner.model_loader.assert_not_called()
    local_runner.processor_loader.assert_not_called()


def test_mps_probe_rejection(monkeypatch):
    monkeypatch.setattr(torch, 'ones', Mock(side_effect=RuntimeError('unsupported BF16')))
    with pytest.raises(ValueError, match='Native MPS bfloat16'):
        joint._check_mps_bfloat16()


@pytest.mark.skipif(os.environ.get('MMJUDGE_TEST_MPS_JOINT') != '1'
                    or not torch.backends.mps.is_available(),
                    reason='Native MPS probe requires explicit MMJUDGE_TEST_MPS_JOINT=1')
def test_native_mps_bfloat16_capability_probe():
    joint._check_mps_bfloat16()


def test_wrong_model_type_rejected_before_custom_load(local_runner):
    local_runner.config_loader.side_effect = lambda *a, **kw: SimpleNamespace(model_type='qwen2_5_vl')
    with pytest.raises(ValueError, match='model_type=qwen3_vl'):
        joint.run_joint_training(local_runner.config)
    local_runner.model_loader.assert_not_called()
    local_runner.processor_loader.assert_not_called()


def test_manifest_pins_base_revision_and_objective(local_runner):
    def cfg(*args, **kwargs):
        config = _tiny_config()
        config._commit_hash = 'abc123'
        return config
    local_runner.config_loader.side_effect = cfg
    local_runner.config['objective'] = {'rationale_weight': .3, 'ce_chunk_size': 2}
    joint.run_joint_training(local_runner.config)
    output = Path(local_runner.config['training']['output_dir'])
    for directory in (output, output / 'checkpoint-1'):
        manifest = json.loads((directory / 'joint_manifest.json').read_text())
        assert manifest['base_model_revision'] == 'abc123'
        assert manifest['base_model_name_or_path'] == 'Qwen/Qwen3-VL-2B-Instruct'
        assert manifest['judge_config'] == manifest['resolved_config']['objective']
        assert manifest['judge_config']['rationale_weight'] == .3
        assert json.loads((directory / 'config.json').read_text())['judge_config'] == manifest['judge_config']
    assert local_runner.model_loader.call_args.kwargs['revision'] == 'abc123'
    assert local_runner.processor_loader.call_args.kwargs['revision'] == 'abc123'
    local_runner.config['training']['max_steps'] = 2
    joint.run_joint_training(local_runner.config, str(output / 'checkpoint-1'))
    assert local_runner.config_loader.call_args.kwargs['revision'] == 'abc123'
    local_runner.processor.save_pretrained.assert_any_call(str(output))


def test_prediction_step_never_gathers_auxiliary_outputs():
    from contextlib import nullcontext
    from multimodal_judge.joint_model import JointJudgeOutput

    class TestTrainer(joint.JointTrainer):
        def __init__(self):
            self.model_accepts_loss_kwargs = False
            self._train_totals = joint._LossTotals()
            self._eval_totals = joint._LossTotals()

        def _prepare_inputs(self, inputs):
            return inputs

        def compute_loss_context_manager(self):
            return nullcontext()

    trainer = TestTrainer()
    model = Mock(training=False, return_value=JointJudgeOutput(
        loss=torch.tensor(1.), logits=torch.tensor([[3.5], [6.5]]),
        score_class_logits=torch.ones(2, 10),
        score_loss=torch.tensor(.25), rationale_loss=torch.tensor(.75),
        rationale_samples=torch.tensor(1), rationale_tokens=torch.tensor(4)))
    inputs = {'scores': torch.tensor([3., 7.]), 'labels': torch.ones(2, 100, dtype=torch.long)}
    loss, predictions, labels = trainer.prediction_step(model, inputs, False)
    assert predictions.shape == (2, 1) and labels.shape == (2,)
    assert loss.item() == 1
    assert trainer._train_totals.samples == 0
    assert trainer._eval_totals.metrics()['rationale_coverage'] == .5
    assert trainer.model_accepts_loss_kwargs is False
    assert trainer.prediction_step(model, inputs, True)[1:] == (None, None)


def test_old_checkpoint_objective_resumes_with_new_defaults(local_runner):
    runner = local_runner
    joint.run_joint_training(runner.config)
    output = Path(runner.config['training']['output_dir'])
    checkpoint = output / 'checkpoint-1'
    # Emulate a checkpoint written before head_type and score_weight existed.
    for directory in (output, checkpoint):
        for filename in ('resolved_config.json', 'config.json', 'joint_manifest.json'):
            path = directory / filename
            if not path.exists():
                continue
            saved = json.loads(path.read_text())
            objectives = [saved.get('objective', {}), saved.get('judge_config', {}),
                          saved.get('resolved_config', {}).get('objective', {})]
            for objective in objectives:
                objective.pop('head_type', None)
                objective.pop('score_weight', None)
            path.write_text(json.dumps(saved))
    runner.config['training']['max_steps'] = 2
    runner.config['objective'] = {'head_type': 'regression', 'score_weight': 1.0}
    result = joint.run_joint_training(runner.config, str(checkpoint))
    assert result['optimizer_step'] == 2


@pytest.mark.parametrize('previous,changed', [
    ({'head_type': 'regression'}, {'head_type': 'classification'}),
    ({'head_type': 'classification'}, {'head_type': 'regression'}),
    ({'score_weight': 1.0}, {'score_weight': 2.0}),
])
def test_resume_rejects_score_objective_change_before_loading(local_runner, previous, changed):
    runner = local_runner
    runner.config['objective'] = previous
    joint.run_joint_training(runner.config)
    checkpoint = Path(runner.config['training']['output_dir']) / 'checkpoint-1'
    runner.model_loader.reset_mock()
    runner.config['objective'] = changed
    with pytest.raises(ValueError, match='objective'):
        joint.run_joint_training(runner.config, str(checkpoint))
    runner.model_loader.assert_not_called()


def test_mps_bfloat16_weights_do_not_enable_trainer_bf16(local_runner, monkeypatch):
    import transformers
    from multimodal_judge import cli
    monkeypatch.setattr(cli, 'select_device', lambda *args: 'mps')
    fraction = Mock()
    probe = Mock()
    monkeypatch.setattr(torch.mps, 'set_per_process_memory_fraction', fraction)
    monkeypatch.setattr(joint, '_check_mps_bfloat16', probe)
    # Stop immediately after capturing TrainingArguments; no accelerator allocation.
    arguments = Mock(side_effect=RuntimeError('captured args'))
    monkeypatch.setattr(transformers, 'TrainingArguments', arguments)
    local_runner.config['model'] = {'dtype': 'bfloat16'}
    with pytest.raises(RuntimeError, match='captured args'):
        joint.run_joint_training(local_runner.config)
    assert arguments.call_args.kwargs['bf16'] is False
    assert arguments.call_args.kwargs['fp16'] is False
    fraction.assert_called_once_with(.75)
    probe.assert_called_once_with()
    local_runner.model_loader.assert_not_called()


@pytest.mark.parametrize('failure', [False, True])
def test_runner_wandb_numeric_logging_and_cleanup(local_runner, monkeypatch, failure):
    run = Mock(url="https://wandb.ai/test/project/runs/training")
    wandb = SimpleNamespace(init=Mock(return_value=run), Settings=Mock())
    monkeypatch.setitem(sys.modules, 'wandb', wandb)
    local_runner.config['wandb']['mode'] = 'offline'
    if failure:
        local_runner.model_loader.side_effect = RuntimeError('mock load failure')
        with pytest.raises(RuntimeError, match='mock load failure'):
            joint.run_joint_training(local_runner.config)
    else:
        joint.run_joint_training(local_runner.config)
        assert run.log.call_count > 0
        for call in run.log.call_args_list:
            assert all(isinstance(value, (int, float)) for value in call.args[0].values())
            assert 'optimizer_step' in call.args[0]
            assert 'step' not in call.kwargs
        assert run.log.call_args.args[0]['optimizer_step'] == 1
    run.define_metric.assert_any_call('*', step_metric='optimizer_step')
    wandb.Settings.assert_called_once_with(disable_code=True, disable_git=True, console='off')
    run.finish.assert_called_once_with(exit_code=1 if failure else 0)


def test_language_attention_regex():
    pattern = training_config.language_attention_pattern(["q_proj", "v_proj"])
    assert re.fullmatch(pattern, "model.language_model.layers.0.self_attn.q_proj")
    assert re.fullmatch(pattern, "base_model.model.model.language_model.layers.12.self_attn.v_proj")
    for name in ("model.visual.blocks.0.attn.q_proj", "model.layers.0.self_attn.q_proj",
                 "model.language_model.layers.0.mlp.q_proj",
                 "model.language_model.layers.0.self_attn.k_proj",
                 "visual.language_model.layers.0.self_attn.q_proj"):
        assert not re.fullmatch(pattern, name)



@pytest.mark.parametrize("section,key,value", [
    ("training", "max_steps", 0), ("training", "learning_rate", float("nan")),
    ("training", "gradient_accumulation_steps", 0), ("training", "generate_eval", "yes"),
    ("training", "max_new_tokens", True), ("data", "max_train_samples", -1),
    ("model", "dtype", "garbage"), ("model", "min_pixels", 100000),
    ("wandb", "mode", "oops"), ("wandb", "raw_examples", []),
    ("lora", "target_modules", ["visual"]), ("lora", "dropout", 1),
    ("runtime", "device", "tpu"),
])
def test_shared_validation_precedes_model_loading(local_runner, section, key, value):
    local_runner.config.setdefault(section, {})[key] = value
    with pytest.raises(ValueError):
        joint.run_joint_training(local_runner.config)
    local_runner.config_loader.assert_not_called()
    local_runner.processor_loader.assert_not_called()
    local_runner.model_loader.assert_not_called()


@pytest.mark.parametrize('config', [None, [], {'unknown': {}}, {'data': []}])
def test_reject_invalid_config_structure(config):
    with pytest.raises(ValueError):
        training_config.resolve_config(config)


def test_training_charts_separate_history_from_final_summaries():
    assert joint.training_chart_metrics({
        'loss': .4, 'rationale_loss': .2, 'score_loss': .1, 'learning_rate': .001,
        'eval_loss': .5, 'eval_mae': .8, 'eval_rmse': 1.1, 'train_loss': .35,
        'train_runtime': 100, 'total_flos': 9000, 'eval_samples': 26,
        'eval_steps_per_second': .2, 'rationale_samples': 205,
    }) == {'train/loss': .4, 'train/reasoning_loss': .2, 'train/score_loss': .1,
           'train/learning_rate': .001, 'validation/loss': .5,
           'validation/mae': .8, 'validation/rmse': 1.1}


def test_validation_components_and_coverage_are_scalar_charts():
    assert joint.training_chart_metrics({
        'eval_score_loss': .01, 'eval_rationale_loss': 2.5,
        'eval_rationale_coverage': .8, 'rationale_coverage': .25,
        'eval_reasoning': 'private', 'eval_rationale_tokens': 12,
    }) == {'validation/score_loss': .01, 'validation/reasoning_loss': 2.5,
           'validation/reasoning_coverage': .8, 'train/reasoning_coverage': .25}


def test_validation_residuals_reuse_ordered_loop_predictions(tmp_path, monkeypatch):
    from torch.utils.data import DataLoader
    from transformers.trainer_utils import EvalLoopOutput
    from multimodal_judge.joint_data import JointScoreDataset

    source = tmp_path / 'validation.jsonl'
    source.write_text('{"text":"private one"}\n{"text":"private two"}\n')
    dataset = JointScoreDataset(source)
    from accelerate import Accelerator
    loader = Accelerator(cpu=True).prepare_data_loader(DataLoader(dataset))
    trainer = object.__new__(joint.JointTrainer)
    trainer.args = SimpleNamespace(output_dir=str(tmp_path / 'run'), world_size=1)
    trainer.state = SimpleNamespace(global_step=100)
    monkeypatch.setattr(trainer, 'is_world_process_zero', lambda: True)
    output = EvalLoopOutput(predictions=np.array([[8.], [2.]]), label_ids=np.array([5., 3.]),
                           metrics={}, num_samples=2)
    parent_loop = Mock(return_value=output)
    monkeypatch.setattr(joint.Trainer, 'evaluation_loop', parent_loop)
    for _ in range(2):
        assert trainer.evaluation_loop(loader, description='Evaluation') is output
    assert parent_loop.call_count == 2
    files = list((tmp_path / 'run' / 'validation').glob('*.jsonl'))
    assert len(files) == 2  # Final evaluation can repeat the last scheduled step.
    rows = [json.loads(line) for line in files[0].read_text().splitlines()]
    assert rows == [dict(row_index=0, prediction=8., target=5., signed_error=3.,
                         absolute_error=3., squared_error=9.),
                    dict(row_index=1, prediction=2., target=3., signed_error=-1.,
                         absolute_error=1., squared_error=1.)]
    summaries = [json.loads(path.read_text()) for path in files[0].parent.glob('*.summary.json')]
    assert summaries[0]['mae'] == 2.
    assert summaries[0]['rmse'] == pytest.approx(5 ** .5)
    assert summaries[0]['dataset_fingerprint'] == summaries[1]['dataset_fingerprint']
    assert 'private' not in files[0].read_text()
    trainer.evaluation_loop(loader, description='Prediction', metric_key_prefix='test')
    parent_loop.return_value = output._replace(predictions=None, label_ids=None)
    trainer.evaluation_loop(loader, description='Evaluation')
    parent_loop.return_value = output
    trainer.evaluation_loop(DataLoader(dataset, shuffle=True), description='Evaluation')
    assert len(list(files[0].parent.glob('*.jsonl'))) == 2
    parent_loop.return_value = output._replace(predictions=np.array([[1.]]))
    with pytest.raises(ValueError, match='align'):
        trainer.evaluation_loop(loader, description='Evaluation')
