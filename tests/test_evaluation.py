"""Evaluation contracts: error units, coverage, provenance and aggregate-only logging."""
import json
import math
from unittest.mock import Mock

import pytest
from PIL import Image

from multimodal_judge import evaluation
from multimodal_judge.evaluation_metrics import parse_base_json, rating_metrics, summarize


def test_rmse_penalizes_a_single_outlier():
    result = rating_metrics([0] * 10, [0] * 9 + [3])
    assert result['mae'] == .3
    assert result['rmse'] == pytest.approx(math.sqrt(.9))
    assert result['bias'] == .3
    assert result['within_one'] == .9
    assert result['p90_absolute_error'] == pytest.approx(.3)


def test_rounding_and_continuous_tolerance_are_distinct():
    result = rating_metrics([5, 6, 0], [4.5, 4.99, .5])
    assert result['rounded_accuracy'] == 1 / 3
    assert result['within_one'] == 2 / 3


@pytest.mark.parametrize('value', [True, float('nan'), float('inf'), -1, 10])
def test_invalid_scores_are_rejected(value):
    with pytest.raises(ValueError):
        rating_metrics([4], [value])


@pytest.mark.parametrize('text', ['Score: 5', '{"rating": true, "reasoning": "x"}',
                                 '{"rating": 5, "reasoning": "cut',
                                 '{"rating": NaN, "reasoning": "x"}'])
def test_invalid_base_json_is_not_zero(text):
    assert parse_base_json(text)['rating'] is None


def test_base_zero_is_a_valid_prediction():
    result = parse_base_json('```json\n{"rating":0,"reasoning":"zero"}\n```')
    assert result['rating'] == 0 and result['parse_error'] is None
    rows = [{'target': 4, 'predictions': {'base': result}},
            {'target': 9, 'predictions': {'base': {'rating': None}}}]
    summary = summarize(rows, 'base')
    assert summary['valid_count'] == 1 and summary['invalid_count'] == 1
    assert summary['mae'] == 4


@pytest.fixture
def local_evaluation(tmp_path, monkeypatch):
    from multimodal_judge import joint_inference, joint_training

    data, checkpoint = tmp_path / 'data', tmp_path / 'checkpoint'
    data.mkdir()
    checkpoint.mkdir()
    Image.new('RGB', (4, 4), 'blue').save(data / 'image.png')
    for split, labels in [('train', [2, 4]), ('test', [0, 8])]:
        records = [{'id': str(i), 'image': 'image.png', 'text': 'PRIVATE_TITLE',
                    'reasoning': 'PRIVATE_REASON', 'score': score} for i, score in enumerate(labels)]
        (data / f'{split}.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in records))
    config = {'prompt': {'system': 'Private rubric'}, 'model': {'dtype': 'float32'},
              'data': {'max_length': 100}, 'training': {'max_new_tokens': 8}}
    (checkpoint / 'joint_manifest.json').write_text(json.dumps({
        'base_model_name_or_path': 'test/base', 'base_model_revision': 'revision',
        'resolved_config': config, 'training_wandb_url': 'https://wandb.ai/test/project/runs/train'}))
    (checkpoint / 'adapter_model.safetensors').write_bytes(b'synthetic weights')
    monkeypatch.setattr(joint_inference, 'load_joint_checkpoint',
                        Mock(return_value=(object(), object(), 'cpu', config)))
    predict = Mock(side_effect=[{'score': 1., 'reasoning': 'PREDICTED_PRIVATE', 'tokens': 1,
                                'hit_token_limit': False},
                               {'score': 6., 'reasoning': 'other', 'tokens': 2,
                                'hit_token_limit': True}])
    monkeypatch.setattr(joint_training, 'predict_joint', predict)
    return checkpoint, data, tmp_path / 'output', predict


def test_pipeline_uses_train_baseline_and_preserves_examples(local_evaluation):
    checkpoint, data, output, predict = local_evaluation
    result = evaluation.run_evaluation(checkpoint, data, output, device='cpu', wandb_mode='disabled')
    assert result['metrics']['joint']['mae'] == 1.5
    assert result['metrics']['joint']['rmse'] == pytest.approx(math.sqrt(2.5))
    assert result['rows'][0]['predictions']['train_median']['rating'] == 3
    assert result['metrics']['joint']['token_limit_count'] == 1
    assert result['dataset_sha256']['test'] == evaluation.file_hash(data / 'test.jsonl')
    assert (output / result['rows'][0]['image']).read_bytes() == (data / 'image.png').read_bytes()
    assert all(call.args[3] == 'PRIVATE_TITLE' for call in predict.call_args_list)
    assert all('target' not in call.kwargs for call in predict.call_args_list)
    assert json.loads((output / 'progress.json').read_text())['status'] == 'finished'
    with pytest.raises(FileExistsError):
        evaluation.run_evaluation(checkpoint, data, output, wandb_mode='disabled')


def test_prediction_failure_is_visible(local_evaluation):
    checkpoint, data, output, predict = local_evaluation
    predict.side_effect = RuntimeError('synthetic inference failure')
    with pytest.raises(RuntimeError, match='synthetic inference'):
        evaluation.run_evaluation(checkpoint, data, output, wandb_mode='disabled')
    assert json.loads((output / 'progress.json').read_text())['status'] == 'failed'
    assert not (output / 'report.json').exists()


def test_wandb_logs_aggregates_without_examples(local_evaluation, monkeypatch):
    import wandb

    checkpoint, data, output, _ = local_evaluation
    evaluation.run_evaluation(checkpoint, data, output, device='cpu', wandb_mode='disabled')
    run = Mock(url='https://wandb.ai/test/project/runs/eval', id='eval')
    init = Mock(return_value=run)
    monkeypatch.setattr(wandb, 'init', init)
    url = evaluation.log_evaluation(output / 'report.json')
    assert url == run.url and init.call_args.kwargs['job_type'] == 'evaluation'
    assert init.call_args.kwargs['project'] == 'multimodal-judge'
    payload = json.dumps(init.call_args.kwargs['config']) + json.dumps(run.log.call_args.args[0])
    assert all(secret not in payload for secret in ['PRIVATE_TITLE', 'PRIVATE_REASON',
                                                   'PREDICTED_PRIVATE', 'Private rubric'])
    assert run.log.call_args.args[0]['evaluation/joint/rmse'] == pytest.approx(math.sqrt(2.5))
    assert init.call_args.kwargs['config'] == {'split': 'test'}
    run.finish.assert_called_once_with(exit_code=0)
    with pytest.raises(ValueError, match='duplicate'):
        evaluation.log_evaluation(output / 'report.json')


def test_center_rejects_paths_outside_catalog(tmp_path, monkeypatch):
    from multimodal_judge.evaluation_center import EvaluationCenter

    center = EvaluationCenter(tmp_path)
    process = Mock()
    monkeypatch.setattr('multimodal_judge.evaluation_center.subprocess.Popen', process)
    with pytest.raises(ValueError, match='catalog'):
        center.launch({'checkpoint': '../../private', 'data_dir': '/tmp'})
    process.assert_not_called()


def test_center_launch_is_single_job_and_uses_argument_list(tmp_path, monkeypatch):
    from multimodal_judge.evaluation_center import EvaluationCenter

    checkpoint = tmp_path / 'artifacts/training/checkpoint'
    data = tmp_path / 'data/training_data/v1'
    checkpoint.mkdir(parents=True)
    data.mkdir(parents=True)
    (checkpoint / 'joint_manifest.json').write_text('{}')
    (data / 'train.jsonl').write_text('{}')
    process = Mock()
    process.poll.return_value = None
    spawn = Mock(return_value=process)
    monkeypatch.setattr('multimodal_judge.evaluation_center.subprocess.Popen', spawn)
    center = EvaluationCenter(tmp_path)
    options = {'checkpoint': str(checkpoint.relative_to(tmp_path)),
               'data_dir': str(data.relative_to(tmp_path)), 'split': 'test',
               'wandb_mode': 'disabled', 'include_base': True}
    result = center.launch(options)
    assert '--include-base' in spawn.call_args.args[0]
    assert isinstance(spawn.call_args.args[0], list)
    assert center.state()['progress']['id'] == result['id']
    with pytest.raises(ValueError, match='already running'):
        center.launch(options)


def test_evaluation_cli_requires_explicit_paths(monkeypatch):
    import sys
    from multimodal_judge.cli import main

    monkeypatch.setattr(sys, 'argv', ['multimodal-judge', 'evaluate'])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2


def test_evaluation_cli_forwards_options(monkeypatch, tmp_path):
    import sys
    from multimodal_judge.cli import main

    run = Mock(return_value={'metrics': {}})
    monkeypatch.setattr(evaluation, 'run_evaluation', run)
    monkeypatch.setattr(sys, 'argv', ['multimodal-judge', 'evaluate', '--checkpoint', 'model',
        '--data-dir', 'data', '--output-dir', str(tmp_path), '--split', 'validation',
        '--include-base', '--max-samples', '2', '--wandb-mode', 'disabled'])
    main()
    assert run.call_args.kwargs['split'] == 'validation'
    assert run.call_args.kwargs['include_base'] is True
    assert run.call_args.kwargs['max_samples'] == 2
    assert run.call_args.kwargs['wandb_mode'] == 'disabled'
