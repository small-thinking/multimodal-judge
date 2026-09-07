"""Training command routing without model downloads or dataset access."""

import json
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml

from multimodal_judge import cli, joint_training, training_data


def test_baseline_command_rejected(monkeypatch, capsys):
    monkeypatch.setattr(sys, 'argv', ['multimodal-judge', 'train'])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    assert "invalid choice: 'train'" in capsys.readouterr().err


def test_training_auto_names_are_unique_and_match_output(monkeypatch):
    run = Mock(return_value={})
    monkeypatch.setattr(joint_training, 'run_joint_training', run)
    monkeypatch.setattr(sys, 'argv', ['multimodal-judge', 'train-joint', '--data-dir',
                                    'data/training_data/v5', '--wandb-mode', 'disabled'])
    cli.main()
    cli.main()
    configs = [call.args[0] for call in run.call_args_list]
    names = [config['wandb']['name'] for config in configs]
    assert names[0] != names[1]
    for config, name in zip(configs, names):
        assert name.startswith('train-Qwen3-VL-2B-Instruct-v5-')
        assert Path(config['training']['output_dir']).name == name


def test_auto_name_accepts_partial_config(monkeypatch, tmp_path):
    path = tmp_path / 'partial.yaml'
    path.write_text('training:\n  max_steps: 2\n')
    run = Mock(return_value={})
    monkeypatch.setattr(joint_training, 'run_joint_training', run)
    monkeypatch.setattr(sys, 'argv', ['multimodal-judge', 'train-joint', '--config', str(path)])
    cli.main()
    config = run.call_args.args[0]
    assert config['wandb']['name'].startswith('train-Qwen3-VL-2B-Instruct-v2-')
    assert config['training']['max_steps'] == 2


def test_inspect_data_uses_data_config(monkeypatch, capsys):
    inspect = Mock(return_value={'splits': {}})
    monkeypatch.setattr(training_data, 'inspect_data', inspect)
    monkeypatch.setattr(sys, 'argv', ['multimodal-judge', 'inspect-data'])
    read_text = Path.read_text
    paths = []

    def read(path, *args, **kwargs):
        paths.append(path)
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'read_text', read)
    cli.main()
    assert paths == [Path('configs/data.yaml')]
    data = yaml.safe_load(read_text(paths[0]))['data']
    inspect.assert_called_once_with(data['directory'], data['train_file'], data['validation_file'])
    assert json.loads(capsys.readouterr().out) == {'splits': {}}


def test_joint_command_preserves_overrides_and_resume(monkeypatch, capsys):
    run = Mock(return_value={'train_loss': 0.25})
    monkeypatch.setattr(joint_training, 'run_joint_training', run)
    monkeypatch.setattr(sys, 'argv', [
        'multimodal-judge', 'train-joint', '--device', 'cpu', '--dtype', 'float32',
        '--model', 'local-model', '--data-dir', 'local-data', '--output-dir', 'local-output',
        '--max-steps', '2', '--wandb-mode', 'disabled', '--resume-from-checkpoint', 'checkpoint-1',
    ])
    cli.main()
    config = yaml.safe_load(Path('configs/train-joint.yaml').read_text())
    config['model'].update(name_or_path='local-model', dtype='float32')
    config['runtime']['device'] = 'cpu'
    config['data']['directory'] = 'local-data'
    config['training'].update(output_dir='local-output', max_steps=2)
    config['wandb']['mode'] = 'disabled'
    run.assert_called_once_with(config, resume_from_checkpoint='checkpoint-1')
    assert json.loads(capsys.readouterr().out) == {'train_loss': 0.25}


@pytest.mark.parametrize('head_type,weight', [('classification', '3.5'), ('regression', '0')])
def test_joint_score_objective_cli_overrides_yaml(monkeypatch, tmp_path, head_type, weight):
    path = tmp_path / 'config.yaml'
    path.write_text('objective:\n  head_type: regression\n  score_weight: 7\n  rationale_weight: 0.25\n')
    run = Mock(return_value={})
    monkeypatch.setattr(joint_training, 'run_joint_training', run)
    monkeypatch.setattr(sys, 'argv', [
        'multimodal-judge', 'train-joint', '--config', str(path),
        '--score-head', head_type, '--score-weight', weight, '--wandb-mode', 'disabled',
    ])
    cli.main()
    objective = run.call_args.args[0]['objective']
    assert objective['head_type'] == head_type
    assert objective['score_weight'] == float(weight)
    assert objective['rationale_weight'] == .25


def test_joint_score_head_cli_rejects_unknown_choice(monkeypatch, capsys):
    run = Mock()
    monkeypatch.setattr(joint_training, 'run_joint_training', run)
    monkeypatch.setattr(sys, 'argv', ['multimodal-judge', 'train-joint', '--score-head', 'ordinal'])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert 'invalid choice' in capsys.readouterr().err
    run.assert_not_called()


@pytest.mark.parametrize('flag,value', [('--score-head', 'classification'), ('--score-weight', '2')])
def test_training_objective_flags_are_not_silently_ignored_by_judge(monkeypatch, capsys, flag, value):
    monkeypatch.setattr(sys, 'argv', ['multimodal-judge', 'judge', flag, value])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert 'only supported for train-joint' in capsys.readouterr().err


@pytest.mark.parametrize('weight', ['-1', 'nan', 'inf'])
def test_joint_score_weight_cli_rejects_invalid_values(monkeypatch, weight):
    # Keep the real runner's config validation; reject before touching data/model.
    validate_paths = Mock(side_effect=AssertionError('invalid weight reached dataset access'))
    monkeypatch.setattr(joint_training, '_validate_paths', validate_paths)
    monkeypatch.setattr(sys, 'argv', ['multimodal-judge', 'train-joint', '--score-weight', weight,
                                    '--wandb-mode', 'disabled'])
    with pytest.raises(ValueError, match='score_weight'):
        cli.main()
    validate_paths.assert_not_called()

def test_judge_returns_rating_reasoning_json(monkeypatch, capsys, tmp_path):
    from PIL import Image
    from multimodal_judge import joint_inference

    image = tmp_path / 'image.png'
    Image.new('RGB', (4, 4), 'blue').save(image)
    config = {'data': {'max_length': 1024}, 'training': {'max_new_tokens': 64},
              'prompt': {'system': 'Use the supplied rubric.'}}
    monkeypatch.setattr(joint_inference, 'load_joint_checkpoint',
                        Mock(return_value=(object(), object(), 'cpu', config)))
    reason = 'A "blue" square.\nClear shape.'
    predict = Mock(return_value={'score': 6.75, 'reasoning': reason})
    monkeypatch.setattr(joint_training, 'predict_joint', predict)
    monkeypatch.setattr(sys, 'argv', ['multimodal-judge', 'judge', '--checkpoint', str(tmp_path),
                                    '--image', str(image), '--text', 'A blue square'])
    cli.main()
    assert json.loads(capsys.readouterr().out) == {'rating': 6.75, 'reasoning': reason}
    assert predict.call_args.kwargs['system_prompt'] == config['prompt']['system']
