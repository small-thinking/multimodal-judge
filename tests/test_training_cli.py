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
