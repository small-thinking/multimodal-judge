"""Portable dataset checks using synthetic records only."""
import importlib.util
import json
from pathlib import Path
import shutil

import pytest

from multimodal_judge.training_data import inspect_data

spec = importlib.util.spec_from_file_location(
    'docker_train', Path(__file__).resolve().parents[1] / 'docker/train.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def source_data(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    image = tmp_path / 'external.png'
    image.write_bytes(b'synthetic-placeholder-no-decoding')
    for split in ('train', 'validation'):
        record = {'image': str(image) if split == 'train' else '../external.png',
                  'text': 'synthetic', 'score': 0, 'reasoning': 'synthetic explanation',
                  'extra_metadata': {'keep': True}}
        (source / f'{split}.jsonl').write_text(json.dumps(record) + '\n')
    return source


def test_bundle_relocates_and_preserves_records(tmp_path):
    source = source_data(tmp_path)
    destination = tmp_path / 'bundle'
    module.bundle(source, destination)
    moved = tmp_path / 'moved'
    destination.rename(moved)
    shutil.rmtree(source)
    (tmp_path / 'external.png').unlink()
    assert len(list((moved / '_portable_images').iterdir())) == 1
    record = json.loads((moved / 'train.jsonl').read_text())
    assert record['score'] == 0
    assert record['extra_metadata'] == {'keep': True}
    assert record['reasoning'] == 'synthetic explanation'
    assert not Path(record['image']).is_absolute()
    inspect_data(moved)


def test_missing_image_fails_atomically_without_private_content(tmp_path):
    source = source_data(tmp_path)
    (tmp_path / 'external.png').unlink()
    destination = tmp_path / 'bundle'
    with pytest.raises(ValueError, match='missing image') as error:
        module.bundle(source, destination)
    assert 'synthetic' not in str(error.value)
    assert not destination.exists()


def test_existing_destination_is_preserved(tmp_path):
    source = source_data(tmp_path)
    with pytest.raises(ValueError, match='already exists'):
        module.bundle(source, source)
    assert (source / 'train.jsonl').exists()


@pytest.mark.parametrize('mounted', [False, True])
def test_run_mounts_outputs_and_forwards_arguments(tmp_path, monkeypatch, mounted):
    config = tmp_path / 'train.yaml'
    config.write_text('')
    argv = ['train.py', 'run', '--config', str(config), '--output', str(tmp_path / 'outputs')]
    if mounted:
        argv += ['--data', str(tmp_path)]
    argv += ['--', '--max-steps', '2']
    monkeypatch.setattr(module.sys, 'argv', argv)
    commands = []
    monkeypatch.setattr(module.os, 'execvp', lambda executable, command: commands.append(command))
    module.main()
    command = commands[0]
    assert any('dst=/app/artifacts' in item for item in command)
    assert any('dst=/app/dataset,readonly' in item for item in command) == mounted
    assert command[-2:] == ['--max-steps', '2']
    assert '--init' in command
    assert '-d' not in command
    assert command[command.index('--wandb-mode') + 1] == 'offline'


def test_build_validates_data_before_touching_image(tmp_path, monkeypatch):
    monkeypatch.setattr(module.sys, 'argv', ['train.py', 'build', '--data', str(tmp_path)])
    commands = []
    monkeypatch.setattr(module, 'call', commands.append)
    with pytest.raises(ValueError, match='Missing required split'):
        module.main()
    assert not commands


def test_bundle_preserves_sidecars_and_auxiliary_image_paths(tmp_path):
    source = source_data(tmp_path)
    (source / 'report.json').write_text('{"synthetic_count": 2}')
    shutil.copyfile(source / 'train.jsonl', source / 'train_original.jsonl')
    destination = tmp_path / 'complete'
    module.bundle(source, destination)
    assert (destination / 'report.json').read_bytes() == (source / 'report.json').read_bytes()
    original = json.loads((destination / 'train_original.jsonl').read_text())
    assert (destination / original['image']).is_file()
    second = tmp_path / 'repacked'
    module.bundle(destination, second)
    inspect_data(second)


def test_bundle_rejects_nested_destination(tmp_path):
    source = source_data(tmp_path)
    with pytest.raises(ValueError, match='outside the source'):
        module.bundle(source, source / 'nested')


def test_env_file_is_runtime_only_and_baked_config_is_default(tmp_path, monkeypatch):
    env_file = tmp_path / 'runtime.env'
    env_file.write_text('SYNTHETIC_SECRET=never-print-this-value\n')
    monkeypatch.setattr(module.sys, 'argv', ['train.py', 'run', '--env-file', str(env_file),
                        '--hf-cache', str(tmp_path), '--output', str(tmp_path / 'out')])
    commands = []
    monkeypatch.setattr(module.os, 'execvp', lambda executable, command: commands.append(command))
    module.main()
    command = commands[0]
    assert command[command.index('--env-file') + 1] == str(env_file)
    assert 'never-print-this-value' not in ' '.join(command)
    assert not any('dst=/app/training.yaml,readonly' in part for part in command)
    assert command[command.index('--config') + 1] == '/app/training.yaml'
    assert any('dst=/app/cache/huggingface' in part for part in command)


def test_bundle_rejects_directory_symlinks_before_copy(tmp_path):
    source = source_data(tmp_path)
    outside = tmp_path / 'external_metadata'
    outside.mkdir()
    (source / 'linked_metadata').symlink_to(outside, target_is_directory=True)
    destination = tmp_path / 'bundled'
    with pytest.raises(ValueError, match='directory symlinks'):
        module.bundle(source, destination)
    assert not destination.exists()
