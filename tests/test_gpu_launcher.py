import argparse
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('gpu_launcher', Path(__file__).resolve().parents[1] / 'docker/gpu.py')
gpu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gpu)


def args(action, tmp_path):
    return argparse.Namespace(action=action, image='private/image@sha256:example', output=tmp_path,
                              env_file=None, run_name='test-run', epochs=10, score_only=True)


def test_pull_uses_explicit_platform_and_image(tmp_path):
    assert gpu.command(args('pull', tmp_path)) == ['docker', 'pull', '--platform', 'linux/amd64', 'private/image@sha256:example']


def test_train_persists_outputs_and_applies_experiment(tmp_path):
    cmd = gpu.command(args('train', tmp_path))
    assert '--gpus' in cmd and 'all' in cmd
    assert f'type=bind,src={tmp_path},dst=/app/artifacts' in cmd
    assert 'training.num_train_epochs=10' in cmd
    assert 'objective.rationale_weight=0' in cmd
    assert '/app/artifacts/training/test-run' in cmd
    assert cmd[cmd.index('--entrypoint') + 1] == 'multimodal-judge'
    assert cmd[cmd.index('--wandb-mode') + 1] == 'online'


def test_evaluation_uses_validation_and_same_checkpoint(tmp_path):
    cmd = gpu.command(args('evaluate', tmp_path))
    assert cmd[cmd.index('--split') + 1] == 'validation'
    assert cmd[cmd.index('--checkpoint') + 1] == '/app/artifacts/training/test-run'


def test_structural_check_needs_no_gpu(tmp_path):
    cmd = gpu.command(args('check', tmp_path))
    assert '--gpus' not in cmd
    assert 'inspect-data' in cmd
