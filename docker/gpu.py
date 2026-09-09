#!/usr/bin/env python3
"""Pull and run the private, data-bundled CUDA image on a rented GPU host."""
import argparse
from pathlib import Path
import subprocess
import sys

IMAGE = 'smallthinking/multimodal-judge:cuda-v6-huber-5ep-20260908'


def command(args):
    if args.action == 'pull':
        return ['docker', 'pull', '--platform', 'linux/amd64', args.image]
    output = args.output.resolve()
    base = ['docker', 'run', '--rm', '--init', '--platform', 'linux/amd64', '--shm-size=2g']
    if args.action != 'check':
        base += ['--gpus', 'all']
    base += ['--mount', f'type=bind,src={output},dst=/app/artifacts',
             '-v', 'multimodal-judge-model-cache:/app/cache']
    if args.env_file:
        base += ['--env-file', str(args.env_file.resolve())]
    if args.action == 'gpu-check':
        return base + ['--entrypoint', 'python', args.image, '-c',
                       'import torch; assert torch.cuda.is_available(), "CUDA GPU unavailable"; '
                       'print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0)); '
                       'x=torch.ones((32,32),device="cuda",requires_grad=True); '
                       '(x@x).sum().backward(); torch.cuda.synchronize(); print("CUDA backward passed")']
    base += [args.image]
    if args.action == 'check':
        return base + ['inspect-data', '--config', '/app/training.yaml', '--data-dir', '/app/dataset']
    if args.action == 'train':
        return base + ['train-joint', '--config', '/app/training.yaml', '--data-dir', '/app/dataset',
                       '--device', 'cuda', '--wandb-mode', 'offline',
                       '--output-dir', f'/app/artifacts/training/{args.run_name}',
                       '--set', f'training.num_train_epochs={args.epochs}',
                       '--set', f'objective.rationale_weight={0 if args.score_only else 0.1}']
    return base + ['evaluate', '--checkpoint', f'/app/artifacts/training/{args.run_name}',
                   '--data-dir', '/app/dataset', '--split', 'validation', '--device', 'cuda',
                   '--wandb-mode', 'offline', '--output-dir', f'/app/artifacts/evaluation/{args.run_name}']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['pull', 'check', 'gpu-check', 'train', 'evaluate'])
    parser.add_argument('--image', default=IMAGE, help='Private tag or immutable digest reference')
    parser.add_argument('--output', type=Path, default=Path('artifacts/gpu'))
    parser.add_argument('--epochs', type=int, choices=[5, 10], default=5)
    parser.add_argument('--score-only', action='store_true')
    parser.add_argument('--run-name', default='v6-joint-huber-5ep')
    parser.add_argument('--env-file', type=Path, help='Docker-format runtime credentials; never uploaded')
    args = parser.parse_args()
    if not args.run_name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in args.run_name):
        parser.error('--run-name must contain only letters, digits, hyphens and underscores')
    if args.env_file and not args.env_file.is_file():
        parser.error('--env-file does not exist')
    if args.action == 'train' and (args.output / 'training' / args.run_name).exists():
        parser.error('Training output already exists; choose a different --run-name')
    if args.action == 'evaluate' and not (args.output / 'training' / args.run_name / 'resolved_config.json').is_file():
        parser.error('Checkpoint not found under --output/training/--run-name')
    if args.action != 'pull':
        args.output.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(command(args), check=True)
    except FileNotFoundError:
        parser.exit(1, 'Docker CLI not found. Use a host with Docker and NVIDIA Container Toolkit installed.\n')
    except subprocess.CalledProcessError as error:
        parser.exit(error.returncode, 'Docker command failed. For private image access, run docker login first.\n')


if __name__ == '__main__':
    main()
