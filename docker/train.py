#!/usr/bin/env python3
"""Build/run training images and prepare relocatable datasets (stdlib only)."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import uuid

ROOT = Path(__file__).resolve().parents[1]


def bundle(source, destination):
    """Copy split records and referenced images without printing private content."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source in destination.parents:
        raise ValueError('Bundle destination must be outside the source directory')
    if destination.exists():
        raise ValueError('Bundle destination already exists; choose a new directory')
    for name in ('train.jsonl', 'validation.jsonl'):
        if not (source / name).is_file():
            raise ValueError(f'Missing required split: {name}')
    for current, directories, _ in os.walk(source):
        if any((Path(current) / name).is_symlink() for name in directories):
            raise ValueError('Dataset directory symlinks must be materialized before bundling')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        stage = Path(temporary) / 'dataset'
        shutil.copytree(source, stage)
        (stage / '_portable_images').mkdir(exist_ok=True)
        copied = {}
        for split_path in sorted(source.rglob('*.jsonl')):
            name = str(split_path.relative_to(source))
            count = 0
            with (source / name).open() as incoming, (stage / name).open('w') as outgoing:
                for line_number, line in enumerate(incoming, 1):
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                        if isinstance(row, dict) and 'image' not in row and name not in (
                                'train.jsonl', 'validation.jsonl', 'test.jsonl'):
                            outgoing.write(line)
                            continue
                        image = Path(row['image'])
                        image = (split_path.parent / image).resolve()
                        if not image.is_file():
                            raise ValueError()
                    except (ValueError, TypeError, KeyError, OSError):
                        raise ValueError(f'{name}:{line_number}: invalid record or missing image') from None
                    if image not in copied:
                        try:
                            relative = str(image.relative_to(source))
                        except ValueError:
                            index = len(copied)
                            relative = f'_portable_images/{index:08d}{image.suffix}'
                            while (stage / relative).exists():
                                index += 1
                                relative = f'_portable_images/{index:08d}{image.suffix}'
                            shutil.copyfile(image, stage / relative)
                        copied[image] = relative
                    row['image'] = os.path.relpath(stage / copied[image], (stage / name).parent)
                    outgoing.write(json.dumps(row, ensure_ascii=False) + '\n')
                    count += 1
            if not count and name in ('train.jsonl', 'validation.jsonl'):
                raise ValueError(f'Empty required split: {name}')
        stage.rename(destination)


def call(command):
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    pack = sub.add_parser('bundle', help='Copy JSONL + images into a portable folder')
    pack.add_argument('--data', required=True, type=Path)
    pack.add_argument('--output', required=True, type=Path)
    build = sub.add_parser('build')
    build.add_argument('--target', choices=['cpu', 'cuda'], default='cpu')
    build.add_argument('--image', default='multimodal-judge:training')
    build.add_argument('--config', type=Path, help='Embed the exact local training YAML')
    build.add_argument('--include', action='append', default=[], metavar='PATH=NAME',
                       help='Copy a supporting file/folder to /app/resources/NAME')
    build.add_argument('--data', type=Path, help='Explicitly embed this dataset in the image')
    run = sub.add_parser('run')
    run.add_argument('--image', default='multimodal-judge:training')
    run.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    run.add_argument('--data', type=Path, help='Portable bundle; omit for an embedded dataset')
    run.add_argument('--config', type=Path, help='Override the image training YAML at runtime')
    run.add_argument('--output', type=Path, default=ROOT / 'artifacts/docker-training')
    run.add_argument('--env-file', type=Path, help='Docker runtime environment file; never baked')
    run.add_argument('--hf-cache', type=Path, help='Reuse an existing Hugging Face cache directory')
    run.add_argument('--cache', default='multimodal-judge-model-cache')
    run.add_argument('--check', action='store_true', help='Validate paths/schema without model loading')
    run.add_argument('training_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command == 'bundle':
        bundle(args.data, args.output)
        print('Portable dataset created.')
    elif args.command == 'build':
        with tempfile.TemporaryDirectory(prefix='judge-image-') as temporary:
            context = Path(temporary)
            if args.data:
                bundle(args.data, context / 'dataset')
            customized = bool(args.data or args.config or args.include)
            copies = 'COPY dataset /app/dataset\n' if args.data else ''
            if args.config:
                shutil.copyfile(args.config, context / 'training.yaml')
                copies += 'COPY training.yaml /app/training.yaml\n'
            for item in args.include:
                source_path, separator, name = item.rpartition('=')
                relative = Path(name)
                if (not separator or relative.is_absolute() or '..' in relative.parts
                        or not relative.parts):
                    raise ValueError('--include requires PATH=NAME with a relative destination')
                source_path = Path(source_path)
                target_path = context / 'resources' / relative
                target_path.parent.mkdir(parents=True, exist_ok=True)
                if source_path.is_dir():
                    shutil.copytree(source_path, target_path)
                else:
                    shutil.copyfile(source_path, target_path)
            if args.include:
                copies += 'COPY resources /app/resources\n'
            base_image = f'multimodal-judge:build-{uuid.uuid4().hex}' if customized else args.image
            command = ['docker', 'build', '--target', args.target, '-t', base_image]
            if args.target == 'cuda':
                command += ['--platform', 'linux/amd64']
            try:
                call(command + [str(ROOT)])
                if customized:
                    (context / 'Dockerfile').write_text(
                        'ARG BASE_IMAGE=multimodal-judge:training\nFROM ${BASE_IMAGE}\n' + copies +
                        'CMD ["train-joint", "--config", "/app/training.yaml", '
                        '"--data-dir", "/app/dataset", "--wandb-mode", "offline"]\n')
                    layer_command = ['docker', 'build']
                    if args.target == 'cuda':
                        layer_command += ['--platform', 'linux/amd64']
                    call(layer_command + ['--build-arg', f'BASE_IMAGE={base_image}',
                                          '-t', args.image, str(context)])
            finally:
                if customized:
                    subprocess.run(['docker', 'image', 'rm', base_image], check=False,
                                   stdout=subprocess.DEVNULL)
    else:
        if args.config and not args.config.is_file():
            raise ValueError('Config file does not exist')
        args.output.mkdir(parents=True, exist_ok=True)
        command = ['docker', 'run', '--rm', '--init', '--shm-size=2g']
        if sys.stdin.isatty() and sys.stdout.isatty():
            command += ['-it']
        if args.device == 'cuda':
            command += ['--gpus', 'all', '--platform', 'linux/amd64']
        if args.config:
            command += ['--mount', f'type=bind,src={args.config.resolve()},dst=/app/training.yaml,readonly']
        if args.env_file:
            if not args.env_file.is_file():
                raise ValueError('Environment file does not exist')
            command += ['--env-file', str(args.env_file.resolve())]
        if args.hf_cache:
            if not args.hf_cache.is_dir():
                raise ValueError('Hugging Face cache directory does not exist')
            command += ['--mount',
                        f'type=bind,src={args.hf_cache.resolve()},dst=/app/cache/huggingface']
        command += ['--mount', f'type=bind,src={args.output.resolve()},dst=/app/artifacts',
                    '-v', f'{args.cache}:/app/cache']
        for key in ('HF_TOKEN', 'WANDB_API_KEY'):
            if key in os.environ:
                command += ['-e', key]
        if args.data:
            if not args.data.is_dir():
                raise ValueError('Dataset directory does not exist; run bundle first')
            command += ['--mount', f'type=bind,src={args.data.resolve()},dst=/app/dataset,readonly']
        extra = args.training_args
        if extra[:1] == ['--']:
            extra = extra[1:]
        command += [args.image, 'inspect-data' if args.check else 'train-joint',
                    '--config', '/app/training.yaml', '--data-dir', '/app/dataset',
                    '--device', args.device, '--wandb-mode', 'offline']
        if args.device == 'cpu' and not args.check:
            command += ['--dtype', 'float32']
        # Replace the wrapper so Ctrl-C and the container exit status propagate directly.
        os.execvp(command[0], command + extra)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, subprocess.CalledProcessError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
