"""Checkpoint evaluation with lazy images, local predictions and aggregate W&B logging."""
import gc
import hashlib
import json
import shutil
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from .evaluation_metrics import parse_base_json, rating_metrics, summarize


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temporary.replace(path)


def file_hash(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def log_evaluation(report_path, mode='online', project='multimodal-judge', entity=None):
    """Publish only aggregate metrics and provenance; never upload local examples."""
    path = Path(report_path)
    report = json.loads(path.read_text())
    if mode == 'disabled':
        return None
    if mode not in ('online', 'offline'):
        raise ValueError('Invalid W&B mode')
    if report.get('wandb_url'):
        raise ValueError('This evaluation already has a W&B run; refusing a duplicate upload')
    import wandb

    metadata = {'split': report['split']}
    run = wandb.init(project=project, entity=entity, mode=mode, job_type='evaluation',
                     name=path.parent.name, config=metadata, dir=str(path.parent),
                     settings=wandb.Settings(disable_code=True, disable_git=True, console='off'))
    exit_code = 1
    try:
        values = {f'evaluation/{method}/{key}': value
                  for method, summary in report['metrics'].items()
                  for key, value in summary.items() if type(value) in (int, float)}
        for method in report['methods']:
            for score in range(10):
                subset = [row for row in report['rows'] if row['target'] == score]
                if subset:
                    values.update({f'evaluation/{method}/score_{score}/{key}': value
                                   for key, value in summarize(subset, method).items()})
        for method, summary in report.get('reasoning_evaluation', {}).get('metrics', {}).items():
            values.update({f'reasoning/{method}/{key}': value for key, value in summary.items()})
        run.log(values)
        run.summary.update(values)
        report['wandb_url'] = run.url if mode == 'online' else None
        report['wandb_run_id'] = run.id
        report['wandb_mode'] = mode
        write_json(path, report)
        print(f'W&B evaluation: {run.url if mode == "online" else "offline run " + run.id}',
              flush=True)
        exit_code = 0
    finally:
        run.finish(exit_code=exit_code)
    return report['wandb_url']


def _base_prompt(saved):
    marker = ' Complete the reasoning for the supplied rating;'
    if marker in saved:
        saved = saved.split(marker)[0]
    return saved + ('\n请自行给出 0–9 分的 rating，并用简体中文写简短的 reasoning。'
                    '只输出一个合法 JSON 对象，恰好包含 rating（数字）和 reasoning（字符串）。'
                    '先写 rating，再写 reasoning；不要 Markdown 或 JSON 之外的文字。')


def _base_predict(model, processor, image, text, system, config, device, limit):
    import torch
    from .joint_data import joint_messages

    prompt = processor.apply_chat_template(joint_messages(text, system),
                                           tokenize=False, add_generation_prompt=True)
    batch = processor(text=[prompt], images=[image], return_tensors='pt', truncation=False)
    length = batch['input_ids'].shape[-1]
    if length > config['data']['max_length']:
        raise ValueError('Evaluation prompt exceeds saved max_length')
    batch = {key: value.to(device) for key, value in batch.items()}
    with torch.inference_mode():
        generated = model.generate(**batch, do_sample=False, num_beams=1,
                                   max_new_tokens=limit, use_cache=True)
    tokens = generated[0, length:].tolist()
    raw = processor.tokenizer.decode(tokens, skip_special_tokens=True)
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else (eos or [])
    return {**parse_base_json(raw), 'raw_output': raw, 'tokens': len(tokens),
            'hit_token_limit': len(tokens) >= limit and (not tokens or tokens[-1] not in eos)}


def run_evaluation(checkpoint, data_dir, output_dir, split='test', device='auto',
                   include_base=False, max_samples=None, max_new_tokens=None,
                   wandb_mode='offline', wandb_project='multimodal-judge', wandb_entity=None,
                   training_run_url=None, reasoning_rubric=None):
    from tqdm import tqdm
    from .joint_data import JointScoreDataset
    from .joint_inference import load_joint_checkpoint
    from .joint_training import predict_joint

    if split not in ('test', 'validation'):
        raise ValueError('Evaluation split must be test or validation')
    for name, value in [('max_samples', max_samples), ('max_new_tokens', max_new_tokens)]:
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError(f'{name} must be a positive integer')
    if reasoning_rubric is not None:
        from .reasoning_evaluation import load_rubric
        load_rubric(reasoning_rubric)
    checkpoint, data_dir, output = map(lambda p: Path(p).resolve(),
                                      (checkpoint, data_dir, output_dir))
    manifest = json.loads((checkpoint / 'joint_manifest.json').read_text())
    config = manifest['resolved_config']
    dataset_path = data_dir / f'{split}.jsonl'
    dataset = JointScoreDataset(dataset_path, max_samples=max_samples)
    if not len(dataset):
        raise ValueError('Evaluation split is empty')
    train_scores = []
    with (data_dir / 'train.jsonl').open() as handle:
        for line in handle:
            if line.strip():
                score = json.loads(line)['score']
                if type(score) is not int or not 0 <= score <= 9:
                    raise ValueError('Training baseline labels must be integers from 0 to 9')
                train_scores.append(score)
    if not train_scores:
        raise ValueError('Training labels are empty; cannot fit constant baselines')
    constants = {'train_median': statistics.median(train_scores),
                 'train_mean': statistics.mean(train_scores)}
    output.mkdir(parents=True, exist_ok=False)
    (output / 'images').mkdir()
    weights = sorted(checkpoint.glob('*.safetensors'))
    if not weights:
        raise ValueError('No checkpoint safetensors found')
    fingerprint = hashlib.sha256()
    processor_files = [checkpoint / name for name in ('preprocessor_config.json',
                       'tokenizer_config.json', 'tokenizer.json', 'chat_template.jinja')
                       if (checkpoint / name).is_file()]
    for path in [checkpoint / 'joint_manifest.json', *weights, *processor_files]:
        fingerprint.update(path.name.encode())
        fingerprint.update(file_hash(path).encode())
    readback = checkpoint / 'wandb-readback.json'
    if not training_run_url and readback.is_file():
        training_run_url = json.loads(readback.read_text()).get('url')
    limit = max_new_tokens or config['training']['max_new_tokens']
    report = {'schema_version': 1, 'created_at': datetime.now(timezone.utc).isoformat(),
              'checkpoint': str(checkpoint), 'checkpoint_sha256': fingerprint.hexdigest(),
              'dataset_sha256': {split: file_hash(dataset_path),
                                 'train': file_hash(data_dir / 'train.jsonl')},
              'split': split, 'model': {'name': manifest['base_model_name_or_path'],
                                       'revision': manifest.get('base_model_revision')},
              'training_wandb_url': training_run_url or manifest.get('training_wandb_url'),
              'settings': {'device': device, 'dtype': config['model']['dtype'],
                           'max_new_tokens': limit, 'max_samples': max_samples,
                           'include_base': include_base},
              'methods': ['joint', 'train_median', 'train_mean'], 'metrics': {}, 'rows': []}
    write_json(output / 'progress.json', {'status': 'loading', 'completed': 0,
                                         'total': len(dataset) * (2 if include_base else 1)})
    completed = 0
    try:
        for method in ['joint', *(['base'] if include_base else [])]:
            if method == 'joint':
                model, processor, device, config = load_joint_checkpoint(checkpoint, device)
                report['settings']['device'] = device
            else:
                import torch
                from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

                processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=False)
                model = Qwen3VLForConditionalGeneration.from_pretrained(
                    manifest['base_model_name_or_path'], revision=manifest.get('base_model_revision'),
                    dtype=getattr(torch, config['model']['dtype']), trust_remote_code=False,
                    attn_implementation=config['model']['attn_implementation']).to(device).eval()
                report['methods'].append('base')
                report['base_output_policy'] = 'zh-json-v1'
                report['base_system_prompt'] = _base_prompt(config.get('prompt', {}).get('system', ''))
            for i in tqdm(range(len(dataset)), desc=f'{split}/{method}'):
                row = dataset[i]
                started = time.perf_counter()
                if method == 'joint':
                    prediction = predict_joint(model, processor, row['image'], row['text'],
                        config['data']['max_length'], limit, device,
                        system_prompt=config.get('prompt', {}).get('system', ''),
                        return_metadata=True)
                    prediction['rating'] = prediction.pop('score')
                    with dataset.path.open('rb') as handle:
                        handle.seek(dataset.offsets[i])
                        raw = json.loads(handle.readline())
                    source = Path(raw['image'])
                    source = source if source.is_absolute() else dataset.path.parent / source
                    image_path = Path('images') / f'{i}{source.suffix.lower()}'
                    shutil.copyfile(source, output / image_path)
                    report['rows'].append({'index': i, 'id': row.get('id', str(i)),
                        'text': row['text'], 'image': str(image_path),
                        'image_sha256': file_hash(source), 'target': row['score'],
                        'target_reasoning': row['reasoning'], 'predictions': {
                            name: {'rating': value} for name, value in constants.items()}})
                else:
                    prediction = _base_predict(model, processor, row['image'], row['text'],
                        report['base_system_prompt'], config, device, limit)
                prediction['seconds'] = time.perf_counter() - started
                report['rows'][i]['predictions'][method] = prediction
                with (output / 'predictions.jsonl').open('a') as handle:
                    handle.write(json.dumps({'index': i, 'method': method, **prediction},
                                            ensure_ascii=False, allow_nan=False) + '\n')
                completed += 1
                write_json(output / 'progress.json', {'status': 'running', 'method': method,
                    'completed': completed, 'total': len(dataset) * (2 if include_base else 1)})
            del model, processor
            gc.collect()
            if device == 'mps':
                import torch
                torch.mps.empty_cache()
        report['dataset_sha256']['evaluated_images'] = hashlib.sha256(
            ''.join(row['image_sha256'] for row in report['rows']).encode()).hexdigest()
        report['metrics'] = {method: summarize(report['rows'], method)
                             for method in report['methods']}
        if include_base:
            valid = [row for row in report['rows'] if row['predictions']['base']['rating'] is not None]
            if valid:
                for method in ['joint', 'train_median']:
                    report['metrics'][method + '_base_valid'] = rating_metrics(
                        [row['target'] for row in valid],
                        [row['predictions'][method]['rating'] for row in valid])
        if reasoning_rubric is not None:
            from .reasoning_evaluation import prepare_reasoning_reviews
            prepare_reasoning_reviews(report, reasoning_rubric, output)
        write_json(output / 'report.json', report)
        log_evaluation(output / 'report.json', wandb_mode, wandb_project, wandb_entity)
        write_json(output / 'progress.json', {'status': 'finished', 'completed': completed,
                                              'total': completed})
        return json.loads((output / 'report.json').read_text())
    except Exception as exc:
        write_json(output / 'progress.json', {'status': 'failed', 'completed': completed,
                                              'error': str(exc)})
        raise
