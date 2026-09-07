"""Rubric-based reasoning reviews, independent of the model's 0–9 content rating."""
import hashlib
import json
import statistics
import shutil
from pathlib import Path

import yaml

from .evaluation import write_json


def load_rubric(path):
    raw = Path(path).read_bytes()
    rubric = yaml.safe_load(raw)
    if not isinstance(rubric, dict) or not isinstance(rubric.get('version'), str):
        raise ValueError('Rubric must have a version')
    dimensions = rubric.get('dimensions')
    if not isinstance(dimensions, dict) or not dimensions:
        raise ValueError('Rubric must define dimensions')
    for key, dimension in dimensions.items():
        if not isinstance(key, str) or not isinstance(dimension, dict):
            raise ValueError('Invalid rubric dimension')
        if (not isinstance(dimension.get('name'), str)
                or set(dimension.get('levels', {})) != {1, 2, 3}
                or any(not isinstance(v, str) or not v.strip()
                       for v in dimension['levels'].values())):
            raise ValueError('Each dimension needs a name and descriptions for levels 1, 2, 3')
    return {**rubric, 'sha256': hashlib.sha256(raw).hexdigest()}


def prepare_reasoning_reviews(report, rubric_path, directory):
    """Export local image/title/candidate inputs without labels or predictor identities."""
    rubric = load_rubric(rubric_path)
    tasks, mapping = [], {}
    for row in report['rows']:
        for method in ('joint', 'base'):
            prediction = row['predictions'].get(method)
            if not prediction:
                continue
            candidate = prediction.get('reasoning')
            status = 'pending' if isinstance(candidate, str) and candidate.strip() else 'unavailable'
            prediction['reasoning_review'] = {'status': status}
            if status == 'unavailable':
                continue
            # Bind reviews to this exact input, output and rubric; never join on row order alone.
            identity = json.dumps([row['id'], row.get('image_sha256'), row['text'], candidate,
                                   rubric['sha256'], method], ensure_ascii=False)
            review_id = hashlib.sha256(identity.encode()).hexdigest()
            mapping[review_id] = {'index': row['index'], 'method': method}
            tasks.append({'review_id': review_id, 'rubric_sha256': rubric['sha256'],
                          'image': row['image'], 'title': row['text'], 'candidate_reasoning': candidate,
                          'response': {'status': 'scored', 'dimensions': {
                              key: {'rating': None, 'reasoning': ''} for key in rubric['dimensions']}}})
    report['reasoning_evaluation'] = {'rubric': rubric, 'reviewer': None,
                                      'status': 'pending', 'mapping': mapping, 'metrics': {}}
    destination = Path(directory)
    write_json(destination / 'reasoning-rubric.json', rubric)
    # Sorting opaque identifiers reduces fixed model-order cues to a future automated judge.
    with (destination / 'reasoning-requests.jsonl').open('w') as handle:
        for task in sorted(tasks, key=lambda item: item['review_id']):
            handle.write(json.dumps(task, ensure_ascii=False, allow_nan=False) + '\n')
    summarize_reasoning(report)


def summarize_reasoning(report):
    evaluation = report['reasoning_evaluation']
    result = {}
    for method in ('joint', 'base'):
        reviews = [row['predictions'][method]['reasoning_review'] for row in report['rows']
                   if method in row['predictions']]
        if not reviews:
            continue
        scored = [review for review in reviews if review['status'] == 'scored']
        summary = {'total_count': len(reviews), 'scored_count': len(scored),
                   'coverage': len(scored) / len(reviews),
                   **{state + '_count': sum(r['status'] == state for r in reviews)
                      for state in ('pending', 'unavailable', 'unscorable')}}
        for dimension in evaluation['rubric']['dimensions']:
            if scored:
                summary[dimension + '_mean'] = statistics.mean(
                    review['dimensions'][dimension]['rating'] for review in scored)
        result[method] = summary
    evaluation['metrics'] = result
    evaluation['status'] = ('pending' if any(v['pending_count'] for v in result.values())
                            else 'complete')
    return result


def import_reasoning_reviews(source_report, scores_path, output_dir, reviewer):
    """Create a new report for human/judge results; preserve prior reports and W&B runs."""
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError('Specify the human or judge model/version as reviewer')
    source = Path(source_report).resolve()
    report = json.loads(source.read_text())
    evaluation = report['reasoning_evaluation']
    if evaluation.get('reviewer') not in (None, reviewer):
        raise ValueError('Do not mix different reviewers in one reasoning evaluation')
    seen = set()
    rows = {row['index']: row for row in report['rows']}
    for line in Path(scores_path).read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        identity = record['review_id']
        if identity not in evaluation['mapping'] or identity in seen:
            raise ValueError('Unknown or duplicate reasoning review ID')
        if record.get('rubric_sha256') != evaluation['rubric']['sha256']:
            raise ValueError('Reasoning rubric hash mismatch')
        seen.add(identity)
        response = record['response']
        validate_review(response, evaluation['rubric'])
        target = evaluation['mapping'][identity]
        rows[target['index']]['predictions'][target['method']]['reasoning_review'] = response
    if not seen:
        raise ValueError('No reasoning reviews provided')
    evaluation['reviewer'] = reviewer
    evaluation['source_report_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
    summarize_reasoning(report)
    for key in ('wandb_url', 'wandb_run_id', 'wandb_mode'):
        report.pop(key, None)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(source.parent / 'images', output / 'images')
    for name in ('reasoning-requests.jsonl', 'reasoning-rubric.json'):
        if (source.parent / name).is_file():
            shutil.copyfile(source.parent / name, output / name)
    write_json(output / 'report.json', report)
    return report


def prepare_reasoning_report(source_report, rubric_path, output_dir):
    """Attach a new rubric to saved predictions without another model inference run."""
    source = Path(source_report).resolve()
    report = json.loads(source.read_text())
    load_rubric(rubric_path)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(source.parent / 'images', output / 'images')
    prepare_reasoning_reviews(report, rubric_path, output)
    for key in ('wandb_url', 'wandb_run_id', 'wandb_mode'):
        report.pop(key, None)
    write_json(output / 'report.json', report)
    return report


def validate_review(response, rubric):
    if not isinstance(response, dict):
        raise ValueError('Reasoning response must be a JSON object')
    if response.get('status') == 'scored':
        values = response.get('dimensions', {})
        if not isinstance(values, dict) or set(values) != set(rubric['dimensions']):
            raise ValueError('Reasoning dimensions must match the rubric exactly')
        for value in values.values():
            if (not isinstance(value, dict) or type(value.get('rating')) is not int or value['rating'] not in (1, 2, 3)
                    or not isinstance(value.get('reasoning'), str) or not value['reasoning'].strip()):
                raise ValueError('Each dimension requires integer 1–3 and a nonempty explanation')
    elif (response.get('status') != 'unscorable'
          or not isinstance(response.get('reason'), str) or not response['reason'].strip()):
        raise ValueError('Expected scored review, or unscorable with a reason')
