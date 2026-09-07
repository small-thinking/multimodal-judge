"""Rubric contracts: no fabricated placeholder scores or mismatched review joins."""
import json
from pathlib import Path

import pytest

from multimodal_judge.evaluation import _base_prompt
from multimodal_judge.reasoning_evaluation import (
    import_reasoning_reviews, load_rubric, prepare_reasoning_reviews,
)

RUBRIC = Path('configs/rubrics/reasoning-v1.yaml')


def test_base_prompt_requests_chinese_json_and_keeps_rubric():
    prompt = _base_prompt('Custom criterion. Complete the reasoning for the supplied rating; app.')
    assert 'Custom criterion.' in prompt and '简体中文' in prompt
    assert 'JSON' in prompt and 'rating' in prompt and 'reasoning' in prompt
    assert 'supplied rating' not in prompt


@pytest.fixture
def prepared(tmp_path):
    (tmp_path / 'images').mkdir()
    report = {'rows': [{'index': 0, 'id': 'a', 'image': 'images/a.png',
        'image_sha256': 'hash', 'text': 'Input title', 'target': 7,
        'target_reasoning': 'SECRET_GOLD', 'predictions': {
            'joint': {'rating': 8, 'reasoning': 'Candidate'},
            'base': {'rating': None, 'reasoning': None}}}], 'methods': ['joint', 'base']}
    prepare_reasoning_reviews(report, RUBRIC, tmp_path)
    (tmp_path / 'report.json').write_text(json.dumps(report))
    task = json.loads((tmp_path / 'reasoning-requests.jsonl').read_text())
    return tmp_path, report, task


def test_placeholder_is_pending_and_judge_inputs_exclude_labels(prepared):
    directory, report, task = prepared
    metrics = report['reasoning_evaluation']['metrics']
    assert metrics['joint']['pending_count'] == 1 and metrics['joint']['scored_count'] == 0
    assert 'visual_accuracy_mean' not in metrics['joint']
    assert metrics['base']['unavailable_count'] == 1
    assert 'SECRET_GOLD' not in json.dumps(task)
    assert 'target' not in task and 'method' not in task
    assert all(v['rating'] is None for v in task['response']['dimensions'].values())
    assert load_rubric(RUBRIC)['sha256'] == task['rubric_sha256']


def test_import_reviews_joins_by_id_and_preserves_source(prepared):
    directory, report, task = prepared
    for value in task['response']['dimensions'].values():
        value.update(rating=2, reasoning='Some support, but incomplete.')
    scores = directory / 'scores.jsonl'
    scores.write_text(json.dumps(task))
    before = (directory / 'report.json').read_bytes()
    result = import_reasoning_reviews(directory / 'report.json', scores, directory / 'reviewed',
                                      'human-reviewer-v1')
    assert result['reasoning_evaluation']['metrics']['joint']['visual_accuracy_mean'] == 2
    assert result['rows'][0]['predictions']['joint']['rating'] == 8
    assert (directory / 'report.json').read_bytes() == before
    assert result['reasoning_evaluation']['reviewer'] == 'human-reviewer-v1'
    assert (directory / 'reviewed/reasoning-requests.jsonl').is_file()


@pytest.mark.parametrize('change', ['id', 'hash', 'bool', 'range', 'empty_reason', 'duplicate'])
def test_invalid_or_mismatched_reviews_fail_before_writing(prepared, change):
    directory, report, task = prepared
    for value in task['response']['dimensions'].values():
        value.update(rating=3, reasoning='Supported by the image.')
    if change == 'id':
        task['review_id'] = 'unknown'
    if change == 'hash':
        task['rubric_sha256'] = 'different'
    if change == 'bool':
        task['response']['dimensions']['visual_accuracy']['rating'] = True
    if change == 'range':
        task['response']['dimensions']['visual_accuracy']['rating'] = 4
    if change == 'empty_reason':
        task['response']['dimensions']['visual_accuracy']['reasoning'] = ''
    scores = directory / 'scores.jsonl'
    scores.write_text((json.dumps(task) + '\n') * (2 if change == 'duplicate' else 1))
    with pytest.raises(ValueError):
        import_reasoning_reviews(directory / 'report.json', scores, directory / 'bad', 'reviewer')
    assert not (directory / 'bad').exists()


def test_unscorable_is_not_a_low_score(prepared):
    directory, report, task = prepared
    task['response'] = {'status': 'unscorable', 'reason': 'Image unavailable to reviewer.'}
    scores = directory / 'scores.jsonl'
    scores.write_text(json.dumps(task))
    result = import_reasoning_reviews(directory / 'report.json', scores, directory / 'reviewed', 'judge')
    metrics = result['reasoning_evaluation']['metrics']['joint']
    assert metrics['unscorable_count'] == 1 and metrics['scored_count'] == 0
    assert 'visual_accuracy_mean' not in metrics
