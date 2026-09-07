import copy
import json
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import pytest
import requests

from multimodal_judge import llm_judge as judge


@pytest.fixture
def sample(tmp_path, monkeypatch):
    image = tmp_path / 'image.png'
    image.write_bytes(b'original-image')
    rubric = {'dimensions': {'accuracy': {}}}
    payload = judge.judge_request(image, 'title', 'candidate', rubric, 'grok-4.6', 'low')
    review = {'status': 'scored', 'dimensions': {'accuracy': {'rating': 2, 'reasoning': 'Evidence'}}}
    response = Mock()
    response.json.return_value = {'choices': [{'finish_reason': 'stop',
                                               'message': {'content': json.dumps(review)}}]}
    post = Mock(return_value=response)
    monkeypatch.setattr(judge.requests, 'post', post)
    monkeypatch.setenv('XAI_API_KEY', 'test-only')
    return payload, rubric, tmp_path / 'cache', post, review


def test_exact_hit_needs_no_key_or_network(sample, monkeypatch):
    payload, rubric, cache, post, review = sample
    assert judge.cached_review(payload, rubric, cache) == (review, False)
    monkeypatch.delenv('XAI_API_KEY')
    assert judge.cached_review(payload, rubric, cache) == (review, True)
    assert post.call_count == 1
    assert not any('test-only' in p.read_text() for p in cache.glob('*.json'))


@pytest.mark.parametrize('change', ['model', 'effort', 'tokens', 'title', 'candidate', 'rubric', 'image'])
def test_changed_request_misses_cache(sample, change):
    payload, rubric, cache, post, _ = sample
    judge.cached_review(payload, rubric, cache)
    changed = copy.deepcopy(payload)
    if change in ('model', 'effort', 'tokens'):
        key = {'model': 'model', 'effort': 'reasoning_effort', 'tokens': 'max_tokens'}[change]
        changed[key] = 'changed'
    elif change == 'image':
        changed['messages'][1]['content'][1]['image_url']['url'] += 'changed'
    else:
        content = changed['messages'][1]['content'][0]
        text = json.loads(content['text'])
        text[{'candidate': 'candidate_reasoning'}.get(change, change)] = 'changed'
        content['text'] = json.dumps(text)
    assert judge.cached_review(changed, rubric, cache)[1] is False
    assert post.call_count == 2


@pytest.mark.parametrize('failure', ['http', 'truncated', 'invalid_json', 'invalid_rating'])
def test_failed_results_are_not_cached(sample, failure):
    payload, rubric, cache, post, _ = sample
    response = post.return_value
    choice = response.json.return_value['choices'][0]
    if failure == 'http':
        response.raise_for_status.side_effect = requests.HTTPError('failure')
    elif failure == 'truncated':
        choice['finish_reason'] = 'length'
    elif failure == 'invalid_json':
        choice['message']['content'] = '{'
    else:
        choice['message']['content'] = json.dumps({'status': 'scored', 'dimensions': {
            'accuracy': {'rating': True, 'reasoning': 'bad'}}})
    with pytest.raises((ValueError, requests.HTTPError)):
        judge.cached_review(payload, rubric, cache)
    assert not list(cache.glob('*.json'))


def test_concurrent_identical_requests_only_call_once(sample):
    payload, rubric, cache, post, _ = sample
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(lambda _: judge.cached_review(payload, rubric, cache), range(2)))
    assert sorted(hit for _, hit in results) == [False, True]
    assert post.call_count == 1


def test_cross_predictor_reuse_and_report_persistence(sample, tmp_path):
    _, rubric, cache, post, review = sample
    report = {'rows': [{'text': 'title', 'image': 'image.png', 'predictions': {
        method: {'reasoning': 'candidate', 'reasoning_review': {'status': 'pending'}}
        for method in ('joint', 'base')}}], 'reasoning_evaluation': {'rubric': rubric}}
    judge.judge_reasoning(report, tmp_path, cache)
    assert post.call_count == 1
    saved = json.loads((tmp_path / 'report.json').read_text())
    assert saved['reasoning_evaluation']['cache'] == {'hits': 1, 'api_calls': 1}
    assert saved['rows'][0]['predictions']['base']['reasoning_review'] == review
    assert saved['reasoning_evaluation']['metrics']['joint']['accuracy_mean'] == 2
