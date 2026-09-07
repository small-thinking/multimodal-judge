"""Pointwise rating metrics; invalid outputs are coverage failures, never zero scores."""
import math
import statistics


def rating_metrics(targets, predictions):
    if len(targets) != len(predictions) or not targets:
        raise ValueError('Expected equally sized, nonempty targets and predictions')
    for value in [*targets, *predictions]:
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 9:
            raise ValueError('Ratings must be finite numbers from 0 to 9')
    errors = [p - y for y, p in zip(targets, predictions)]
    absolute = sorted(map(abs, errors))

    def quantile(q):
        position = q * (len(absolute) - 1)
        low = math.floor(position)
        return absolute[low] + (absolute[math.ceil(position)] - absolute[low]) * (position - low)

    return {
        'count': len(errors), 'mae': statistics.mean(absolute),
        'rmse': math.sqrt(statistics.mean(e * e for e in errors)),
        'bias': statistics.mean(errors), 'median_absolute_error': quantile(.5),
        'p90_absolute_error': quantile(.9), 'max_absolute_error': absolute[-1],
        'rounded_accuracy': statistics.mean(math.floor(p + .5) == y
                                             for y, p in zip(targets, predictions)),
        'within_one': statistics.mean(e <= 1 for e in absolute),
    }


def summarize(rows, method):
    valid = [r for r in rows if r['predictions'][method]['rating'] is not None]
    summary = {'total_count': len(rows), 'valid_count': len(valid),
               'valid_rate': len(valid) / len(rows), 'invalid_count': len(rows) - len(valid),
               'token_limit_count': sum(r['predictions'][method].get('hit_token_limit', False)
                                        for r in rows)}
    if valid:
        summary.update(rating_metrics([r['target'] for r in valid],
                                      [r['predictions'][method]['rating'] for r in valid]))
    return summary


def parse_base_json(text):
    import json

    text = text.strip()
    if text.startswith('```json\n') and text.endswith('```'):
        text = text[8:-3].strip()
    elif text.startswith('```\n') and text.endswith('```'):
        text = text[4:-3].strip()
    try:
        value = json.loads(text)
    except ValueError:
        return {'rating': None, 'reasoning': '', 'parse_error': 'invalid_json'}
    if not isinstance(value, dict) or set(value) != {'rating', 'reasoning'}:
        return {'rating': None, 'reasoning': '', 'parse_error': 'invalid_schema'}
    score = value['rating']
    if (type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 9
            or not isinstance(value['reasoning'], str)):
        return {'rating': None, 'reasoning': '', 'parse_error': 'invalid_value'}
    return {**value, 'parse_error': None}
