"""Readable, unique names shared by local outputs and W&B runs."""

from datetime import datetime
from pathlib import Path
import re
from uuid import uuid4


def make_run_name(kind, model, dataset, split=None):
    parts = [kind, Path(model).name, Path(dataset).name]
    if split:
        parts.append(split)
    parts.extend([datetime.now().astimezone().strftime('%Y%m%d-%H%M%S%z'), uuid4().hex[:8]])
    return '-'.join(re.sub(r'[^a-zA-Z0-9_.-]+', '-', str(part)).strip('-') for part in parts)
