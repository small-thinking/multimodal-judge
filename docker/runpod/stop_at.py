#!/usr/bin/env python3
"""Stop this Pod at an absolute UTC deadline, even after SSH disconnects."""
import argparse
from datetime import datetime, timezone
import shutil
import subprocess
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('pod_id')
p.add_argument('deadline', help='ISO 8601 UTC timestamp')
a = p.parse_args()
deadline = datetime.fromisoformat(a.deadline.replace('Z', '+00:00'))
if deadline.tzinfo is None:
    p.error('deadline must include a timezone')
cli = shutil.which('runpodctl')
if not cli:
    p.error('runpodctl is not installed; do not rely on this timer')
# Pod-scoped credentials may deny read APIs while allowing stop.
# The stop operation was verified on this deployment before arming.
print('Stop timer armed:', a.pod_id, deadline.isoformat(), flush=True)
while True:
    seconds = (deadline - datetime.now(timezone.utc)).total_seconds()
    if seconds <= 0:
        break
    time.sleep(min(seconds, 30))
for attempt in range(60):
    # Legacy order is supported by the CLI injected into Pods.
    try:
        result = subprocess.run([cli, 'stop', 'pod', a.pod_id], capture_output=True, timeout=20)
    except subprocess.TimeoutExpired:
        print('Stop request timed out', flush=True)
        continue
    print('Stop request attempt', attempt + 1, 'exit', result.returncode, flush=True)
    if result.returncode == 0:
        break
    time.sleep(5)
else:
    raise SystemExit('Failed to stop Pod; provider UI intervention required')
