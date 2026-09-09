#!/bin/bash
set -euo pipefail
mode=${1:-smoke}
if [ "$#" -gt 0 ]; then shift; fi
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
case "$mode" in
  smoke) exec bash "$script_dir/train-current.sh" --set training.max_steps=2 "$@" ;;
  train) exec bash "$script_dir/train-current.sh" "$@" ;;
  *) echo 'Usage: bash docker/runpod/train.sh smoke|train [training CLI overrides]' >&2; exit 2 ;;
esac
