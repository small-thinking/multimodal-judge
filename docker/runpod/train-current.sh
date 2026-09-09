#!/bin/bash
# Use the image's current config/data; later CLI arguments can override defaults.
set -euo pipefail
mkdir -p /workspace/artifacts
exec multimodal-judge train-joint \
  --config "${TRAIN_CONFIG:-/app/configs/train-joint.yaml}" \
  --data-dir "${TRAIN_DATA_DIR:-/app/dataset}" \
  --device cuda \
  --wandb-mode "${WANDB_MODE:-online}" \
  --output-dir "${TRAIN_OUTPUT_DIR:-/workspace/artifacts/train-$(date -u +%Y%m%dT%H%M%SZ)}" \
  "$@"
