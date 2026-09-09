#!/bin/bash
# Run inside the RunPod image; do not invoke Docker within a Pod.
set -euo pipefail
mode=${1:-smoke}
export HF_HOME=/workspace/cache/huggingface
export WANDB_MODE=offline
mkdir -p /workspace/artifacts
case "$mode" in
  smoke) run=v6-smoke; extra=(--set training.max_steps=2) ;;
  train) run=v6-joint-huber-5ep; extra=(--set training.num_train_epochs=5) ;;
  evaluate)
    exec multimodal-judge evaluate --checkpoint /workspace/artifacts/v6-joint-huber-5ep --data-dir /app/dataset --split validation --device cuda --wandb-mode offline --output-dir /workspace/artifacts/evaluation-v6-joint-huber-5ep ;;
  *) echo 'Usage: bash docker/runpod/train.sh smoke|train|evaluate' >&2; exit 2 ;;
esac
if [ -e "/workspace/artifacts/$run" ]; then
  echo "Output already exists: /workspace/artifacts/$run" >&2
  exit 1
fi
python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0)); x=torch.ones((32,32),device="cuda",requires_grad=True); (x@x).sum().backward(); torch.cuda.synchronize()'
multimodal-judge train-joint --config /app/training.yaml --data-dir /app/dataset --device cuda --wandb-mode offline --output-dir "/workspace/artifacts/$run" "${extra[@]}" 2>&1 | tee "/workspace/artifacts/$run.log"
