# RunPod training

Use the image reference in `docker/runpod/image.txt` as the Pod image. It includes current code, all configs and the portable dataset; no separate data download is required. The image starts SSH and waits for an explicit training command.

This release defaults to Joint + MSE, learning rate 1e-4, 10 epochs, batch 16, original-only training samples, validation/save every 25 optimizer steps, and W&B online. Config: `/app/configs/train-joint.yaml`. Dataset: `/app/dataset`. Credentials must be provided at runtime.

On an A40 Pod with 20 GB container disk and 50 GB volume at `/workspace`:

```bash
git clone https://github.com/small-thinking/multimodal-judge.git /workspace/multimodal-judge
cd /workspace/multimodal-judge
bash docker/runpod/train.sh train
```

Or run the bundled `train-current` directly. Override parameters after the command, e.g. `train-current --config /app/configs/another.yaml`. Output defaults to a unique directory under `/workspace/artifacts`.

Use tmux for training. Before training, arm an independent provider stop timer with an absolute deadline within the approved budget. The current user-authorized limit is three hours, including setup and evaluation. Preserve final/best weights, logs and evaluations locally and verify archive hashes before terminating the Pod. Stopping releases compute but volume storage continues billing until termination.

For a conventional GPU VM with Docker installed, `python docker/gpu.py pull` downloads the same image; `python docker/gpu.py train --env-file /path/to/runtime.env` runs it. Do not run Docker inside a RunPod Pod.
