# RunPod training

RunPod Pods are containers. Select the image as the Pod image; do not run `docker/gpu.py` inside a Pod.

Image: `smallthinking/multimodal-judge:runpod-v6-20260908` (linux/amd64, now public with the owner's authorization; includes the v6 dataset).

Configure one A40, 20 GB container disk, 50 GB volume mounted at `/workspace`, and TCP port 22. The image starts SSH and waits; it does not automatically train. Model cache and outputs belong on `/workspace`.

Use the managed SSH proxy from the Connect tab first. If direct SSH fails, install your existing public key in `/root/.ssh/authorized_keys` through that authenticated session (mode 600). Never upload the private key.

```sh
git clone https://github.com/small-thinking/multimodal-judge.git /workspace/multimodal-judge
cd /workspace/multimodal-judge
cp docker/runpod/train.sh docker/runpod/stop_at.py /workspace/
export PATH=/app/.venv/bin:/usr/local/bin:/usr/bin:/bin
# Run the supplied train.sh in the Pod environment:
bash /workspace/train.sh smoke
bash /workspace/train.sh train
bash /workspace/train.sh evaluate
```

Run long commands under tmux so SSH disconnects do not end training. The two-step smoke uses the same batch 16 joint Huber configuration as the five-epoch run. Use distinct outputs; the launcher refuses to overwrite them.

Before training, arm `stop_at.py` under tmux using the actual Pod ID and an absolute UTC deadline within the approved budget. Example (replace both values):

```sh
tmux new-session -d -s budget-stop 'python /workspace/stop_at.py POD_ID 2026-09-09T10:04:14Z > /workspace/stop-timer.log 2>&1'
```

Verify the process and log. RunPod's injected CLI uses `runpodctl stop pod POD_ID`. Its scoped credential may deny `get pod` even when stopping itself works; a failed read is not proof that stopping is unauthorized. A timer sends a provider stop request, which can fail if the control plane/network is unavailable. Keep an independent control-plane check and a time margin. Stopping only the training process or SSH server does not stop GPU billing.

Stopping releases the GPU, and resuming may require migration if it has been rented by someone else. Export outputs before releasing the Pod. Volume storage continues to cost money after stopping; terminate only after required results are safely copied.
