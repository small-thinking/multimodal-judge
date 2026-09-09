# Run the private training image on an NVIDIA GPU host

The GitHub repository contains only the launcher and documentation. Training data stays in the **private** Docker Hub repository `smallthinking/multimodal-judge`. You must have pull access; cloning this public repository does not grant it.

Prerequisites: Linux amd64, Python 3.11+, Docker daemon access, an NVIDIA GPU/compatible driver, and NVIDIA Container Toolkit. This launcher does not rent hardware or install/alter the host driver. Choose a GPU with enough memory for the configured batch16 workload; capacity must be confirmed with a short run on the actual host.

```sh
git clone https://github.com/small-thinking/multimodal-judge.git
cd multimodal-judge
docker login -u smallthinking
# Enter a Docker Hub read-only access token at the password prompt on the rented host.
# Do not put tokens in shell commands, the repo, or this document.
python3 docker/gpu.py pull
python3 docker/gpu.py check
python3 docker/gpu.py gpu-check
python3 docker/gpu.py train --epochs 5 --run-name v6-joint-huber-5ep
python3 docker/gpu.py evaluate --run-name v6-joint-huber-5ep
```

The launcher defaults to `smallthinking/multimodal-judge:cuda-v6-huber-5ep-20260908`; `--image` accepts a digest-pinned reference too. It uses the embedded configuration/data, downloads base-model weights into a persistent Docker cache volume, and saves checkpoints/reports under `artifacts/gpu/`. No local Python ML environment or local dataset is needed. The image includes CUDA12.8 PyTorch wheels and the application; the host provides the GPU driver. `gpu-check` verifies CUDA availability and a small backward pass, not full training memory requirements.

Default training is joint Huber with rationale weight0.1, batch16, LR1e-4 and five epochs. For ten epochs:

```sh
python3 docker/gpu.py train --epochs 10 --run-name v6-joint-huber-10ep
python3 docker/gpu.py evaluate --run-name v6-joint-huber-10ep
```

For score-only, add `--score-only` to training and select a distinct run name. `--output /path/on/persistent/disk` relocates outputs; use the same path when evaluating. Checkpoints are saved every50 steps and the last3 are retained. The launcher refuses to train into an existing run directory. It does not automatically resume interrupted runs.

Use `--env-file /path/to/docker.env` only when runtime credentials are needed. W&B runs offline. On a rented host, use a dedicated read-only Docker Hub token; never share your write token. Run in tmux or the provider's persistent job interface so an SSH disconnect does not terminate training. Copy checkpoints/reports to durable storage before releasing the machine, and log out/revoke the temporary token afterward.

The Mac-built image was checked for amd64 architecture, dependencies and portable data paths. Actual NVIDIA execution must be checked on the rented GPU. A successful image pull alone does not validate training.
