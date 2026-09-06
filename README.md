# Multimodal Judge

Qwen3-VL image-text judge: a continuous **0–9 score** and optional rationale.
Training uses a frozen 2B backbone, language-attention LoRA and a scalar scoring
head. BF16 base weights keep FP32 adapters, head and loss calculations.
See [architecture and measured Mac performance](docs/joint-model.md).

## Local training (Mac)

Use Python 3.11 and [uv](https://docs.astral.sh/uv/getting-started/installation/).
Run from the repository root:

```bash
uv sync --locked --extra cpu --extra vlm
uv run --no-sync multimodal-judge inspect-data --config configs/train-joint.yaml
uv run --no-sync multimodal-judge train-joint --config configs/train-joint.yaml \
  --max-steps 2 --output-dir artifacts/training/local-smoke
```

The first training run downloads the model. On macOS, the `cpu` extra installs
PyTorch with native MPS support; device selection is automatic. **Use native uv
for the Mac GPU.** Docker on Mac runs this project on CPU.

Edit [configs/train-joint.yaml](configs/train-joint.yaml) for model size, dataset,
image/token limits, batch size, learning rate and epochs. Remove `--max-steps 2`
for a full run and choose a fresh output directory. Defaults are batch 1 with
8-step gradient accumulation; tqdm shows optimizer updates. The current trainer
uses one GPU/process. Larger Qwen3-VL models need their own memory check.

Data lives in `data/training_data/v2/{train,validation}.jsonl`. Images resolve
relative to each JSONL file; preserve the whole `data/` layout when copying it.
Images are decoded per batch, so 10,000 images do not all enter RAM at once.
Datasets and weights are not included in the repository or Docker image.

W&B defaults to offline mode, project `multimodal-judge`. Set `wandb.entity` and
`wandb.name` in the config; for online tracking, supply `WANDB_API_KEY` and add
`--wandb-mode online`. Logs include total/score/rationale losses, score MAE/RMSE,
learning rate, gradient norm, throughput and sampled MPS memory. Sample images
and text are not uploaded. Results, adapters and Trainer checkpoints go to the
output directory. Resume with `--resume-from-checkpoint <run>/checkpoint-N`.

```bash
uv run --no-sync multimodal-judge judge \
  --checkpoint artifacts/training/local-smoke \
  --image /path/to/image.jpg --text 'Text to evaluate'
```

The earlier score-token SFT baseline remains available through `train` and
`configs/train.yaml`.

## Build Docker locally

```bash
docker compose build cpu
docker compose run --rm cpu  # Small forward/backward environment check
```

On a Linux x86_64 NVIDIA host, build the CUDA target instead:

```bash
docker compose --profile gpu build gpu
docker compose --profile gpu run --rm gpu
docker compose --profile gpu run --rm gpu train-joint \
  --config configs/train-joint.yaml --device cuda --max-steps 2 \
  --output-dir artifacts/training/cuda-smoke
```

Compose mounts data/configs read-only, outputs under `artifacts/`, and a persistent
model cache. Copy `.env.example` to `.env` for optional runtime credentials.
Do not put tokens in Docker build arguments.

## Publish an image

Recommended registry: **GitHub Container Registry (GHCR)**, alongside this repo.
Once the workflows are merged into `main`:

1. Open **Actions → Publish Docker image → Run workflow**, selecting `main`.
2. The workflow builds the CUDA image, runs a CPU smoke check, then publishes
   `ghcr.io/small-thinking/multimodal-judge:latest` and `:sha-<full-commit>`.
3. Copy the image digest from the job summary for a reproducible training run.

PRs run Ruff and CPU tests; they do **not** build Docker. Publishing is manual,
so `latest` means the latest successful publish, not necessarily the newest code.
GitHub's runner does not test NVIDIA execution. The workflow uses its built-in
`GITHUB_TOKEN`; no registry password needs to be added to repository secrets.

A new GHCR package is private by default. Keep it private and authenticate on the
GPU host with a classic PAT granting `read:packages`, or explicitly change the
package visibility to public for anonymous pulls. See the
[GHCR authentication guide](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry).

```bash
# Only for private images; GHCR_TOKEN contains your read:packages token.
printf '%s' "$GHCR_TOKEN" | docker login ghcr.io -u YOUR_GITHUB_USER --password-stdin
```

## Train on a rented GPU

Choose a **Linux x86_64 NVIDIA** machine with Docker, a CUDA 12.8-compatible driver
and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
The host does not need a separate CUDA toolkit. Transfer your data and config
from your local machine, preserving relative image paths:

```bash
ssh user@gpu-host 'mkdir -p ~/multimodal-judge/{data,artifacts,configs}'
rsync -av data/ user@gpu-host:~/multimodal-judge/data/
scp configs/train-joint.yaml user@gpu-host:~/multimodal-judge/configs/
```

On the GPU host (after the first successful image publication):

```bash
cd ~/multimodal-judge
export IMAGE=ghcr.io/small-thinking/multimodal-judge:latest
docker pull "$IMAGE"
docker run --rm --gpus device=0 "$IMAGE" smoke --config configs/gpu.yaml

# Run inside tmux if you need the job to survive SSH disconnection.
docker run --rm --gpus device=0 --shm-size=2g \
  -v "$PWD/data:/app/data:ro" \
  -v "$PWD/configs:/app/configs:ro" \
  -v "$PWD/artifacts:/app/artifacts" \
  -v model-cache:/app/cache \
  -e HF_TOKEN -e WANDB_API_KEY \
  "$IMAGE" train-joint --config configs/train-joint.yaml --device cuda \
  --max-steps 2 --output-dir artifacts/training/remote-smoke
```

After the smoke run, remove `--max-steps 2`, choose a new output directory and add
`--wandb-mode online` if desired. Export runtime tokens on the host before running
Docker. For repeatable experiments, set `IMAGE` to `:sha-<full-commit>` or, more
strictly, `ghcr.io/small-thinking/multimodal-judge@sha256:<digest>`. Pulling a new
image does not change an already-running container. Keep outputs/cache on a
persistent disk and copy `artifacts/` back before terminating the rental.

## Tests

```bash
uv run --no-sync ruff check .
uv run --no-sync pytest tests -q
```

Default tests use synthetic fixtures and skip checks requiring a saved processor
or MPS. To include those checks on a Mac with a local Qwen3-VL processor:

```bash
MMJUDGE_TEST_PROCESSOR=/absolute/path/to/processor MMJUDGE_TEST_MPS_JOINT=1 \
  uv run --no-sync pytest tests -q
```

The full Mac suite passed 146 tests; Linux CPU Docker passed 142 with 4 skipped
hardware/processor checks. Separate full-2B MPS profiling used **one
synthetic image/text example repeated for five updates per run**, not the real
8/1/1 dataset. Saved-adapter inference was also checked in a fresh process.
This validates plumbing and short-run performance, not held-out quality or a
10,000-example training run. See the [benchmark details](docs/joint-model.md).
