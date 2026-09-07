# Multimodal Judge

Qwen3-VL image-text judge: a continuous **0–9 score** and optional rationale.
Training uses a frozen 2B backbone, language-attention LoRA and a scalar scoring
head. BF16 base weights keep FP32 adapters, head and loss calculations.
See [architecture and measured Mac performance](docs/joint-model.md).
The architecture PNG is stored in Git LFS. After cloning, install Git LFS and
run `git lfs install --local && git lfs pull` to download documentation images.

## Local training (Mac)

Use Python 3.11 and [uv](https://docs.astral.sh/uv/getting-started/installation/).
Run from the repository root:

```bash
uv sync --locked --extra cpu --extra vlm
uv run --no-sync multimodal-judge inspect-data --config configs/train-joint.yaml
uv run --no-sync multimodal-judge train-joint --config configs/train-joint.yaml \
  --max-steps 2 --output-dir artifacts/training/local-smoke
```

The config includes a general English rating-assistant system prompt. Edit it
in `configs/train-joint.yaml` to add your scoring criteria:

```yaml
prompt:
  system: |
    Evaluate the supplied title and image together.
    Follow the scoring rubric below.
```

Put your actual scoring rubric in this field. It is sent as a separate `system`
message, with the title/image in the `user` message.
The default prompt describes the output contract: `judge` returns exactly
`{"reasoning": "...", "rating": 7.2}`. The scoring head supplies the number; the
application JSON-encodes it with the generated rationale. Empty text omits the system message. Training, evaluation and `judge` use the
same prompt saved in the checkpoint; resume rejects a changed prompt. Older
checkpoints without this field keep their original behavior.

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

`train-joint` is the single training entry point. `joint_training.py` runs the
Trainer; `joint_model.py` defines the heads and losses; `joint_inference.py`
restores a saved model for scoring and rationale generation. `training_config.py`
holds configuration validation and small helpers; it is not another trainer.

## Evaluate a checkpoint

Evaluation is independent of training. It uses the saved prompt and image processor,
reads one image at a time, and reports MAE/RMSE, bias, P90/max error, rounded accuracy
and the fraction within one rating point. Constant baselines use **training** labels.
Choose a fresh output directory for every run:

```bash
uv run --no-sync --env-file .env multimodal-judge evaluate \
  --checkpoint artifacts/training/my-run --data-dir data/training_data/v4 \
  --split test --device mps --output-dir artifacts/evaluation/runs/my-eval \
  --wandb-mode online
```

For a local UI with checkpoint/dataset selection, progress and saved comparisons:

```bash
uv run --no-sync --env-file .env multimodal-judge evaluation-center --port 8877
```

Open http://127.0.0.1:8877. Checkpoints are discovered under `artifacts/training/`,
datasets under `data/training_data/`, and reports under `artifacts/evaluation/runs/`.
One evaluation runs at a time. Use native MPS on Mac; use `--device cuda` in a GPU
container for the CLI. `HF_HOME` should point to the same cache used for training.

Add `--include-base` for original-Qwen JSON inference. Base has no learned scalar
head; its output instruction is adapted for direct scoring. Invalid JSON is counted
separately, never as a zero rating. Compare the matching base-valid subset when some
outputs fail. Use `--max-samples` for a smoke run, `--max-new-tokens` to override the
saved generation budget, or `--split validation` while tuning. Keep test for held-out
measurement.

Online logging creates a separate W&B **evaluation** run in `multimodal-judge`, with
final aggregate metrics and per-score aggregates. Checkpoint/data hashes and the
training-run link stay in the local report. Override `--wandb-project`, `--wandb-entity` or
`--training-run-url` as needed. Raw samples and predictions remain local.
Use `--wandb-mode offline` or `disabled` without network access. If an upload fails,
local `report.json` remains available; retry logging without rerunning inference:

```bash
uv run --no-sync --env-file .env multimodal-judge log-evaluation \
  --report artifacts/evaluation/runs/my-eval/report.json --wandb-mode online
```

RMSE has the same unit as the score, but penalizes large errors more than MAE.
Nine exact predictions and one error of 3 points give RMSE ≈ 0.95, not an error of
0.95 on every sample. A value of 1 is not a universal quality threshold.

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

GitHub Actions runs Ruff and CPU tests on PR creation/updates and pushes to
`main`. Full-checkpoint training and GPU checks are separate local runs.
Previous full-2B MPS profiling used **one
synthetic image/text example repeated for five updates per run**, not the real
8/1/1 dataset. Saved-adapter inference was also checked in a fresh process.
This validates plumbing and short-run performance, not held-out quality or a
10,000-example training run. See the [model notes](docs/joint-model.md).
