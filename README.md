# Multimodal Judge

Portable image-text scoring research, managed with **uv** and Python 3.11.
Current implementation: a joint **continuous regression + optional rationale**
Qwen3-VL model, with lazy loading, LoRA, Transformers Trainer and W&B. The earlier
score-token SFT baseline remains available through `train`.

## Joint model on Apple Silicon

See [architecture, losses, measured M4 memory/speed and inference](docs/joint-model.md).
The diagram matches the implemented two-head model:

![Joint model architecture](docs/assets/joint-architecture.png)

```bash
# Reuse this workspace's downloaded public checkpoint cache.
HF_HOME="$PWD/artifacts/hf" uv run --locked --extra cpu --extra vlm \
  multimodal-judge train-joint --config configs/train-joint.yaml \
  --max-steps 2 --output-dir artifacts/training/joint-first-run
```

The joint config uses native MPS when available, BF16 frozen base weights with
FP32 adapters/head, SDPA, batch 1 and gradient accumulation 8. Full 2B synthetic
forward/backward was verified on this M4 Pro (48 GiB); use `--dtype float32` for
the conservative fallback. `train-joint` is Qwen3-VL only. GPU memory fraction
defaults to 0.75 of MPS's recommended device budget, not 75% of physical RAM.

`judge --checkpoint <run-or-checkpoint-directory> --image <local-image> --text
<input-text>` loads the saved adapter **and regression head** with the matching
base, returning a continuous 0–9 `score` and generated `reasoning`.

## Training MVP

Run commands from the repository root. Install the locked environment first:

```bash
uv sync --locked --extra cpu --extra vlm
uv run --locked --extra cpu --extra vlm multimodal-judge inspect-data --config configs/train.yaml
# Short first run; the first training call downloads the model weights.
uv run --locked --extra cpu --extra vlm multimodal-judge train \
  --config configs/train.yaml --max-steps 2 \
  --output-dir artifacts/training/first-run
```

The default is [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct),
the 2B member of Qwen3-VL, with language-attention LoRA. This is a **discrete
0–9 score-only SFT baseline**: image + text goes in, a single score comes out.
The original plan's continuous regression head and rationale supervision are
future work. Prompt, image, and padding tokens are masked from the training
loss; optional annotation rationales are not model inputs or training targets.

Edit [configs/train.yaml](configs/train.yaml) for model ID or local checkpoint,
data directory, image pixel budget, sequence length, LoRA settings, batch sizes,
gradient accumulation, learning rate, epochs/steps, checkpoints, device, and W&B.
CLI overrides include `--model`, `--data-dir`, `--output-dir`, `--max-steps`, and
`--wandb-mode`. Paths resolve from the current working directory. Larger Qwen3-VL
or Qwen2.5-VL checkpoints can use the same interface; each checkpoint still needs
its own hardware/runtime smoke test. Arbitrary VLM families are not guaranteed.

`float32`, batch size 1, and a small image budget provide a conservative starting
configuration for native Mac development. LoRA reduces trainable parameters and
optimizer memory, but the base model still occupies memory (roughly 8 GB for 2B
float32 parameters alone, plus activations and runtime overhead). On CUDA, use
`--extra cuda` and consider `model.dtype: bfloat16` on supported hardware.
Gradient accumulation increases the effective batch size without holding all
microbatches at once. Trainer displays a tqdm optimizer-step progress bar;
`max_steps` counts optimizer updates, not individual images.

### Data loading

Point `data.directory` to a release containing `train.jsonl` and
`validation.jsonl`, currently `data/training_data/v2`. The loader resolves each
`image` path relative to its JSONL file, so `../../images/example.jpg` resolves
to `data/images/example.jpg`. Preserve this layout when moving data. Image bytes
are decoded with Pillow regardless of the filename extension.

The dataset stores JSONL byte offsets in memory and reads/decode images only
when a batch requests them. **10,000 images do not require 10,000 decoded images
in RAM or GPU memory.** Memory scales mainly with the current batch, image
resolution, sequence length, and worker prefetch. The default uses zero loader
workers for easy Mac debugging; raise `dataloader_num_workers` after measuring
I/O throughput. For millions of examples, a sharded/streaming dataset can replace
the offset index later.

`inspect-data` checks schema, paths, counts, score distributions and recorded
cross-split identity/lineage overlap without decoding all images or downloading
a model. It cannot establish perceptual deduplication or detect changed bytes
from hashes alone. Training uses only the train split; validation is for loss
and generated-score evaluation. Test is reserved for later evaluation. Input
files must remain immutable during a run. Overlong tokenized examples fail
explicitly; reduce image budget or curate long text instead of silently cutting
image tokens or answer labels.

### W&B and results

The default W&B project is **`multimodal-judge`**, with `mode: offline` for local
runs. Set `wandb.entity` to your account/team and `wandb.name` for a named run.
For live dashboards, authenticate once and use online mode:

```bash
uv run --locked --extra cpu --extra vlm wandb login
uv run --locked --extra cpu --extra vlm multimodal-judge train \
  --config configs/train.yaml --wandb-mode online \
  --output-dir artifacts/training/online-run
```

Alternatively supply `WANDB_API_KEY` through the environment; `uv run --env-file
.env ...` loads the local file explicitly. `--wandb-mode disabled` turns tracking
off. Offline runs can be uploaded later with `wandb sync <offline-run-directory>`.
An aggregate-only Trainer callback sends training loss, gradient norm, learning rate, epoch,
validation loss, and
training/evaluation runtime and throughput. Resolved configuration and dataset
counts identify the run. No sample images, titles, rationale text, or model
artifacts are deliberately logged to W&B.

After training, deterministic validation generation reports score MAE, RMSE,
exact-match accuracy, and invalid-output rate. MAE/RMSE use valid parsed scores;
accuracy counts invalid outputs as wrong. These are actual generated predictions,
not teacher-forced token accuracy. An empty validation set disables evaluation.
The current 8/1/1 release is sufficient for a plumbing check, not a reliable
quality estimate.

Outputs stay under `training.output_dir`: configuration, metrics, processor,
final adapter (or full model when LoRA is disabled), and Trainer checkpoints.
Resume using a checkpoint directory, not just the final adapter:

```bash
uv run --locked --extra cpu --extra vlm multimodal-judge train \
  --config configs/train.yaml --output-dir artifacts/training/first-run \
  --resume-from-checkpoint artifacts/training/first-run/checkpoint-2
```

Use a fresh output directory for each experiment. Checkpoint resume restores
training state; offline W&B tracking may create a separate run.

### Validation without full model weights

`uv run --locked --extra cpu --extra vlm pytest` runs synthetic unit tests.
The opt-in `tests/test_training_integration.py` uses a real Qwen3-VL processor,
a tiny randomly initialized Qwen architecture and synthetic colored images to
test batched processing, LoRA updates, generation, offline W&B and checkpoint
resume. Save a Qwen3-VL processor locally, then run:

```bash
MMJUDGE_TEST_PROCESSOR=/absolute/path/to/saved-processor \
  uv run --locked --extra cpu --extra vlm pytest tests/test_training_integration.py -q
```

This checks the training stack; it does not establish full 2B checkpoint memory
fit, pretrained quality, or CUDA/MPS compatibility.

## Native development (Apple Silicon)

Install uv >= 0.8.22 (for an older Homebrew install: `brew upgrade uv`), then:

```bash
uv sync --locked --extra cpu --extra vlm
uv run --locked --extra cpu --extra vlm multimodal-judge smoke --config configs/local.yaml
uv run --locked --extra cpu --extra vlm pytest
uv run --locked --extra cpu --extra vlm ruff check .
```

On macOS, `cpu` selects the standard PyPI PyTorch wheel, which also supports MPS.
`auto` prefers CUDA, then MPS, then CPU. An explicitly unavailable device fails.
The smoke test uses float32 and no model download. The commands above install
PyTorch, Transformers, PEFT, Accelerate, Datasets, Safetensors, and Pillow, plus
pytest and Ruff for development. Keep both extras on subsequent `uv sync` and
`uv run` commands to preserve the full environment. For a minimal environment,
omit `--extra vlm`.
The current VLM dependency lock is a starting
point, not evidence that a particular checkpoint works on MPS.

### Managing the environment

`uv sync` creates the project-local `.venv` using Python 3.11 from
`.python-version`; uv can download that interpreter if necessary. Point your
editor at `.venv/bin/python`. `uv run` uses this environment automatically, so
shell activation is optional (`source .venv/bin/activate`).

Declare dependencies in `pyproject.toml` and keep the generated `uv.lock` in Git.
Do not commit `.venv` or install project dependencies with global pip.

```bash
# Add a runtime dependency, then restore the full local environment.
uv add <package>
uv sync --locked --extra cpu --extra vlm

# Add a development tool.
uv add --dev <package>
uv sync --locked --extra cpu --extra vlm

# Deliberately update one dependency and synchronize the environment.
uv lock --upgrade-package transformers
uv sync --locked --extra cpu --extra vlm
```

For native Linux NVIDIA GPU development, replace `--extra cpu` with
`--extra cuda`; these two backend extras are mutually exclusive.

## Docker CPU (Mac or Linux)

Docker Desktop/Engine and Compose >= 2.29 are required for these Compose files.

```bash
docker compose build cpu
docker compose run --rm cpu
```

Mac Linux containers use CPU; run native uv for PyTorch MPS acceleration.
The same Dockerfile has CPU and CUDA targets; they are distinct architecture/backend
builds, not one binary image that runs everywhere.

## Docker CUDA (rented Linux x86_64 NVIDIA GPU)

The host needs Docker, a CUDA 12.8-compatible NVIDIA driver and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
Host CUDA toolkit installation is not required by this wheel-based image.

```bash
nvidia-smi
docker compose --profile gpu build gpu
docker compose --profile gpu run --rm gpu
```

The GPU config explicitly requests CUDA, so a missing GPU cannot pass via CPU fallback.
First validate the same small model/checkpoint before increasing model size.
Linux ARM GPU hosts are outside the initial CUDA support scope.

## Data and secrets

Optional: copy `.env.example` to `.env` and set HF_TOKEN for gated models.
Compose passes it at runtime; native uv can use `uv run --env-file .env ...`.
Do not bake tokens into images. For native use, remove the container-specific HF_HOME
entry or replace it with a local cache path.

The base image tags and uv version are pinned by tag; release images should also
record image digests. Rebuilding later may pick up base-image security updates.

`data/` and `artifacts/` are bind-mounted; the Hugging Face cache uses a named volume.
Keep SQLite and datasets on persistent storage. Do not delete volumes containing
needed artifacts. On shared Linux hosts, use `docker compose run --rm --user
"$(id -u):$(id -g)" ...` only after making the mounted directories/cache writable.

The private canonical plan is `docs/PROJECT_PLAN.md`; the original is
`docs/PROJECT_PLAN.original.md`. Both are intentionally Git-ignored and excluded
from the Docker build context. They will not appear in a fresh clone: transfer or
back them up separately. `.env`, datasets, weights, SQLite files and run outputs
are also ignored. `.env.example`, configuration and `uv.lock` belong in Git.

Dependency backend routing follows the [uv PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/).
Docker GPU platform constraints: [Docker documentation](https://docs.docker.com/desktop/features/gpu/).
