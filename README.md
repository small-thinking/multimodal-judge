# Multimodal Judge

Project scaffold for portable image-text scoring research. The scaffold provides the
Python environment, runtime device selection, a synthetic PyTorch smoke check,
and container/CI setup. Training is deferred to a later PR.

## Install and native smoke

Run commands from the repository root. Use Python 3.11 and uv 0.8.22 or newer
(Docker pins uv 0.12.10).

```sh
uv sync --locked --extra cpu --extra vlm --group dev
uv run --locked --extra cpu --extra vlm multimodal-judge smoke
uv run --locked --extra cpu --extra vlm multimodal-judge smoke --device cpu
uv run --locked --extra cpu --extra vlm pytest tests -q
uv run --locked --extra cpu --extra vlm ruff check src tests
```

The local config selects CUDA, then MPS, then CPU according to availability.
Explicit device requests fail when unavailable. The smoke check uses random
inputs and a small linear layer to verify forward/backward execution and an
optimizer update, then prints JSON. It needs no dataset, checkpoint, or token.

CPU and CUDA extras are mutually exclusive. The VLM extra is retained as a
foundation for later work; the scaffold does not load a pretrained model.

## Docker

CPU (also the default Docker build target):

```sh
docker compose build cpu
docker compose run --rm cpu
```

CUDA requires a Linux x86-64 NVIDIA host with a compatible driver and NVIDIA
Container Toolkit:

```sh
docker compose --profile gpu build gpu
docker compose --profile gpu run --rm gpu
```

The CUDA target uses PyTorch CUDA 12.8 wheels. Compose mounts local `configs`,
`data`, and `artifacts` directories and a named model cache. These runtime mounts
are separate from the image; private data and model files are excluded from the
build context. Optional environment variable names are in `.env.example`.

## CI and manual GHCR publication

The Checks workflow runs Ruff and pytest on pull requests and pushes to `main`.
To publish after merge, open GitHub Actions → **Publish Docker image** →
**Run workflow**, selecting `main`. Publication is manual and restricted to
`main`; it uses `GITHUB_TOKEN` with package write permission.

The workflow builds the Linux amd64 CUDA image, runs its smoke check on CPU
with networking disabled, then pushes `ghcr.io/<owner>/<repository>:sha-<commit>`
and `:latest`. This checks CPU execution in the CUDA image; GPU execution must
be validated on an NVIDIA host.

On Mac, use native uv for MPS acceleration; these Linux containers use CPU.

## Documentation assets

PNG files under `docs/assets/` use Git LFS. Install Git LFS, then run:

```sh
git lfs install --local
git lfs pull
```

Git stores pointers; Git LFS stores the image bytes. Data and model checkpoints
remain external runtime files.

## Data pipeline

Put `train.jsonl` and optional `validation.jsonl`/`test.jsonl` in a dataset directory.
Each row has `image`, nonempty `text`, and an integer `score` from 0 to 9;
`reasoning` is optional. Example (synthetic):

```json
{"image":"../../images/example.png","text":"A blue square","score":5,"reasoning":"Clear shape."}
```

Image paths resolve relative to the JSONL file. Copy the whole `data/` directory
when moving machines so those paths still resolve.

```bash
uv run --no-sync multimodal-judge inspect-data --config configs/data.yaml
# Or point to a different release:
uv run --no-sync multimodal-judge inspect-data --data-dir /path/to/release
```

The inspector checks schema, paths, score counts and recorded identity/hash
cross-split overlap. It does not decode every image or prove perceptual deduplication.
JSONL byte offsets are indexed in RAM; images are opened and decoded per sample,
so memory does not scale with 10,000 decoded images. Keep input files immutable
during a run.

The collators prepare score-only or score-plus-rationale batches with masked
prompt/padding tokens and a score readout position before the answer. Missing
rationale leaves score supervision only; overlong sequences fail explicitly.
Tests use generated images and fake processors; an optional saved-processor
check requires `MMJUDGE_TEST_PROCESSOR`. No model training is included in this layer.
