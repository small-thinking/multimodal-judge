# Multimodal Judge

Qwen3-VL image-text judge: a continuous **0–9 score** and optional rationale.
Training uses a frozen 2B backbone, language-attention LoRA and a scalar scoring
head, or an optional ten-class head whose probability-weighted rating is continuous.
BF16 base weights keep FP32 adapters, head and loss calculations.
See [architecture and measured Mac performance](docs/joint-model.md).
The architecture PNG is stored in Git LFS. After cloning, install Git LFS and
run `git lfs install --local && git lfs pull` to download documentation images.

## Local training (Mac)

For a full run on dataset v5 with online W&B tracking:

```bash
uv run --no-sync --env-file .env multimodal-judge train-joint --config configs/train-joint.yaml --data-dir data/training_data/v5 --wandb-mode online
```

Without `--output-dir`, new training runs create a unique directory alongside the
configured output directory. The default W&B name matches it:
`train-<model>-<dataset>-<local timestamp with timezone>-<random ID>`.
Explicit output directories and configured W&B names remain supported; resume
keeps the configured output directory. Evaluation W&B names use
`eval-<model>-<dataset>-<split>-<timestamp>-<random ID>` and are saved in the report.
W&B job types distinguish `training` from `evaluation`.

The default run trains for two epochs, evaluates every 100 optimizer steps, and
evaluates again after training. With 1,456 v5 samples and effective batch size 8,
this gives 364 updates and validation at steps 100, 200, 300, and 364.

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

### Select the scoring head and loss weight

Use the batch-16 reference config for new local comparisons. These commands each
create a fresh run; they do not convert an existing regression checkpoint:

```bash
# Increase only the regression-loss weight. 5 is an experiment value, not a tuned default.
uv run --no-sync multimodal-judge train-joint --config configs/train-joint-batch16.yaml \
  --score-head regression --score-weight 5 --wandb-mode disabled

# Ten-class (0–9) cross-entropy head, reporting a continuous expected rating.
uv run --no-sync multimodal-judge train-joint --config configs/train-joint-batch16.yaml \
  --score-head classification --score-weight 1 --wandb-mode disabled
```

The same settings are `objective.head_type` and `objective.score_weight` in YAML.
The default remains regression with score weight 1. Total loss is
`score_weight * score_loss + rationale_weight * rationale_loss`; logged component
losses remain unweighted. CE and normalized Huber have different scales, so the
same score weight does not imply matched task balance across heads. Classification
does not guarantee better MAE/RMSE and is not an ordinal objective.

The batch-16 config uses gradient accumulation 1, learning rate 1e-4, two epochs,
dataset v5 and disabled W&B. Change one factor per comparison and retain the same
update budget. Inference and Evaluation Center restore the head from the checkpoint;
no head flag is needed. Old checkpoints remain compatible. Switching head type or
loss weights is rejected on resume: use a fresh training run.

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

### Reasoning review (draft rubric)

Rating error and explanation quality are separate. The draft
`configs/rubrics/reasoning-v1.yaml` scores visual accuracy, grounded style description,
fluency/repetition and inference reliability, each on **1–3** with a short Chinese
explanation. Replace its anchors and version when your rubric is ready. Unsupported
inferences should not earn credit; appropriate uncertainty or abstention can score well.

Add `--reasoning-rubric configs/rubrics/reasoning-v1.yaml` to `evaluate`, or select the
rubric in Evaluation Center. To prepare reviews from existing predictions without
rerunning the model:

```bash
uv run --no-sync multimodal-judge prepare-reasoning-reviews \
  --report artifacts/evaluation/runs/my-eval/report.json \
  --reasoning-rubric configs/rubrics/reasoning-v1.yaml \
  --output-dir artifacts/evaluation/runs/my-review-queue
```

`reasoning-requests.jsonl` contains local image paths, titles, candidate explanations
and an empty JSON response template. Give a human or future multimodal judge the images
and rubric alongside those requests. It excludes human labels and predictor names.
No judge is configured or invoked automatically: pending/unscorable reviews have no scores.

Fill `response.dimensions` with `{ "rating": 1, "reasoning": "简短依据" }` for every
rubric dimension, retaining `review_id` and `rubric_sha256`. If review is impossible,
use `response: { "status": "unscorable", "reason": "原因" }` instead. Import completed
reviews into a new report and optionally log their aggregate scores:

```bash
uv run --no-sync --env-file .env multimodal-judge import-reasoning-reviews \
  --report artifacts/evaluation/runs/my-review-queue/report.json \
  --reviews /path/to/completed-reviews.jsonl --reviewer human-v1 \
  --output-dir artifacts/evaluation/runs/my-reviewed --wandb-mode online
```

The page shows per-dimension results and coverage; W&B logs `reasoning/<method>/*`
aggregates only. Rubric scores depend on the reviewer and are not ground truth. Do not
mix reviewers or rubric versions in a comparison. New base inference explicitly asks
for Chinese reasoning and JSON; old runs retain their original prompt and are labeled legacy.

### Optional reasoning judge

In Evaluation Center, select a rubric and enable **LLM judge**. The default is
`grok-4.6` with `reasoning_effort=low` (4.6 does not support `none`). Set
`GROK_API_KEY` in the project `.env` or process environment before starting the center. Only the
image, title, candidate explanation and rubric are sent to xAI.

You can also grade saved predictions without running the VLM again:

```bash
uv run --no-sync multimodal-judge prepare-reasoning-reviews \
  --report artifacts/evaluation/runs/<run>/report.json \
  --reasoning-rubric configs/rubrics/reasoning-v1.yaml \
  --output-dir artifacts/evaluation/runs/<new-run> \
  --enable-llm-judge --judge-model grok-4.6 --judge-effort low \
  --wandb-mode online
```

The same judge flags work with `evaluate`. Valid results are cached under
`artifacts/evaluation/judge-cache`; use `--judge-cache-dir` to change it. Keep this
directory between runs (mount it as a volume in Docker). Exact requests reuse the
saved result without an API call, including after an interrupted run. Changing
image bytes, text, rubric, model or generation settings produces a new cache key.
Model aliases can change server-side; clear the cache or use a pinned model ID
when intentionally refreshing a judge version.

W&B training charts use `train/*` for step losses, learning rate and gradient norm,
and `validation/*` for loss, MAE and RMSE. Evaluation logs retain rating error,
accuracy and coverage under `evaluation/*`, plus rubric means and coverage under
`reasoning/*`. Counts, timing, memory diagnostics and per-score details stay local.
The old `train_loss` was one final average; `train/loss` is the step time series.
Existing W&B runs keep their historical charts; this change applies to new runs.

Validation also logs `validation/score_loss`, `validation/reasoning_loss`, and
`validation/reasoning_coverage`; `train/reasoning_coverage` shows how much of each
training interval has rationale supervision. The combined loss can improve while
rating errors worsen, so compare the components and MAE/RMSE separately.
Each ordered single-process validation on a JSONL dataset saves numeric per-row
predictions, targets, and residuals in the run's `validation/step-<N>-<unique>.jsonl`,
with a companion summary containing a dataset fingerprint and MAE/RMSE. Row indices
refer to zero-based dataset order. These files remain local; no sample content is
uploaded to W&B. Repeated evaluations at the same step get separate files.

Base evaluation reads `src/multimodal_judge/prompts/base-evaluation.txt` by default.
Edit that file, or pass `--base-system-prompt path/to/prompt.txt` to `evaluate`.
This is the complete base system prompt, independent of the checkpoint's training
prompt. Each report saves the exact prompt used; edits affect future evaluations.

### Merge annotation partitions

From the repository root, run:

```bash
uv run python -m multimodal_judge.merge_annotations
```

This merges `data/annotations/**/*.json` into
`data/merged_annotations/merged_annotations.json`. Use `--annotation-dir` and
`--output` to override these paths. Outputs must be outside the input directory
and named `merged_*.json`. Backups, paths beginning with `merged_`, and documents
marked as merge artifacts are excluded. Re-running atomically replaces the output
with the same content when inputs are unchanged.

The merge preserves all version-2 records, including repeated IDs, unscored records,
and pending-review flags; it does not deduplicate or approve labels. `record_sources`
provides the source file and index for each record, and `sources` includes input
hashes. Legacy pointwise documents are retained under `legacy_partitions`, and
partition-specific pairwise labels under `pairwise_partitions` to avoid key collisions.
These retained sections are not converted into version-2 records or global pairwise
labels. Continue using the original partitions for training-data preparation.
Generated data stays local under the ignored `data/` directory.
