# Joint pointwise judge: continuous score and rationale

The earlier `train` baseline generates a score token with the original language
model head. The new `train-joint` model adds a regression head while using the
original LM head to generate a short annotated assessment. These are shared
backbone tasks, not two separate 2B models. Pairwise/DPO is outside this change.

![Architecture](assets/joint-architecture.png)

## Architecture and causal contract

The selected [Qwen3-VL-2B configuration](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/blob/89644892e4d85e24eaac8bacfd4f463576704203/config.json)
has a 28-layer language decoder of hidden width 2048. The vision encoder, visual
mergers/DeepStack, text embeddings and original LM head are frozen. Language
attention q/v projections receive rank-8 LoRA; a new `Linear(2048, 1)` head has
2,049 trainable parameters. The complete wrapped model measured 2,129,141,762
parameters, including 1,607,681 trainable LoRA/head parameters (about 0.076%).

`joint_model.py` subclasses the native Qwen3-VL implementation and retains its
generation and KV-cache behavior. Scoring reads the last token of the user plus
assistant-generation prefix, **before any score or rationale answer tokens**.
It returns `score = 9 * sigmoid(head(hidden))`. Changing later gold score or
rationale tokens cannot change that score under causal attention. This is tested
with altered suffixes, unequal lengths and padding, including real vision inputs.

Training constructs this conceptual sequence:

```text
[image + text + assistant prefix] [Score: 5] [Reasoning: rationale text + EOS]
                              ^
                     regression pooling position
```

All rows supervise regression. Rationale rows additionally supervise only the
rationale body and EOS; prompt, score condition, format prefix and padding have
CE labels `-100`. Rows without a rationale use the input prefix alone and have
zero rationale loss. The loader does not use `reasoning_alternatives` as inputs.

The objective is

```text
loss = Huber(predicted_score / 9, annotated_score / 9; delta=0.1)
       + rationale_weight * rationale_CE
```

Default `rationale_weight=0.1` is a starting hyperparameter, not a calibrated
balance. CE first averages valid tokens within each example, then averages over
rationale-bearing examples in the microbatch. Logged score/rationale losses and
coverage should guide later tuning. Scores at the API boundary remain on 0–9.

Inference first computes the continuous score using inputs alone. It rounds that
prediction half-up to an integer bin, inserts it in `Score: ...\nReasoning:\n`,
then greedily generates the rationale. The result retains the continuous score;
it does not reparse a score from generated text. Training uses the annotated bin,
so score-condition exposure mismatch remains an evaluation question. A generated
assessment is not proof of faithful internal reasoning or correct scoring.

## Memory and computation

JSONL byte offsets are indexed; images are decoded per batch. A dataset of 10,000
pairs does not require all decoded images or token tensors in memory. Default
workers=0, microbatch=1, accumulation=8; about 1,250 optimizer steps process
10,000 examples once, while forward/backward still occurs for every example.

The runner freezes base weights and stores optimizer state only for adapters/head.
Decoder gradient checkpointing recomputes activations. The joint forward obtains
only the final hidden states, without accumulating every decoder layer. It projects
only supervised rationale tokens through the vocabulary head, in checkpointed
chunks of 32. This avoids retaining `[batch, full_sequence, 151936]` logits and
their CE activations. It costs recomputation during backward. Score-only rows
need no vocabulary projection at all.

`configs/train-joint.yaml` uses BF16 base weights, FP32 trainable parameters,
SDPA, max image pixels 65,536, total length 1,024, and at most 128 rationale-body
tokens. Oversized full sequences fail rather than silently cutting image tokens.
Long rationales are bounded explicitly; increasing their budget affects cost.
MPS BF16 is gated by a native operation/backward probe; use `--dtype float32` if
the host does not support it. No bitsandbytes, FlashAttention or CPU operator
fallback is enabled by this configuration.

## Measured on this Mac, 2026-09-06

Hardware: Apple M4 Pro, 20 GPU cores, 48 GiB unified memory, macOS 26.6.2.
Software: PyTorch 2.8.0, Transformers 4.57.6, PEFT 0.17.1. Public checkpoint
revision: `89644892e4d85e24eaac8bacfd4f463576704203`.

All rows below use the **full pretrained 2B checkpoint**, frozen vision,
language LoRA + scalar head, SDPA, checkpointing, a synthetic 256×256 image,
batch 1, one warmup and four measured optimizer/microbatch steps. No private
training examples were used. All losses/gradients were finite; the head and all
56 LoRA-B tensors changed. Each run repeats one synthetic example; the three
runs use the same image with two text-length variants, not 15 distinct examples.
The device is one integrated GPU with 20 cores. GPU utilization was not recorded.

| Base precision | Text / rationale tokens | Total sequence | Max sampled Metal allocation | Mean compute seconds/pair | 10k-pair compute extrapolation |
|---|---:|---:|---:|---:|---:|
| BF16 | 256 / 64 | 443 | 4.82 GiB | 1.08 | 3.00 hours |
| FP32 | 256 / 64 | 443 | 9.41 GiB | 1.26 | 3.49 hours |
| BF16 | 768 / 128 | 1019 | 4.84 GiB | 3.15 | 8.75 hours |

These are short-run measurements, not a sustained 10,000-example run. Times
include synchronized model forward, backward and AdamW, excluding data loading,
preprocessing, profiling checks, eval, saving and generation. Real titles,
reasoning lengths, images, background apps and thermals change throughput.
The benchmark updates every sample; production accumulation reduces optimizer
frequency. BF16 clearly reduced sampled memory here; a large speedup over FP32
is not established by four steps.

Memory values are synchronized stage samples from
[`torch.mps.driver_allocated_memory`](https://docs.pytorch.org/docs/2.8/generated/torch.mps.driver_allocated_memory.html),
including allocator/driver allocations. They are **not true peaks or total
system RAM**; CPU RSS and GPU allocations are not safely additive on unified
memory. Sampled system swap increased by 2.46 GiB in the first BF16 run and
2.00 GiB in the FP32 run, and did not increase during the longer BF16 run.
These are system-wide observations, not attributable solely to this process;
watch macOS memory pressure during a longer run with other applications open.

The observed workloads fit and run on this machine. The data count mainly adds
time; longer sequences are the stronger local throughput constraint. Prefer BF16
for this Mac, keep the current token/pixel bounds initially, and measure a longer
representative run before committing to a full epoch schedule.

A follow-up smoke run saved the adapter after five updates on that same synthetic
square example (target score 5). Total loss went from 0.1305 to 0.0628, normalized
score loss from 0.0457 to 0.0049, and rationale CE from 0.8481 to 0.5792. These
losses are recorded before each update; they are fitting diagnostics, not held-out
quality measurements. None of these runs trained on the released private dataset.

With that adapter, scoring plus generation capped at 64 new rationale tokens took
2.86 seconds on average over three warmed requests. The model used the full
64-token budget, so the explanation was truncated. Post-request Metal allocation
samples reached 4.79 GiB. The output score was 5.13 on the training example; this
does not establish generalization. The saved adapter is approximately 6.15 MiB at
`artifacts/training/synthetic-square-demo/`, alongside its local timing reports.

Raw local reports: `artifacts/benchmarks/m4pro-joint-bf16-sdpa.json`,
`m4pro-joint-fp32-sdpa.json`, and `m4pro-joint-bf16-long.json`. These remain ignored
alongside cached weights. Reproduce with `benchmarks/profile_joint.py`:

```bash
uv run --locked --extra cpu --extra vlm python benchmarks/profile_joint.py \
  --dtype bfloat16 --attention sdpa --steps 4 --warmup-steps 1 \
  --output artifacts/benchmarks/new-profile.json
```

## Training, inference, checkpoints and logging

Run from the repository root, with uv >=0.8.22:

```bash
HF_HOME="$PWD/artifacts/hf" uv run --locked --extra cpu --extra vlm \
  multimodal-judge train-joint --config configs/train-joint.yaml \
  --max-steps 2 --output-dir artifacts/training/joint-first-run

HF_HOME="$PWD/artifacts/hf" uv run --locked --extra cpu --extra vlm \
  multimodal-judge judge --checkpoint artifacts/training/joint-first-run \
  --image /absolute/path/to/image.png --text 'Input description'
```

Each checkpoint/final directory stores the processor, native model config,
`joint_manifest.json`, resolved settings, and PEFT adapter with
`modules_to_save=["score_head"]`. The manifest records the base and resolved
Hub revision so later loading reconstructs the custom model before loading its
adapter/head. Local bases must retain their path; move/copy them together when
relocating an experiment. Resume uses `--resume-from-checkpoint <checkpoint-dir>`.

W&B project remains `multimodal-judge`, default offline. Numeric metrics include
total loss, normalized score Huber loss, rationale CE, rationale sample/token
counts and coverage, raw-scale MAE/RMSE, learning rate, gradient norm, parameters,
elapsed steps/time and sampled MPS tensor/driver bytes. Multiple logs at one
optimizer step are retained through an explicit `optimizer_step` axis. No images,
input text or generated rationale are sent to W&B by the runner.

The current dataset has 8 train / 1 validation / 1 test examples. Existing labels
are usable for plumbing checks, but neither this split nor synthetic benchmarks
establish scoring or rationale quality. Final rationale generation is opt-in via
`training.generate_eval`; it uses predicted scores and logs only aggregate score
metrics. Grounding and explanation quality still need their own held-out review.

## Verification

Synthetic numerical tests compare the sparse CE value and gradients against a
full vocabulary oracle, check missing/mixed rationale batches, causal isolation,
padding, and head serialization. Opt-in real-processor integration tests train a
tiny Qwen vision model, save/reload LoRA **and** head, infer in a fresh process,
and resume training. Full checkpoint profiling separately verifies MPS forward,
backward, finite gradients and parameter updates. CUDA remains untested.
