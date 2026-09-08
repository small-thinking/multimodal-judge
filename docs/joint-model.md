# Joint pointwise judge

One Qwen3-VL backbone produces a continuous 0–9 score and an optional rationale.
Choose `objective.head_type: regression` (default) or `classification` when training;
inference restores the saved head automatically. The illustration shows regression.

![Architecture](assets/joint-architecture.png)

## Model

The 2B configuration has 28 decoder layers and hidden width 2048. The vision
encoder, embeddings and base decoder/LM-head weights are frozen. Attention q/v
projections use rank-8 LoRA. The scoring head has two alternatives:

| Head | Layer | Training loss | Continuous rating |
|---|---|---|---|
| `regression` | `Linear(2048, 1)` | Huber on normalized 0–1 scores | `9 * sigmoid(logit)` |
| `classification` | `Linear(2048, 10)` | Cross-entropy on integer labels 0–9 | `sum(k * softmax(logits)[k])` |

Classification uses one head with ten mutually exclusive classes, including zero.
The expected rating, rather than the argmax class, preserves continuous MAE/RMSE
and the existing inference JSON contract. Its probabilities are model predictions,
not an observed distribution of human ratings. This is a new randomly initialized
head, not a reuse of the pretrained LM vocabulary head or an ordinal-loss method.
Base weights use BF16; adapters, scoring head and loss calculations retain FP32.

The scoring head reads the final input-prefix token, before the annotated score
or rationale. Causal attention prevents those future labels from influencing
its prediction. Both outputs share the same backbone.

## Supervision and inference

```text
[image + text + assistant prefix] [Score: 5] [Reasoning: rationale + EOS]
                              ^
                       score readout
```

Every sample supervises the score. Samples with a rationale also supervise its
text and EOS; prompt, score-prefix and padding tokens are masked from CE.
Missing rationale means zero rationale loss.

```text
score_loss = Huber(predicted_score / 9, target_score / 9; delta=0.1)  # regression
score_loss = cross_entropy(class_logits, target_score)              # classification
loss = score_weight * score_loss + rationale_weight * rationale_CE
```

Defaults are `score_weight: 1.0` and `rationale_weight: 0.1`; loss component logs
remain unweighted. Increasing score weight increases the relative score gradient,
but does not guarantee better validation MAE/RMSE. `huber_delta` only affects
regression. Ten-class CE and normalized Huber have different scales and optimize
different targets: equal weights across head types do not mean equal task balance.
For a comparison that removes rationale-task balancing, set `rationale_weight: 0`
in both configurations, keeping data, batch size, seed and update budget fixed.

Reasoning CE uses teacher-forced reference prefixes. Lower CE can coexist with
repetition in free generation. Exponentiating that same CE adds no independent
evidence; standard token-weighted perplexity also differs from this project's
mean of per-sample token losses. Assess generation with fixed decoding settings,
repetition and termination checks, and a separate quality review.

CE averages tokens per sample, then samples with rationale. At inference, the
model predicts the continuous score, rounds it to an integer conditioning bin,
and generates the rationale. Training uses the gold bin. This mismatch needs
held-out evaluation; generated explanations do not establish reasoning fidelity.

## Repetition monitoring

Set `training.repetition_eval: true` to generate a rationale for every validation
row at each `eval_steps` interval and at final evaluation. This adds one W&B curve,
`validation/reasoning_rep4` (0–1, lower is less literal repetition). Generation uses
the predicted score, greedy decoding and `training.max_new_tokens`, matching joint
inference; it never receives the gold score or reference rationale. This adds a
full generation pass to validation, so evaluation takes longer.

For each answer, Rep-4 is `1 - unique_4grams / total_4grams`, computed on generated
token IDs after excluding the prompt and tokenizer special tokens. The reported
value is the mean over answers with at least four body tokens. Shorter answers
are undefined and excluded; eligible/short counts are stored in local Trainer
logs and `metrics.json`, not additional W&B curves. If every answer is too short,
the metric is omitted rather than reported as zero. Generated text/token IDs are
discarded, never uploaded or written by this metric. Rep-4 measures literal token
repetition, not the percentage of answers with loops or reasoning correctness.
Compare runs with the same validation rows, tokenizer and generation budget;
shorter or incorrect answers can also have lower repetition.

The batch-16 config enables this monitor. Its generation budget is 129 tokens:
128 supervised rationale-body tokens plus one EOS. The collator explicitly
supervises EOS and shares the inference chat prefix. Long reference reasons are
still truncated at the token limit, potentially mid-sentence; the extra generation
token fixes the EOS budget boundary, not that truncation policy or repetition.

Start a fresh classification experiment with live aggregate W&B logging:

```bash
uv run --env-file .env multimodal-judge train-joint \
  --config configs/train-joint-batch16.yaml \
  --score-head classification --score-weight 1 \
  --wandb-mode online
```

The CLI creates a new timestamped output directory. Use `--score-head regression`
for the existing Huber head; changing the head requires a fresh run.

## Runtime

Defaults: batch 1, gradient accumulation 8, gradient checkpointing, at most
65,536 image pixels and 1,024 sequence tokens. Vocabulary logits are computed
in chunks only for supervised rationale tokens. Images load on demand.

Checkpoints include LoRA, the scoring head, processor, configuration and a
manifest identifying the base model/revision and score-head configuration. Legacy
checkpoints without `head_type` or `score_weight` keep regression and weight 1.
Changing head type or objective weights requires a fresh run; resume checks reject
these changes rather than reinterpret saved weights. Resume from a Trainer checkpoint,
not the final inference adapter. The [README](../README.md) covers commands,
Docker and W&B; tests cover masking, losses/gradients and save/load/resume.

The M4 Pro 48 GiB smoke used one synthetic example repeated for five updates:
about 1.07 s/update and 4.82 GiB sampled Metal allocation. Scoring plus generation
capped at 64 tokens averaged 2.86 s over three warm requests. These are short-run
measurements, not peak memory, held-out quality or a 10,000-example training run.
