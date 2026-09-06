# Joint pointwise judge

One Qwen3-VL backbone produces a continuous 0–9 score and an optional rationale.

![Architecture](assets/joint-architecture.png)

## Model

The 2B configuration has 28 decoder layers and hidden width 2048. The vision
encoder, embeddings and base decoder/LM-head weights are frozen. Attention q/v
projections use rank-8 LoRA; a trainable `Linear(2048, 1)` head returns
`9 * sigmoid(score_head(hidden))`. Base weights use BF16; adapters, scoring head and loss
calculations retain FP32.

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
loss = Huber(predicted_score / 9, target_score / 9; delta=0.1)
       + 0.1 * rationale_CE
```

CE averages tokens per sample, then samples with rationale. At inference, the
model predicts the continuous score, rounds it to an integer conditioning bin,
and generates the rationale. Training uses the gold bin. This mismatch needs
held-out evaluation; generated explanations do not establish reasoning fidelity.

## Runtime

Defaults: batch 1, gradient accumulation 8, gradient checkpointing, at most
65,536 image pixels and 1,024 sequence tokens. Vocabulary logits are computed
in chunks only for supervised rationale tokens. Images load on demand.

Checkpoints include LoRA, the scoring head, processor, configuration and a
manifest identifying the base model/revision. Resume from a Trainer checkpoint,
not the final inference adapter. The [README](../README.md) covers commands,
Docker and W&B; tests cover masking, losses/gradients and save/load/resume.

The M4 Pro 48 GiB smoke used one synthetic example repeated for five updates:
about 1.07 s/update and 4.82 GiB sampled Metal allocation. Scoring plus generation
capped at 64 tokens averaged 2.86 s over three warm requests. These are short-run
measurements, not peak memory, held-out quality or a 10,000-example training run.
