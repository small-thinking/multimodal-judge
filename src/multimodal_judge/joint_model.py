"""Qwen3-VL with configurable scoring and sparse, score-conditioned rationale SFT."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers import Qwen3VLForConditionalGeneration
from transformers.utils import ModelOutput

from .training_config import validate_number


@dataclass
class JointJudgeOutput(ModelOutput):
    loss: torch.Tensor | None = None
    # Trainer predictions are scores [batch, 1], not vocabulary logits.
    logits: torch.Tensor | None = None
    score_class_logits: torch.Tensor | None = None
    score_loss: torch.Tensor | None = None
    rationale_loss: torch.Tensor | None = None
    rationale_samples: torch.Tensor | None = None
    rationale_tokens: torch.Tensor | None = None


class JointQwen3VLForConditionalGeneration(Qwen3VLForConditionalGeneration):
    """Score the input prefix and generate reasons with the original LM head.

    Regression uses normalized Huber; classification uses ten-class CE and returns
    the probability-weighted expected score. Both expose continuous 0..9 logits.
    Rationale CE projects only supervised token states, in checkpointed chunks.
    """

    def __init__(self, config):
        super().__init__(config)
        defaults = {"head_type": "regression", "score_weight": 1.0,
                    "rationale_weight": 0.1, "huber_delta": 0.1,
                    "score_min": 0.0, "score_max": 9.0, "ce_chunk_size": 32}
        defaults.update(getattr(config, "judge_config", {}))
        if (defaults["score_min"], defaults["score_max"]) != (0.0, 9.0):
            raise ValueError("This dataset/model contract uses the raw 0..9 score scale")
        if defaults["head_type"] not in ("regression", "classification"):
            raise ValueError("head_type must be regression or classification")
        for key in ("score_weight", "rationale_weight", "huber_delta"):
            validate_number(defaults[key], key)
        if defaults["huber_delta"] == 0:
            raise ValueError("huber_delta must be positive")
        if type(defaults["ce_chunk_size"]) is not int or defaults["ce_chunk_size"] <= 0:
            raise ValueError("ce_chunk_size must be a positive integer")
        config.judge_config = defaults
        outputs = 10 if defaults["head_type"] == "classification" else 1
        self.score_head = nn.Linear(config.text_config.hidden_size, outputs)
        self._init_weights(self.score_head)
        # Set after head creation; the strict list preserves FP32 on BF16 loads.
        self._keep_in_fp32_modules_strict = ["score_head"]

    def _rationale_loss(self, hidden, labels):
        zero = hidden[:, 0, 0].sum() * 0.0
        sample_losses = []
        token_count = 0
        if labels is None:
            return zero, 0, 0
        # Shift labels: each hidden state predicts the next token.
        for sample_hidden, sample_labels in zip(hidden[:, :-1], labels[:, 1:]):
            selected = sample_labels != -100
            count = selected.sum().item()
            if not count:
                continue
            states, targets = sample_hidden[selected], sample_labels[selected]
            total = zero

            def token_loss(states_chunk, targets_chunk):
                logits = self.lm_head(states_chunk).float()
                return F.cross_entropy(logits, targets_chunk, reduction="sum")

            chunk_size = self.config.judge_config["ce_chunk_size"]
            for start in range(0, count, chunk_size):
                args = (states[start:start + chunk_size], targets[start:start + chunk_size])
                if self.training and torch.is_grad_enabled() and states.requires_grad:
                    # Recompute chunk logits in backward instead of retaining them.
                    total = total + checkpoint(token_loss, *args, use_reentrant=False)
                else:
                    total = total + token_loss(*args)
            sample_losses.append(total / count)
            token_count += count
        if not sample_losses:
            return zero, 0, 0
        return torch.stack(sample_losses).mean(), len(sample_losses), token_count

    def forward(
        self, input_ids=None, attention_mask=None, position_ids=None,
        past_key_values=None, inputs_embeds=None, labels=None, pixel_values=None,
        pixel_values_videos=None, image_grid_thw=None, video_grid_thw=None,
        cache_position=None, logits_to_keep=0, score_positions=None, scores=None,
        **kwargs,
    ):
        # Generation keeps HF's native KV-cache/rope handling and vocabulary output.
        if score_positions is None:
            if scores is not None:
                raise ValueError("scores require score_positions")
            return super().forward(
                input_ids=input_ids, attention_mask=attention_mask,
                position_ids=position_ids, past_key_values=past_key_values,
                inputs_embeds=inputs_embeds, labels=labels, pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos, image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw, cache_position=cache_position,
                logits_to_keep=logits_to_keep, **kwargs,
            )
        if input_ids is None or input_ids.ndim != 2:
            raise ValueError("Joint scoring requires input_ids [batch, sequence]")
        batch_size, length = input_ids.shape
        if score_positions.shape != (batch_size,) or score_positions.dtype != torch.long:
            raise ValueError("score_positions must be int64 [batch]")
        if ((score_positions < 0) | (score_positions >= length)).any():
            raise ValueError("score_positions must point inside the input sequence")
        rows = torch.arange(batch_size, device=input_ids.device)
        if attention_mask is not None and not attention_mask[rows, score_positions].bool().all():
            raise ValueError("score_positions cannot point to padding")
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("labels must match input_ids shape")
            prefix = torch.arange(length, device=input_ids.device)[None] <= score_positions[:, None]
            if (labels[prefix] != -100).any():
                raise ValueError("Prompt and score-position labels must be masked")
        if scores is not None:
            if scores.shape != (batch_size,) or not torch.isfinite(scores).all():
                raise ValueError("scores must be finite [batch]")
            if ((scores < 0) | (scores > 9)).any():
                raise ValueError("scores must be on the raw 0..9 scale")
            if self.config.judge_config["head_type"] == "classification" and (
                scores != scores.round()
            ).any():
                raise ValueError("Classification scores must be integer labels from 0 to 9")
        kwargs.pop("num_items_in_batch", None)
        kwargs.pop("output_hidden_states", None)
        kwargs.pop("return_dict", None)
        kwargs.pop("use_cache", None)
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
            inputs_embeds=inputs_embeds, pixel_values=pixel_values,
            image_grid_thw=image_grid_thw, pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw, use_cache=False,
            output_hidden_states=False, return_dict=True, **kwargs,
        )
        hidden = outputs.last_hidden_state
        # Read the head dtype through PEFT's wrapper; keep the scalar loss FP32.
        head_dtype = next(self.score_head.parameters()).dtype
        # Causal attention keeps this prefix state blind to the score and rationale.
        pooled = hidden[rows, score_positions].to(head_dtype)
        head_logits = self.score_head(pooled).float()
        class_logits = None
        if self.config.judge_config["head_type"] == "classification":
            class_logits = head_logits
            levels = torch.arange(10, device=head_logits.device, dtype=torch.float32)
            predicted = (head_logits.softmax(dim=-1) * levels).sum(dim=-1).clamp(0.0, 9.0)
        else:
            normalized = torch.sigmoid(head_logits).squeeze(-1)
            predicted = normalized * 9.0
        score_loss = predicted.sum() * 0.0
        if scores is not None:
            if class_logits is not None:
                score_loss = F.cross_entropy(class_logits, scores.long())
            else:
                score_loss = F.huber_loss(
                    normalized, scores.float() / 9.0,
                    delta=self.config.judge_config["huber_delta"], reduction="mean",
                )
        rationale_loss, sample_count, token_count = self._rationale_loss(hidden, labels)
        loss = None
        if scores is not None or labels is not None:
            objective = self.config.judge_config
            loss = (objective["score_weight"] * score_loss
                    + objective["rationale_weight"] * rationale_loss)
        return JointJudgeOutput(
            loss=loss, logits=predicted[:, None], score_class_logits=class_logits,
            score_loss=score_loss,
            rationale_loss=rationale_loss,
            rationale_samples=predicted.new_tensor(sample_count),
            rationale_tokens=predicted.new_tensor(token_count),
        )
