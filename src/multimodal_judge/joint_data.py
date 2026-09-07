"""Score regression data with bounded rationale supervision."""

import json
import math
from numbers import Real

from PIL import Image
import torch

from .training_data import JsonlScoreDataset


JOINT_INSTRUCTION = (
    "Assess the image together with the supplied text. Give a score from 0 to 9 "
    "and explain your assessment. Use this format:\nScore: <integer>\nReasoning:\n"
    "<explanation>\n\n"
)


def joint_messages(text: str, system_prompt: str = "") -> list[dict]:
    """Build fresh user messages for training and inference."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a nonempty string")
    if not isinstance(system_prompt, str):
        raise ValueError("system_prompt must be a string")
    messages = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})
    return messages + [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": JOINT_INSTRUCTION + text},
    ]}]


def reasoning_prefix(processor, text: str, score: float, system_prompt: str = "") -> str:
    """Render a generation prefix conditioned on a clipped, half-up rounded score."""
    if isinstance(score, bool) or not isinstance(score, Real) or not math.isfinite(score):
        raise ValueError("score must be a finite number")
    integer = math.floor(min(9, max(0, score)) + 0.5)
    prefix = processor.apply_chat_template(
        joint_messages(text, system_prompt), tokenize=False, add_generation_prompt=True)
    return prefix + f"Score: {integer}\nReasoning:\n"


def _reasoning(value):
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError("reasoning must be a string or None")
    return value.strip()


class JointScoreDataset(JsonlScoreDataset):
    """Add stripped reasoning to the parent's lazy, RGB-validated records.

    Missing, null, empty, and whitespace-only reasoning mean regression-only.
    Like the parent, the JSONL and image files must not change during use.
    """

    def __getitem__(self, index):
        row = super().__getitem__(index)
        with self.path.open("rb") as handle:
            handle.seek(self.offsets[index])
            raw = json.loads(handle.readline())
        row["reasoning"] = _reasoning(raw.get("reasoning"))
        return row


class JointScoreCollator:
    """Collate model inputs, raw scores, score positions, and rationale labels.

    score_positions selects the final generation-prefix token, before the answer.
    Only the rationale body and one EOS receive CE labels; rows without reasoning
    remain prefix-only. Oversize vision-expanded sequences fail without truncation.
    """

    def __init__(self, processor, max_length=1024, max_reasoning_tokens=128, system_prompt=""):
        if not isinstance(system_prompt, str):
            raise ValueError("system_prompt must be a string")
        self.system_prompt = system_prompt
        for name, value in (("max_length", max_length),
                            ("max_reasoning_tokens", max_reasoning_tokens)):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.processor = processor
        self.max_length = max_length
        self.max_reasoning_tokens = max_reasoning_tokens
        self.processor.tokenizer.padding_side = "right"

    def _bounded_reasoning(self, reason):
        tokenizer = self.processor.tokenizer
        ids = tokenizer(reason, add_special_tokens=False, truncation=True,
                        max_length=self.max_reasoning_tokens)["input_ids"]
        reason = tokenizer.decode(ids, skip_special_tokens=False,
                                  clean_up_tokenization_spaces=False).strip()
        # A partial Unicode token can retokenize longer after decoding.
        while reason and len(tokenizer(reason, add_special_tokens=False)["input_ids"]) > (
            self.max_reasoning_tokens
        ):
            reason = reason[:-1].rstrip()
        if not reason:
            raise ValueError("reasoning has no tokens after bounding")
        return reason

    def __call__(self, examples):
        if not examples:
            raise ValueError("Cannot collate an empty batch")
        processor = self.processor
        processor.tokenizer.padding_side = "right"
        texts, prefixes, conditioning, images, scores, present = [], [], [], [], [], []
        for example in examples:
            score = example.get("score")
            if type(score) is not int or not 0 <= score <= 9:
                raise ValueError("score must be an integer from 0 to 9")
            if not isinstance(example.get("image"), Image.Image):
                raise ValueError("image must be a PIL image")
            prefix = processor.apply_chat_template(
                joint_messages(example.get("text"), self.system_prompt), tokenize=False, add_generation_prompt=True)
            reason = _reasoning(example.get("reasoning"))
            conditioned = prefix
            full = prefix
            if reason:
                reason = self._bounded_reasoning(reason)
                if not processor.tokenizer.eos_token or processor.tokenizer.eos_token_id is None:
                    raise ValueError("Processor tokenizer must define EOS for rationale supervision")
                conditioned = reasoning_prefix(processor, example["text"], score, self.system_prompt)
                full = conditioned + reason + processor.tokenizer.eos_token
            prefixes.append(prefix)
            conditioning.append(conditioned)
            texts.append(full)
            images.append(example["image"])
            scores.append(score)
            present.append(bool(reason))

        def encode(rendered):
            return processor(text=rendered, images=images, padding=True,
                             truncation=False, return_tensors="pt")

        full, prefix, conditioned = encode(texts), encode(prefixes), encode(conditioning)
        labels = torch.full_like(full["input_ids"], -100, dtype=torch.long)
        positions = []
        for index, has_reason in enumerate(present):
            lengths = []
            for batch in (full, prefix, conditioned):
                mask = batch["attention_mask"][index].bool()
                length = int(mask.sum())
                if length == 0 or not mask[:length].all() or mask[length:].any():
                    raise ValueError("Processor must right-pad nonempty sequences")
                lengths.append(length)
            length, boundary, rationale_start = lengths
            if length > self.max_length:
                raise ValueError(f"Batch item {index} exceeds max_length; truncation is disabled")
            for other, end in ((prefix, boundary), (conditioned, rationale_start)):
                if end > length or not torch.equal(
                    full["input_ids"][index, :end], other["input_ids"][index, :end]
                ):
                    raise ValueError(f"Batch item {index}: full/prefix token mismatch")
            positions.append(boundary - 1)
            if has_reason:
                body_length = length - rationale_start - 1
                if not 0 < body_length <= self.max_reasoning_tokens:
                    raise ValueError(f"Batch item {index}: expanded rationale exceeds token bound "
                                     "or is empty")
                if full["input_ids"][index, length - 1] != processor.tokenizer.eos_token_id:
                    raise ValueError("Rationale must end in exactly one EOS token")
                labels[index, rationale_start:length] = full["input_ids"][
                    index, rationale_start:length
                ]
        full["labels"] = labels
        full["score_positions"] = torch.tensor(positions, dtype=torch.long)
        full["scores"] = torch.tensor(scores, dtype=torch.float32)
        return full
