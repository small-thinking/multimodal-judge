"""Lazy score-only SFT data and metadata-only split inspection."""

import json
from itertools import combinations
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset

SCORE_MIN = 0
SCORE_MAX = 9
SCORE_INSTRUCTION = (
    "Evaluate the image together with the supplied text. "
    "Return only an integer score from 0 to 9, with no explanation.\n\n"
)
_ID_FIELDS = ("id", "parent_id", "original_id")
_HASH_FIELDS = ("image_sha256", "parent_image_sha256", "original_image_sha256")
_SAFE_FIELDS = _ID_FIELDS + _HASH_FIELDS



def score_messages(text: str) -> list[dict]:
    """Return a fresh user chat for score-only training or generation.

    Render with tokenize=False, add_generation_prompt=True and pass the PIL
    image separately to processor(images=..., text=...).
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a nonempty string")
    return [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": SCORE_INSTRUCTION + text},
    ]}]


def _row(raw: bytes, path: Path, location: int) -> dict:
    """Validate one row without echoing its contents in errors."""
    context = f"{path} at byte {location}"
    try:
        row = json.loads(raw)
    except (ValueError, UnicodeError):
        raise ValueError(f"{context}: invalid JSON") from None
    if not isinstance(row, dict):
        raise ValueError(f"{context}: expected a JSON object")
    if not isinstance(row.get("text"), str) or not row["text"].strip():
        raise ValueError(f"{context}: text must be a nonempty string")
    if type(row.get("score")) is not int or not SCORE_MIN <= row["score"] <= SCORE_MAX:
        raise ValueError(f"{context}: score must be an integer from 0 to 9")
    if not isinstance(row.get("image"), str) or not row["image"].strip():
        raise ValueError(f"{context}: image must be a nonempty path string")
    for field in _SAFE_FIELDS:
        if field in row and (not isinstance(row[field], str) or not row[field].strip()):
            raise ValueError(f"{context}: {field} must be a nonempty string")
    try:
        image = Path(row["image"])
        if not image.is_absolute():
            image = path.parent / image
        exists = image.is_file()
    except (OSError, ValueError):
        exists = False
    if not exists:
        raise ValueError(f"{context}: image file is missing or inaccessible")
    return {"image": image, "text": row["text"], "score": row["score"],
            **{key: row[key] for key in _SAFE_FIELDS if key in row}}


class JsonlScoreDataset(Dataset):
    """Index byte offsets only; validate and decode a single RGB image on access.

    Blank lines are ignored. The JSONL and images must remain unchanged while
    this dataset is used. Each access opens its own files, including in workers.
    Use inspect_data for exhaustive schema validation before training.
    """

    def __init__(self, path: Path, max_samples: int | None = None):
        if max_samples is not None and (type(max_samples) is not int or max_samples < 0):
            raise ValueError("max_samples must be a nonnegative integer or None")
        self.path = Path(path).resolve()
        self.offsets: list[int] = []
        with self.path.open("rb") as handle:
            while max_samples is None or len(self.offsets) < max_samples:
                offset = handle.tell()
                raw = handle.readline()
                if not raw:
                    break
                if raw.strip():
                    self.offsets.append(offset)

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, index):
        offset = self.offsets[index]
        with self.path.open("rb") as handle:
            handle.seek(offset)
            row = _row(handle.readline(), self.path, offset)
        try:
            with Image.open(row["image"]) as image:
                row["image"] = image.convert("RGB")
        except (OSError, ValueError):
            raise ValueError(f"{self.path} at byte {offset}: image decoding failed") from None
        return row


def inspect_data(directory, train_file="train.jsonl", validation_file="validation.jsonl"):
    """Stream schema/path checks without decoding images or returning raw records.

    Training is required; validation and test may be absent or empty. Overlaps
    count distinct shared identifiers/hashes, not row pairs. Lineage comparisons
    include current, parent, and original fields. Test is inspected only.
    """
    directory = Path(directory).resolve()
    splits, identities = {}, {}
    for name, filename in (("train", train_file), ("validation", validation_file),
                           ("test", "test.jsonl")):
        path = directory / filename
        exists = path.is_file()
        if name == "train" and not exists:
            raise FileNotFoundError(f"Training JSONL is missing: {path}")
        info = {"path": str(path), "exists": exists, "count": 0,
                "score_histogram": {str(i): 0 for i in range(SCORE_MIN, SCORE_MAX + 1)}}
        sets = {key: set() for key in ("id", "image_sha256", "id_lineage", "image_lineage")}
        if exists:
            with path.open("rb") as handle:
                while True:
                    offset = handle.tell()
                    raw = handle.readline()
                    if not raw:
                        break
                    if not raw.strip():
                        continue
                    row = _row(raw, path, offset)
                    info["count"] += 1
                    info["score_histogram"][str(row["score"])] += 1
                    for key in ("id", "image_sha256"):
                        if key in row:
                            sets[key].add(row[key])
                    sets["id_lineage"].update(row[k] for k in _ID_FIELDS if k in row)
                    sets["image_lineage"].update(row[k] for k in _HASH_FIELDS if k in row)
        splits[name], identities[name] = info, sets
    overlaps = {
        f"{left}_{right}": {key: len(identities[left][key] & identities[right][key])
                           for key in identities[left]}
        for left, right in combinations(splits, 2)
    }
    return {"splits": splits, "overlaps": overlaps}


class ScoreCollator:
    """Use processor-expanded prefixes to supervise only assistant score replies."""

    def __init__(self, processor, max_length=2048):
        if type(max_length) is not int or max_length <= 0:
            raise ValueError("max_length must be a positive integer")
        self.processor = processor
        self.max_length = max_length
        self.processor.tokenizer.padding_side = "right"

    def __call__(self, examples):
        if not examples:
            raise ValueError("Cannot collate an empty batch")
        full_texts, prefix_texts, images = [], [], []
        for example in examples:
            if type(example.get("score")) is not int or not 0 <= example["score"] <= 9:
                raise ValueError("score must be an integer from 0 to 9")
            if not isinstance(example.get("text"), str) or not example["text"].strip():
                raise ValueError("text must be a nonempty string")
            if not isinstance(example.get("image"), Image.Image):
                raise ValueError("image must be a PIL image")
            messages = score_messages(example["text"])
            assistant = {"role": "assistant", "content": [
                {"type": "text", "text": str(example["score"])}
            ]}
            prefix_texts.append(self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True))
            full_texts.append(self.processor.apply_chat_template(
                messages + [assistant], tokenize=False, add_generation_prompt=False))
            images.append(example["image"])
        self.processor.tokenizer.padding_side = "right"
        full = self.processor(text=full_texts, images=images, padding=True,
                              truncation=False, return_tensors="pt")
        prefix = self.processor(text=prefix_texts, images=images, padding=True,
                                truncation=False, return_tensors="pt")
        labels = full["input_ids"].clone()
        for index in range(len(examples)):
            mask = full["attention_mask"][index].bool()
            prefix_mask = prefix["attention_mask"][index].bool()
            length, boundary = int(mask.sum()), int(prefix_mask.sum())
            if length > self.max_length:
                raise ValueError(f"Batch item {index} exceeds max_length; truncation is disabled")
            if not mask[:length].all() or mask[length:].any():
                raise ValueError("Processor must right-pad full sequences")
            if not prefix_mask[:boundary].all() or prefix_mask[boundary:].any():
                raise ValueError("Processor must right-pad prefix sequences")
            if boundary > length or not torch.equal(
                full["input_ids"][index, :boundary], prefix["input_ids"][index, :boundary]
            ):
                raise ValueError(f"Batch item {index}: full/prefix token mismatch")
            labels[index, :boundary] = -100
            labels[index, ~mask] = -100
            if not (labels[index] != -100).any():
                raise ValueError(f"Batch item {index}: all target tokens are masked")
        full["labels"] = labels
        return full
