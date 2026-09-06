import json
import shutil
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image
import pytest
import torch

from multimodal_judge.training_data import (
    JsonlScoreDataset, ScoreCollator, inspect_data, score_messages,
)


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


@pytest.fixture
def data(tmp_path):
    Image.new("L", (3, 4), 100).save(tmp_path / "image.png")
    row = {"image": "image.png", "text": "Synthetic café", "score": 9, "id": "sample-a"}
    write_rows(tmp_path / "train.jsonl", [row, dict(row, text="Synthetic second", score=0)])
    return tmp_path, row


def test_lazy_index_rgb_and_relocated_paths(data, tmp_path):
    root, row = data
    destination = root / "relocated"
    destination.mkdir()
    shutil.copy(root / "image.png", destination)
    shutil.copy(root / "train.jsonl", destination)
    with patch("multimodal_judge.training_data.Image.open") as opened, \
         patch("multimodal_judge.training_data.json.loads") as decoded:
        dataset = JsonlScoreDataset(destination / "train.jsonl")
        assert len(dataset) == 2
        assert all(type(offset) is int for offset in dataset.offsets)
        opened.assert_not_called()
        decoded.assert_not_called()
    (root / "image.png").unlink()
    with patch("multimodal_judge.training_data.Image.open", wraps=Image.open) as opened:
        item = dataset[1]
        assert opened.call_count == 1
    assert item["image"].mode == "RGB"
    assert item["image"].getpixel((0, 0)) == (100, 100, 100)
    assert item["score"] == 0
    assert item["text"] == "Synthetic second"
    assert dataset[0]["id"] == row["id"]
    with pytest.raises(IndexError):
        dataset[2]


def test_limits_and_blank_lines(data):
    root, row = data
    path = root / "train.jsonl"
    path.write_bytes(b"\n" + path.read_bytes() + b"\n")
    assert len(JsonlScoreDataset(path, 0)) == 0
    assert len(JsonlScoreDataset(path, 1)) == 1
    assert len(JsonlScoreDataset(path)) == 2
    for value in (-1, True, 1.5):
        with pytest.raises(ValueError, match="max_samples"):
            JsonlScoreDataset(path, value)


@pytest.mark.parametrize("changes,match", [
    ({"score": True}, "score"), ({"score": 10}, "score"),
    ({"score": -1}, "score"), ({"score": 3.0}, "score"),
    ({"score": "3"}, "score"), ({"score": None}, "score"),
    ({"text": None}, "text"), ({"text": " "}, "text"),
    ({"image": None}, "image"), ({"image": "missing.png"}, "image file"),
    ({"image": "\x00"}, "image file"), ({"id": []}, "id"),
])
def test_schema_errors(data, changes, match):
    root, row = data
    row.update(changes)
    write_rows(root / "train.jsonl", [row])
    with pytest.raises(ValueError, match=match):
        JsonlScoreDataset(root / "train.jsonl")[0]
    with pytest.raises(ValueError, match=match):
        inspect_data(root)


@pytest.mark.parametrize("raw", [b"{PRIVATE_SENTINEL", b"[]", b"\xff"])
def test_invalid_json_safe_errors(data, raw):
    root, _ = data
    (root / "train.jsonl").write_bytes(raw)
    with pytest.raises(ValueError) as error:
        inspect_data(root)
    assert "PRIVATE_SENTINEL" not in str(error.value)


def test_inspect_no_decode_and_lineage_overlap(data):
    root, row = data
    row.update(image_sha256="original-hash", reasoning="PRIVATE_SENTINEL")
    write_rows(root / "train.jsonl", [row])
    variant = dict(row, id="variant", image_sha256="changed-hash",
                   parent_id=row["id"], parent_image_sha256=row["image_sha256"])
    write_rows(root / "validation.jsonl", [variant])
    write_rows(root / "test.jsonl", [row])
    # Path existence is sufficient: inspection must not try to decode this file.
    (root / "image.png").write_bytes(b"not an image")
    before = {p.name: p.read_bytes() for p in root.glob("*.jsonl")}
    with patch("multimodal_judge.training_data.Image.open") as opened:
        result = inspect_data(root)
        opened.assert_not_called()
    assert result["splits"]["train"]["count"] == 1
    assert result["splits"]["test"]["count"] == 1
    assert result["splits"]["train"]["score_histogram"]["9"] == 1
    assert result["overlaps"]["train_validation"] == {
        "id": 0, "image_sha256": 0, "id_lineage": 1, "image_lineage": 1,
    }
    assert result["overlaps"]["train_test"]["image_sha256"] == 1
    serialized = json.dumps(result)
    for secret in (row["text"], "PRIVATE_SENTINEL", "original-hash", "sample-a"):
        assert secret not in serialized
    assert before == {p.name: p.read_bytes() for p in root.glob("*.jsonl")}
    with pytest.raises(ValueError, match="decoding failed"):
        JsonlScoreDataset(root / "train.jsonl")[0]


def test_inspect_missing_empty_and_custom_splits(data):
    root, _ = data
    (root / "validation.jsonl").touch()
    result = inspect_data(root)
    assert result["splits"]["validation"]["count"] == 0
    assert not result["splits"]["test"]["exists"]
    (root / "train.jsonl").rename(root / "custom.jsonl")
    with pytest.raises(FileNotFoundError):
        inspect_data(root)
    assert inspect_data(root, train_file="custom.jsonl")["splits"]["train"]["count"] == 2


class FakeProcessor:
    """Expand an image into multiple IDs and preserve extra model inputs."""

    def __init__(self, mismatch=False, no_target=False):
        self.tokenizer = SimpleNamespace(padding_side="left")
        self.calls = []
        self.mismatch = mismatch
        self.no_target = no_target

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is False
        assert messages[0]["content"][0] == {"type": "image"}
        text = messages[0]["content"][1]["text"]
        if add_generation_prompt:
            assert len(messages) == 1
            return text + "|assistant|"
        assert len(messages) == 2
        score = messages[1]["content"][0]["text"]
        assert score in [str(i) for i in range(10)]
        return text + "|assistant|" + score

    def __call__(self, *, text, images, padding, truncation, return_tensors):
        assert self.tokenizer.padding_side == "right"
        assert padding is True and truncation is False and return_tensors == "pt"
        assert all(isinstance(image, Image.Image) for image in images)
        self.calls.append((text, images))
        rows = []
        for rendered in text:
            prompt, score = rendered.split("|assistant|")
            ids = [1, 99, 99, 99] + [ord(char) + 100 for char in prompt] + [2]
            if score and not self.no_target:
                ids += [10 + int(score), 3]
            if self.mismatch and not score:
                ids[0] = 7
            rows.append(ids)
        width = max(map(len, rows))
        # Pad shares the EOS ID: labels must use the attention mask, not ID equality.
        return {
            "input_ids": torch.tensor([r + [3] * (width - len(r)) for r in rows]),
            "attention_mask": torch.tensor([[1] * len(r) + [0] * (width - len(r))
                                            for r in rows]),
            "pixel_values": torch.ones(len(rows), 3),
            "image_grid_thw": torch.ones(len(rows), 3, dtype=torch.long),
        }


def examples():
    return [{"image": Image.new("RGB", (2, 2)), "text": text, "score": score}
            for text, score in (("Synthetic", 0), ("Longer synthetic example", 9))]


def test_collator_boundaries_padding_and_model_inputs():
    processor = FakeProcessor()
    batch = ScoreCollator(processor)(examples())
    assert len(processor.calls) == 2
    for index, score in enumerate((0, 9)):
        assert batch["labels"][index][batch["labels"][index] != -100].tolist() == [10 + score, 3]
        assert (batch["labels"][index, :4] == -100).all()
        assert (batch["labels"][index][batch["attention_mask"][index] == 0] == -100).all()
    assert "pixel_values" in batch and "image_grid_thw" in batch
    assert processor.calls[0][1] == processor.calls[1][1]
    assert (batch["input_ids"] != -100).all()


def test_collator_rejects_oversize_without_truncating():
    processor = FakeProcessor()
    full = ScoreCollator(processor)(examples())
    length = int(full["attention_mask"].sum(dim=1).max())
    ScoreCollator(processor, max_length=length)(examples())
    with pytest.raises(ValueError, match="exceeds max_length"):
        ScoreCollator(processor, max_length=length - 1)(examples())


@pytest.mark.parametrize("options,match", [
    ({"mismatch": True}, "token mismatch"),
    ({"no_target": True}, "all target tokens are masked"),
])
def test_collator_rejects_bad_prefix_and_empty_target(options, match):
    with pytest.raises(ValueError, match=match):
        ScoreCollator(FakeProcessor(**options))(examples())


def test_score_messages_shared_and_fresh():
    first = score_messages("Synthetic")
    assert "0 to 9" in first[0]["content"][1]["text"]
    first[0]["content"].clear()
    assert len(score_messages("Synthetic")[0]["content"]) == 2
    with pytest.raises(ValueError):
        score_messages(123)
