"""Synthetic tests only; local processor integration never loads sample records."""

import json
import os
import shutil
from unittest.mock import patch

from PIL import Image
import pytest
import torch

from multimodal_judge.joint_data import (
    JointScoreCollator, JointScoreDataset, joint_messages, reasoning_prefix,
)


class Tokenizer:
    padding_side = "left"
    eos_token = "~"
    eos_token_id = ord("~")

    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None):
        assert not add_special_tokens
        ids = list(map(ord, text))
        return {"input_ids": ids[:max_length] if truncation else ids}

    def decode(self, ids, **kwargs):
        return "".join(chr(int(i)) for i in ids)


class Processor:
    def __init__(self, mismatch=False, left_pad=False):
        self.tokenizer = Tokenizer()
        self.calls = []
        self.mismatch = mismatch
        self.left_pad = left_pad

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is False and add_generation_prompt is True
        system = messages[0]["content"] if messages[0]["role"] == "system" else ""
        assert messages[-1]["content"][0] == {"type": "image"}
        return system + "<image>" + messages[-1]["content"][1]["text"] + "|assistant|"

    def __call__(self, *, text, images, padding, truncation, return_tensors):
        assert padding is True and truncation is False and return_tensors == "pt"
        assert self.tokenizer.padding_side == "right"
        assert all(isinstance(image, Image.Image) for image in images)
        self.calls.append(text)
        rows = []
        for rendered in text:
            # Expanded vision length must count in all boundaries.
            ids = [900, 901, 902, 903] + list(map(ord, rendered.removeprefix("<image>")))
            if self.mismatch and rendered.endswith("|assistant|"):
                ids[0] = 999
            rows.append(ids)
        width = max(map(len, rows))
        if self.left_pad:
            ids = [[126] * (width - len(r)) + r for r in rows]
            masks = [[0] * (width - len(r)) + [1] * len(r) for r in rows]
        else:
            ids = [r + [126] * (width - len(r)) for r in rows]
            masks = [[1] * len(r) + [0] * (width - len(r)) for r in rows]
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(masks),
                "pixel_values": torch.ones(len(rows), 3),
                "image_grid_thw": torch.ones(len(rows), 3, dtype=torch.long)}


def examples():
    return [{"text": "Synthetic", "score": 0, "image": Image.new("RGB", (56, 56)),
             "reasoning": "Clear image."},
            {"text": "Longer synthetic text", "score": 9,
             "image": Image.new("RGB", (56, 56))}]


def targets(processor, batch, index=0):
    row = batch["labels"][index]
    return processor.tokenizer.decode(row[row != -100])


def test_mixed_boundaries_padding_and_dtypes():
    processor = Processor()
    batch = JointScoreCollator(processor)(examples())
    assert targets(processor, batch) == "Clear image.~"
    assert targets(processor, batch, 1) == ""
    assert batch["score_positions"].dtype == torch.long
    assert batch["score_positions"].shape == (2,)
    assert batch["scores"].dtype == torch.float32
    assert batch["scores"].tolist() == [0., 9.]
    assert batch["labels"].dtype == torch.long
    assert batch["labels"].shape == batch["input_ids"].shape
    assert "pixel_values" in batch and "image_grid_thw" in batch
    for i, rendered in enumerate(processor.calls[1]):
        expected = 4 + len(rendered.removeprefix("<image>")) - 1
        assert batch["score_positions"][i] == expected
        assert batch["attention_mask"][i, expected] == 1
        assert (batch["labels"][i, :expected + 1] == -100).all()
    assert (batch["labels"][batch["attention_mask"] == 0] == -100).all()
    assert processor.calls[0][1].endswith("|assistant|")
    assert processor.calls[0][0].endswith("Score: 0\nReasoning:\nClear image.~")


@pytest.mark.parametrize("reason", [None, "", " \n "])
def test_all_no_rationale_are_prefix_only_and_masked(reason):
    rows = [dict(row, reasoning=reason) for row in examples()]
    processor = Processor()
    batch = JointScoreCollator(processor)(rows)
    assert (batch["labels"] == -100).all()
    assert processor.calls[0] == processor.calls[1]
    assert torch.equal(batch["score_positions"], batch["attention_mask"].sum(1) - 1)


def test_score_position_prefix_independent_of_all_labels():
    row = examples()[0]
    batches = [JointScoreCollator(Processor())([dict(row, score=s, reasoning=r)])
               for s, r in [(0, "First"), (9, "Other explanation"), (4, "")]]
    end = int(batches[0]["score_positions"][0]) + 1
    for batch in batches[1:]:
        assert int(batch["score_positions"][0]) + 1 == end
        assert torch.equal(batch["input_ids"][0, :end], batches[0]["input_ids"][0, :end])


def test_reasoning_bound_and_full_length_failure():
    processor = Processor()
    rows = [dict(examples()[0], reasoning="abcdefghijk")]
    batch = JointScoreCollator(processor, max_reasoning_tokens=4)(rows)
    assert targets(processor, batch) == "abcd~"
    length = batch["input_ids"].shape[1]
    JointScoreCollator(processor, max_length=length, max_reasoning_tokens=4)(rows)
    with pytest.raises(ValueError, match="exceeds max_length"):
        JointScoreCollator(processor, max_length=length - 1, max_reasoning_tokens=4)(rows)
    with pytest.raises(ValueError, match="exceeds max_length"):
        JointScoreCollator(processor, max_length=10)([examples()[1]])


@pytest.mark.parametrize("kwargs,match", [
    ({"mismatch": True}, "token mismatch"), ({"left_pad": True}, "right-pad"),
])
def test_processor_alignment_and_padding_required(kwargs, match):
    with pytest.raises(ValueError, match=match):
        JointScoreCollator(Processor(**kwargs))(examples())


@pytest.mark.parametrize("name", ["max_length", "max_reasoning_tokens"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_positive_integer_limits(name, value):
    with pytest.raises(ValueError, match=name):
        JointScoreCollator(Processor(), **{name: value})


@pytest.mark.parametrize("score,integer", [(-8, 0), (0.49, 0), (0.5, 1), (4.5, 5), (20, 9)])
def test_shared_helpers_half_up_and_fresh_messages(score, integer):
    processor = Processor()
    prefix = processor.apply_chat_template(joint_messages("Synthetic"), False, True)
    assert reasoning_prefix(processor, "Synthetic", score) == (
        prefix + f"Score: {integer}\nReasoning:\n")
    joint_messages("Synthetic")[0]["content"].clear()
    assert len(joint_messages("Synthetic")[0]["content"]) == 2


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "5"])
def test_invalid_inference_score(value):
    with pytest.raises(ValueError, match="finite"):
        reasoning_prefix(Processor(), "Synthetic", value)


def test_lazy_relocated_dataset_and_optional_reasoning(tmp_path):
    original = tmp_path / "original"
    original.mkdir()
    Image.new("L", (3, 4), 100).save(original / "image.png")
    row = {"text": "Synthetic", "score": 9, "image": "image.png", "id": "synthetic"}
    rows = [dict(row, reasoning=" Explanation. "), row, dict(row, reasoning=None),
            dict(row, reasoning=" \n"), dict(row, reasoning=42)]
    (original / "train.jsonl").write_text("\n" + "\n".join(map(json.dumps, rows)))
    relocated = tmp_path / "relocated"
    shutil.copytree(original, relocated)
    (original / "image.png").unlink()
    with patch("multimodal_judge.training_data.Image.open") as opened, \
         patch("multimodal_judge.joint_data.json.loads") as decoded:
        dataset = JointScoreDataset(relocated / "train.jsonl")
        assert len(dataset) == 5
        opened.assert_not_called()
        decoded.assert_not_called()
    assert dataset[0]["reasoning"] == "Explanation."
    assert dataset[0]["image"].mode == "RGB"
    assert dataset[0]["id"] == "synthetic"
    assert [dataset[i]["reasoning"] for i in (1, 2, 3)] == ["", "", ""]
    with pytest.raises(ValueError, match="reasoning"):
        dataset[4]
    assert len(JointScoreDataset(relocated / "train.jsonl", max_samples=2)) == 2


@pytest.mark.parametrize("changes,match", [
    ({"reasoning": []}, "reasoning"), ({"score": 2.5}, "score"),
    ({"score": True}, "score"), ({"text": " "}, "text"), ({"image": None}, "PIL"),
])
def test_invalid_examples(changes, match):
    with pytest.raises(ValueError, match=match):
        JointScoreCollator(Processor())([dict(examples()[0], **changes)])


def test_empty_batch():
    with pytest.raises(ValueError, match="empty batch"):
        JointScoreCollator(Processor())([])


@pytest.mark.skipif(not os.environ.get('MMJUDGE_TEST_PROCESSOR'),
                    reason="Local processor fixture unavailable")
def test_real_processor_expanded_boundaries_and_bounded_unicode():
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(os.environ['MMJUDGE_TEST_PROCESSOR'],
                                               local_files_only=True)
    rows = examples()
    rows[0]["reasoning"] = "Synthetic image has a neutral appearance."
    batch = JointScoreCollator(processor)(rows)
    assert targets(processor, batch) == rows[0]["reasoning"] + processor.tokenizer.eos_token
    assert (batch["labels"][1] == -100).all()
    for i, row in enumerate(rows):
        text = processor.apply_chat_template(joint_messages(row["text"]),
                                              tokenize=False, add_generation_prompt=True)
        prefix = processor(text=[text], images=[row["image"]], return_tensors="pt")
        length = prefix["input_ids"].shape[1]
        assert batch["score_positions"][i] == length - 1
        assert torch.equal(batch["input_ids"][i, :length], prefix["input_ids"][0])
    assert (batch["labels"][batch["attention_mask"] == 0] == -100).all()
    for reason in ["A sentence. " * 50, "图像颜色协调。" * 20, "🙂🚀" * 20]:
        bounded = JointScoreCollator(processor, max_reasoning_tokens=3)(
            [dict(rows[0], reasoning=reason)])
        assert 1 < (bounded["labels"] != -100).sum() <= 4
        assert bounded["input_ids"][0, -1] == processor.tokenizer.eos_token_id


def test_system_prompt_is_separate_and_not_supervised():
    system = "Use this rubric.\nOnly assess the title and image."
    messages = joint_messages("Synthetic title", system)
    assert messages[0] == {"role": "system", "content": system}
    assert messages[1]["role"] == "user"
    assert system not in messages[1]["content"][1]["text"]
    assert len(joint_messages("Synthetic title")) == 1
    processor = Processor()
    batch = JointScoreCollator(processor, system_prompt=system)(examples())
    assert all(text.startswith(system) for call in processor.calls for text in call)
    assert targets(processor, batch) == "Clear image.~"
    for i, position in enumerate(batch["score_positions"]):
        assert (batch["labels"][i, :position + 1] == -100).all()
