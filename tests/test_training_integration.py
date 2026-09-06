"""Opt-in real Qwen stack test, using synthetic data and random tiny weights.

Set MMJUDGE_TEST_PROCESSOR to a locally saved Qwen3-VL processor directory.
No pretrained model weights or dataset content are downloaded by this test.
"""

import json
import os
from pathlib import Path

import pytest


@pytest.mark.skipif(not os.environ.get("MMJUDGE_TEST_PROCESSOR"),
                    reason="Set MMJUDGE_TEST_PROCESSOR to a local Qwen3-VL processor")
def test_real_qwen_training_roundtrip(tmp_path):
    import torch
    import yaml
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLConfig, Qwen3VLForConditionalGeneration

    from multimodal_judge.training import run_training
    from multimodal_judge.training_data import JsonlScoreDataset, ScoreCollator

    torch.set_num_threads(2)
    processor = AutoProcessor.from_pretrained(os.environ["MMJUDGE_TEST_PROCESSOR"])
    model_dir = tmp_path / "tiny-base"
    model_config = Qwen3VLConfig(
        text_config={
            "vocab_size": len(processor.tokenizer), "hidden_size": 32,
            "intermediate_size": 64, "num_hidden_layers": 2,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
            "rope_scaling": {"rope_type": "default", "mrope_section": [2, 1, 1]},
        },
        vision_config={
            "depth": 1, "hidden_size": 32, "intermediate_size": 64, "num_heads": 4,
            "out_hidden_size": 32, "num_position_embeddings": 16,
            "deepstack_visual_indexes": [0],
        },
        eos_token_id=processor.tokenizer.eos_token_id,
        pad_token_id=processor.tokenizer.pad_token_id,
    )
    model = Qwen3VLForConditionalGeneration(model_config)
    model.save_pretrained(model_dir)
    processor.save_pretrained(model_dir)
    data = tmp_path / "data"
    data.mkdir()
    for split, color, text in [("train", "red", "A red square"),
                               ("validation", "blue", "A blue square with a longer title")]:
        Image.new("RGB", (64, 64), color).save(data / f"{split}.png")
        (data / f"{split}.jsonl").write_text(json.dumps({
            "image": f"{split}.png", "text": text, "score": 5,
            "id": split, "image_sha256": split,
        }) + "\n")
    examples = [JsonlScoreDataset(data / f"{s}.jsonl")[0]
                for s in ("train", "validation")]
    batch = ScoreCollator(processor)(examples)
    assert (batch["labels"] == -100).any()
    for labels in batch["labels"]:
        assert processor.tokenizer.decode(
            labels[labels != -100], skip_special_tokens=True
        ).strip() == "5"
    model.eval()
    with torch.no_grad():
        loss = model(**batch).loss
    assert torch.isfinite(loss)

    config = yaml.safe_load(Path("configs/train.yaml").read_text())
    config["model"]["name_or_path"] = str(model_dir)
    config["data"]["directory"] = str(data)
    config["runtime"]["device"] = os.environ.get("MMJUDGE_TEST_DEVICE", "cpu")
    output = tmp_path / "output"
    config["training"].update({
        "output_dir": str(output), "max_steps": 1,
        "gradient_accumulation_steps": 1, "save_steps": 1,
        "eval_steps": 1, "max_new_tokens": 2,
    })
    config["wandb"]["mode"] = "offline"
    run_training(config)
    assert list((output / "wandb").glob("offline-run-*/run-*.wandb"))
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["generation_count"] == 1
    assert 0 <= metrics["generation_invalid_rate"] <= 1
    assert metrics["train_samples"] == metrics["eval_samples"] == 1
    checkpoint = output / "checkpoint-1"
    assert (checkpoint / "trainer_state.json").is_file()
    from safetensors.torch import load_file

    weights = load_file(str(checkpoint / "adapter_model.safetensors"))
    assert any("lora_B" in key and value.abs().sum() > 0 for key, value in weights.items())
    assert not any("visual" in key for key in weights)
    # Cross the serialization boundary: reload optimizer/adapter and advance.
    config["training"]["max_steps"] = 2
    run_training(config, resume_from_checkpoint=str(checkpoint))
    state = json.loads((output / "checkpoint-2" / "trainer_state.json").read_text())
    assert state["global_step"] == 2
