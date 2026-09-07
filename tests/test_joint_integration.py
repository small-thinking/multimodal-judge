"""Opt-in native Qwen vision + dual loss + LoRA/head serialization integration."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(not os.environ.get("MMJUDGE_TEST_PROCESSOR"),
                    reason="Set MMJUDGE_TEST_PROCESSOR to a local Qwen3-VL processor")
def test_joint_training_reload_and_inference(tmp_path):
    import torch
    import yaml
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from safetensors.torch import load_file

    from test_joint_model import tiny_config
    from multimodal_judge.joint_data import JointScoreDataset, JointScoreCollator
    from multimodal_judge.joint_training import run_joint_training, predict_joint
    from multimodal_judge.joint_inference import load_joint_checkpoint

    torch.set_num_threads(2)
    processor = AutoProcessor.from_pretrained(os.environ["MMJUDGE_TEST_PROCESSOR"])
    config = tiny_config(len(processor.tokenizer))
    for key in ("image_token_id", "video_token_id", "vision_start_token_id",
                "vision_end_token_id"):
        setattr(config, key, {
            "image_token_id": 151655, "video_token_id": 151656,
            "vision_start_token_id": 151652, "vision_end_token_id": 151653,
        }[key])
    config.eos_token_id = processor.tokenizer.eos_token_id
    config.pad_token_id = processor.tokenizer.pad_token_id
    base = tmp_path / "base"
    # Mimic a pretrained base checkpoint with no newly introduced scalar head.
    Qwen3VLForConditionalGeneration(config).save_pretrained(base)
    processor.save_pretrained(base)
    data = tmp_path / "data"
    data.mkdir()
    for split, color in (("train", "red"), ("validation", "blue")):
        Image.new("RGB", (64, 64), color).save(data / f"{split}.png")
        (data / f"{split}.jsonl").write_text(json.dumps({
            "id": split, "image_sha256": split, "image": f"{split}.png",
            "text": f"A {color} square", "score": 7,
            "reasoning": f"The image shows a {color} square.",
        }) + "\n")
    settings = yaml.safe_load(Path("configs/train-joint.yaml").read_text())
    settings["prompt"] = {"system": "Evaluate the title and image with the supplied rubric."}
    settings["model"]["name_or_path"] = str(base)
    settings["model"]["dtype"] = os.environ.get("MMJUDGE_TEST_DTYPE", "float32")
    settings["runtime"]["device"] = os.environ.get("MMJUDGE_TEST_DEVICE", "cpu")
    settings["data"]["directory"] = str(data)
    output = tmp_path / "run"
    settings["training"].update(output_dir=str(output), max_steps=1, save_steps=1,
                                  eval_steps=1, gradient_accumulation_steps=1,
                                  max_new_tokens=2, generate_eval=True)
    settings["wandb"]["mode"] = "offline"
    metrics = run_joint_training(settings)
    assert 0 <= metrics["eval_mae"] <= 9
    assert metrics["eval_rationale_loss"] > 0
    weights = load_file(str(output / "adapter_model.safetensors"))
    assert any("score_head" in key for key in weights)
    assert any("lora_B" in key and value.abs().sum() > 0 for key, value in weights.items())
    assert list((output / "wandb").glob("offline-run-*/run-*.wandb"))
    # Fresh base + saved adapter/head must score identically across suffix changes.
    dtype = getattr(torch, settings["model"]["dtype"])
    device = settings["runtime"]["device"]
    restored, processor, _, _ = load_joint_checkpoint(output, device=device)
    head = restored.score_head.modules_to_save["default"]
    for suffix in ("weight", "bias"):
        saved = next(value for key, value in weights.items() if key.endswith(f"score_head.{suffix}"))
        torch.testing.assert_close(getattr(head, suffix).detach().cpu(), saved, rtol=0, atol=0)
    record = JointScoreDataset(data / "validation.jsonl")[0]
    collator = JointScoreCollator(processor, system_prompt=settings["prompt"]["system"])
    first = collator([record])
    second = collator([{**record, "score": 1, "reasoning": "A completely different rationale."}])
    with torch.no_grad():
        a = restored(**{k: v.to(device) for k, v in first.items()}).logits
        b = restored(**{k: v.to(device) for k, v in second.items()}).logits
    torch.testing.assert_close(a, b, atol=0.005 if dtype == torch.bfloat16 else 1e-5, rtol=0)
    assert abs(a.item() - 7) == pytest.approx(metrics["eval_mae"], abs=0.005)
    result = predict_joint(restored, processor, record["image"], record["text"],
                           max_new_tokens=2, device=device, system_prompt=settings["prompt"]["system"])
    assert 0 <= result["score"] <= 9 and isinstance(result["reasoning"], str)
    # Independent CPU process loads only the saved manifest + base + adapter/head.
    cli = Path(sys.executable).with_name("multimodal-judge")
    reply = subprocess.run(
        [str(cli), "judge", "--checkpoint", str(output), "--image",
         str(data / "validation.png"), "--text", record["text"],
         "--device", "cpu", "--dtype", "float32"],
        check=True, capture_output=True, text=True, timeout=90,
    )
    saved_result = json.loads(reply.stdout)
    assert 0 <= saved_result["score"] <= 9
    assert saved_result["score_scale"] == [0, 9]
    if dtype == torch.float32:
        assert saved_result["score"] == pytest.approx(result["score"], abs=1e-5)
    del restored
    if device == "mps":
        torch.mps.empty_cache()
    settings["training"]["max_steps"] = 2
    run_joint_training(settings, resume_from_checkpoint=str(output / "checkpoint-1"))
    state = json.loads((output / "checkpoint-2" / "trainer_state.json").read_text())
    assert state["global_step"] == 2
