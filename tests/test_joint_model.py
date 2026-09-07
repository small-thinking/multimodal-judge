"""Numerical checks for causal pooling and the two supervised objectives."""

import pytest
import torch
from torch.nn import functional as F
from transformers import Qwen3VLConfig

from multimodal_judge.joint_model import JointQwen3VLForConditionalGeneration


def tiny_config(vocab_size=128):
    return Qwen3VLConfig(
        text_config={
            "vocab_size": vocab_size, "hidden_size": 32, "intermediate_size": 64,
            "num_hidden_layers": 2, "num_attention_heads": 4,
            "num_key_value_heads": 2, "head_dim": 8,
            "rope_scaling": {"rope_type": "default", "mrope_section": [2, 1, 1]},
        },
        vision_config={
            "depth": 1, "hidden_size": 32, "intermediate_size": 64, "num_heads": 4,
            "out_hidden_size": 32, "num_position_embeddings": 16,
            "deepstack_visual_indexes": [0],
        },
        image_token_id=120, video_token_id=121, vision_start_token_id=122,
        vision_end_token_id=123, pad_token_id=0, eos_token_id=2,
        judge_config={"ce_chunk_size": 1},
    )


@pytest.fixture
def model(request):
    torch.manual_seed(7)
    config = tiny_config()
    config.judge_config['head_type'] = getattr(request, 'param', 'regression')
    return JointQwen3VLForConditionalGeneration(config).eval()


def batch():
    return {
        "input_ids": torch.tensor([[1, 8, 9, 10, 11, 12, 13, 2],
                                   [1, 7, 10, 11, 4, 5, 2, 0]]),
        "attention_mask": torch.tensor([[1] * 8, [1] * 7 + [0]]),
        "score_positions": torch.tensor([3, 2]), "scores": torch.tensor([7., 2.]),
        "labels": torch.tensor([[-100] * 6 + [13, 2], [-100] * 5 + [5, 2, -100]]),
    }


def test_score_cannot_read_gold_score_or_reason(model):
    original = batch()
    changed = {key: value.clone() for key, value in original.items()}
    changed["input_ids"][0, 4:] = torch.tensor([30, 31, 32, 33])
    changed["input_ids"][1, 3:7] = torch.tensor([40, 41, 42, 43])
    changed["scores"] = torch.tensor([0., 9.])
    with torch.no_grad():
        a, b = model(**original), model(**changed)
    torch.testing.assert_close(a.logits, b.logits, atol=1e-6, rtol=1e-6)
    assert (a.logits >= 0).all() and (a.logits <= 9).all()


def test_rationale_loss_matches_independent_full_vocabulary_oracle(model):
    data = batch()
    out = model(**data)
    hidden = model.model(input_ids=data["input_ids"], attention_mask=data["attention_mask"],
                         use_cache=False).last_hidden_state
    full_logits = model.lm_head(hidden).float()[:, :-1]
    labels = data["labels"][:, 1:]
    losses = F.cross_entropy(full_logits.reshape(-1, 128), labels.reshape(-1),
                             ignore_index=-100, reduction="none").reshape(2, -1)
    oracle = (losses.sum(-1) / (labels != -100).sum(-1)).mean()
    torch.testing.assert_close(out.rationale_loss, oracle)
    expected_score = F.huber_loss(out.logits[:, 0] / 9, data["scores"] / 9, delta=0.1)
    torch.testing.assert_close(out.score_loss, expected_score)
    torch.testing.assert_close(out.loss, expected_score + 0.1 * oracle)


def test_sparse_and_absent_rationale_are_finite(model):
    data = batch()
    data["labels"][1] = -100
    mixed = model(**data)
    assert mixed.rationale_samples == 1 and mixed.rationale_tokens == 2
    data["labels"][:] = -100
    absent = model(**data)
    assert torch.isfinite(absent.loss) and absent.rationale_loss == 0
    torch.testing.assert_close(absent.loss, absent.score_loss)
    absent.loss.backward()
    assert model.score_head.weight.grad.abs().sum() > 0


def test_checkpointed_sparse_ce_gradients_match_oracle(model):
    data = batch()
    model.train()
    # Checkpointed chunk-size 1 versus a full CE oracle on the same parameters.
    model(**data).rationale_loss.backward()
    actual = model.model.language_model.layers[0].self_attn.q_proj.weight.grad.clone()
    model.zero_grad()
    hidden = model.model(input_ids=data["input_ids"], attention_mask=data["attention_mask"],
                         use_cache=False).last_hidden_state
    logits = model.lm_head(hidden)[:, :-1]
    labels = data["labels"][:, 1:]
    ce = F.cross_entropy(logits.reshape(-1, 128), labels.reshape(-1),
                          reduction="none").reshape(2, -1)
    (ce.sum(-1) / (labels != -100).sum(-1)).mean().backward()
    torch.testing.assert_close(
        actual, model.model.language_model.layers[0].self_attn.q_proj.weight.grad,
        atol=1e-7, rtol=1e-4,
    )


def test_padding_does_not_change_scores(model):
    data = batch()
    with torch.no_grad():
        combined = model(**data).logits
        single = model(input_ids=data["input_ids"][1:2, :7],
                       attention_mask=torch.ones(1, 7, dtype=torch.long),
                       score_positions=torch.tensor([2])).logits
    torch.testing.assert_close(combined[1:], single)


@pytest.mark.parametrize('model', ['regression', 'classification'], indirect=True)
def test_head_checkpoint_roundtrip(model, tmp_path):
    model.config.judge_config['score_weight'] = 2.5
    model.save_pretrained(tmp_path)
    restored = JointQwen3VLForConditionalGeneration.from_pretrained(tmp_path).eval()
    with torch.no_grad():
        torch.testing.assert_close(model(**batch()).logits, restored(**batch()).logits)
    assert restored.config.judge_config["ce_chunk_size"] == 1
    assert restored.config.judge_config['head_type'] == model.config.judge_config['head_type']
    assert restored.config.judge_config['score_weight'] == 2.5
    assert restored.score_head.out_features == model.score_head.out_features


@pytest.mark.parametrize('model', ['regression', 'classification'], indirect=True)
def test_bfloat16_base_load_preserves_fp32_head_exactly(model, tmp_path):
    with torch.no_grad():
        model.score_head.weight.fill_(0.1234567)
        model.score_head.bias.fill_(0.2345678)
    model.save_pretrained(tmp_path)
    restored = JointQwen3VLForConditionalGeneration.from_pretrained(tmp_path, dtype=torch.bfloat16)
    assert restored.score_head.weight.dtype == torch.float32
    torch.testing.assert_close(restored.score_head.weight, model.score_head.weight, rtol=0, atol=0)
    torch.testing.assert_close(restored.score_head.bias, model.score_head.bias, rtol=0, atol=0)


def test_legacy_model_config_keeps_regression_defaults():
    model = JointQwen3VLForConditionalGeneration(tiny_config()).eval()
    assert model.config.judge_config['head_type'] == 'regression'
    assert model.config.judge_config['score_weight'] == 1.0
    assert model.score_head.out_features == 1
    assert model(**batch()).score_class_logits is None


@pytest.mark.parametrize('key,value', [
    ('head_type', 'ordinal'), ('head_type', None), ('score_weight', -1),
    ('score_weight', float('nan')), ('score_weight', float('inf')), ('score_weight', True),
])
def test_direct_model_load_rejects_invalid_objective(key, value):
    config = tiny_config()
    config.judge_config[key] = value
    with pytest.raises(ValueError):
        JointQwen3VLForConditionalGeneration(config)


@pytest.mark.parametrize('model', ['classification'], indirect=True)
def test_classification_ce_and_expected_score_known_distribution(model):
    # A fixed, asymmetric distribution distinguishes expectation from argmax,
    # including the endpoints of the public 0..9 label contract.
    probabilities = torch.tensor([.20, .05, .05, .05, .05, .05, .05, .05, .05, .40])
    with torch.no_grad():
        model.score_head.weight.zero_()
        model.score_head.bias.copy_(probabilities.log())
    data = batch()
    data['scores'] = torch.tensor([0., 9.])
    out = model(**data)
    assert out.score_class_logits.shape == (2, 10)
    torch.testing.assert_close(out.score_class_logits.softmax(-1), probabilities.expand(2, -1))
    # sum(p[k] * k) = .05 * (1 + ... + 8) + .40 * 9 = 5.4
    torch.testing.assert_close(out.logits, torch.full((2, 1), 5.4))
    assert not torch.equal(out.logits[:, 0], out.score_class_logits.argmax(-1).float())
    expected_ce = -(torch.log(torch.tensor(.20)) + torch.log(torch.tensor(.40))) / 2
    torch.testing.assert_close(out.score_loss, expected_ce)
    torch.testing.assert_close(out.loss, expected_ce + .1 * out.rationale_loss)


@pytest.mark.parametrize('model', ['classification'], indirect=True)
@pytest.mark.parametrize('bad_score', [.5, -1., 10., float('nan'), float('inf')])
def test_classification_rejects_invalid_score_labels(model, bad_score):
    data = batch()
    data['scores'][0] = bad_score
    with pytest.raises(ValueError):
        model(**data)


@pytest.mark.parametrize('model', ['regression', 'classification'], indirect=True)
def test_score_weight_scales_head_and_shared_gradients(model):
    data = batch()
    # Isolate the score contribution to shared layers; the LM objective has
    # a separate path and must not masquerade as scoring progress.
    data['labels'][:] = -100
    gradients = []
    losses = []
    for weight in (1., 3., 0.):
        model.zero_grad(set_to_none=True)
        model.config.judge_config['score_weight'] = weight
        out = model(**data)
        out.loss.backward()
        losses.append(out.loss.detach())
        gradients.append((model.score_head.weight.grad.clone(),
                          model.model.language_model.layers[0].self_attn.q_proj.weight.grad.clone()))
    torch.testing.assert_close(losses[1], 3 * losses[0])
    assert losses[0] > 0 and losses[2] == 0
    for index in (0, 1):
        assert gradients[0][index].abs().sum() > 0
        torch.testing.assert_close(gradients[1][index], 3 * gradients[0][index], atol=1e-7, rtol=1e-4)
        assert gradients[2][index].count_nonzero() == 0


@pytest.mark.parametrize('model', ['regression', 'classification'], indirect=True)
def test_weighted_objectives_keep_rationale_term_independent(model):
    model.config.judge_config.update(score_weight=2.5, rationale_weight=.3)
    out = model(**batch())
    torch.testing.assert_close(out.loss, 2.5 * out.score_loss + .3 * out.rationale_loss)


def test_rejects_supervised_prompt_and_padding_pool(model):
    data = batch()
    data["labels"][0, 2] = 8
    with pytest.raises(ValueError, match="masked"):
        model(**data)
    data = batch()
    data["score_positions"][1] = 7
    with pytest.raises(ValueError, match="padding"):
        model(**data)
