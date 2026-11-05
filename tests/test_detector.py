import math

import numpy as np
import pytest
import torch

from config import GenerationConfig, WatermarkConfig
from detect import detect_prefixes, detect_token_ids, z_score
from watermark import context_seed, get_green_mask_np

V = 2_000
CFG = WatermarkConfig(secret_key="test-key", green_fraction=0.5, delta=2.0, context_width=4)


def _all_green_sequence(n, cfg=CFG, seed=0):
    """Build a sequence where every token (after the first w) is green."""
    rng = np.random.default_rng(seed)
    ids = list(rng.integers(0, V, cfg.context_width))
    for _ in range(n):
        mask = get_green_mask_np(V, context_seed(ids[-cfg.context_width:], cfg.secret_key), cfg.green_fraction)
        ids.append(int(rng.choice(np.flatnonzero(mask))))
    return ids


def test_z_score_formula():
    assert z_score(124, 200, 0.5) == pytest.approx((124 - 100) / math.sqrt(50))
    assert z_score(0, 0, 0.5) == 0.0


def test_all_green_text_detected():
    ids = _all_green_sequence(200)
    res = detect_token_ids(ids, V, CFG)
    assert res["tokens_evaluated"] == 200
    assert res["green_tokens"] == 200
    assert res["green_ratio"] == 1.0
    assert res["z_score"] == pytest.approx(math.sqrt(200))
    assert res["watermark_detected"]


def test_random_text_null_distribution():
    rng = np.random.default_rng(1)
    zs = [detect_token_ids(list(rng.integers(0, V, 204)), V, CFG)["z_score"] for _ in range(200)]
    assert abs(np.mean(zs)) < 0.25          # ~N(0,1): SE of mean ~0.07
    assert 0.8 < np.std(zs) < 1.2
    assert max(zs) < CFG.z_threshold        # no false positives at z>=4 here


def test_wrong_key_not_detected():
    ids = _all_green_sequence(200)
    res = detect_token_ids(ids, V, WatermarkConfig(secret_key="other-key", context_width=4))
    assert not res["watermark_detected"]


def test_short_text_evaluates_nothing():
    res = detect_token_ids([1, 2, 3], V, CFG)
    assert res["tokens_evaluated"] == 0 and not res["watermark_detected"]


def test_output_keeps_all_statistics():
    res = detect_token_ids(_all_green_sequence(20), V, CFG)
    for k in ("tokens_evaluated", "green_tokens", "green_ratio", "expected_ratio",
              "z_score", "p_value", "watermark_detected"):
        assert k in res


def test_prefixes_match_truncation():
    ids = _all_green_sequence(100)
    pref = detect_prefixes(ids, V, CFG, [10, 50, 104])
    for L, st in pref.items():
        assert st == detect_token_ids(ids, V, CFG, max_tokens=L)


def test_generator_and_detector_agree():
    """End-to-end: the real generation loop with a tiny random model."""
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from generate import generate_ids

    torch.manual_seed(0)
    model = Qwen2ForCausalLM(Qwen2Config(
        vocab_size=V, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512)).eval()
    input_ids = torch.tensor([[0, 0, 11, 12, 13], [21, 22, 23, 24, 25]])  # row 0 left-padded
    attn = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])
    gen_cfg = GenerationConfig(max_new_tokens=120, temperature=1.0, top_p=1.0)

    strong = replace_delta(CFG, 8.0)
    wm = generate_ids(model, input_ids, attn, strong, gen_cfg, seed=0)
    normal = generate_ids(model, input_ids, attn, None, gen_cfg, seed=0)

    for row in wm:
        res = detect_token_ids(row["token_ids"], V, strong)
        assert res["green_ratio"] > 0.9 and res["watermark_detected"]
    for row in normal:
        res = detect_token_ids(row["token_ids"], V, strong)
        assert abs(res["z_score"]) < 4.0


def replace_delta(cfg, delta):
    from dataclasses import replace

    return replace(cfg, delta=delta)
