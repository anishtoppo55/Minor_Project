"""Statistical watermark detector (one-proportion z-test on green tokens).

Null hypothesis: the text is not watermarked, so each scored token is green
with probability gamma independently:  G ~ Binomial(n, gamma).

    z = (G - n*gamma) / sqrt(n * gamma * (1 - gamma))

Only the secret key, tokenizer and vocabulary size are required; the LLM
itself is not needed for detection.

Usage:
    python detect.py --text "some text"
    python detect.py --file outputs/sample.txt
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Sequence

import numpy as np
from scipy.stats import norm

import config
from config import WatermarkConfig
from watermark import context_seed, get_green_mask_np, green_list_size


def z_score(green_tokens: int, n: int, gamma: float) -> float:
    if n == 0:
        return 0.0
    return (green_tokens - n * gamma) / math.sqrt(n * gamma * (1.0 - gamma))


def green_flags(token_ids: Sequence[int], vocab_size: int, wm_cfg: WatermarkConfig) -> np.ndarray:
    """Per-position green flag for every scorable token.

    The first `context_width` tokens are skipped: their generation context
    included prompt tokens the detector never sees.
    """
    w = wm_cfg.context_width
    flags = []
    for i in range(w, len(token_ids)):
        seed = context_seed(token_ids[i - w:i], wm_cfg.secret_key)
        tok = int(token_ids[i])
        if not 0 <= tok < vocab_size:
            raise ValueError(f"token id {tok} outside vocabulary of size {vocab_size}")
        flags.append(get_green_mask_np(vocab_size, seed, wm_cfg.green_fraction)[tok])
    return np.asarray(flags, dtype=bool)


def stats_from_flags(flags: np.ndarray, vocab_size: int, wm_cfg: WatermarkConfig) -> dict:
    # Use the exact green fraction realised by int(V * gamma).
    gamma = green_list_size(vocab_size, wm_cfg.green_fraction) / vocab_size
    n = int(flags.size)
    g = int(flags.sum())
    z = z_score(g, n, gamma)
    return {
        "tokens_evaluated": n,
        "green_tokens": g,
        "green_ratio": g / n if n else 0.0,
        "expected_ratio": gamma,
        "z_score": z,
        "p_value": float(norm.sf(z)) if n else 1.0,
        "z_threshold": wm_cfg.z_threshold,
        "watermark_detected": bool(n > 0 and z >= wm_cfg.z_threshold),
    }


def detect_token_ids(
    token_ids: Sequence[int],
    vocab_size: int,
    wm_cfg: WatermarkConfig,
    max_tokens: int | None = None,
) -> dict:
    """Detect on raw token ids. `max_tokens` truncates the sequence first."""
    if max_tokens is not None:
        token_ids = token_ids[:max_tokens]
    return stats_from_flags(green_flags(token_ids, vocab_size, wm_cfg), vocab_size, wm_cfg)


def detect_prefixes(
    token_ids: Sequence[int], vocab_size: int, wm_cfg: WatermarkConfig, lengths: Sequence[int]
) -> dict[int, dict]:
    """Detection statistics for several prefix lengths from one hashing pass."""
    flags = green_flags(token_ids, vocab_size, wm_cfg)
    w = wm_cfg.context_width
    return {L: stats_from_flags(flags[: max(0, L - w)], vocab_size, wm_cfg) for L in lengths}


class WatermarkDetector:
    """Text-level detector: tokenises text and runs the z-test."""

    def __init__(self, tokenizer, vocab_size: int, wm_cfg: WatermarkConfig | None = None):
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.cfg = wm_cfg or WatermarkConfig()

    def tokenize(self, text: str) -> list[int]:
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def detect(self, text: str, max_tokens: int | None = None) -> dict:
        return detect_token_ids(self.tokenize(text), self.vocab_size, self.cfg, max_tokens)

    @classmethod
    def from_pretrained(cls, model_name: str = config.MODEL_NAME, wm_cfg: WatermarkConfig | None = None):
        """Loads only the tokenizer and model config (no weights)."""
        from transformers import AutoConfig, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        vocab_size = AutoConfig.from_pretrained(model_name).vocab_size
        return cls(tokenizer, vocab_size, wm_cfg)


def main() -> None:
    p = argparse.ArgumentParser(description="Detect the green-list watermark in text.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--text")
    src.add_argument("--file")
    p.add_argument("--gamma", type=float, default=config.GREEN_FRACTION)
    p.add_argument("--context-width", type=int, default=config.CONTEXT_WIDTH)
    p.add_argument("--threshold", type=float, default=config.Z_THRESHOLD)
    args = p.parse_args()

    text = args.text if args.text is not None else open(args.file, encoding="utf-8").read()
    cfg = WatermarkConfig(green_fraction=args.gamma, context_width=args.context_width,
                          z_threshold=args.threshold)
    print(json.dumps(WatermarkDetector.from_pretrained(wm_cfg=cfg).detect(text), indent=2))


if __name__ == "__main__":
    main()
