"""Core watermark logic: context hashing, green/red partition, logit bias.

Generator and detector both use these functions, so any change here changes
both sides consistently.
"""

from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np
import torch


def context_seed(previous_tokens: Sequence[int], secret_key: str) -> int:
    """Deterministic 64-bit seed from the secret key and the context tokens.

    Uses SHA-256 (not Python's salted hash()) so the value is identical
    across processes, machines and runs.
    """
    message = secret_key + "|" + ",".join(str(int(t)) for t in previous_tokens)
    digest = hashlib.sha256(message.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big")


def green_list_size(vocab_size: int, green_fraction: float) -> int:
    return int(vocab_size * green_fraction)


def get_green_list(vocab_size: int, seed: int, green_fraction: float) -> np.ndarray:
    """Token ids in the green list: first gamma-fraction of a seeded permutation."""
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(vocab_size)
    return permutation[: green_list_size(vocab_size, green_fraction)]


def get_green_mask_np(vocab_size: int, seed: int, green_fraction: float) -> np.ndarray:
    mask = np.zeros(vocab_size, dtype=bool)
    mask[get_green_list(vocab_size, seed, green_fraction)] = True
    return mask


def get_green_mask(
    vocab_size: int,
    seed: int,
    green_fraction: float,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Boolean tensor of shape [vocab_size]; True = green token."""
    mask = get_green_mask_np(vocab_size, seed, green_fraction)
    return torch.from_numpy(mask).to(device)


def is_green(
    token_id: int,
    context_tokens: Sequence[int],
    vocab_size: int,
    secret_key: str,
    green_fraction: float,
) -> bool:
    seed = context_seed(context_tokens, secret_key)
    return bool(get_green_mask_np(vocab_size, seed, green_fraction)[token_id])


def apply_watermark(logits: torch.Tensor, green_mask: torch.Tensor, delta: float) -> torch.Tensor:
    """Return logits with `delta` added to green tokens (input is not mutated).

    `logits` may be [vocab] or [batch, vocab]; `green_mask` must broadcast.
    """
    if green_mask.dtype != torch.bool:
        raise TypeError("green_mask must be a boolean tensor")
    return logits + green_mask.to(logits.dtype) * delta
