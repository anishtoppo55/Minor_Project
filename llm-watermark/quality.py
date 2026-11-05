"""Automated text-quality metric for V0: conditional perplexity.

PPL of the generated tokens given the prompt, scored by a fixed reference
LM (by default the same, unwatermarked model). Lower is more "likely" text;
it is not a complete measure of quality and should be read alongside
manual inspection.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch


@torch.inference_mode()
def conditional_perplexity(model, prompt_ids: Sequence[int], gen_ids: Sequence[int]) -> float:
    if len(gen_ids) == 0:
        return float("nan")
    device = next(model.parameters()).device
    ids = torch.tensor([list(prompt_ids) + list(gen_ids)], device=device)
    logits = model(input_ids=ids).logits[0, :-1].float()
    targets = ids[0, 1:]
    nll = torch.nn.functional.cross_entropy(logits, targets, reduction="none")
    gen_nll = nll[len(prompt_ids) - 1:]  # only score the generated continuation
    return math.exp(gen_nll.mean().item())


def repetition_rate(token_ids: Sequence[int], n: int = 3) -> float:
    """Fraction of repeated n-grams: a cheap signal of degenerate text."""
    grams = [tuple(token_ids[i:i + n]) for i in range(len(token_ids) - n + 1)]
    if not grams:
        return 0.0
    return 1.0 - len(set(grams)) / len(grams)
