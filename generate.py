"""Custom generation loop with optional green-list watermarking.

We do not use `model.generate()` so that the logits can be inspected and
modified explicitly at every step. The loop is batched, uses the KV cache,
and keeps token ids directly (no re-tokenisation per step).

Order of operations at every step (same as HF logits processors -> warpers):
    raw logits -> +delta on green tokens -> /temperature -> top-p -> sample

Usage:
    python generate.py --prompt "Explain how GPS works."
    python generate.py --prompt "Explain how GPS works." --no-watermark
"""

from __future__ import annotations

import argparse
import json
from typing import Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import config
from config import GenerationConfig, WatermarkConfig
from watermark import apply_watermark, context_seed, get_green_mask


def load_model(model_name: str = config.MODEL_NAME, device: str | None = None, dtype=None):
    device = device or config.get_device()
    if dtype is None:
        dtype = torch.float16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device)
    model.eval()
    return model, tokenizer


def build_chat_prompt(tokenizer, prompt: str) -> str:
    """Wrap a user prompt in the model's chat template (if it has one)."""
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
    return prompt


def eos_token_ids(model, tokenizer) -> list[int]:
    ids = set()
    gen_eos = getattr(model.generation_config, "eos_token_id", None)
    if isinstance(gen_eos, int):
        ids.add(gen_eos)
    elif gen_eos:
        ids.update(gen_eos)
    if tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)
    return sorted(ids)


def sample_next_token(
    logits: torch.Tensor, temperature: float, top_p: float, generator: torch.Generator | None
) -> torch.Tensor:
    """Temperature + nucleus sampling. logits: [batch, vocab] -> [batch]."""
    logits = logits / temperature
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = sorted_probs.cumsum(dim=-1)
        # Drop a token if the mass *before* it already exceeds top_p
        # (the most likely token is therefore always kept).
        remove = (cumulative - sorted_probs) > top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)


@torch.inference_mode()
def generate_ids(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    wm_cfg: WatermarkConfig | None,
    gen_cfg: GenerationConfig,
    eos_ids: Sequence[int] = (),
    pad_id: int = 0,
    seed: int | None = None,
    debug_steps: int = 0,
    tokenizer=None,
) -> list[dict]:
    """Generate for a left-padded batch. Returns one dict per row.

    wm_cfg=None (or delta == 0) produces normal, unwatermarked text.
    """
    device = input_ids.device
    vocab_size = model.config.vocab_size
    batch = input_ids.shape[0]
    watermark_on = wm_cfg is not None and wm_cfg.delta != 0.0

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)

    # Unpadded token histories: the watermark context is taken from these,
    # so prompt tokens seed the first few generated positions.
    histories = [row[mask.bool()].tolist() for row, mask in zip(input_ids, attention_mask)]
    generated: list[list[int]] = [[] for _ in range(batch)]
    finished = torch.zeros(batch, dtype=torch.bool, device=device)
    stopped_by_eos = [False] * batch
    eos_tensor = torch.tensor(list(eos_ids), device=device, dtype=torch.long)

    position_ids = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=True,
    )
    past = out.past_key_values
    last_pos = position_ids[:, -1:]

    for step in range(gen_cfg.max_new_tokens):
        logits = out.logits[:, -1, :].float()
        if logits.shape[-1] != vocab_size:
            raise RuntimeError(f"logit dim {logits.shape[-1]} != vocab_size {vocab_size}")
        raw_logits = logits

        green_masks = None
        if watermark_on:
            green_masks = torch.zeros_like(logits, dtype=torch.bool)
            for b in range(batch):
                if finished[b]:
                    continue
                ctx = histories[b][-wm_cfg.context_width:]
                s = context_seed(ctx, wm_cfg.secret_key)
                green_masks[b] = get_green_mask(vocab_size, s, wm_cfg.green_fraction, device)
            logits = apply_watermark(logits, green_masks, wm_cfg.delta)

        if step < gen_cfg.min_new_tokens and len(eos_tensor):
            logits[:, eos_tensor] = float("-inf")

        next_tokens = sample_next_token(logits, gen_cfg.temperature, gen_cfg.top_p, generator)
        next_tokens = torch.where(finished, torch.full_like(next_tokens, pad_id), next_tokens)

        if step < debug_steps:
            _print_debug(step, histories[0], next_tokens[0].item(), raw_logits[0], logits[0],
                         green_masks[0] if green_masks is not None else None, wm_cfg, tokenizer)

        for b in range(batch):
            if finished[b]:
                continue
            tok = next_tokens[b].item()
            if tok in eos_ids:
                finished[b] = True
                stopped_by_eos[b] = True
                continue
            histories[b].append(tok)
            generated[b].append(tok)

        if bool(finished.all()) or step == gen_cfg.max_new_tokens - 1:
            break

        attention_mask = torch.cat(
            [attention_mask, torch.ones((batch, 1), dtype=attention_mask.dtype, device=device)], dim=-1
        )
        last_pos = last_pos + 1
        out = model(
            input_ids=next_tokens.unsqueeze(-1),
            attention_mask=attention_mask,
            position_ids=last_pos,
            past_key_values=past,
            use_cache=True,
        )
        past = out.past_key_values

    return [
        {"token_ids": generated[b], "num_tokens": len(generated[b]), "stopped_by_eos": stopped_by_eos[b]}
        for b in range(batch)
    ]


def _print_debug(step, history, token, raw_logits, mod_logits, green_mask, wm_cfg, tokenizer):
    """Failure-mode debugging aid (Section 40)."""
    tok_str = tokenizer.decode([token]) if tokenizer is not None else ""
    line = f"[step {step:3d}] token={token:6d} {tok_str!r:14}"
    if green_mask is not None:
        ctx = history[-wm_cfg.context_width:]
        s = context_seed(ctx, wm_cfg.secret_key)
        line += (
            f" seed={s:20d} greens={int(green_mask.sum()):6d} is_green={bool(green_mask[token])!s:5}"
            f" logit {raw_logits[token].item():7.3f} -> {mod_logits[token].item():7.3f}"
        )
    print(line)


def generate_texts(
    model,
    tokenizer,
    prompts: Sequence[str],
    wm_cfg: WatermarkConfig | None,
    gen_cfg: GenerationConfig,
    seed: int | None = None,
    debug_steps: int = 0,
) -> list[dict]:
    """Prompt strings -> list of dicts with generated text and token ids."""
    chat_prompts = [build_chat_prompt(tokenizer, p) for p in prompts]
    enc = tokenizer(chat_prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    device = next(model.parameters()).device
    results = generate_ids(
        model,
        enc["input_ids"].to(device),
        enc["attention_mask"].to(device),
        wm_cfg,
        gen_cfg,
        eos_ids=eos_token_ids(model, tokenizer),
        pad_id=tokenizer.pad_token_id,
        seed=seed,
        debug_steps=debug_steps,
        tokenizer=tokenizer,
    )
    for prompt, chat_prompt, res in zip(prompts, chat_prompts, results):
        res["prompt"] = prompt
        res["prompt_ids"] = tokenizer(chat_prompt, add_special_tokens=False)["input_ids"]
        res["text"] = tokenizer.decode(res["token_ids"], skip_special_tokens=True)
    return results


def main() -> None:
    p = argparse.ArgumentParser(description="Generate normal or watermarked text.")
    p.add_argument("--prompt", required=True)
    p.add_argument("--no-watermark", action="store_true")
    p.add_argument("--delta", type=float, default=config.WATERMARK_DELTA)
    p.add_argument("--gamma", type=float, default=config.GREEN_FRACTION)
    p.add_argument("--context-width", type=int, default=config.CONTEXT_WIDTH)
    p.add_argument("--max-new-tokens", type=int, default=config.MAX_NEW_TOKENS)
    p.add_argument("--temperature", type=float, default=config.TEMPERATURE)
    p.add_argument("--top-p", type=float, default=config.TOP_P)
    p.add_argument("--seed", type=int, default=config.EXPERIMENT_SEED)
    p.add_argument("--debug-steps", type=int, default=0, help="print per-step watermark info")
    p.add_argument("--output", help="optional path to write the result as JSON")
    args = p.parse_args()

    model, tokenizer = load_model()
    wm_cfg = None if args.no_watermark else WatermarkConfig(
        green_fraction=args.gamma, delta=args.delta, context_width=args.context_width
    )
    gen_cfg = GenerationConfig(
        max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p
    )
    res = generate_texts(model, tokenizer, [args.prompt], wm_cfg, gen_cfg,
                         seed=args.seed, debug_steps=args.debug_steps)[0]
    print(res["text"])
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({"prompt": res["prompt"], "text": res["text"], "watermarked": wm_cfg is not None},
                      f, indent=2)


if __name__ == "__main__":
    main()
