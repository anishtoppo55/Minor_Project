"""V0 experiments: baseline, watermark strength, green fraction, text length.

Every generation is stored as one CSV row; summaries, a baseline table,
plots and reproducibility metadata are written to experiments/.

Examples:
    python evaluate.py baseline --num-prompts 10 --seeds 3
    python evaluate.py delta --deltas 0.5 1 1.5 2 --num-prompts 10 --seeds 2
    python evaluate.py gamma --gammas 0.25 0.5 0.75
    python evaluate.py length --lengths 50 100 200 400
    python evaluate.py all
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import time
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402

import config  # noqa: E402
from config import GenerationConfig, WatermarkConfig  # noqa: E402
from detect import WatermarkDetector, detect_prefixes  # noqa: E402
from generate import generate_texts, load_model  # noqa: E402
from quality import conditional_perplexity, repetition_rate  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_prompts(n: int | None) -> list[str]:
    lines = [l.strip() for l in config.PROMPTS_FILE.read_text(encoding="utf-8").splitlines()]
    prompts = [l for l in lines if l]
    return prompts[:n] if n else prompts


def metadata(args, wm_cfg: WatermarkConfig, gen_cfg: GenerationConfig, model) -> dict:
    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": config.MODEL_NAME,
        "model_revision": getattr(model.config, "_commit_hash", None),
        "model_dtype": str(next(model.parameters()).dtype),
        "vocab_size": model.config.vocab_size,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "hardware": platform.processor() or platform.machine(),
        "device": str(next(model.parameters()).device),
        "cuda": torch.version.cuda,
        "experiment_seed": config.EXPERIMENT_SEED,
        "secret_key": "<redacted>",
        "watermark": wm_cfg.public_dict(),
        "generation": asdict(gen_cfg),
        "cli_args": {k: v for k, v in vars(args).items() if k != "func"},
    }


class Runner:
    """Generates samples and scores them; shared by all experiments."""

    def __init__(self, args):
        self.args = args
        set_global_seed(config.EXPERIMENT_SEED)
        print(f"Loading {config.MODEL_NAME} on {config.get_device()} ...", flush=True)
        self.model, self.tokenizer = load_model()
        self.vocab_size = self.model.config.vocab_size
        self.prompts = load_prompts(args.num_prompts)
        self.gen_cfg = GenerationConfig(max_new_tokens=args.max_new_tokens,
                                        temperature=args.temperature, top_p=args.top_p)
        self.wm_cfg = WatermarkConfig(delta=args.delta, green_fraction=args.gamma,
                                      context_width=args.context_width, z_threshold=args.threshold)
        self.sample_counter = 0

    def generate(self, wm_cfg: WatermarkConfig | None, gen_cfg: GenerationConfig) -> list[dict]:
        """All prompts x all seeds. Sampler seeds are shared across conditions so
        normal/watermarked samples are paired (same prompt, same RNG stream)."""
        samples = []
        bs = self.args.batch_size
        label = "normal" if wm_cfg is None else f"wm(delta={wm_cfg.delta}, gamma={wm_cfg.green_fraction})"
        for s in range(self.args.seeds):
            for start in range(0, len(self.prompts), bs):
                batch = self.prompts[start:start + bs]
                seed = config.EXPERIMENT_SEED + 1000 * s + start
                t0 = time.time()
                outs = generate_texts(self.model, self.tokenizer, batch, wm_cfg, gen_cfg, seed=seed)
                ntok = sum(o["num_tokens"] for o in outs)
                print(f"  {label} seed#{s} prompts {start}-{start + len(batch) - 1}: "
                      f"{ntok} tokens in {time.time() - t0:.1f}s", flush=True)
                for i, o in enumerate(outs):
                    o.update(prompt_id=start + i, seed=seed)
                    if not self.args.no_ppl:
                        o["perplexity"] = conditional_perplexity(self.model, o["prompt_ids"], o["token_ids"])
                    else:
                        o["perplexity"] = float("nan")
                    o["repetition_3gram"] = repetition_rate(o["token_ids"])
                samples.extend(outs)
        return samples

    def score(self, sample: dict, det_cfg: WatermarkConfig, watermarked: bool, experiment: str,
              condition: str, gen_delta: float, gen_cfg: GenerationConfig,
              lengths: list[int] | None = None) -> list[dict]:
        """Detect on re-tokenised *text* (what a real detector sees)."""
        detector = WatermarkDetector(self.tokenizer, self.vocab_size, det_cfg)
        ids = detector.tokenize(sample["text"])
        lengths = lengths or [len(ids)]
        rows = []
        for L, st in detect_prefixes(ids, self.vocab_size, det_cfg, lengths).items():
            self.sample_counter += 1
            rows.append({
                "sample_id": self.sample_counter,
                "experiment": experiment,
                "condition": condition,
                "prompt_id": sample["prompt_id"],
                "seed": sample["seed"],
                "watermarked": watermarked,
                "model": config.MODEL_NAME,
                "temperature": gen_cfg.temperature,
                "top_p": gen_cfg.top_p,
                "max_new_tokens": gen_cfg.max_new_tokens,
                "delta": gen_delta,
                "green_fraction": det_cfg.green_fraction,
                "context_width": det_cfg.context_width,
                "length_limit": L,
                "num_generated_tokens": sample["num_tokens"],
                "stopped_by_eos": sample["stopped_by_eos"],
                "retokenization_match": ids == sample["token_ids"],
                "tokens_evaluated": st["tokens_evaluated"],
                "green_tokens": st["green_tokens"],
                "green_ratio": st["green_ratio"],
                "z_score": st["z_score"],
                "p_value": st["p_value"],
                "detected": st["watermark_detected"],
                "perplexity": sample["perplexity"],
                "repetition_3gram": sample["repetition_3gram"],
                "generated_text": sample["text"],
            })
        return rows


def summarize(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    return (
        df.groupby(keys, dropna=False)
        .agg(n=("z_score", "size"),
             mean_tokens=("tokens_evaluated", "mean"),
             mean_green_ratio=("green_ratio", "mean"),
             mean_z=("z_score", "mean"),
             std_z=("z_score", "std"),
             median_z=("z_score", "median"),
             detection_rate=("detected", "mean"),
             mean_ppl=("perplexity", "mean"),
             median_ppl=("perplexity", "median"),
             mean_rep3=("repetition_3gram", "mean"))
        .reset_index()
    )


def save(df: pd.DataFrame, name: str, meta: dict, keys: list[str]) -> pd.DataFrame:
    config.EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.EXPERIMENTS_DIR / f"{name}_results.csv", index=False)
    summary = summarize(df, keys)
    summary.to_csv(config.EXPERIMENTS_DIR / f"{name}_summary.csv", index=False)
    (config.EXPERIMENTS_DIR / f"{name}_metadata.json").write_text(json.dumps(meta, indent=2))
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(f"\n=== {name} summary ===\n{summary.round(3).to_string(index=False)}\n", flush=True)
    return summary


def save_texts(samples: list[dict], subdir: str, name: str) -> None:
    out = config.OUTPUTS_DIR / subdir
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps({k: s[k] for k in ("prompt_id", "seed", "prompt", "text",
                                                   "num_tokens", "perplexity")}) + "\n")


def _plot_path(name: str) -> Path:
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    return config.PLOTS_DIR / name


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------
def exp_baseline(r: Runner) -> None:
    """Normal vs watermarked at the configured delta (Sections 28/34/52)."""
    print("\n##### Baseline: normal vs watermarked #####")
    normal = r.generate(None, r.gen_cfg)
    wm = r.generate(r.wm_cfg, r.gen_cfg)
    save_texts(normal, "normal", "baseline")
    save_texts(wm, "watermarked", "baseline")

    rows = [row for s in normal for row in r.score(s, r.wm_cfg, False, "baseline", "normal", 0.0, r.gen_cfg)]
    rows += [row for s in wm for row in r.score(s, r.wm_cfg, True, "baseline", "watermarked",
                                                  r.wm_cfg.delta, r.gen_cfg)]
    df = pd.DataFrame(rows)
    save(df, "baseline", metadata(r.args, r.wm_cfg, r.gen_cfg, r.model), ["condition"])

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(min(-4, df.z_score.min()), max(8, df.z_score.max()), 40)
    for cond, color in (("normal", "tab:blue"), ("watermarked", "tab:green")):
        ax.hist(df[df.condition == cond].z_score, bins=bins, alpha=0.6, label=cond, color=color)
    ax.axvline(r.wm_cfg.z_threshold, color="red", ls="--", label=f"threshold z={r.wm_cfg.z_threshold}")
    ax.set(xlabel="z-score", ylabel="count",
           title=f"Z-score distribution (delta={r.wm_cfg.delta}, gamma={r.wm_cfg.green_fraction})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(_plot_path("z_distribution.png"), dpi=150)
    plt.close(fig)


def exp_delta(r: Runner) -> None:
    """Watermark strength vs detection and quality (Sections 30/38/39)."""
    print("\n##### Delta sweep #####")
    rows = []
    control = r.generate(None, r.gen_cfg)
    rows += [row for s in control for row in r.score(s, r.wm_cfg, False, "delta", "normal", 0.0, r.gen_cfg)]
    for d in r.args.deltas:
        if d == 0:
            continue  # delta=0 is the normal control above
        cfg = replace(r.wm_cfg, delta=d)
        wm = r.generate(cfg, r.gen_cfg)
        rows += [row for s in wm for row in r.score(s, cfg, True, "delta", f"delta={d}", d, r.gen_cfg)]
    df = pd.DataFrame(rows)
    summary = save(df, "delta", metadata(r.args, r.wm_cfg, r.gen_cfg, r.model), ["watermarked", "delta"])

    # Baseline table (Section 38): FPR comes from the unwatermarked control.
    ctrl = summary[~summary.watermarked].iloc[0]
    table = summary[summary.watermarked].assign(
        normal_z=ctrl.mean_z, fpr=ctrl.detection_rate, normal_ppl=ctrl.mean_ppl)
    table = table[["delta", "normal_z", "mean_z", "detection_rate", "fpr", "normal_ppl", "mean_ppl"]]
    table.columns = ["delta", "normal_mean_z", "watermarked_mean_z", "detection_rate", "fpr",
                     "normal_ppl", "watermarked_ppl"]
    table.to_csv(config.EXPERIMENTS_DIR / "baseline_table.csv", index=False)
    print("Baseline table:\n" + table.round(3).to_string(index=False))

    s = summary.sort_values("delta")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(s.delta, s.detection_rate, "o-", label="detection rate (TPR; delta=0 is FPR)")
    ax.set(xlabel="delta (logit bias)", ylabel="rate", ylim=(-0.05, 1.05),
           title=f"Watermark strength vs detection (z >= {r.wm_cfg.z_threshold})")
    ax2 = ax.twinx()
    ax2.plot(s.delta, s.mean_z, "s--", color="tab:orange", label="mean z")
    ax2.set_ylabel("mean z-score")
    fig.legend(loc="upper left", bbox_to_anchor=(0.12, 0.88))
    fig.tight_layout()
    fig.savefig(_plot_path("delta_vs_detection.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    g = df.groupby("delta").perplexity
    ax.errorbar(g.mean().index, g.mean().values, yerr=g.sem().values, fmt="o-", capsize=3)
    ax.set(xlabel="delta (logit bias)", ylabel="conditional perplexity (mean ± SEM)",
           title="Watermark strength vs text quality")
    fig.tight_layout()
    fig.savefig(_plot_path("delta_vs_quality.png"), dpi=150)
    plt.close(fig)


def exp_gamma(r: Runner) -> None:
    """Green fraction sweep at fixed delta (Section 31)."""
    print("\n##### Gamma sweep #####")
    rows = []
    control = r.generate(None, r.gen_cfg)
    for gm in r.args.gammas:
        cfg = replace(r.wm_cfg, green_fraction=gm)
        rows += [row for s in control for row in r.score(s, cfg, False, "gamma", "normal", 0.0, r.gen_cfg)]
        wm = r.generate(cfg, r.gen_cfg)
        rows += [row for s in wm for row in r.score(s, cfg, True, "gamma", f"gamma={gm}", cfg.delta, r.gen_cfg)]
    df = pd.DataFrame(rows)
    summary = save(df, "gamma", metadata(r.args, r.wm_cfg, r.gen_cfg, r.model), ["watermarked", "green_fraction"])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for wm_flag, label in ((True, "watermarked"), (False, "normal")):
        s = summary[summary.watermarked == wm_flag].sort_values("green_fraction")
        axes[0].plot(s.green_fraction, s.mean_z, "o-", label=label)
        if wm_flag:
            axes[1].plot(s.green_fraction, s.mean_ppl, "o-", label=label)
    axes[1].axhline(df[~df.watermarked].perplexity.mean(), color="gray", ls="--", label="normal")
    axes[0].set(xlabel="gamma (green fraction)", ylabel="mean z-score", title=f"delta={r.wm_cfg.delta}")
    axes[1].set(xlabel="gamma (green fraction)", ylabel="mean perplexity", title="quality")
    for a in axes:
        a.legend()
    fig.tight_layout()
    fig.savefig(_plot_path("gamma_sweep.png"), dpi=150)
    plt.close(fig)


def exp_length(r: Runner) -> None:
    """Detection vs number of tokens (Section 32).

    Generates max(lengths) tokens with EOS suppressed, then scores prefixes
    of each length, so every length uses the same underlying samples.
    """
    print("\n##### Length sweep #####")
    L = max(r.args.lengths)
    gen_cfg = replace(r.gen_cfg, max_new_tokens=L, min_new_tokens=L)
    normal = r.generate(None, gen_cfg)
    wm = r.generate(r.wm_cfg, gen_cfg)
    rows = [row for s in normal for row in r.score(s, r.wm_cfg, False, "length", "normal", 0.0,
                                                   gen_cfg, r.args.lengths)]
    rows += [row for s in wm for row in r.score(s, r.wm_cfg, True, "length", "watermarked",
                                                r.wm_cfg.delta, gen_cfg, r.args.lengths)]
    df = pd.DataFrame(rows)
    summary = save(df, "length", metadata(r.args, r.wm_cfg, gen_cfg, r.model), ["watermarked", "length_limit"])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for wm_flag, label in ((True, "watermarked (TPR)"), (False, "normal (FPR)")):
        s = summary[summary.watermarked == wm_flag].sort_values("length_limit")
        axes[0].plot(s.length_limit, s.detection_rate, "o-", label=label)
        axes[1].plot(s.length_limit, s.mean_z, "o-", label=label)
    axes[0].set(xlabel="tokens", ylabel="detection rate", ylim=(-0.05, 1.05),
                title=f"Text length vs detection (delta={r.wm_cfg.delta})")
    axes[1].set(xlabel="tokens", ylabel="mean z-score", title="z grows ~ sqrt(n)")
    for a in axes:
        a.legend()
    fig.tight_layout()
    fig.savefig(_plot_path("length_vs_detection.png"), dpi=150)
    plt.close(fig)


EXPERIMENTS = {"baseline": exp_baseline, "delta": exp_delta, "gamma": exp_gamma, "length": exp_length}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("experiment", choices=[*EXPERIMENTS, "all"])
    p.add_argument("--num-prompts", type=int, default=10)
    p.add_argument("--seeds", type=int, default=3, help="generations per prompt per condition")
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--max-new-tokens", type=int, default=config.MAX_NEW_TOKENS)
    p.add_argument("--temperature", type=float, default=config.TEMPERATURE)
    p.add_argument("--top-p", type=float, default=config.TOP_P)
    p.add_argument("--delta", type=float, default=config.WATERMARK_DELTA)
    p.add_argument("--gamma", type=float, default=config.GREEN_FRACTION)
    p.add_argument("--context-width", type=int, default=config.CONTEXT_WIDTH)
    p.add_argument("--threshold", type=float, default=config.Z_THRESHOLD)
    p.add_argument("--deltas", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0, 2.5])
    p.add_argument("--gammas", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    p.add_argument("--lengths", type=int, nargs="+", default=[25, 50, 100, 200, 400])
    p.add_argument("--no-ppl", action="store_true", help="skip perplexity computation")
    args = p.parse_args()

    r = Runner(args)
    t0 = time.time()
    for name in (EXPERIMENTS if args.experiment == "all" else [args.experiment]):
        EXPERIMENTS[name](r)
    print(f"Done in {(time.time() - t0) / 60:.1f} min. Results in {config.EXPERIMENTS_DIR}")


if __name__ == "__main__":
    main()
