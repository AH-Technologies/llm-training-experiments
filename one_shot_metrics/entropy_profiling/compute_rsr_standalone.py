#!/usr/bin/env python3
"""Compute Rank-Surprisal Ratio (RSR) for existing rollout pickles.

RSR measures how "informatively surprising" a reasoning trajectory is to
the base model. Per the paper (arxiv 2601.14249):
    RSR = sum(min(rank_i, r_max)) / sum(surprisal_i)
where rank_i is the rank of the actual token in the model's predictions
and surprisal_i = -log(P(token_i)).

Usage:
    python compute_rsr_standalone.py \
        --results-dir results/entropy_profiles \
        --model Qwen/Qwen2.5-Math-1.5B
"""

import argparse
import gc
import pickle
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer


def compute_rsr_for_example(
    rollouts: list[dict],
    prompt_ids: torch.Tensor,
    model: torch.nn.Module,
    batch_size: int = 4,
    rank_clip_r: int = 100,
) -> list[dict]:
    """Compute RSR for each rollout via forward pass.

    Returns list of dicts with {rsr, mean_rank, mean_surprisal, is_correct}.
    """
    prompt_len = len(prompt_ids)
    pad_id = model.config.eos_token_id or 0
    results = []

    for batch_start in range(0, len(rollouts), batch_size):
        batch_rollouts = rollouts[batch_start:batch_start + batch_size]

        sequences = []
        gen_lengths = []
        for rollout in batch_rollouts:
            gen_ids = torch.tensor(rollout["token_ids"], dtype=torch.long)
            full_seq = torch.cat([prompt_ids, gen_ids])
            sequences.append(full_seq)
            gen_lengths.append(len(gen_ids))

        max_len = max(len(s) for s in sequences)
        padded = torch.full((len(sequences), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(sequences), max_len), dtype=torch.long)
        for i, seq in enumerate(sequences):
            padded[i, :len(seq)] = seq
            attention_mask[i, :len(seq)] = 1

        padded = padded.to(model.device)
        attention_mask = attention_mask.to(model.device)

        with torch.no_grad():
            out = model(input_ids=padded, attention_mask=attention_mask)

        logits = out.logits  # (batch, seq_len, vocab_size)

        for i, rollout in enumerate(batch_rollouts):
            gen_len = gen_lengths[i]
            if gen_len < 2:
                results.append({
                    "rsr": 0.0, "mean_rank": 0.0, "mean_surprisal": 0.0,
                    "is_correct": rollout.get("is_correct", False),
                })
                continue

            # Logits at positions [prompt_len-1 .. prompt_len+gen_len-2]
            # predict tokens at positions [prompt_len .. prompt_len+gen_len-1]
            gen_logits = logits[i, prompt_len - 1:prompt_len - 1 + gen_len, :]  # (gen_len, vocab)
            target_ids = padded[i, prompt_len:prompt_len + gen_len]  # (gen_len,)

            # Surprisal: -log P(target)
            log_probs = F.log_softmax(gen_logits.float(), dim=-1)
            token_log_probs = log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)  # (gen_len,)
            surprisals = -token_log_probs  # (gen_len,)

            # Rank: how many tokens have higher logit than the target?
            target_logits = gen_logits.gather(1, target_ids.unsqueeze(1))  # (gen_len, 1)
            # Use top-k for efficiency (only need to know if rank <= r_max)
            topk_vals, _ = torch.topk(gen_logits, k=min(rank_clip_r, gen_logits.shape[-1]), dim=-1)
            ranks = 1 + (topk_vals.float() > target_logits.float()).sum(dim=-1)  # (gen_len,)
            ranks = torch.clamp(ranks, max=rank_clip_r)

            sum_rank = ranks.sum().item()
            sum_surprisal = surprisals.sum().item()

            rsr = sum_rank / max(sum_surprisal, 1e-8)
            mean_rank = sum_rank / gen_len
            mean_surprisal = sum_surprisal / gen_len

            results.append({
                "rsr": float(rsr),
                "mean_rank": float(mean_rank),
                "mean_surprisal": float(mean_surprisal),
                "is_correct": rollout.get("is_correct", False),
            })

        del out, logits, padded, attention_mask
        torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(description="Compute RSR from existing rollout pickles")
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--rank-clip-r", type=int, default=100)
    parser.add_argument("--max-rollouts", type=int, default=None,
                        help="Max rollouts per example (default: all)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    pkl_files = sorted(results_dir.glob("entropy_*.pkl"))

    if not pkl_files:
        print(f"ERROR: No entropy_*.pkl files found in {results_dir}")
        return

    print(f"Found {len(pkl_files)} pickle files")
    print(f"Model: {args.model}")
    print(f"Rank clip r_max: {args.rank_clip_r}")
    print()

    # Load model
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    model.eval()
    print(f"Model loaded. Vocab size: {model.config.vocab_size}")
    print()

    # Process each example
    all_example_metrics = {}
    t0 = time.time()

    for idx, pkl_path in enumerate(pkl_files):
        name = pkl_path.stem.replace("entropy_", "")

        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        rollouts = data["rollouts"]
        example = data["example"]

        if args.max_rollouts and len(rollouts) > args.max_rollouts:
            rng = np.random.RandomState(42)
            indices = rng.choice(len(rollouts), args.max_rollouts, replace=False)
            rollouts = [rollouts[i] for i in sorted(indices)]

        # Tokenize prompt
        messages = [{"role": "user", "content": example["prompt_text"]}]
        prompt_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).squeeze(0)

        print(f"[{idx+1}/{len(pkl_files)}] {name}: {len(rollouts)} rollouts", end="", flush=True)

        t_ex = time.time()
        rsr_results = compute_rsr_for_example(
            rollouts, prompt_ids, model,
            batch_size=args.batch_size,
            rank_clip_r=args.rank_clip_r,
        )
        elapsed_ex = time.time() - t_ex

        rsrs = [r["rsr"] for r in rsr_results]
        ranks = [r["mean_rank"] for r in rsr_results]
        surprisals = [r["mean_surprisal"] for r in rsr_results]
        correct_rsrs = [r["rsr"] for r in rsr_results if r["is_correct"]]
        wrong_rsrs = [r["rsr"] for r in rsr_results if not r["is_correct"]]

        metrics = {
            "rsr_mean": float(np.mean(rsrs)),
            "rsr_std": float(np.std(rsrs)),
            "rsr_median": float(np.median(rsrs)),
            "mean_rank_mean": float(np.mean(ranks)),
            "mean_surprisal_mean": float(np.mean(surprisals)),
            "rsr_correct_mean": float(np.mean(correct_rsrs)) if correct_rsrs else float("nan"),
            "rsr_wrong_mean": float(np.mean(wrong_rsrs)) if wrong_rsrs else float("nan"),
        }

        # Score for plotting
        score = example.get("math500_score") or example.get("avg_all") or 0
        metrics["_score"] = score
        metrics["_name"] = name

        all_example_metrics[name] = metrics
        print(f" → {elapsed_ex:.1f}s | RSR={metrics['rsr_mean']:.3f}, "
              f"rank={metrics['mean_rank_mean']:.1f}, surprisal={metrics['mean_surprisal_mean']:.2f}")

        gc.collect()
        torch.cuda.empty_cache()

    elapsed_total = time.time() - t0
    print(f"\nDone in {elapsed_total:.1f}s")

    # Build DataFrame
    rows = []
    for pkl_path in pkl_files:
        name = pkl_path.stem.replace("entropy_", "")
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        example = data["example"]

        row = {"example": name}
        for score_key in ["math500_score", "avg_all"]:
            if example.get(score_key) is not None:
                row[score_key] = example[score_key]
        row["pass_rate"] = float(
            sum(r.get("is_correct", False) for r in data["rollouts"]) / len(data["rollouts"])
        )
        row.update({k: v for k, v in all_example_metrics[name].items() if not k.startswith("_")})
        rows.append(row)

    df = pd.DataFrame(rows)

    # Determine score column
    score_col = "math500_score" if "math500_score" in df.columns else "avg_all" if "avg_all" in df.columns else None
    score_label = {"math500_score": "MATH500", "avg_all": "Avg All"}.get(score_col, "Score")

    # Print correlations
    rsr_cols = ["rsr_mean", "rsr_std", "rsr_median", "mean_rank_mean",
                "mean_surprisal_mean", "rsr_correct_mean", "rsr_wrong_mean", "pass_rate"]

    if score_col:
        print(f"\n{'=' * 70}")
        print(f"Correlations with {score_label} (n={len(df)})")
        print(f"{'=' * 70}")
        print(f"{'metric':<25} {'Spearman r':>10} {'p-value':>10} {'Pearson r':>10} {'p-value':>10}")
        print("-" * 70)
        for col in rsr_cols:
            if col not in df.columns:
                continue
            valid = df[[col, score_col]].dropna()
            if len(valid) < 5:
                continue
            r_s, p_s = stats.spearmanr(valid[col], valid[score_col])
            r_p, p_p = stats.pearsonr(valid[col], valid[score_col])
            sig = "***" if p_s < 0.01 else "**" if p_s < 0.05 else "*" if p_s < 0.1 else ""
            print(f"{col:<25} {r_s:>10.3f} {p_s:>10.4f} {r_p:>10.3f} {p_p:>10.4f} {sig}")

    # Save CSV
    csv_path = results_dir / "rsr_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved metrics to {csv_path}")

    # Plot RSR vs score
    if score_col:
        fig_dir = results_dir.parent / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for ax, (metric, label) in zip(axes, [
            ("rsr_mean", "Mean RSR"),
            ("mean_rank_mean", "Mean Token Rank"),
            ("mean_surprisal_mean", "Mean Surprisal"),
        ]):
            x = df[metric].values
            y = df[score_col].values

            ax.scatter(x, y, s=80, c="tab:blue", edgecolors="black", linewidth=0.8, alpha=0.9)

            for j, name in enumerate(df["example"].values):
                ax.annotate(name, (x[j], y[j]), fontsize=7, ha="left", va="bottom",
                            xytext=(4, 4), textcoords="offset points")

            if len(set(x)) > 1:
                r_s, p_s = stats.spearmanr(x, y)
                r_p, p_p = stats.pearsonr(x, y)
                ax.set_title(f"{label}\nSpearman \u03c1={r_s:.3f} (p={p_s:.3f}) | Pearson r={r_p:.3f}",
                             fontsize=10)
                z = np.polyfit(x, y, 1)
                x_line = np.linspace(x.min(), x.max(), 50)
                ax.plot(x_line, np.polyval(z, x_line), "r--", alpha=0.6)
            else:
                ax.set_title(label, fontsize=10)

            ax.set_xlabel(label, fontsize=9)
            ax.set_ylabel(score_label, fontsize=9)

        fig.suptitle(f"Rank-Surprisal Ratio vs {score_label} (n={len(df)})", fontsize=12)
        fig.tight_layout()
        fig.savefig(fig_dir / "rsr_vs_score.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot to {fig_dir / 'rsr_vs_score.png'}")


if __name__ == "__main__":
    main()
