#!/usr/bin/env python3
"""Standalone script: compute hidden state variance from existing rollout pickles.

Re-runs HF forward passes with output_hidden_states=True to extract
final-token hidden states per layer, then computes variance metrics
across rollouts for each example.

Usage:
    python compute_hidden_states_standalone.py \
        --results-dir results/entropy_profiles \
        --model Qwen/Qwen2.5-Math-1.5B

    python compute_hidden_states_standalone.py \
        --results-dir results/li_run/entropy_profiles \
        --model Qwen/Qwen2.5-7B
"""

import argparse
import gc
import json
import pickle
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer

from hidden_state_variance import compute_hidden_state_metrics, SCALAR_METRIC_KEYS


def extract_hidden_states_for_example(
    rollouts: list[dict],
    prompt_ids: torch.Tensor,
    model: torch.nn.Module,
    batch_size: int = 4,
    max_rollouts: int | None = None,
) -> np.ndarray:
    """Run forward passes and extract final-token hidden states per layer.

    Args:
        rollouts: List of rollout dicts with 'token_ids' key.
        prompt_ids: Tokenized prompt (1D tensor).
        model: HuggingFace model with output_hidden_states support.
        batch_size: Batch size for forward passes.
        max_rollouts: Maximum rollouts to process (None = all).

    Returns:
        Array of shape (n_rollouts, n_layers, hidden_dim).
    """
    if max_rollouts and len(rollouts) > max_rollouts:
        # Subsample deterministically
        rng = np.random.RandomState(42)
        indices = rng.choice(len(rollouts), max_rollouts, replace=False)
        rollouts = [rollouts[i] for i in sorted(indices)]

    prompt_len = len(prompt_ids)
    pad_id = model.config.eos_token_id or 0
    all_hidden = []

    for batch_start in range(0, len(rollouts), batch_size):
        batch_rollouts = rollouts[batch_start:batch_start + batch_size]

        # Build full sequences: prompt + generated tokens
        sequences = []
        gen_lengths = []
        for rollout in batch_rollouts:
            gen_ids = torch.tensor(rollout["token_ids"], dtype=torch.long)
            full_seq = torch.cat([prompt_ids, gen_ids])
            sequences.append(full_seq)
            gen_lengths.append(len(gen_ids))

        # Pad to same length
        max_len = max(len(s) for s in sequences)
        padded = torch.full((len(sequences), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(sequences), max_len), dtype=torch.long)
        for i, seq in enumerate(sequences):
            padded[i, :len(seq)] = seq
            attention_mask[i, :len(seq)] = 1

        padded = padded.to(model.device)
        attention_mask = attention_mask.to(model.device)

        with torch.no_grad():
            out = model(
                input_ids=padded,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        # Extract final-token hidden state per layer per rollout
        # out.hidden_states is tuple of (n_layers+1,) each (batch, seq_len, hidden_dim)
        # Index 0 = embeddings, 1..n = transformer layers
        n_transformer_layers = len(out.hidden_states) - 1

        for i in range(len(batch_rollouts)):
            final_pos = prompt_len + gen_lengths[i] - 1
            layer_states = []
            for layer_idx in range(1, n_transformer_layers + 1):
                hs = out.hidden_states[layer_idx][i, final_pos, :].float().cpu().numpy()
                layer_states.append(hs)
            all_hidden.append(np.stack(layer_states))  # (n_layers, hidden_dim)

        del out, padded, attention_mask
        torch.cuda.empty_cache()

    return np.stack(all_hidden)  # (n_rollouts, n_layers, hidden_dim)


def plot_layer_profiles(all_metrics: dict, score_col: str, score_label: str, save_dir: Path):
    """Plot per-layer variance and cosine distance profiles, colored by score."""
    names = list(all_metrics.keys())
    scores = [all_metrics[n]["_score"] for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Normalize scores for colormap
    score_arr = np.array(scores)
    vmin, vmax = score_arr.min(), score_arr.max()
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.RdYlGn

    for name in names:
        m = all_metrics[name]
        score = m["_score"]
        color = cmap(norm(score))

        layer_vars = m["hs_layer_variances"]
        layer_cos = m["hs_layer_cosine_dists"]
        x = range(len(layer_vars))

        axes[0].plot(x, layer_vars, color=color, alpha=0.7, linewidth=1.5)
        axes[1].plot(x, layer_cos, color=color, alpha=0.7, linewidth=1.5)

    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Mean Per-Dim Variance")
    axes[0].set_title("Hidden State Variance by Layer")

    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Mean Pairwise Cosine Distance")
    axes[1].set_title("Cosine Distance by Layer")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    plt.colorbar(sm, ax=axes, label=score_label, shrink=0.8)

    fig.tight_layout()
    save_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_dir / "hidden_state_layer_profiles.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_correlations(df: pd.DataFrame, score_col: str, score_label: str, save_dir: Path):
    """Scatter plots of key hidden state metrics vs score."""
    plot_keys = [
        ("hs_var_late", "Late-Layer Variance"),
        ("hs_var_late_early_ratio", "Late/Early Variance Ratio"),
        ("hs_cosine_late", "Late-Layer Cosine Distance"),
        ("hs_cosine_late_early_ratio", "Late/Early Cosine Ratio"),
        ("hs_var_total", "Total Variance"),
        ("hs_peak_layer", "Peak Variance Layer"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, (key, label) in enumerate(plot_keys):
        ax = axes[i]
        x = df[key].values
        y = df[score_col].values

        ax.scatter(x, y, s=60, c="tab:blue", edgecolors="black", linewidth=0.8, alpha=0.9)

        if len(df) <= 20:
            for j, name in enumerate(df["example"].values):
                ax.annotate(name, (x[j], y[j]), fontsize=6, ha="left", va="bottom")

        if len(set(x)) > 1:
            r_s, p_s = stats.spearmanr(x, y)
            r_p, p_p = stats.pearsonr(x, y)
            ax.set_title(f"{label}\nSpearman \u03c1={r_s:.3f} (p={p_s:.3f})", fontsize=9)
            # Regression line
            z = np.polyfit(x, y, 1)
            x_line = np.linspace(x.min(), x.max(), 50)
            ax.plot(x_line, np.polyval(z, x_line), "r--", alpha=0.6)
        else:
            ax.set_title(label, fontsize=9)

        ax.set_xlabel(label, fontsize=8)
        ax.set_ylabel(score_label, fontsize=8)

    fig.suptitle(f"Hidden State Variance vs {score_label}", fontsize=12)
    fig.tight_layout()
    save_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_dir / "hidden_state_correlations.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compute hidden state variance from existing rollout pickles")
    parser.add_argument("--results-dir", type=str, required=True,
                        help="Directory containing entropy_*.pkl files")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Math-1.5B",
                        help="Model to use for forward passes")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size for forward passes (smaller = less GPU memory)")
    parser.add_argument("--max-rollouts", type=int, default=None,
                        help="Max rollouts per example (default: use all)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path (default: <results-dir>/hidden_state_metrics.csv)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    pkl_files = sorted(results_dir.glob("entropy_*.pkl"))

    if not pkl_files:
        print(f"ERROR: No entropy_*.pkl files found in {results_dir}")
        return

    print(f"Found {len(pkl_files)} pickle files")
    print(f"Model: {args.model}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max rollouts: {args.max_rollouts or 'all'}")
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
    print(f"Model loaded. Layers: {model.config.num_hidden_layers}, "
          f"Hidden dim: {model.config.hidden_size}")
    print()

    # Process each example
    all_metrics = {}
    t0 = time.time()

    for idx, pkl_path in enumerate(pkl_files):
        name = pkl_path.stem.replace("entropy_", "")

        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        rollouts = data["rollouts"]
        example = data["example"]

        # Tokenize prompt
        messages = [{"role": "user", "content": example["prompt_text"]}]
        prompt_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).squeeze(0)

        n_use = min(len(rollouts), args.max_rollouts) if args.max_rollouts else len(rollouts)
        print(f"[{idx+1}/{len(pkl_files)}] {name}: {n_use} rollouts, "
              f"prompt_len={len(prompt_ids)}", end="", flush=True)

        t_ex = time.time()
        hidden_states = extract_hidden_states_for_example(
            rollouts, prompt_ids, model,
            batch_size=args.batch_size,
            max_rollouts=args.max_rollouts,
        )

        metrics = compute_hidden_state_metrics(hidden_states)
        elapsed_ex = time.time() - t_ex

        # Store score for plotting
        score = (example.get("math500_score")
                 or example.get("avg_all")
                 or example.get("historical_variance")
                 or 0)
        metrics["_score"] = score

        all_metrics[name] = metrics
        print(f" → {elapsed_ex:.1f}s | var_late={metrics['hs_var_late']:.4f}, "
              f"ratio={metrics['hs_var_late_early_ratio']:.2f}")

        del hidden_states
        gc.collect()
        torch.cuda.empty_cache()

    elapsed_total = time.time() - t0
    print(f"\nDone in {elapsed_total:.1f}s")

    # Build results DataFrame
    rows = []
    for pkl_path in pkl_files:
        name = pkl_path.stem.replace("entropy_", "")
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        example = data["example"]

        row = {"example": name}
        for score_key in ["math500_score", "historical_variance", "avg_all"]:
            if example.get(score_key) is not None:
                row[score_key] = example[score_key]
        row["pass_rate"] = float(
            sum(r.get("is_correct", False) for r in data["rollouts"]) / len(data["rollouts"])
        )
        for key in SCALAR_METRIC_KEYS:
            row[key] = all_metrics[name][key]
        rows.append(row)

    df = pd.DataFrame(rows)

    # Determine score column
    score_col = None
    score_label = None
    for col, label in [("math500_score", "MATH500"), ("avg_all", "Avg All"),
                        ("historical_variance", "Historical Variance")]:
        if col in df.columns:
            score_col = col
            score_label = label
            break

    # Print correlation table
    if score_col:
        print(f"\n{'=' * 70}")
        print(f"Correlations with {score_label} (n={len(df)})")
        print(f"{'=' * 70}")
        print(f"{'metric':<30} {'Spearman r':>10} {'p-value':>10} {'Pearson r':>10} {'p-value':>10}")
        print("-" * 70)
        for key in SCALAR_METRIC_KEYS:
            r_s, p_s = stats.spearmanr(df[key], df[score_col])
            r_p, p_p = stats.pearsonr(df[key], df[score_col])
            sig = "***" if p_s < 0.01 else "**" if p_s < 0.05 else "*" if p_s < 0.1 else ""
            print(f"{key:<30} {r_s:>10.3f} {p_s:>10.4f} {r_p:>10.3f} {p_p:>10.4f} {sig}")

    # Save CSV
    output_path = Path(args.output) if args.output else results_dir / "hidden_state_metrics.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved metrics to {output_path}")

    # Save per-layer profiles as JSON
    profiles = {name: {
        "layer_variances": all_metrics[name]["hs_layer_variances"],
        "layer_cosine_dists": all_metrics[name]["hs_layer_cosine_dists"],
    } for name in all_metrics}
    json_path = results_dir / "hidden_state_profiles.json"
    with open(json_path, "w") as f:
        json.dump(profiles, f, indent=2)
    print(f"Saved per-layer profiles to {json_path}")

    # Generate plots
    if score_col:
        fig_dir = results_dir.parent / "figures"
        print(f"\nGenerating plots to {fig_dir}...")
        plot_layer_profiles(all_metrics, score_col, score_label, fig_dir)
        plot_correlations(df, score_col, score_label, fig_dir)
        print("Done!")


if __name__ == "__main__":
    main()
