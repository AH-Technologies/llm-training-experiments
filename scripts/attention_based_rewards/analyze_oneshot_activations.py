#!/usr/bin/env python3
"""Analyze activations for 14 one-shot pi examples from Wang et al.

For each pi example, generates N rollouts with the base model, runs forward
passes with hooks, and computes per-example activation features. These are
then correlated with the known math500 training scores to understand why some
examples are better for training than others.

Usage:
  sbatch scripts/attention_based_rewards/slurm/analyze_oneshot_activations.slurm
"""

import os
import sys
import time
import pickle
import glob
import re
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

os.environ.setdefault("HF_HOME", "/cluster/projects/nn12068k/haaklau/.cache/huggingface")

BASE_DIR = Path("attention_based_rewards")
MODEL_PATH = "/cluster/projects/nn12068k/haaklau/.cache/huggingface/hub/models--Qwen--Qwen2.5-Math-1.5B/snapshots/4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2"
REASONING_HEADS_PATH = BASE_DIR / "data" / "base_model_reasoning_heads.pt"
ENTROPY_SUMMARY_PATH = Path("one_shot_metrics/entropy_profiling/results/entropy_summary.csv")
PICKLE_DIR = Path("one_shot_metrics/entropy_profiling/results/entropy_profiles")
SAVE_PATH = BASE_DIR / "data" / "oneshot_activation_features.csv"
PLOT_DIR = BASE_DIR / "plots"

N_ROLLOUTS = 16
MAX_NEW_TOKENS = 1024
N_LAYERS = 28
N_HEADS = 12
HEAD_DIM = 128
TOP_K_HEADS = 10


def load_pi_examples():
    """Load all 14 pi examples from pickle files."""
    pkl_files = sorted(glob.glob(str(PICKLE_DIR / "entropy_pi_*.pkl")))
    examples = []
    for pkl_path in pkl_files:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        ex = data["example"]
        examples.append({
            "name": ex["name"],
            "prompt_text": ex["prompt_text"],
            "ground_truth": ex["ground_truth"],
            "math500_score": ex["math500_score"],
            "pass_rate": data["pass_rate"],
        })
        print(f"  Loaded {ex['name']}: math500={ex['math500_score']}, pass_rate={data['pass_rate']:.3f}")
    print(f"Total: {len(examples)} pi examples loaded")
    return examples


def load_reasoning_heads():
    """Load top-K EAP-IG reasoning heads."""
    data = torch.load(REASONING_HEADS_PATH, map_location="cpu", weights_only=False)
    top_heads = [(l, h) for l, h, _ in data["selected_heads"][:TOP_K_HEADS]]
    all_heads = set((l, h) for l, h, _ in data["selected_heads"])
    print(f"Top-{TOP_K_HEADS} reasoning heads: {top_heads}")
    return top_heads, all_heads


def check_answer(response: str, ground_truth: str) -> bool:
    """Simple answer extraction and comparison."""
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", response, re.DOTALL)
    if match:
        extracted = match.group(1).strip()
    else:
        match = re.search(r"\\boxed\{([^}]*)\}", response)
        if match:
            extracted = match.group(1).strip()
        else:
            numbers = re.findall(r"[-+]?\d*\.?\d+", response)
            extracted = numbers[-1] if numbers else ""
    gt = ground_truth.strip().replace(",", "").replace("$", "")
    ext = extracted.strip().replace(",", "").replace("$", "")
    try:
        return abs(float(ext) - float(gt)) < 1e-6
    except (ValueError, TypeError):
        return ext == gt


def analyze_example(model, tokenizer, example, top_heads, device):
    """Run rollouts and compute activation features for one pi example."""
    prompt_text = example["prompt_text"]

    # Tokenize
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        msgs = [{"role": "user", "content": prompt_text}]
        prompt_str = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
    else:
        prompt_str = f"Question: {prompt_text}\nLet's solve this step by step.\n"

    prompt_ids = tokenizer(prompt_str, return_tensors="pt", truncation=True, max_length=1024)
    prompt_len = prompt_ids["input_ids"].shape[1]

    # Storage for hook captures
    hook_data = {}

    def make_attn_output_hook(layer_idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hook_data[f"attn_out_{layer_idx}"] = output[0].detach()
            else:
                hook_data[f"attn_out_{layer_idx}"] = output.detach()
        return hook_fn

    def make_mlp_output_hook(layer_idx):
        def hook_fn(module, input, output):
            hook_data[f"mlp_out_{layer_idx}"] = output.detach()
        return hook_fn

    def make_resid_pre_hook(layer_idx):
        def hook_fn(module, input):
            if isinstance(input, tuple):
                hook_data[f"resid_pre_{layer_idx}"] = input[0].detach()
            else:
                hook_data[f"resid_pre_{layer_idx}"] = input.detach()
        return hook_fn

    def make_mlp_gate_hook(layer_idx):
        def hook_fn(module, input, output):
            hook_data[f"mlp_gate_{layer_idx}"] = output.detach()
        return hook_fn

    # Register hooks
    hooks = []
    for layer_idx in range(N_LAYERS):
        layer = model.model.layers[layer_idx]
        hooks.append(layer.self_attn.register_forward_hook(make_attn_output_hook(layer_idx)))
        hooks.append(layer.mlp.register_forward_hook(make_mlp_output_hook(layer_idx)))
        hooks.append(layer.register_forward_pre_hook(make_resid_pre_hook(layer_idx)))
        if hasattr(layer.mlp, 'gate_proj'):
            hooks.append(layer.mlp.gate_proj.register_forward_hook(make_mlp_gate_hook(layer_idx)))

    # Accumulators for per-rollout features
    all_head_fai = []          # (n_rollouts, N_LAYERS, N_HEADS)
    all_head_norms = []        # same shape
    all_mlp_norms = []         # (n_rollouts, N_LAYERS)
    all_gate_activity = []     # (n_rollouts, N_LAYERS)
    all_resid_growth = []      # (n_rollouts,) total growth
    all_resp_lengths = []
    all_answers = []
    n_correct = 0

    for rollout_idx in range(N_ROLLOUTS):
        # Generate one response
        with torch.no_grad():
            outputs = model.generate(
                prompt_ids["input_ids"].to(device),
                attention_mask=prompt_ids["attention_mask"].to(device),
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.7,
                top_p=1.0,
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
            )

        full_ids = outputs[0:1]
        resp_text = tokenizer.decode(full_ids[0, prompt_len:], skip_special_tokens=True)
        correct = check_answer(resp_text, example["ground_truth"])
        seq_len = full_ids.shape[1]
        resp_len = seq_len - prompt_len

        if resp_len < 5:
            continue

        all_resp_lengths.append(resp_len)
        if correct:
            n_correct += 1

        # Extract answer for diversity count
        match = re.search(r"\\boxed\{([^}]*)\}", resp_text)
        if match:
            all_answers.append(match.group(1).strip())
        else:
            match = re.search(r"<answer>\s*(.*?)\s*</answer>", resp_text, re.DOTALL)
            if match:
                all_answers.append(match.group(1).strip())
            else:
                all_answers.append(resp_text[-50:])  # fallback

        # Forward pass with hooks + attention
        hook_data.clear()
        with torch.no_grad():
            out = model(
                input_ids=full_ids.to(device),
                output_attentions=True,
                use_cache=False,
            )

        attentions = out.attentions
        future_mask = torch.tril(torch.ones(seq_len, seq_len, device=device), diagonal=-1)
        future_count = future_mask.sum(dim=0).clamp(min=1)

        rollout_fai = np.zeros((N_LAYERS, N_HEADS))
        rollout_head_norms = np.zeros((N_LAYERS, N_HEADS))
        rollout_mlp_norms = np.zeros(N_LAYERS)
        rollout_gate_activity = np.zeros(N_LAYERS)
        total_resid_growth = 0.0

        for layer_idx in range(N_LAYERS):
            attn = attentions[layer_idx][0]  # (n_heads, seq_len, seq_len)

            # FAI per head
            for head_idx in range(N_HEADS):
                a = attn[head_idx]
                future_attn = a * future_mask
                received = future_attn.sum(dim=0) / future_count
                fai_val = received[prompt_len:].mean().item()
                rollout_fai[layer_idx, head_idx] = fai_val

            # Attention head output norms
            attn_out = hook_data.get(f"attn_out_{layer_idx}")
            if attn_out is not None:
                ao = attn_out[0, prompt_len:, :]
                ao_heads = ao.reshape(ao.shape[0], N_HEADS, HEAD_DIM)
                head_norms = ao_heads.float().norm(dim=-1).mean(dim=0).cpu().numpy()
                rollout_head_norms[layer_idx] = head_norms

            # MLP output norm
            mlp_out = hook_data.get(f"mlp_out_{layer_idx}")
            if mlp_out is not None:
                mlp_resp = mlp_out[0, prompt_len:, :]
                rollout_mlp_norms[layer_idx] = mlp_resp.float().norm(dim=-1).mean().item()

            # Gate activity
            gate_out = hook_data.get(f"mlp_gate_{layer_idx}")
            if gate_out is not None:
                gate_resp = gate_out[0, prompt_len:, :]
                rollout_gate_activity[layer_idx] = (gate_resp > 0).float().mean().item()

            # Residual growth
            resid_pre = hook_data.get(f"resid_pre_{layer_idx}")
            if resid_pre is not None and attn_out is not None and mlp_out is not None:
                rp = resid_pre[0, prompt_len:, :].float()
                ao_f = attn_out[0, prompt_len:, :].float()
                mo_f = mlp_out[0, prompt_len:, :].float()
                resid_post_mlp = rp + ao_f + mo_f
                growth = resid_post_mlp.norm(dim=-1).mean().item() - rp.norm(dim=-1).mean().item()
                total_resid_growth += growth

        all_head_fai.append(rollout_fai)
        all_head_norms.append(rollout_head_norms)
        all_mlp_norms.append(rollout_mlp_norms)
        all_gate_activity.append(rollout_gate_activity)
        all_resid_growth.append(total_resid_growth)

        del out, attentions
        hook_data.clear()
        torch.cuda.empty_cache()

    # Clean up hooks
    for h in hooks:
        h.remove()

    if len(all_head_fai) == 0:
        print(f"  WARNING: No valid rollouts for {example['name']}")
        return None

    # Aggregate across rollouts
    mean_fai = np.mean(all_head_fai, axis=0)         # (N_LAYERS, N_HEADS)
    mean_head_norms = np.mean(all_head_norms, axis=0)
    mean_mlp_norms = np.mean(all_mlp_norms, axis=0)  # (N_LAYERS,)
    mean_gate = np.mean(all_gate_activity, axis=0)

    # Top-K reasoning head indices
    top_head_set = set(top_heads)
    non_reasoning = [(l, h) for l in range(N_LAYERS) for h in range(N_HEADS) if (l, h) not in top_head_set]

    # Compute features
    reasoning_fai_vals = [mean_fai[l, h] for l, h in top_heads]
    non_reasoning_fai_vals = [mean_fai[l, h] for l, h in non_reasoning]
    total_fai = mean_fai.sum()
    reasoning_fai_sum = sum(reasoning_fai_vals)

    features = {
        "example": example["name"],
        "math500_score": example["math500_score"],
        "pass_rate": n_correct / len(all_head_fai),
        "response_length": np.mean(all_resp_lengths),
        "answer_diversity": len(set(all_answers)),
        "reasoning_head_fai": np.mean(reasoning_fai_vals),
        "reasoning_head_norm": np.mean([mean_head_norms[l, h] for l, h in top_heads]),
        "non_reasoning_head_fai": np.mean(non_reasoning_fai_vals),
        "fai_concentration": reasoning_fai_sum / total_fai if total_fai > 0 else 0,
        "mlp_norm_mean": mean_mlp_norms.mean(),
        "mlp_L27_norm": mean_mlp_norms[27],
        "residual_growth": np.mean(all_resid_growth),
        "late_layer_fai": mean_fai[20:28, :].mean(),
        "early_layer_fai": mean_fai[0:14, :].mean(),
        "late_vs_early_fai": mean_fai[20:28, :].mean() / max(mean_fai[0:14, :].mean(), 1e-10),
        "gate_activity": mean_gate.mean(),
    }

    # Per-reasoning-head FAI for the heatmap
    per_head_fai = {}
    for i, (l, h) in enumerate(top_heads):
        per_head_fai[f"rh{i}_L{l}H{h}_fai"] = mean_fai[l, h]
    features.update(per_head_fai)

    return features


def merge_with_entropy(df_features):
    """Merge activation features with existing entropy metrics."""
    if not ENTROPY_SUMMARY_PATH.exists():
        print(f"  Warning: {ENTROPY_SUMMARY_PATH} not found, skipping merge")
        return df_features

    df_entropy = pd.read_csv(ENTROPY_SUMMARY_PATH)
    # Entropy CSV has 'example' column with names like 'pi_13'
    # Our features also use that naming
    merged = df_features.merge(df_entropy, on="example", how="left", suffixes=("", "_entropy"))
    # If math500_score appears in both, drop the entropy version
    if "math500_score_entropy" in merged.columns:
        merged.drop(columns=["math500_score_entropy"], inplace=True)
    print(f"  Merged: {len(merged)} rows, {len(merged.columns)} columns")
    return merged


def create_plots(df):
    """Create all four visualization plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import matplotlib.gridspec as gridspec

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # Identify activation feature columns (exclude metadata and per-head columns)
    meta_cols = {"example", "math500_score", "ground_truth", "historical_variance"}
    per_head_cols = {c for c in df.columns if c.startswith("rh")}
    entropy_cols_to_use = {
        "mean_entropy_mean", "cross_rollout_entropy_var", "entropy_trend_mean",
        "num_tokens_mean", "high_entropy_ratio_mean", "num_spikes_mean",
    }

    activation_features = [
        "reasoning_head_fai", "reasoning_head_norm", "non_reasoning_head_fai",
        "fai_concentration", "mlp_norm_mean", "mlp_L27_norm", "residual_growth",
        "late_vs_early_fai", "gate_activity", "pass_rate", "response_length",
        "answer_diversity",
    ]
    # Add entropy features that exist in the merged df
    for c in entropy_cols_to_use:
        if c in df.columns:
            activation_features.append(c)

    # Filter to features that exist
    activation_features = [f for f in activation_features if f in df.columns]

    # ================================================================
    # Plot 1: Correlation matrix heatmap
    # ================================================================
    corr_cols = ["math500_score"] + activation_features
    df_corr = df[corr_cols].apply(pd.to_numeric, errors="coerce")
    corr_matrix = df_corr.corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(corr_matrix.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(corr_cols)))
    ax.set_xticklabels(corr_cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(corr_cols)))
    ax.set_yticklabels(corr_cols, fontsize=8)

    # Annotate cells with correlation values
    for i in range(len(corr_cols)):
        for j in range(len(corr_cols)):
            val = corr_matrix.values[i, j]
            if not np.isnan(val):
                color = "white" if abs(val) > 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=6, color=color)

    plt.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
    ax.set_title("Feature Correlation Matrix (14 Pi Examples)")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "oneshot_correlation_matrix.png", dpi=150, bbox_inches="tight")
    print("Saved oneshot_correlation_matrix.png")
    plt.close()

    # ================================================================
    # Plot 2: Top 6 scatterplots vs math500_score
    # ================================================================
    # Compute correlations with math500
    correlations = {}
    for feat in activation_features:
        vals = df[feat].apply(pd.to_numeric, errors="coerce")
        valid = vals.notna() & df["math500_score"].notna()
        if valid.sum() >= 4:
            r, p = stats.pearsonr(vals[valid], df["math500_score"][valid])
            correlations[feat] = (r, p)

    # Sort by absolute correlation
    sorted_feats = sorted(correlations.keys(), key=lambda x: abs(correlations[x][0]), reverse=True)
    top6 = sorted_feats[:6]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes_flat = axes.flatten()

    for idx, feat in enumerate(top6):
        ax = axes_flat[idx]
        x = df[feat].apply(pd.to_numeric, errors="coerce")
        y = df["math500_score"]
        valid = x.notna() & y.notna()

        ax.scatter(x[valid], y[valid], s=80, c="#3498db", edgecolors="black", linewidths=0.5, zorder=5)

        # Add labels
        for _, row in df[valid].iterrows():
            ax.annotate(row["example"].replace("pi_", ""),
                        (float(row[feat]), row["math500_score"]),
                        fontsize=7, ha="left", va="bottom",
                        xytext=(3, 3), textcoords="offset points")

        # Regression line
        r, p = correlations[feat]
        slope, intercept = np.polyfit(x[valid], y[valid], 1)
        x_line = np.linspace(x[valid].min(), x[valid].max(), 100)
        ax.plot(x_line, slope * x_line + intercept, "r--", alpha=0.7, linewidth=2)

        ax.set_xlabel(feat, fontsize=10)
        ax.set_ylabel("math500_score", fontsize=10)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        ax.set_title(f"r={r:.3f}, p={p:.3f} {sig}", fontsize=11)
        ax.grid(alpha=0.3)

    plt.suptitle("Top 6 Features Correlated with math500 Training Score", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "oneshot_scatter_vs_math500.png", dpi=150, bbox_inches="tight")
    print("Saved oneshot_scatter_vs_math500.png")
    plt.close()

    # ================================================================
    # Plot 3: Per-example reasoning head FAI heatmap
    # ================================================================
    rh_cols = sorted([c for c in df.columns if c.startswith("rh")],
                     key=lambda x: int(x.split("_")[0].replace("rh", "")))

    if len(rh_cols) > 0:
        # Sort by math500_score
        df_sorted = df.sort_values("math500_score", ascending=False)
        heatmap_data = df_sorted[rh_cols].apply(pd.to_numeric, errors="coerce").values
        example_labels = [f"{r['example']} ({r['math500_score']}%)" for _, r in df_sorted.iterrows()]
        head_labels = [c.split("_", 1)[1] for c in rh_cols]

        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(heatmap_data, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(len(head_labels)))
        ax.set_xticklabels(head_labels, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(len(example_labels)))
        ax.set_yticklabels(example_labels, fontsize=9)

        # Annotate
        for i in range(heatmap_data.shape[0]):
            for j in range(heatmap_data.shape[1]):
                val = heatmap_data[i, j]
                if not np.isnan(val):
                    color = "white" if val > np.nanmean(heatmap_data) + np.nanstd(heatmap_data) else "black"
                    ax.text(j, i, f"{val:.4f}", ha="center", va="center", fontsize=7, color=color)

        plt.colorbar(im, ax=ax, shrink=0.8, label="Mean FAI")
        ax.set_title(f"Reasoning Head FAI Profiles (Top-{TOP_K_HEADS} EAP-IG Heads)\nSorted by math500 score (high→low)")
        ax.set_xlabel("Reasoning Head")
        ax.set_ylabel("Pi Example (math500 score)")
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "oneshot_reasoning_head_profiles.png", dpi=150, bbox_inches="tight")
        print("Saved oneshot_reasoning_head_profiles.png")
        plt.close()

    # ================================================================
    # Plot 4: Radar/spider chart — top-3 vs bottom-3 examples
    # ================================================================
    df_sorted = df.sort_values("math500_score", ascending=False)
    top3 = df_sorted.head(3)
    bot3 = df_sorted.tail(3)

    radar_features = [f for f in [
        "reasoning_head_fai", "fai_concentration", "gate_activity",
        "late_vs_early_fai", "residual_growth", "mlp_norm_mean",
        "reasoning_head_norm", "response_length",
    ] if f in df.columns]

    if len(radar_features) >= 4:
        # Normalize features to [0, 1] for radar
        vals_all = df[radar_features].apply(pd.to_numeric, errors="coerce")
        mins = vals_all.min()
        maxs = vals_all.max()
        ranges = maxs - mins
        ranges[ranges == 0] = 1

        top3_norm = ((top3[radar_features].apply(pd.to_numeric, errors="coerce") - mins) / ranges).mean()
        bot3_norm = ((bot3[radar_features].apply(pd.to_numeric, errors="coerce") - mins) / ranges).mean()

        angles = np.linspace(0, 2 * np.pi, len(radar_features), endpoint=False).tolist()
        angles += angles[:1]

        top3_vals = top3_norm.values.tolist() + [top3_norm.values[0]]
        bot3_vals = bot3_norm.values.tolist() + [bot3_norm.values[0]]

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        ax.fill(angles, top3_vals, alpha=0.2, color="#2ecc71")
        ax.plot(angles, top3_vals, "o-", color="#2ecc71", linewidth=2,
                label=f"Top 3 (mean score={top3['math500_score'].mean():.1f}%)")
        ax.fill(angles, bot3_vals, alpha=0.2, color="#e74c3c")
        ax.plot(angles, bot3_vals, "s--", color="#e74c3c", linewidth=2,
                label=f"Bottom 3 (mean score={bot3['math500_score'].mean():.1f}%)")

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(radar_features, fontsize=9)
        ax.set_ylim(0, 1.1)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=11)
        ax.set_title("Feature Profiles: Best vs Worst Pi Examples\n(normalized to [0,1])", fontsize=13, pad=20)
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "oneshot_feature_radar.png", dpi=150, bbox_inches="tight")
        print("Saved oneshot_feature_radar.png")
        plt.close()


def print_summary(df):
    """Print a text summary of the results."""
    print(f"\n{'='*70}")
    print("ONE-SHOT ACTIVATION ANALYSIS SUMMARY")
    print(f"{'='*70}")
    print(f"\n{len(df)} pi examples analyzed with {N_ROLLOUTS} rollouts each\n")

    # Print feature table
    cols_to_show = [
        "example", "math500_score", "pass_rate", "reasoning_head_fai",
        "fai_concentration", "late_vs_early_fai", "residual_growth",
        "gate_activity", "response_length", "answer_diversity",
    ]
    cols_to_show = [c for c in cols_to_show if c in df.columns]

    print(df[cols_to_show].sort_values("math500_score", ascending=False).to_string(index=False))

    # Correlations with math500
    print(f"\n--- Correlations with math500_score ---")
    activation_features = [c for c in df.columns
                           if c not in {"example", "math500_score", "ground_truth", "historical_variance"}
                           and not c.startswith("rh")]
    for feat in activation_features:
        vals = pd.to_numeric(df[feat], errors="coerce")
        valid = vals.notna() & df["math500_score"].notna()
        if valid.sum() >= 4:
            r, p = stats.pearsonr(vals[valid], df["math500_score"][valid])
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            if abs(r) > 0.3:
                print(f"  {feat:<35s} r={r:+.3f}  p={p:.4f} {sig}")


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    print("\n--- Loading pi examples ---")
    examples = load_pi_examples()

    print("\n--- Loading reasoning heads ---")
    top_heads, all_heads = load_reasoning_heads()

    # Load model
    print("\n--- Loading model ---")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, attn_implementation="eager",
    ).to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Analyze each example
    print(f"\n--- Analyzing {len(examples)} examples × {N_ROLLOUTS} rollouts ---")
    all_features = []
    t0 = time.time()

    for i, example in enumerate(examples):
        print(f"\n[{i+1}/{len(examples)}] {example['name']} (math500={example['math500_score']})")
        features = analyze_example(model, tokenizer, example, top_heads, device)
        if features is not None:
            all_features.append(features)
            print(f"  reasoning_fai={features['reasoning_head_fai']:.6f}, "
                  f"fai_conc={features['fai_concentration']:.4f}, "
                  f"pass_rate={features['pass_rate']:.3f}, "
                  f"resp_len={features['response_length']:.0f}")

        elapsed = time.time() - t0
        rate = (i + 1) / elapsed * 60
        remaining = (len(examples) - i - 1) / (rate / 60) if rate > 0 else 0
        print(f"  Rate: {rate:.1f} examples/min, ~{remaining:.0f}s remaining")

    # Create DataFrame
    df = pd.DataFrame(all_features)

    # Merge with entropy metrics
    print("\n--- Merging with entropy metrics ---")
    df = merge_with_entropy(df)

    # Save CSV
    BASE_DIR_DATA = BASE_DIR / "data"
    BASE_DIR_DATA.mkdir(parents=True, exist_ok=True)
    df.to_csv(SAVE_PATH, index=False)
    print(f"\nSaved features to {SAVE_PATH}")

    # Create plots
    print("\n--- Creating plots ---")
    create_plots(df)

    # Print summary
    print_summary(df)

    print(f"\nTotal time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
