#!/usr/bin/env python3
"""Suffix activation comparison: how do different prompt suffixes change internal activations?

Runs the same 100 problems with 3 suffix conditions (no_suffix, step_suffix, random_suffix),
4 responses each. Compares attention head FAI, output norms, MLP norms, gate activity,
and residual stream growth across suffix conditions on the base model.

Uses 4 GPUs with data parallelism.

Usage:
  sbatch scripts/attention_based_rewards/slurm/analyze_suffix_activations.slurm
"""

import os
import sys
import random
import time
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field

os.environ.setdefault("HF_HOME", "/cluster/projects/nn12068k/haaklau/.cache/huggingface")

BASE_DIR = Path("attention_based_rewards")
MODEL_PATH = "/cluster/projects/nn12068k/haaklau/.cache/huggingface/hub/models--Qwen--Qwen2.5-Math-1.5B/snapshots/4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2"
DATA_PATH = BASE_DIR / "data" / "dapo_math_17k.parquet"
SAVE_PATH = BASE_DIR / "data" / "suffix_activation_analysis.pt"
PLOT_DIR = BASE_DIR / "plots"

N_EXAMPLES = 100
N_SAMPLES = 4
MAX_NEW_TOKENS = 512
N_LAYERS = 28
N_HEADS = 12
HEAD_DIM = 128  # 1536 / 12

SUFFIX_CONDITIONS = ["no_suffix", "step_suffix", "random_suffix"]

STEP_SUFFIX = " Let's think step by step and output the final answer within \\boxed{}."

# Import the suffix list from suffix_patch
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from attention_based_rewards.suffix_patch import SUFFIXES as RANDOM_SUFFIXES


def check_answer(response: str, ground_truth: str) -> bool:
    """Simple answer extraction and comparison."""
    import re
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


@dataclass
class SuffixAccumulators:
    """Stores running sums for all circuit components, per suffix condition."""
    # Attention heads: FAI per head — {condition: tensor(N_LAYERS, N_HEADS)}
    attn_fai: dict = field(default_factory=lambda: {c: torch.zeros(N_LAYERS, N_HEADS) for c in SUFFIX_CONDITIONS})
    attn_norm: dict = field(default_factory=lambda: {c: torch.zeros(N_LAYERS, N_HEADS) for c in SUFFIX_CONDITIONS})
    mlp_norm: dict = field(default_factory=lambda: {c: torch.zeros(N_LAYERS) for c in SUFFIX_CONDITIONS})
    mlp_gate: dict = field(default_factory=lambda: {c: torch.zeros(N_LAYERS) for c in SUFFIX_CONDITIONS})
    resid_post_attn: dict = field(default_factory=lambda: {c: torch.zeros(N_LAYERS) for c in SUFFIX_CONDITIONS})
    resid_post_mlp: dict = field(default_factory=lambda: {c: torch.zeros(N_LAYERS) for c in SUFFIX_CONDITIONS})
    resid_attn_growth: dict = field(default_factory=lambda: {c: torch.zeros(N_LAYERS) for c in SUFFIX_CONDITIONS})
    resid_mlp_growth: dict = field(default_factory=lambda: {c: torch.zeros(N_LAYERS) for c in SUFFIX_CONDITIONS})
    # Counts and accuracy
    n_samples: dict = field(default_factory=lambda: {c: 0 for c in SUFFIX_CONDITIONS})
    n_correct: dict = field(default_factory=lambda: {c: 0 for c in SUFFIX_CONDITIONS})


def apply_suffix(prompt_msgs: list, condition: str) -> list:
    """Return a modified copy of prompt_msgs with the appropriate suffix."""
    import copy
    msgs = copy.deepcopy(prompt_msgs)
    if condition == "no_suffix":
        return msgs
    for msg in reversed(msgs):
        if msg["role"] == "user":
            if condition == "step_suffix":
                msg["content"] = msg["content"] + STEP_SUFFIX
            elif condition == "random_suffix":
                msg["content"] = msg["content"] + " " + random.choice(RANDOM_SUFFIXES)
            break
    return msgs


def analyze_on_gpu(gpu_id: int, problem_indices: list, return_dict: dict):
    """Run analysis on a single GPU for a subset of problems."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)
    print(f"[GPU {gpu_id}] Loading model...", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, attn_implementation="eager",
    ).to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    df = pd.read_parquet(DATA_PATH)
    acc = SuffixAccumulators()

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

    t0 = time.time()
    for idx_i, data_idx in enumerate(problem_indices):
        row = df.iloc[data_idx]
        prompt_msgs = row["prompt"]
        gt = row["reward_model"]["ground_truth"]

        # Run all 3 suffix conditions for this problem
        for condition in SUFFIX_CONDITIONS:
            modified_msgs = apply_suffix(prompt_msgs, condition)

            # Build prompt
            if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
                prompt_text = tokenizer.apply_chat_template(
                    modified_msgs, tokenize=False, add_generation_prompt=True
                )
            else:
                for msg in modified_msgs:
                    if msg["role"] == "user":
                        prompt_text = f"Question: {msg['content']}\nLet's solve this step by step.\n"

            prompt_ids = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=1024)
            prompt_len = prompt_ids["input_ids"].shape[1]

            # Generate responses
            with torch.no_grad():
                outputs = model.generate(
                    prompt_ids["input_ids"].to(device),
                    attention_mask=prompt_ids["attention_mask"].to(device),
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=0.7,
                    top_p=1.0,
                    do_sample=True,
                    num_return_sequences=N_SAMPLES,
                    pad_token_id=tokenizer.pad_token_id,
                )

            # Analyze each response
            for s in range(N_SAMPLES):
                full_ids = outputs[s:s+1]
                resp_text = tokenizer.decode(full_ids[0, prompt_len:], skip_special_tokens=True)
                correct = check_answer(resp_text, gt)
                seq_len = full_ids.shape[1]
                resp_len = seq_len - prompt_len
                if resp_len < 5:
                    continue

                # Forward pass with hooks + attention
                hook_data.clear()
                with torch.no_grad():
                    out = model(
                        input_ids=full_ids.to(device),
                        output_attentions=True,
                        use_cache=False,
                    )

                attentions = out.attentions

                # FAI computation
                future_mask = torch.tril(torch.ones(seq_len, seq_len, device=device), diagonal=-1)
                future_count = future_mask.sum(dim=0).clamp(min=1)

                for layer_idx in range(N_LAYERS):
                    attn = attentions[layer_idx][0]

                    # Attention heads: FAI
                    for head_idx in range(N_HEADS):
                        a = attn[head_idx]
                        future_attn = a * future_mask
                        received = future_attn.sum(dim=0) / future_count
                        fai_val = received[prompt_len:].mean().item()
                        acc.attn_fai[condition][layer_idx, head_idx] += fai_val

                    # Attention head output norms
                    attn_out = hook_data.get(f"attn_out_{layer_idx}")
                    if attn_out is not None:
                        ao = attn_out[0, prompt_len:, :]
                        ao_heads = ao.reshape(ao.shape[0], N_HEADS, HEAD_DIM)
                        head_norms = ao_heads.float().norm(dim=-1).mean(dim=0)
                        acc.attn_norm[condition][layer_idx] += head_norms.cpu()

                    # MLP output norm
                    mlp_out = hook_data.get(f"mlp_out_{layer_idx}")
                    if mlp_out is not None:
                        mlp_resp = mlp_out[0, prompt_len:, :]
                        mlp_norm_val = mlp_resp.float().norm(dim=-1).mean().item()
                        acc.mlp_norm[condition][layer_idx] += mlp_norm_val

                    # MLP gate activation
                    gate_out = hook_data.get(f"mlp_gate_{layer_idx}")
                    if gate_out is not None:
                        gate_resp = gate_out[0, prompt_len:, :]
                        gate_active = (gate_resp > 0).float().mean().item()
                        acc.mlp_gate[condition][layer_idx] += gate_active

                    # Residual stream norms
                    resid_pre = hook_data.get(f"resid_pre_{layer_idx}")
                    if resid_pre is not None and attn_out is not None and mlp_out is not None:
                        rp = resid_pre[0, prompt_len:, :].float()
                        ao_f = attn_out[0, prompt_len:, :].float()
                        mo = mlp_out[0, prompt_len:, :].float()

                        resid_post_attn = rp + ao_f
                        resid_post_mlp = resid_post_attn + mo

                        rp_norm = rp.norm(dim=-1).mean().item()
                        rpa_norm = resid_post_attn.norm(dim=-1).mean().item()
                        rpm_norm = resid_post_mlp.norm(dim=-1).mean().item()

                        acc.resid_post_attn[condition][layer_idx] += rpa_norm
                        acc.resid_post_mlp[condition][layer_idx] += rpm_norm
                        acc.resid_attn_growth[condition][layer_idx] += (rpa_norm - rp_norm)
                        acc.resid_mlp_growth[condition][layer_idx] += (rpm_norm - rpa_norm)

                acc.n_samples[condition] += 1
                if correct:
                    acc.n_correct[condition] += 1

                del out, attentions
                hook_data.clear()
                torch.cuda.empty_cache()

        if (idx_i + 1) % 5 == 0:
            elapsed = time.time() - t0
            rate = (idx_i + 1) / elapsed * 60
            counts = {c: acc.n_samples[c] for c in SUFFIX_CONDITIONS}
            accs = {c: f"{acc.n_correct[c]}/{acc.n_samples[c]}" for c in SUFFIX_CONDITIONS}
            print(f"  [GPU {gpu_id}] [{idx_i+1}/{len(problem_indices)}] "
                  f"samples={counts}, accuracy={accs}, "
                  f"rate={rate:.1f} problems/min", flush=True)

    for h in hooks:
        h.remove()

    # Build result dict
    result = {}
    for metric in ["attn_fai", "attn_norm", "mlp_norm", "mlp_gate",
                    "resid_post_attn", "resid_post_mlp", "resid_attn_growth", "resid_mlp_growth"]:
        for cond in SUFFIX_CONDITIONS:
            result[f"{metric}_{cond}"] = getattr(acc, metric)[cond]
    for cond in SUFFIX_CONDITIONS:
        result[f"n_samples_{cond}"] = acc.n_samples[cond]
        result[f"n_correct_{cond}"] = acc.n_correct[cond]

    return_dict[gpu_id] = result
    print(f"[GPU {gpu_id}] Done. Samples per condition: "
          + ", ".join(f"{c}={acc.n_samples[c]}" for c in SUFFIX_CONDITIONS), flush=True)


def merge_results(results: dict) -> dict:
    """Merge results from multiple GPUs."""
    merged = None
    for gpu_id, r in results.items():
        if merged is None:
            merged = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in r.items()}
        else:
            for k, v in r.items():
                if isinstance(v, torch.Tensor):
                    merged[k] += v
                elif isinstance(v, (int, float)):
                    merged[k] += v
    return merged


def compute_means(data: dict) -> dict:
    """Normalize accumulated sums by sample counts to get means per condition."""
    results = {}
    metrics = ["attn_fai", "attn_norm", "mlp_norm", "mlp_gate",
               "resid_post_attn", "resid_post_mlp", "resid_attn_growth", "resid_mlp_growth"]

    for metric in metrics:
        results[metric] = {}
        for cond in SUFFIX_CONDITIONS:
            n = max(data[f"n_samples_{cond}"], 1)
            results[metric][cond] = data[f"{metric}_{cond}"] / n

    # Accuracy
    results["accuracy"] = {}
    for cond in SUFFIX_CONDITIONS:
        n = max(data[f"n_samples_{cond}"], 1)
        results["accuracy"][cond] = data[f"n_correct_{cond}"] / n

    # Sample counts
    results["n_samples"] = {cond: data[f"n_samples_{cond}"] for cond in SUFFIX_CONDITIONS}
    results["n_correct"] = {cond: data[f"n_correct_{cond}"] for cond in SUFFIX_CONDITIONS}

    return results


def create_plots(results: dict):
    """Create comprehensive visualizations comparing suffix conditions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    COLORS = {"no_suffix": "#3498db", "step_suffix": "#2ecc71", "random_suffix": "#e74c3c"}
    LABELS = {"no_suffix": "No suffix", "step_suffix": "Step-by-step suffix", "random_suffix": "Random suffix"}

    x = np.arange(N_LAYERS)

    # ================================================================
    # Plot 1: Attention FAI heatmap trio (one per suffix, side-by-side)
    # ================================================================
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    # Get shared color range
    all_fai = [results["attn_fai"][c].numpy() for c in SUFFIX_CONDITIONS]
    vmin = min(f.min() for f in all_fai)
    vmax = max(f.max() for f in all_fai)

    for ax_idx, cond in enumerate(SUFFIX_CONDITIONS):
        ax = axes[ax_idx]
        fai = all_fai[ax_idx]
        im = ax.imshow(fai, cmap="YlOrRd", aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(N_HEADS))
        ax.set_xticklabels([f"H{h}" for h in range(N_HEADS)], fontsize=7)
        ax.set_yticks(range(N_LAYERS))
        ax.set_yticklabels([f"L{l}" for l in range(N_LAYERS)], fontsize=7)
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")
        n = results["n_samples"][cond]
        acc_pct = results["accuracy"][cond] * 100
        ax.set_title(f"{LABELS[cond]}\n(n={n}, acc={acc_pct:.1f}%)")

    plt.colorbar(im, ax=axes, shrink=0.8, label="Mean FAI")
    plt.suptitle("Attention Head FAI by Suffix Condition", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "suffix_fai_heatmaps.png", dpi=150, bbox_inches="tight")
    print("Saved suffix_fai_heatmaps.png")
    plt.close()

    # ================================================================
    # Plot 2: Difference heatmaps — (step - no_suffix) and (random - no_suffix)
    # ================================================================
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    no_fai = results["attn_fai"]["no_suffix"].numpy()
    no_norm = results["attn_norm"]["no_suffix"].numpy()

    diffs = [
        ("step_suffix", "Step - No suffix"),
        ("random_suffix", "Random - No suffix"),
    ]
    for row, (cond, label) in enumerate(diffs):
        # FAI difference
        ax = axes[row, 0]
        diff = results["attn_fai"][cond].numpy() - no_fai
        vabs = max(abs(diff.min()), abs(diff.max())) or 1e-6
        im = ax.imshow(diff, cmap="RdBu_r", aspect="auto", vmin=-vabs, vmax=vabs)
        ax.set_xticks(range(N_HEADS))
        ax.set_xticklabels([f"H{h}" for h in range(N_HEADS)], fontsize=7)
        ax.set_yticks(range(N_LAYERS))
        ax.set_yticklabels([f"L{l}" for l in range(N_LAYERS)], fontsize=7)
        ax.set_title(f"FAI: {label}")
        plt.colorbar(im, ax=ax, shrink=0.8)

        # Norm difference
        ax = axes[row, 1]
        diff = results["attn_norm"][cond].numpy() - no_norm
        vabs = max(abs(diff.min()), abs(diff.max())) or 1e-6
        im = ax.imshow(diff, cmap="RdBu_r", aspect="auto", vmin=-vabs, vmax=vabs)
        ax.set_xticks(range(N_HEADS))
        ax.set_xticklabels([f"H{h}" for h in range(N_HEADS)], fontsize=7)
        ax.set_yticks(range(N_LAYERS))
        ax.set_yticklabels([f"L{l}" for l in range(N_LAYERS)], fontsize=7)
        ax.set_title(f"Attn Output Norm: {label}")
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle("Activation Differences Relative to No-Suffix Baseline\n(red = higher with suffix, blue = lower)",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "suffix_difference_heatmaps.png", dpi=150, bbox_inches="tight")
    print("Saved suffix_difference_heatmaps.png")
    plt.close()

    # ================================================================
    # Plot 3: MLP norm per layer, 3 conditions overlaid
    # ================================================================
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    ax = axes[0]
    for cond in SUFFIX_CONDITIONS:
        vals = results["mlp_norm"][cond].numpy()
        ax.plot(x, vals, "o-", color=COLORS[cond], label=LABELS[cond], markersize=4, linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{l}" for l in range(N_LAYERS)], fontsize=7, rotation=45)
    ax.set_ylabel("MLP output norm")
    ax.set_title("MLP Output Norm Per Layer")
    ax.legend(fontsize=9)

    ax = axes[1]
    for cond in SUFFIX_CONDITIONS:
        vals = results["mlp_gate"][cond].numpy()
        ax.plot(x, vals, "o-", color=COLORS[cond], label=LABELS[cond], markersize=4, linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{l}" for l in range(N_LAYERS)], fontsize=7, rotation=45)
    ax.set_ylabel("Fraction of active gate neurons")
    ax.set_title("MLP Gate Activity Per Layer")
    ax.legend(fontsize=9)

    plt.suptitle("MLP Activations by Suffix Condition", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "suffix_mlp_comparison.png", dpi=150, bbox_inches="tight")
    print("Saved suffix_mlp_comparison.png")
    plt.close()

    # ================================================================
    # Plot 4: Residual norm growth per layer, 3 conditions overlaid
    # ================================================================
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    ax = axes[0]
    for cond in SUFFIX_CONDITIONS:
        vals = results["resid_attn_growth"][cond].numpy()
        ax.plot(x, vals, "o-", color=COLORS[cond], label=LABELS[cond], markersize=4, linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{l}" for l in range(N_LAYERS)], fontsize=7, rotation=45)
    ax.set_ylabel("Residual norm growth from attention")
    ax.set_title("Attention Contribution to Residual Stream")
    ax.legend(fontsize=9)
    ax.axhline(0, color="black", linewidth=0.5)

    ax = axes[1]
    for cond in SUFFIX_CONDITIONS:
        vals = results["resid_mlp_growth"][cond].numpy()
        ax.plot(x, vals, "o-", color=COLORS[cond], label=LABELS[cond], markersize=4, linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{l}" for l in range(N_LAYERS)], fontsize=7, rotation=45)
    ax.set_ylabel("Residual norm growth from MLP")
    ax.set_title("MLP Contribution to Residual Stream")
    ax.legend(fontsize=9)
    ax.axhline(0, color="black", linewidth=0.5)

    plt.suptitle("Residual Stream Norm Growth by Suffix Condition", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "suffix_residual_growth.png", dpi=150, bbox_inches="tight")
    print("Saved suffix_residual_growth.png")
    plt.close()

    # ================================================================
    # Plot 5: Top 50 most affected components ranked by max divergence
    # ================================================================
    fig, ax = plt.subplots(figsize=(14, 12))

    components = []
    # Attention heads
    for l in range(N_LAYERS):
        for h in range(N_HEADS):
            vals = [results["attn_fai"][c][l, h].item() for c in SUFFIX_CONDITIONS]
            max_div = max(vals) - min(vals)
            components.append({
                "name": f"Attn L{l}H{h}",
                "type": "attention",
                "max_div": max_div,
                "no_suffix": vals[0],
                "step_suffix": vals[1],
                "random_suffix": vals[2],
            })
    # MLP layers
    for l in range(N_LAYERS):
        vals = [results["mlp_norm"][c][l].item() for c in SUFFIX_CONDITIONS]
        max_div = max(vals) - min(vals)
        components.append({
            "name": f"MLP L{l}",
            "type": "mlp",
            "max_div": max_div,
            "no_suffix": vals[0],
            "step_suffix": vals[1],
            "random_suffix": vals[2],
        })

    components.sort(key=lambda c: c["max_div"], reverse=True)
    top_n = 50
    top = components[:top_n]

    y_pos = np.arange(top_n)
    bar_h = 0.25
    for i, cond in enumerate(SUFFIX_CONDITIONS):
        vals = [c[cond] for c in top]
        ax.barh(y_pos + i * bar_h - bar_h, vals, bar_h,
                color=COLORS[cond], label=LABELS[cond], alpha=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([c["name"] for c in top], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Activation value")
    ax.set_title(f"Top {top_n} Components with Largest Cross-Suffix Divergence")
    ax.legend(fontsize=9, loc="lower right")

    n_attn = sum(1 for c in top if c["type"] == "attention")
    n_mlp = sum(1 for c in top if c["type"] == "mlp")
    ax.text(0.02, 0.02, f"Top {top_n}: {n_attn} attention heads, {n_mlp} MLP layers",
            transform=ax.transAxes, fontsize=10, verticalalignment="bottom",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "suffix_top_divergent_components.png", dpi=150, bbox_inches="tight")
    print("Saved suffix_top_divergent_components.png")
    plt.close()

    # ================================================================
    # Plot 6: Accuracy bar chart per suffix
    # ================================================================
    fig, ax = plt.subplots(figsize=(8, 5))

    conds = SUFFIX_CONDITIONS
    acc_vals = [results["accuracy"][c] * 100 for c in conds]
    bars = ax.bar(range(len(conds)), acc_vals, color=[COLORS[c] for c in conds], edgecolor="white", linewidth=1.5)

    for bar, val, cond in zip(bars, acc_vals, conds):
        n = results["n_samples"][cond]
        nc = results["n_correct"][cond]
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%\n({nc}/{n})", ha="center", va="bottom", fontsize=11)

    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([LABELS[c] for c in conds], fontsize=11)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Base Model Accuracy by Suffix Condition")
    ax.set_ylim(0, max(acc_vals) * 1.2 + 5)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "suffix_accuracy.png", dpi=150, bbox_inches="tight")
    print("Saved suffix_accuracy.png")
    plt.close()

    return components


def print_summary(results: dict, components: list):
    """Print summary of the analysis."""
    print(f"\n{'='*70}")
    print("SUFFIX ACTIVATION COMPARISON SUMMARY")
    print(f"{'='*70}")

    print(f"\n--- Sample Counts & Accuracy ---")
    for cond in SUFFIX_CONDITIONS:
        n = results["n_samples"][cond]
        nc = results["n_correct"][cond]
        acc = results["accuracy"][cond] * 100
        print(f"  {cond:<15} n={n}, correct={nc}, accuracy={acc:.1f}%")

    print(f"\n--- Top 30 Most Suffix-Sensitive Components ---")
    print(f"{'Rank':<6} {'Component':<18} {'Type':<10} {'MaxDiv':>10} "
          f"{'no_sfx':>10} {'step_sfx':>10} {'rand_sfx':>10}")
    print("-" * 80)
    for i, c in enumerate(components[:30]):
        print(f"  #{i+1:<4} {c['name']:<18} {c['type']:<10} {c['max_div']:>10.6f} "
              f"{c['no_suffix']:>10.6f} {c['step_suffix']:>10.6f} {c['random_suffix']:>10.6f}")

    # Layer-level summary
    print(f"\n--- Per-Layer MLP Norm (mean across conditions) ---")
    for l in range(N_LAYERS):
        vals = {c: results["mlp_norm"][c][l].item() for c in SUFFIX_CONDITIONS}
        spread = max(vals.values()) - min(vals.values())
        if spread > 0.01:  # only show layers with notable spread
            print(f"  L{l:02d}: " + ", ".join(f"{c}={v:.4f}" for c, v in vals.items()) + f" (spread={spread:.4f})")


def main():
    n_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {n_gpus}")

    df = pd.read_parquet(DATA_PATH)
    random.seed(42)
    indices = random.sample(range(len(df)), min(N_EXAMPLES, len(df)))

    if n_gpus <= 1:
        print("Running on single GPU...")
        return_dict = {}
        analyze_on_gpu(0, indices, return_dict)
        merged = return_dict[0]
    else:
        chunks = [[] for _ in range(n_gpus)]
        for i, idx in enumerate(indices):
            chunks[i % n_gpus].append(idx)

        print(f"Splitting {len(indices)} problems across {n_gpus} GPUs: "
              + ", ".join(f"GPU{i}={len(c)}" for i, c in enumerate(chunks)))

        manager = mp.Manager()
        return_dict = manager.dict()

        processes = []
        for gpu_id in range(n_gpus):
            p = mp.Process(target=analyze_on_gpu, args=(gpu_id, chunks[gpu_id], return_dict))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        for gpu_id in range(n_gpus):
            if gpu_id not in return_dict:
                print(f"WARNING: GPU {gpu_id} did not produce results!")

        merged = merge_results(dict(return_dict))

    # Save raw data
    torch.save(merged, SAVE_PATH)
    print(f"\nSaved raw data to {SAVE_PATH}")

    # Compute means
    results = compute_means(merged)

    # Create plots
    components = create_plots(results)

    # Print summary
    print_summary(results, components)

    # Save processed results
    processed_path = BASE_DIR / "data" / "suffix_activation_results.pt"
    torch.save({
        "results": results,
        "components": components,
        "n_examples": N_EXAMPLES,
        "n_samples": N_SAMPLES,
    }, processed_path)
    print(f"\nSaved processed results to {processed_path}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
