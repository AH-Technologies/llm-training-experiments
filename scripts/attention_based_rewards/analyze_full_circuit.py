#!/usr/bin/env python3
"""Full circuit analysis: attention heads + MLP layers + residual stream.

Measures all component contributions on correct vs incorrect rollouts.
Uses 4 GPUs with data parallelism for fast processing.

Components measured per response:
  1. Attention heads (28 layers × 12 heads): FAI score per head
  2. MLP layers (28 layers): output activation norm on response tokens
  3. Residual stream (28 layers): norm growth after attn and after MLP
  4. Layer-level: total contribution (attn + MLP) per layer

For each component, we track mean activation on correct vs incorrect
responses, then compute divergence to identify reasoning-critical components.

Usage (4 GPUs, ~20-30 min):
  sbatch scripts/attention_based_rewards/slurm/analyze_full_circuit.slurm
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
SAVE_PATH = BASE_DIR / "data" / "full_circuit_analysis.pt"
PLOT_DIR = BASE_DIR / "plots"

N_EXAMPLES = 200      # more problems for better statistics
N_SAMPLES = 4         # responses per problem
MAX_NEW_TOKENS = 512
N_LAYERS = 28
N_HEADS = 12
HEAD_DIM = 128        # 1536 / 12


def check_answer(response: str, ground_truth: str) -> bool:
    """Simple answer extraction and comparison."""
    import re
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", response, re.DOTALL)
    if match:
        extracted = match.group(1).strip()
    else:
        # boxed format
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
class CircuitAccumulators:
    """Stores running sums for all circuit components."""
    # Attention heads: FAI per head
    attn_fai_correct: torch.Tensor = field(default_factory=lambda: torch.zeros(N_LAYERS, N_HEADS))
    attn_fai_incorrect: torch.Tensor = field(default_factory=lambda: torch.zeros(N_LAYERS, N_HEADS))
    # Attention heads: output norm per head
    attn_norm_correct: torch.Tensor = field(default_factory=lambda: torch.zeros(N_LAYERS, N_HEADS))
    attn_norm_incorrect: torch.Tensor = field(default_factory=lambda: torch.zeros(N_LAYERS, N_HEADS))
    # MLP: output activation norm per layer
    mlp_norm_correct: torch.Tensor = field(default_factory=lambda: torch.zeros(N_LAYERS))
    mlp_norm_incorrect: torch.Tensor = field(default_factory=lambda: torch.zeros(N_LAYERS))
    # MLP: gating activation (how "active" the gating is)
    mlp_gate_correct: torch.Tensor = field(default_factory=lambda: torch.zeros(N_LAYERS))
    mlp_gate_incorrect: torch.Tensor = field(default_factory=lambda: torch.zeros(N_LAYERS))
    # Residual stream: norm after attn, after mlp
    resid_post_attn_correct: torch.Tensor = field(default_factory=lambda: torch.zeros(N_LAYERS))
    resid_post_attn_incorrect: torch.Tensor = field(default_factory=lambda: torch.zeros(N_LAYERS))
    resid_post_mlp_correct: torch.Tensor = field(default_factory=lambda: torch.zeros(N_LAYERS))
    resid_post_mlp_incorrect: torch.Tensor = field(default_factory=lambda: torch.zeros(N_LAYERS))
    # Residual stream: norm *growth* from attn and mlp contributions
    resid_attn_growth_correct: torch.Tensor = field(default_factory=lambda: torch.zeros(N_LAYERS))
    resid_attn_growth_incorrect: torch.Tensor = field(default_factory=lambda: torch.zeros(N_LAYERS))
    resid_mlp_growth_correct: torch.Tensor = field(default_factory=lambda: torch.zeros(N_LAYERS))
    resid_mlp_growth_incorrect: torch.Tensor = field(default_factory=lambda: torch.zeros(N_LAYERS))
    # Counts
    n_correct: int = 0
    n_incorrect: int = 0


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
    acc = CircuitAccumulators()

    # Storage for hook captures
    hook_data = {}

    def make_attn_output_hook(layer_idx):
        """Capture attention output before residual add."""
        def hook_fn(module, input, output):
            # output is (hidden_states, attn_weights) or just hidden_states
            if isinstance(output, tuple):
                hook_data[f"attn_out_{layer_idx}"] = output[0].detach()
            else:
                hook_data[f"attn_out_{layer_idx}"] = output.detach()
        return hook_fn

    def make_mlp_output_hook(layer_idx):
        """Capture MLP output before residual add."""
        def hook_fn(module, input, output):
            hook_data[f"mlp_out_{layer_idx}"] = output.detach()
        return hook_fn

    def make_resid_pre_hook(layer_idx):
        """Capture input to each decoder layer (residual stream before layer)."""
        def hook_fn(module, input):
            if isinstance(input, tuple):
                hook_data[f"resid_pre_{layer_idx}"] = input[0].detach()
            else:
                hook_data[f"resid_pre_{layer_idx}"] = input.detach()
        return hook_fn

    def make_mlp_gate_hook(layer_idx):
        """Capture MLP gate activations (measures gating sparsity)."""
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
        # Gate activation (gate_proj in Qwen2MLP)
        if hasattr(layer.mlp, 'gate_proj'):
            hooks.append(layer.mlp.gate_proj.register_forward_hook(make_mlp_gate_hook(layer_idx)))

    t0 = time.time()
    for idx_i, data_idx in enumerate(problem_indices):
        row = df.iloc[data_idx]
        prompt_msgs = row["prompt"]
        gt = row["reward_model"]["ground_truth"]

        # Build prompt
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            prompt_text = tokenizer.apply_chat_template(
                prompt_msgs, tokenize=False, add_generation_prompt=True
            )
        else:
            for msg in prompt_msgs:
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

            attentions = out.attentions  # tuple of (1, n_heads, seq_len, seq_len)

            # FAI computation
            future_mask = torch.tril(torch.ones(seq_len, seq_len, device=device), diagonal=-1)
            future_count = future_mask.sum(dim=0).clamp(min=1)

            for layer_idx in range(N_LAYERS):
                attn = attentions[layer_idx][0]  # (n_heads, seq_len, seq_len)

                # --- Attention heads: FAI ---
                for head_idx in range(N_HEADS):
                    a = attn[head_idx]
                    future_attn = a * future_mask
                    received = future_attn.sum(dim=0) / future_count
                    fai_val = received[prompt_len:].mean().item()

                    if correct:
                        acc.attn_fai_correct[layer_idx, head_idx] += fai_val
                    else:
                        acc.attn_fai_incorrect[layer_idx, head_idx] += fai_val

                # --- Attention head output norms (per head) ---
                attn_out = hook_data.get(f"attn_out_{layer_idx}")
                if attn_out is not None:
                    # attn_out: (1, seq_len, hidden_size)
                    # Reshape to per-head: (1, seq_len, n_heads, head_dim)
                    ao = attn_out[0, prompt_len:, :]  # (resp_len, hidden)
                    ao_heads = ao.reshape(ao.shape[0], N_HEADS, HEAD_DIM)  # (resp_len, n_heads, head_dim)
                    head_norms = ao_heads.float().norm(dim=-1).mean(dim=0)  # (n_heads,)
                    if correct:
                        acc.attn_norm_correct[layer_idx] += head_norms.cpu()
                    else:
                        acc.attn_norm_incorrect[layer_idx] += head_norms.cpu()

                # --- MLP output norm ---
                mlp_out = hook_data.get(f"mlp_out_{layer_idx}")
                if mlp_out is not None:
                    mlp_resp = mlp_out[0, prompt_len:, :]  # (resp_len, hidden)
                    mlp_norm = mlp_resp.float().norm(dim=-1).mean().item()
                    if correct:
                        acc.mlp_norm_correct[layer_idx] += mlp_norm
                    else:
                        acc.mlp_norm_incorrect[layer_idx] += mlp_norm

                # --- MLP gate activation ---
                gate_out = hook_data.get(f"mlp_gate_{layer_idx}")
                if gate_out is not None:
                    gate_resp = gate_out[0, prompt_len:, :]
                    # Fraction of gate neurons that are "active" (> 0 after SiLU)
                    gate_active = (gate_resp > 0).float().mean().item()
                    if correct:
                        acc.mlp_gate_correct[layer_idx] += gate_active
                    else:
                        acc.mlp_gate_incorrect[layer_idx] += gate_active

                # --- Residual stream norms ---
                resid_pre = hook_data.get(f"resid_pre_{layer_idx}")
                if resid_pre is not None and attn_out is not None and mlp_out is not None:
                    rp = resid_pre[0, prompt_len:, :].float()  # (resp_len, hidden)
                    ao = attn_out[0, prompt_len:, :].float()
                    mo = mlp_out[0, prompt_len:, :].float()

                    resid_post_attn = rp + ao
                    resid_post_mlp = resid_post_attn + mo

                    rp_norm = rp.norm(dim=-1).mean().item()
                    rpa_norm = resid_post_attn.norm(dim=-1).mean().item()
                    rpm_norm = resid_post_mlp.norm(dim=-1).mean().item()

                    attn_growth = rpa_norm - rp_norm
                    mlp_growth = rpm_norm - rpa_norm

                    if correct:
                        acc.resid_post_attn_correct[layer_idx] += rpa_norm
                        acc.resid_post_mlp_correct[layer_idx] += rpm_norm
                        acc.resid_attn_growth_correct[layer_idx] += attn_growth
                        acc.resid_mlp_growth_correct[layer_idx] += mlp_growth
                    else:
                        acc.resid_post_attn_incorrect[layer_idx] += rpa_norm
                        acc.resid_post_mlp_incorrect[layer_idx] += rpm_norm
                        acc.resid_attn_growth_incorrect[layer_idx] += attn_growth
                        acc.resid_mlp_growth_incorrect[layer_idx] += mlp_growth

            if correct:
                acc.n_correct += 1
            else:
                acc.n_incorrect += 1

            del out, attentions
            hook_data.clear()
            torch.cuda.empty_cache()

        if (idx_i + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = (idx_i + 1) / elapsed * 60
            print(f"  [GPU {gpu_id}] [{idx_i+1}/{len(problem_indices)}] "
                  f"correct={acc.n_correct}, incorrect={acc.n_incorrect}, "
                  f"rate={rate:.1f} problems/min", flush=True)

    # Clean up hooks
    for h in hooks:
        h.remove()

    # Store results — convert to CPU tensors
    result = {
        "attn_fai_correct": acc.attn_fai_correct,
        "attn_fai_incorrect": acc.attn_fai_incorrect,
        "attn_norm_correct": acc.attn_norm_correct,
        "attn_norm_incorrect": acc.attn_norm_incorrect,
        "mlp_norm_correct": acc.mlp_norm_correct,
        "mlp_norm_incorrect": acc.mlp_norm_incorrect,
        "mlp_gate_correct": acc.mlp_gate_correct,
        "mlp_gate_incorrect": acc.mlp_gate_incorrect,
        "resid_post_attn_correct": acc.resid_post_attn_correct,
        "resid_post_attn_incorrect": acc.resid_post_attn_incorrect,
        "resid_post_mlp_correct": acc.resid_post_mlp_correct,
        "resid_post_mlp_incorrect": acc.resid_post_mlp_incorrect,
        "resid_attn_growth_correct": acc.resid_attn_growth_correct,
        "resid_attn_growth_incorrect": acc.resid_attn_growth_incorrect,
        "resid_mlp_growth_correct": acc.resid_mlp_growth_correct,
        "resid_mlp_growth_incorrect": acc.resid_mlp_growth_incorrect,
        "n_correct": acc.n_correct,
        "n_incorrect": acc.n_incorrect,
    }
    return_dict[gpu_id] = result
    print(f"[GPU {gpu_id}] Done. {acc.n_correct} correct, {acc.n_incorrect} incorrect.", flush=True)


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


def compute_divergences(data: dict) -> dict:
    """Compute divergences and rankings from merged data."""
    nc = data["n_correct"]
    ni = data["n_incorrect"]
    print(f"\nTotal: {nc} correct, {ni} incorrect")

    def norm(tensor_c, tensor_i, n_c, n_i):
        """Normalize by count and compute absolute divergence."""
        c = tensor_c / max(n_c, 1)
        i = tensor_i / max(n_i, 1)
        return c, i, (c - i).abs(), c - i  # mean_c, mean_i, abs_div, signed_div

    results = {}

    # Attention FAI
    c, i, ad, sd = norm(data["attn_fai_correct"], data["attn_fai_incorrect"], nc, ni)
    results["attn_fai"] = {"correct": c, "incorrect": i, "abs_div": ad, "signed_div": sd}

    # Attention norm
    c, i, ad, sd = norm(data["attn_norm_correct"], data["attn_norm_incorrect"], nc, ni)
    results["attn_norm"] = {"correct": c, "incorrect": i, "abs_div": ad, "signed_div": sd}

    # MLP norm
    c, i, ad, sd = norm(data["mlp_norm_correct"], data["mlp_norm_incorrect"], nc, ni)
    results["mlp_norm"] = {"correct": c, "incorrect": i, "abs_div": ad, "signed_div": sd}

    # MLP gate activity
    c, i, ad, sd = norm(data["mlp_gate_correct"], data["mlp_gate_incorrect"], nc, ni)
    results["mlp_gate"] = {"correct": c, "incorrect": i, "abs_div": ad, "signed_div": sd}

    # Residual norm growth (attn contribution)
    c, i, ad, sd = norm(data["resid_attn_growth_correct"], data["resid_attn_growth_incorrect"], nc, ni)
    results["resid_attn_growth"] = {"correct": c, "incorrect": i, "abs_div": ad, "signed_div": sd}

    # Residual norm growth (MLP contribution)
    c, i, ad, sd = norm(data["resid_mlp_growth_correct"], data["resid_mlp_growth_incorrect"], nc, ni)
    results["resid_mlp_growth"] = {"correct": c, "incorrect": i, "abs_div": ad, "signed_div": sd}

    # Residual post-MLP (full layer output norm)
    c, i, ad, sd = norm(data["resid_post_mlp_correct"], data["resid_post_mlp_incorrect"], nc, ni)
    results["resid_post_mlp"] = {"correct": c, "incorrect": i, "abs_div": ad, "signed_div": sd}

    return results


def create_plots(results: dict, data: dict):
    """Create comprehensive visualizations."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    nc, ni = data["n_correct"], data["n_incorrect"]

    # Load EAP-IG for comparison
    eapig_path = BASE_DIR / "data" / "base_model_reasoning_heads.pt"
    eapig_top20 = set()
    if eapig_path.exists():
        eapig = torch.load(eapig_path, map_location="cpu", weights_only=False)
        eapig_top20 = {(l, h) for l, h, _ in eapig["selected_heads"][:20]}

    # ================================================================
    # Figure 1: Full Circuit Overview — Layer-by-layer component breakdown
    # ================================================================
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    # Panel A: Attention vs MLP contribution per layer (correct vs incorrect)
    ax = axes[0, 0]
    x = np.arange(N_LAYERS)
    attn_grow_c = results["resid_attn_growth"]["correct"].numpy()
    attn_grow_i = results["resid_attn_growth"]["incorrect"].numpy()
    mlp_grow_c = results["resid_mlp_growth"]["correct"].numpy()
    mlp_grow_i = results["resid_mlp_growth"]["incorrect"].numpy()

    w = 0.2
    ax.bar(x - 1.5*w, attn_grow_c, w, label="Attn (correct)", color="#2ecc71", alpha=0.8)
    ax.bar(x - 0.5*w, attn_grow_i, w, label="Attn (incorrect)", color="#27ae60", alpha=0.5)
    ax.bar(x + 0.5*w, mlp_grow_c, w, label="MLP (correct)", color="#e74c3c", alpha=0.8)
    ax.bar(x + 1.5*w, mlp_grow_i, w, label="MLP (incorrect)", color="#c0392b", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{l}" for l in range(N_LAYERS)], fontsize=7, rotation=45)
    ax.set_ylabel("Residual norm growth")
    ax.set_title("A) Per-Layer Contribution: Attention vs MLP")
    ax.legend(fontsize=8, ncol=2)
    ax.axhline(0, color="black", linewidth=0.5)

    # Panel B: MLP output norm — correct vs incorrect
    ax = axes[0, 1]
    mlp_c = results["mlp_norm"]["correct"].numpy()
    mlp_i = results["mlp_norm"]["incorrect"].numpy()
    ax.bar(x - 0.2, mlp_c, 0.4, label="Correct", color="#3498db", alpha=0.8)
    ax.bar(x + 0.2, mlp_i, 0.4, label="Incorrect", color="#e74c3c", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{l}" for l in range(N_LAYERS)], fontsize=7, rotation=45)
    ax.set_ylabel("MLP output norm (mean over response)")
    ax.set_title("B) MLP Activation Norm Per Layer")
    ax.legend(fontsize=9)

    # Panel C: MLP gate activity — fraction of active neurons
    ax = axes[1, 0]
    gate_c = results["mlp_gate"]["correct"].numpy()
    gate_i = results["mlp_gate"]["incorrect"].numpy()
    ax.plot(x, gate_c, "o-", color="#2ecc71", label="Correct", markersize=5)
    ax.plot(x, gate_i, "s--", color="#e74c3c", label="Incorrect", markersize=5)
    ax.fill_between(x, gate_c, gate_i, alpha=0.15, color="purple",
                    where=gate_c > gate_i, interpolate=True)
    ax.fill_between(x, gate_c, gate_i, alpha=0.15, color="orange",
                    where=gate_c <= gate_i, interpolate=True)
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{l}" for l in range(N_LAYERS)], fontsize=7, rotation=45)
    ax.set_ylabel("Fraction of active gate neurons")
    ax.set_title("C) MLP Gating Activity (correct vs incorrect)")
    ax.legend(fontsize=9)

    # Panel D: Residual stream norm growth through the network
    ax = axes[1, 1]
    resid_c = results["resid_post_mlp"]["correct"].numpy()
    resid_i = results["resid_post_mlp"]["incorrect"].numpy()
    ax.plot(x, resid_c, "o-", color="#2ecc71", label="Correct", linewidth=2, markersize=5)
    ax.plot(x, resid_i, "s--", color="#e74c3c", label="Incorrect", linewidth=2, markersize=5)
    ax.fill_between(x, resid_c, resid_i, alpha=0.15, color="gray")
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{l}" for l in range(N_LAYERS)], fontsize=7, rotation=45)
    ax.set_ylabel("Residual stream norm")
    ax.set_title("D) Residual Stream Norm Through Network")
    ax.legend(fontsize=9)

    plt.suptitle(f"Full Circuit Analysis: Correct vs Incorrect Rollouts\n"
                 f"(n_correct={nc}, n_incorrect={ni}, {N_EXAMPLES} problems × {N_SAMPLES} samples)",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "full_circuit_overview.png", dpi=150, bbox_inches="tight")
    print("Saved full_circuit_overview.png")
    plt.close()

    # ================================================================
    # Figure 2: Divergence Heatmap — which components differ most
    # ================================================================
    fig, axes = plt.subplots(1, 3, figsize=(22, 8))

    # Panel A: Attention head FAI divergence heatmap
    ax = axes[0]
    fai_sd = results["attn_fai"]["signed_div"].numpy()
    vmax = max(abs(fai_sd.min()), abs(fai_sd.max()))
    im = ax.imshow(fai_sd, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(N_HEADS))
    ax.set_xticklabels([f"H{h}" for h in range(N_HEADS)], fontsize=8)
    ax.set_yticks(range(N_LAYERS))
    ax.set_yticklabels([f"L{l}" for l in range(N_LAYERS)], fontsize=8)
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_title("Attention Head FAI\n(red = more active on correct)")
    plt.colorbar(im, ax=ax, shrink=0.8, label="FAI_correct - FAI_incorrect")
    # Mark EAP-IG heads
    for l, h in eapig_top20:
        rect = plt.Rectangle((h-0.5, l-0.5), 1, 1, linewidth=2,
                              edgecolor="lime", facecolor="none")
        ax.add_patch(rect)

    # Panel B: Attention head output norm divergence
    ax = axes[1]
    norm_sd = results["attn_norm"]["signed_div"].numpy()
    vmax = max(abs(norm_sd.min()), abs(norm_sd.max()))
    im = ax.imshow(norm_sd, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(N_HEADS))
    ax.set_xticklabels([f"H{h}" for h in range(N_HEADS)], fontsize=8)
    ax.set_yticks(range(N_LAYERS))
    ax.set_yticklabels([f"L{l}" for l in range(N_LAYERS)], fontsize=8)
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_title("Attention Head Output Norm\n(red = larger norm on correct)")
    plt.colorbar(im, ax=ax, shrink=0.8, label="norm_correct - norm_incorrect")
    for l, h in eapig_top20:
        rect = plt.Rectangle((h-0.5, l-0.5), 1, 1, linewidth=2,
                              edgecolor="lime", facecolor="none")
        ax.add_patch(rect)

    # Panel C: Layer-level divergence bar chart (all components)
    ax = axes[2]
    mlp_div = results["mlp_norm"]["signed_div"].numpy()
    attn_fai_layer_div = results["attn_fai"]["signed_div"].numpy().mean(axis=1)  # avg across heads
    attn_norm_layer_div = results["attn_norm"]["signed_div"].numpy().mean(axis=1)
    gate_div = results["mlp_gate"]["signed_div"].numpy()

    y = np.arange(N_LAYERS)
    h = 0.2
    ax.barh(y - 1.5*h, attn_fai_layer_div, h, label="Attn FAI", color="#3498db", alpha=0.8)
    ax.barh(y - 0.5*h, attn_norm_layer_div, h, label="Attn Norm", color="#2980b9", alpha=0.8)
    ax.barh(y + 0.5*h, mlp_div, h, label="MLP Norm", color="#e74c3c", alpha=0.8)
    ax.barh(y + 1.5*h, gate_div, h, label="MLP Gate", color="#c0392b", alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([f"L{l}" for l in range(N_LAYERS)], fontsize=8)
    ax.set_xlabel("Signed divergence (correct - incorrect)")
    ax.set_title("Per-Layer Component Divergence\n(positive = more active on correct)")
    ax.axvline(0, color="black", linewidth=0.5)
    ax.legend(fontsize=8, loc="best")
    ax.invert_yaxis()

    plt.suptitle("Circuit Divergence Map: What Differs Between Correct & Incorrect?",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "circuit_divergence_map.png", dpi=150, bbox_inches="tight")
    print("Saved circuit_divergence_map.png")
    plt.close()

    # ================================================================
    # Figure 3: Component Importance Ranking (all components unified)
    # ================================================================
    fig, ax = plt.subplots(figsize=(14, 10))

    # Build a unified ranking of all components by absolute divergence
    components = []

    # Attention heads (336 components)
    fai_ad = results["attn_fai"]["abs_div"]
    fai_sd_t = results["attn_fai"]["signed_div"]
    for l in range(N_LAYERS):
        for h in range(N_HEADS):
            components.append({
                "name": f"Attn L{l}H{h}",
                "type": "attention",
                "layer": l,
                "abs_div": fai_ad[l, h].item(),
                "signed_div": fai_sd_t[l, h].item(),
                "is_eapig": (l, h) in eapig_top20,
            })

    # MLP layers (28 components)
    mlp_ad = results["mlp_norm"]["abs_div"]
    mlp_sd_t = results["mlp_norm"]["signed_div"]
    for l in range(N_LAYERS):
        components.append({
            "name": f"MLP L{l}",
            "type": "mlp",
            "layer": l,
            "abs_div": mlp_ad[l].item(),
            "signed_div": mlp_sd_t[l].item(),
            "is_eapig": False,
        })

    # Sort by absolute divergence
    components.sort(key=lambda x: x["abs_div"], reverse=True)

    # Plot top 50
    top_n = 50
    top = components[:top_n]
    names = [c["name"] for c in top]
    divs = [c["signed_div"] for c in top]
    colors = []
    for c in top:
        if c["is_eapig"]:
            colors.append("#f1c40f")  # gold for EAP-IG
        elif c["type"] == "mlp":
            colors.append("#e74c3c")  # red for MLP
        else:
            colors.append("#3498db")  # blue for attention

    bars = ax.barh(range(top_n-1, -1, -1), divs, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(top_n-1, -1, -1))
    ax.set_yticklabels(names, fontsize=7)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Signed divergence (correct - incorrect)")
    ax.set_title(f"Top {top_n} Most Divergent Components (Attention Heads + MLP Layers)")

    legend_elements = [
        Patch(facecolor="#3498db", label="Attention head"),
        Patch(facecolor="#e74c3c", label="MLP layer"),
        Patch(facecolor="#f1c40f", label="EAP-IG top-20 head"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10)

    # Count types in top 50
    n_attn = sum(1 for c in top if c["type"] == "attention")
    n_mlp = sum(1 for c in top if c["type"] == "mlp")
    n_eapig_in_top = sum(1 for c in top if c["is_eapig"])
    ax.text(0.02, 0.02, f"Top {top_n}: {n_attn} attention heads, {n_mlp} MLP layers\n"
            f"EAP-IG top-20 in top {top_n}: {n_eapig_in_top}/20",
            transform=ax.transAxes, fontsize=10, verticalalignment="bottom",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "component_importance_ranking.png", dpi=150, bbox_inches="tight")
    print("Saved component_importance_ranking.png")
    plt.close()

    # ================================================================
    # Figure 4: Attention heads scatter — correct vs incorrect activation
    # ================================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Panel A: FAI scatter
    ax = axes[0]
    fai_c = results["attn_fai"]["correct"].numpy()
    fai_i = results["attn_fai"]["incorrect"].numpy()
    for l in range(N_LAYERS):
        for h in range(N_HEADS):
            if (l, h) in eapig_top20:
                ax.scatter(fai_i[l, h], fai_c[l, h], c="#f1c40f", s=80, zorder=10,
                           edgecolors="black", linewidths=0.8)
                ax.annotate(f"L{l}H{h}", (fai_i[l, h], fai_c[l, h]),
                            fontsize=6, ha="left", va="bottom",
                            xytext=(2, 2), textcoords="offset points")
            else:
                ax.scatter(fai_i[l, h], fai_c[l, h], c="#95a5a6", s=15, alpha=0.4)
    lims = [min(fai_c.min(), fai_i.min()), max(fai_c.max(), fai_i.max())]
    ax.plot(lims, lims, "k--", alpha=0.3)
    ax.set_xlabel("Mean FAI (incorrect)")
    ax.set_ylabel("Mean FAI (correct)")
    ax.set_title("Attention Head FAI: Correct vs Incorrect")
    legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#f1c40f", markersize=10, label="EAP-IG top-20"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#95a5a6", markersize=6, label="Other heads"),
    ]
    ax.legend(handles=legend, fontsize=9)

    # Panel B: MLP norm scatter
    ax = axes[1]
    mlp_c = results["mlp_norm"]["correct"].numpy()
    mlp_i = results["mlp_norm"]["incorrect"].numpy()
    for l in range(N_LAYERS):
        color = f"C{l % 10}"
        ax.scatter(mlp_i[l], mlp_c[l], c=color, s=80, zorder=5, edgecolors="black", linewidths=0.5)
        ax.annotate(f"L{l}", (mlp_i[l], mlp_c[l]), fontsize=8, ha="left",
                    xytext=(3, 3), textcoords="offset points")
    lims = [min(mlp_c.min(), mlp_i.min()), max(mlp_c.max(), mlp_i.max())]
    ax.plot(lims, lims, "k--", alpha=0.3)
    ax.set_xlabel("MLP norm (incorrect)")
    ax.set_ylabel("MLP norm (correct)")
    ax.set_title("MLP Output Norm: Correct vs Incorrect")

    plt.suptitle("Component Activation: Correct vs Incorrect Rollouts", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "circuit_scatter_correct_incorrect.png", dpi=150, bbox_inches="tight")
    print("Saved circuit_scatter_correct_incorrect.png")
    plt.close()

    # ================================================================
    # Figure 5: "Circuit Fingerprint" — radar/profile of the reasoning circuit
    # ================================================================
    fig, ax = plt.subplots(figsize=(16, 6))

    # For each layer, show stacked: attn contribution (mean across heads) + MLP contribution
    attn_div_signed = results["resid_attn_growth"]["signed_div"].numpy()
    mlp_div_signed = results["resid_mlp_growth"]["signed_div"].numpy()

    x = np.arange(N_LAYERS)
    # Positive = more on correct, negative = more on incorrect
    p_attn = np.maximum(attn_div_signed, 0)
    n_attn = np.minimum(attn_div_signed, 0)
    p_mlp = np.maximum(mlp_div_signed, 0)
    n_mlp = np.minimum(mlp_div_signed, 0)

    ax.bar(x, p_attn, 0.8, label="Attn growth (correct > incorrect)", color="#2ecc71", alpha=0.8)
    ax.bar(x, p_mlp, 0.8, bottom=p_attn, label="MLP growth (correct > incorrect)", color="#3498db", alpha=0.8)
    ax.bar(x, n_attn, 0.8, label="Attn growth (incorrect > correct)", color="#e74c3c", alpha=0.8)
    ax.bar(x, n_mlp, 0.8, bottom=n_attn, label="MLP growth (incorrect > correct)", color="#e67e22", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"L{l}" for l in range(N_LAYERS)], fontsize=8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Residual norm growth difference\n(correct - incorrect)")
    ax.set_xlabel("Layer")
    ax.set_title("Reasoning Circuit Fingerprint: Where Does the Model Think Harder on Correct Responses?")
    ax.legend(fontsize=9, ncol=2, loc="upper left")

    # Annotate key layers
    total_div = attn_div_signed + mlp_div_signed
    top_layers = np.argsort(np.abs(total_div))[-5:]
    for l in top_layers:
        ax.annotate(f"L{l}", (l, total_div[l]), fontsize=9, fontweight="bold",
                    ha="center", va="bottom" if total_div[l] > 0 else "top")

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "reasoning_circuit_fingerprint.png", dpi=150, bbox_inches="tight")
    print("Saved reasoning_circuit_fingerprint.png")
    plt.close()

    return components


def print_summary(components: list):
    """Print a text summary of the analysis."""
    print(f"\n{'='*70}")
    print("FULL CIRCUIT ANALYSIS SUMMARY")
    print(f"{'='*70}")

    print(f"\n--- Top 30 Most Divergent Components ---")
    print(f"{'Rank':<6} {'Component':<18} {'Type':<10} {'Abs Div':>10} {'Signed Div':>12} {'EAP-IG':>8}")
    print("-" * 70)
    for i, c in enumerate(components[:30]):
        eapig = "YES" if c["is_eapig"] else ""
        print(f"  #{i+1:<4} {c['name']:<18} {c['type']:<10} {c['abs_div']:>10.6f} "
              f"{c['signed_div']:>+12.6f} {eapig:>8}")

    # Count MLP vs attention in top-N
    for n in [20, 30, 50]:
        top = components[:n]
        n_attn = sum(1 for c in top if c["type"] == "attention")
        n_mlp = sum(1 for c in top if c["type"] == "mlp")
        n_eapig = sum(1 for c in top if c["is_eapig"])
        print(f"\nTop {n}: {n_attn} attention heads, {n_mlp} MLP layers, {n_eapig} EAP-IG heads")

    # Identify the "reasoning circuit" — components with large positive divergence
    print(f"\n--- Reasoning Circuit (components more active on CORRECT) ---")
    reasoning = [c for c in components if c["signed_div"] > 0]
    reasoning.sort(key=lambda x: x["signed_div"], reverse=True)
    for c in reasoning[:20]:
        eapig = " [EAP-IG]" if c["is_eapig"] else ""
        print(f"  {c['name']:<18} +{c['signed_div']:.6f}{eapig}")

    print(f"\n--- Anti-Reasoning (components more active on INCORRECT) ---")
    anti = [c for c in components if c["signed_div"] < 0]
    anti.sort(key=lambda x: x["signed_div"])
    for c in anti[:10]:
        eapig = " [EAP-IG]" if c["is_eapig"] else ""
        print(f"  {c['name']:<18} {c['signed_div']:.6f}{eapig}")


def main():
    n_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {n_gpus}")

    # Load data indices
    df = pd.read_parquet(DATA_PATH)
    random.seed(42)
    indices = random.sample(range(len(df)), min(N_EXAMPLES, len(df)))

    if n_gpus <= 1:
        # Single GPU fallback
        print("Running on single GPU...")
        return_dict = {}
        analyze_on_gpu(0, indices, return_dict)
        merged = return_dict[0]
    else:
        # Split problems across GPUs
        chunks = [[] for _ in range(n_gpus)]
        for i, idx in enumerate(indices):
            chunks[i % n_gpus].append(idx)

        print(f"Splitting {len(indices)} problems across {n_gpus} GPUs: "
              + ", ".join(f"GPU{i}={len(c)}" for i, c in enumerate(chunks)))

        # Use mp.spawn-compatible approach with shared dict
        manager = mp.Manager()
        return_dict = manager.dict()

        processes = []
        for gpu_id in range(n_gpus):
            p = mp.Process(target=analyze_on_gpu, args=(gpu_id, chunks[gpu_id], return_dict))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        # Check for failures
        for gpu_id in range(n_gpus):
            if gpu_id not in return_dict:
                print(f"WARNING: GPU {gpu_id} did not produce results!")

        merged = merge_results(dict(return_dict))

    # Save raw data
    torch.save(merged, SAVE_PATH)
    print(f"\nSaved raw data to {SAVE_PATH}")

    # Compute divergences
    results = compute_divergences(merged)

    # Create plots
    components = create_plots(results, merged)

    # Print summary
    print_summary(components)

    # Save processed results
    processed_path = BASE_DIR / "data" / "full_circuit_results.pt"
    torch.save({
        "results": results,
        "components": components,
        "n_correct": merged["n_correct"],
        "n_incorrect": merged["n_incorrect"],
        "n_examples": N_EXAMPLES,
        "n_samples": N_SAMPLES,
    }, processed_path)
    print(f"\nSaved processed results to {processed_path}")


if __name__ == "__main__":
    # Required for multiprocessing with CUDA
    mp.set_start_method("spawn", force=True)
    main()
