"""Analyze how GRPO training conditions change attention head patterns.

For each trained checkpoint vs the original model:
1. Load model into TransformerLens
2. Run 20 math problems, collect per-head activation norms
3. Compute per-head divergence from original model
4. Compare: Do FAI conditions change reasoning heads more/less than others?

Requires 1 GPU, ~15-20 min total.
"""

import json
import os
import random
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = "/cluster/projects/nn12068k/haaklau/llm-training-experiments/attention_based_rewards"
PROJECT_DIR = "/cluster/projects/nn12068k/haaklau/llm-training-experiments"

# These are Instruct model checkpoints
ORIGINAL_MODEL = "/cluster/projects/nn12068k/haaklau/.cache/huggingface/hub/models--Qwen--Qwen2.5-Math-1.5B-Instruct/snapshots/aafeb0fc6f22cbf0eaeed126eff8be45b0360a35"
TL_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

CHECKPOINTS = OrderedDict([
    ("entropy_v2", "checkpoints/dapo-grpo/dapo_entropy_v2/global_step_100"),
    ("fai_allheads_v2", "checkpoints/dapo-grpo/dapo_fai_allheads_v2/global_step_100"),
    ("fai_asymmetric_v2", "checkpoints/dapo-grpo/dapo_fai_asymmetric_v2/global_step_100"),
    ("uniform_v2", "checkpoints/dapo-grpo/dapo_uniform_v2/global_step_50"),
    ("fai_v2", "checkpoints/dapo-grpo/dapo_fai_v2/global_step_50"),
])

N_PROBLEMS = 20
SYSTEM_PROMPT = "Please reason step by step, and put your final answer within <answer> </answer> tags."


MERGED_BASE = Path(f"{PROJECT_DIR}/attention_based_rewards/merged_models_v2")


def merge_fsdp_checkpoint(name, ckpt_dir):
    """Merge FSDP sharded checkpoint using verl's model_merger.

    Returns path to merged HF model directory.
    """
    actor_dir = os.path.join(PROJECT_DIR, ckpt_dir, "actor")
    step = ckpt_dir.split("step_")[1]
    target = MERGED_BASE / name / f"step_{step}"

    if (target / "config.json").exists():
        print(f"  Already merged: {target}")
        return str(target)

    target.mkdir(parents=True, exist_ok=True)
    print(f"  Merging FSDP shards via verl.model_merger: {actor_dir} -> {target}")

    result = subprocess.run(
        [
            sys.executable, "-m", "verl.model_merger", "merge",
            "--backend", "fsdp",
            "--local_dir", str(actor_dir),
            "--target_dir", str(target),
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  Merge stderr: {result.stderr[-2000:]}")
        raise RuntimeError(f"Failed to merge {actor_dir}")

    print(f"  Merged successfully to {target}")
    return str(target)


def load_model_tl(model_path, device="cuda"):
    """Load a HF model directory into TransformerLens."""
    hf_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
    model = HookedTransformer.from_pretrained(
        TL_MODEL_NAME, hf_model=hf_model, device=device, dtype=torch.float16,
    )
    del hf_model
    model.cfg.use_attn_result = True
    model.setup()
    return model


def collect_head_activations(model, prompts, tokenizer):
    """Run prompts through model and collect per-head activation norms.

    Returns: (n_layers, n_heads) tensor of mean activation L2 norms.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    # Accumulate per-head activation norms
    head_norms = torch.zeros(n_layers, n_heads, device="cpu")
    total_tokens = 0

    for i, prompt in enumerate(prompts):
        tokens = model.to_tokens(prompt)
        seq_len = tokens.shape[1]
        total_tokens += seq_len

        # Run with hooks to capture attention head outputs
        cache = {}
        def make_hook(layer):
            def hook_fn(activation, hook):
                # activation shape: (batch, pos, n_heads, d_head)
                cache[layer] = activation.detach().cpu()
            return hook_fn

        hooks = [(f"blocks.{l}.attn.hook_result", make_hook(l)) for l in range(n_layers)]

        with torch.no_grad():
            with model.hooks(fwd_hooks=hooks):
                model(tokens)

        for l in range(n_layers):
            act = cache[l]  # (1, pos, n_heads, d_head)
            # Mean L2 norm per head across positions
            per_head_norm = act[0].float().norm(dim=-1).mean(dim=0)  # (n_heads,)
            head_norms[l] += per_head_norm

        if (i + 1) % 5 == 0:
            print(f"    Processed {i+1}/{len(prompts)} prompts", flush=True)

    head_norms /= len(prompts)
    return head_norms


def collect_head_attention_entropy(model, prompts):
    """Collect per-head attention entropy (how focused/diffuse attention is).

    Returns: (n_layers, n_heads) tensor of mean attention entropy.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    head_entropy = torch.zeros(n_layers, n_heads, device="cpu")

    for i, prompt in enumerate(prompts):
        tokens = model.to_tokens(prompt)

        cache = {}
        def make_hook(layer):
            def hook_fn(activation, hook):
                cache[layer] = activation.detach().cpu()
            return hook_fn

        hooks = [(f"blocks.{l}.attn.hook_pattern", make_hook(l)) for l in range(n_layers)]

        with torch.no_grad():
            with model.hooks(fwd_hooks=hooks):
                model(tokens)

        for l in range(n_layers):
            attn = cache[l]  # (1, n_heads, pos, pos)
            # Entropy of attention distribution at each position, averaged
            attn_clamped = attn[0].float().clamp(min=1e-10)
            entropy = -(attn_clamped * attn_clamped.log()).sum(dim=-1)  # (n_heads, pos)
            head_entropy[l] += entropy.mean(dim=-1)  # (n_heads,)

        if (i + 1) % 5 == 0:
            print(f"    Processed {i+1}/{len(prompts)} prompts", flush=True)

    head_entropy /= len(prompts)
    return head_entropy


def main():
    random.seed(42)
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load reasoning heads info
    heads_path = os.path.join(BASE_DIR, "results/reasoning_heads.pt")
    heads_data = torch.load(heads_path, weights_only=False)
    head_importance = heads_data["head_scores"]
    n_layers, n_heads = head_importance.shape
    flat = head_importance.flatten()
    sorted_idx = flat.argsort(descending=True)
    top10_heads = set()
    for idx in sorted_idx[:10]:
        l, h = idx.item() // n_heads, idx.item() % n_heads
        top10_heads.add((l, h))
    print(f"Top 10 reasoning heads (Instruct EAP-IG): {sorted(top10_heads)}")

    # Prepare prompts
    tokenizer = AutoTokenizer.from_pretrained(ORIGINAL_MODEL)
    print(f"Loading {N_PROBLEMS} DAPO-Math-17k problems...")
    ds = load_dataset("open-r1/DAPO-Math-17k-Processed", "en", split="train")
    indices = list(range(len(ds)))
    random.shuffle(indices)
    prompts = []
    for idx in indices[:N_PROBLEMS]:
        problem = ds[idx]["prompt"]
        # Chat format
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": problem},
        ]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt_text)
    del tokenizer

    # Collect activations from original model
    print(f"\n{'='*60}")
    print("Loading ORIGINAL model...")
    print(f"{'='*60}")
    model = load_model_tl(ORIGINAL_MODEL, device=device)
    print("Collecting head activation norms...")
    original_norms = collect_head_activations(model, prompts, None)
    print("Collecting head attention entropy...")
    original_entropy = collect_head_attention_entropy(model, prompts)
    del model
    torch.cuda.empty_cache()

    # Collect activations from each checkpoint
    results = {
        "original_norms": original_norms,
        "original_entropy": original_entropy,
        "checkpoints": {},
    }

    for name, ckpt_path in CHECKPOINTS.items():
        print(f"\n{'='*60}")
        print(f"Loading checkpoint: {name} ({ckpt_path})")
        print(f"{'='*60}")

        # Merge FSDP shards using verl
        merged_path = merge_fsdp_checkpoint(name, ckpt_path)
        model = load_model_tl(merged_path, device=device)

        print("Collecting head activation norms...")
        ckpt_norms = collect_head_activations(model, prompts, None)
        print("Collecting head attention entropy...")
        ckpt_entropy = collect_head_attention_entropy(model, prompts)
        del model
        torch.cuda.empty_cache()

        # Compute divergence from original
        norm_diff = (ckpt_norms - original_norms).abs()
        entropy_diff = (ckpt_entropy - original_entropy).abs()
        # Relative change
        norm_rel = norm_diff / (original_norms.abs() + 1e-8)

        results["checkpoints"][name] = {
            "norms": ckpt_norms,
            "entropy": ckpt_entropy,
            "norm_diff": norm_diff,
            "entropy_diff": entropy_diff,
            "norm_rel_change": norm_rel,
        }

        # Print summary for this checkpoint
        print(f"\n  Top 10 most changed heads (by relative norm change):")
        flat_rel = norm_rel.flatten()
        sorted_rel = flat_rel.argsort(descending=True)
        for rank, idx in enumerate(sorted_rel[:10]):
            l, h = idx.item() // n_heads, idx.item() % n_heads
            is_reasoning = "(REASONING)" if (l, h) in top10_heads else ""
            print(f"    #{rank+1}: L{l}H{h} = {flat_rel[idx]:.4f} (abs: {norm_diff[l,h]:.4f}) {is_reasoning}")

        # How much did reasoning heads vs non-reasoning heads change?
        reasoning_changes = []
        nonreasoning_changes = []
        for l in range(n_layers):
            for h in range(n_heads):
                if (l, h) in top10_heads:
                    reasoning_changes.append(norm_rel[l, h].item())
                else:
                    nonreasoning_changes.append(norm_rel[l, h].item())

        r_mean = sum(reasoning_changes) / len(reasoning_changes)
        nr_mean = sum(nonreasoning_changes) / len(nonreasoning_changes)
        print(f"\n  Mean relative change - Reasoning heads: {r_mean:.4f}")
        print(f"  Mean relative change - Other heads:     {nr_mean:.4f}")
        print(f"  Ratio (reasoning/other):                {r_mean/nr_mean:.2f}x")

    # ── Summary comparison across conditions ──────────────────────────
    print(f"\n{'='*70}")
    print("CROSS-CONDITION COMPARISON")
    print(f"{'='*70}")
    print(f"\n{'Condition':<25} {'Reasoning Δ':>12} {'Other Δ':>12} {'Ratio':>8} {'Step':>6}")
    print("-" * 70)
    for name, data in results["checkpoints"].items():
        reasoning_changes = []
        nonreasoning_changes = []
        for l in range(n_layers):
            for h in range(n_heads):
                val = data["norm_rel_change"][l, h].item()
                if (l, h) in top10_heads:
                    reasoning_changes.append(val)
                else:
                    nonreasoning_changes.append(val)
        r_mean = sum(reasoning_changes) / len(reasoning_changes)
        nr_mean = sum(nonreasoning_changes) / len(nonreasoning_changes)
        step = "100" if "step_100" in CHECKPOINTS[name] else "50"
        print(f"  {name:<23} {r_mean:>12.4f} {nr_mean:>12.4f} {r_mean/nr_mean:>8.2f}x {step:>6}")

    # ── Plots ─────────────────────────────────────────────────────────
    os.makedirs(f"{BASE_DIR}/plots", exist_ok=True)

    # Plot 1: Head change heatmaps for each condition
    n_ckpts = len(results["checkpoints"])
    fig, axes = plt.subplots(1, n_ckpts, figsize=(6 * n_ckpts, 8), squeeze=False)
    for i, (name, data) in enumerate(results["checkpoints"].items()):
        ax = axes[0, i]
        im = ax.imshow(data["norm_rel_change"].numpy(), cmap="Reds", aspect="auto")
        # Mark reasoning heads
        for (l, h) in top10_heads:
            ax.plot(h, l, 'b*', markersize=8)
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")
        step = "100" if "step_100" in CHECKPOINTS[name] else "50"
        ax.set_title(f"{name}\n(step {step})")
        plt.colorbar(im, ax=ax, shrink=0.6)
    fig.suptitle("Relative Head Activation Change from Original\n(blue stars = reasoning heads)", fontsize=14)
    plt.tight_layout()
    plot1_path = f"{BASE_DIR}/plots/head_divergence_heatmaps.png"
    plt.savefig(plot1_path, dpi=150)
    print(f"\nSaved {plot1_path}")

    # Plot 2: Bar chart comparing reasoning vs non-reasoning head changes
    fig, ax = plt.subplots(figsize=(10, 6))
    conditions = list(results["checkpoints"].keys())
    reasoning_means = []
    other_means = []
    for name, data in results["checkpoints"].items():
        r_vals = [data["norm_rel_change"][l, h].item() for l, h in top10_heads]
        o_vals = [data["norm_rel_change"][l, h].item()
                  for l in range(n_layers) for h in range(n_heads)
                  if (l, h) not in top10_heads]
        reasoning_means.append(sum(r_vals) / len(r_vals))
        other_means.append(sum(o_vals) / len(o_vals))

    x = np.arange(len(conditions))
    width = 0.35
    ax.bar(x - width/2, reasoning_means, width, label="Reasoning Heads (top 10)", color="tab:blue")
    ax.bar(x + width/2, other_means, width, label="Other Heads (326)", color="tab:gray")
    ax.set_xlabel("Training Condition")
    ax.set_ylabel("Mean Relative Activation Change")
    ax.set_title("How Much Did Each Condition Change Reasoning vs Other Heads?")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=30, ha="right")
    ax.legend()
    plt.tight_layout()
    plot2_path = f"{BASE_DIR}/plots/reasoning_vs_other_head_changes.png"
    plt.savefig(plot2_path, dpi=150)
    print(f"Saved {plot2_path}")

    # Plot 3: Per-head change scatter — reasoning heads highlighted
    fig, axes = plt.subplots(1, n_ckpts, figsize=(5 * n_ckpts, 5), squeeze=False)
    for i, (name, data) in enumerate(results["checkpoints"].items()):
        ax = axes[0, i]
        all_changes = data["norm_rel_change"].flatten().numpy()
        colors = ["tab:blue" if (l, h) in top10_heads else "tab:gray"
                  for l in range(n_layers) for h in range(n_heads)]
        alphas = [0.9 if (l, h) in top10_heads else 0.2
                  for l in range(n_layers) for h in range(n_heads)]
        for j in range(len(all_changes)):
            ax.scatter(j, all_changes[j], c=colors[j], alpha=alphas[j], s=10)
        ax.set_xlabel("Head index (layer × n_heads + head)")
        ax.set_ylabel("Relative change")
        step = "100" if "step_100" in CHECKPOINTS[name] else "50"
        ax.set_title(f"{name} (step {step})")
    fig.suptitle("Per-Head Activation Change (blue = reasoning heads)", fontsize=12)
    plt.tight_layout()
    plot3_path = f"{BASE_DIR}/plots/per_head_change_scatter.png"
    plt.savefig(plot3_path, dpi=150)
    print(f"Saved {plot3_path}")

    # Save raw results
    save_data = {
        "top10_reasoning_heads": sorted(list(top10_heads)),
        "n_problems": N_PROBLEMS,
        "conditions": {},
    }
    for name, data in results["checkpoints"].items():
        r_vals = [data["norm_rel_change"][l, h].item() for l, h in top10_heads]
        o_vals = [data["norm_rel_change"][l, h].item()
                  for l in range(n_layers) for h in range(n_heads)
                  if (l, h) not in top10_heads]
        save_data["conditions"][name] = {
            "reasoning_head_mean_change": sum(r_vals) / len(r_vals),
            "other_head_mean_change": sum(o_vals) / len(o_vals),
            "ratio": (sum(r_vals) / len(r_vals)) / (sum(o_vals) / len(o_vals)),
            "step": int(CHECKPOINTS[name].split("step_")[1]),
        }

    results_path = f"{BASE_DIR}/results/head_divergence_analysis.json"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"Saved {results_path}")

    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
