#!/usr/bin/env python3
"""Compare head classification: prompt-only vs prompt+generated response.

Method A (our current): Forward pass on raw prompts only
Method B (paper's): Generate responses with T=0.7, then forward on prompt+response

Outputs:
  1. Head classification comparison (overlap, turnover)
  2. Attention heatmaps for a sample response showing local vs global heads
  3. WAAD/FAI curves for a sample response under both methods
"""

import json
import logging
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/cluster/projects/nn12068k/haaklau/llm-training-experiments")
from attention_sparks_thinking.src.attention_rhythm import compute_fai, compute_gamma, compute_waad
from attention_sparks_thinking.src.head_classifier import aggregate_attention_hooks, classify_heads

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

OUT_DIR = "attention_sparks_thinking/analysis/head_classification_comparison"
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-Math-1.5B")
NUM_PROMPTS = 50
NUM_GENERATE = 20  # prompts to actually generate responses for
MAX_NEW_TOKENS = 512
GEN_TEMPERATURE = 0.7
HEAD_QUANTILE = 0.3


def load_prompts(n=NUM_PROMPTS):
    """Load math prompts from training data."""
    data_path = "attention_sparks_thinking/data/dapo_math_17k.parquet"
    if not os.path.exists(data_path):
        data_path = "attention_based_rewards/data/dapo_math_17k.parquet"
    df = pd.read_parquet(data_path)

    prompts = []
    for _, row in df.head(n).iterrows():
        prompt_msgs = row["prompt"]
        if isinstance(prompt_msgs, str):
            prompt_msgs = json.loads(prompt_msgs)
        text = "\n".join(m["content"] for m in prompt_msgs if m["role"] in ("user", "system"))
        prompts.append(text)
    return prompts


def classify_prompt_only(model, tokenizer, prompts):
    """Method A: classify heads using forward pass on prompts only."""
    logger.info("=== Method A: Prompt-only classification ===")
    H_loc, H_glob, d_matrix = classify_heads(
        model, tokenizer, prompts, head_quantile=HEAD_QUANTILE
    )
    logger.info(f"  H_loc: {len(H_loc)} heads, H_glob: {len(H_glob)} heads")
    return H_loc, H_glob, d_matrix


def classify_with_responses(model, tokenizer, prompts, device):
    """Method B: generate responses, then classify on full prompt+response."""
    logger.info("=== Method B: Prompt+Response classification ===")

    # Generate responses
    full_texts = []
    for i, prompt in enumerate(prompts[:NUM_GENERATE]):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=GEN_TEMPERATURE,
                do_sample=True,
                top_p=1.0,
            )
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        full_texts.append(full_text)
        if i < 3:
            resp_part = full_text[len(prompt):].strip()
            logger.info(f"  Sample {i}: prompt={len(prompt)} chars, response={len(resp_part)} chars")
            logger.info(f"    Response preview: {resp_part[:200]}...")

    logger.info(f"  Generated {len(full_texts)} responses")

    # Classify using full texts (prompt+response)
    H_loc, H_glob, d_matrix = classify_heads(
        model, tokenizer, full_texts, head_quantile=HEAD_QUANTILE
    )
    logger.info(f"  H_loc: {len(H_loc)} heads, H_glob: {len(H_glob)} heads")
    return H_loc, H_glob, d_matrix, full_texts


def compare_classifications(H_loc_A, H_glob_A, H_loc_B, H_glob_B, d_matrix_A, d_matrix_B):
    """Compare the two classification methods."""
    set_loc_A = set(H_loc_A)
    set_loc_B = set(H_loc_B)
    set_glob_A = set(H_glob_A)
    set_glob_B = set(H_glob_B)

    loc_overlap = set_loc_A & set_loc_B
    glob_overlap = set_glob_A & set_glob_B
    loc_jaccard = len(loc_overlap) / max(len(set_loc_A | set_loc_B), 1)
    glob_jaccard = len(glob_overlap) / max(len(set_glob_A | set_glob_B), 1)

    report = []
    report.append("=" * 60)
    report.append("HEAD CLASSIFICATION COMPARISON")
    report.append("=" * 60)
    report.append(f"Method A (prompt-only): {len(H_loc_A)} local, {len(H_glob_A)} global")
    report.append(f"Method B (prompt+response): {len(H_loc_B)} local, {len(H_glob_B)} global")
    report.append(f"")
    report.append(f"Local heads overlap: {len(loc_overlap)}/{len(set_loc_A)} (Jaccard={loc_jaccard:.3f})")
    report.append(f"Global heads overlap: {len(glob_overlap)}/{len(set_glob_A)} (Jaccard={glob_jaccard:.3f})")
    report.append(f"")
    report.append(f"Local heads ONLY in A: {sorted(set_loc_A - set_loc_B)}")
    report.append(f"Local heads ONLY in B: {sorted(set_loc_B - set_loc_A)}")
    report.append(f"Global heads ONLY in A: {sorted(set_glob_A - set_glob_B)}")
    report.append(f"Global heads ONLY in B: {sorted(set_glob_B - set_glob_A)}")

    # Compare d_matrix correlation
    corr = torch.corrcoef(torch.stack([d_matrix_A.flatten(), d_matrix_B.flatten()]))[0, 1].item()
    report.append(f"")
    report.append(f"d[l,h] correlation between methods: {corr:.4f}")

    # Layer distribution
    report.append(f"")
    report.append("Layer distribution of local heads:")
    for method, H_loc in [("A", H_loc_A), ("B", H_loc_B)]:
        layers = [l for l, h in H_loc]
        report.append(f"  Method {method}: {sorted(set(layers))}")
    report.append("Layer distribution of global heads:")
    for method, H_glob in [("A", H_glob_A), ("B", H_glob_B)]:
        layers = [l for l, h in H_glob]
        report.append(f"  Method {method}: {sorted(set(layers))}")

    report_text = "\n".join(report)
    print(report_text)

    with open(f"{OUT_DIR}/comparison_report.txt", "w") as f:
        f.write(report_text)

    return loc_jaccard, glob_jaccard


def plot_d_matrix_comparison(d_matrix_A, d_matrix_B, H_loc_A, H_glob_A, H_loc_B, H_glob_B):
    """Plot d[l,h] heatmaps side by side with head classifications marked."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, d_matrix, H_loc, H_glob, title in [
        (axes[0], d_matrix_A, H_loc_A, H_glob_A, "Method A: Prompt-only"),
        (axes[1], d_matrix_B, H_loc_B, H_glob_B, "Method B: Prompt+Response"),
    ]:
        d_np = d_matrix.cpu().numpy()
        im = ax.imshow(d_np, aspect="auto", cmap="viridis")
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label="Avg backward distance d[l,h]")

        # Mark local heads (blue circles) and global heads (red circles)
        for l, h in H_loc:
            ax.plot(h, l, "bo", markersize=4, alpha=0.7)
        for l, h in H_glob:
            ax.plot(h, l, "ro", markersize=4, alpha=0.7)
        ax.legend(["Local (H_loc)", "Global (H_glob)"], loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/d_matrix_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved d_matrix comparison to {OUT_DIR}/d_matrix_comparison.png")


def plot_attention_maps(model, tokenizer, text, H_loc_A, H_glob_A, H_loc_B, H_glob_B, device, prompt_len_tokens):
    """Plot aggregated attention maps for a sample text under both classifications."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(device)
    seq_len = inputs["input_ids"].shape[1]

    # Get aggregated attention under both sets of heads
    A_loc_A, A_glob_A = aggregate_attention_hooks(
        model, inputs["input_ids"], inputs["attention_mask"], H_loc_A, H_glob_A
    )
    A_loc_B, A_glob_B = aggregate_attention_hooks(
        model, inputs["input_ids"], inputs["attention_mask"], H_loc_B, H_glob_B
    )

    # Focus on response portion
    resp_start = prompt_len_tokens
    resp_end = min(seq_len, resp_start + 200)  # first 200 response tokens

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    titles = [
        ("A_bar_loc (Method A: prompt-only)", A_loc_A),
        ("A_bar_loc (Method B: prompt+resp)", A_loc_B),
        ("A_bar_glob (Method A: prompt-only)", A_glob_A),
        ("A_bar_glob (Method B: prompt+resp)", A_glob_B),
    ]

    for idx, (title, A) in enumerate(titles):
        ax = axes[idx // 2][idx % 2]
        A_resp = A[resp_start:resp_end, resp_start:resp_end].cpu().numpy()
        im = ax.imshow(A_resp, aspect="auto", cmap="hot", vmin=0)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Source token (response)")
        ax.set_ylabel("Target token (response)")
        plt.colorbar(im, ax=ax)

    plt.suptitle(f"Attention maps: response tokens {resp_start}-{resp_end}", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/attention_maps_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved attention maps to {OUT_DIR}/attention_maps_comparison.png")

    return A_loc_A, A_glob_A, A_loc_B, A_glob_B, resp_start


def plot_waad_fai_gamma(A_loc_A, A_glob_A, A_loc_B, A_glob_B, resp_start, tokenizer, input_ids, seq_len):
    """Plot WAAD, FAI, and gamma curves under both head classifications."""
    # Compute metrics under both methods
    waad_A = compute_waad(A_loc_A, resp_start, W=10)
    fai_A = compute_fai(A_glob_A, resp_start, H_lo=10, H_hi=50)
    gamma_A, stats_A = compute_gamma(waad_A, fai_A, q=0.4, gamma_amp=1.5, alpha=0.5, k=3)

    waad_B = compute_waad(A_loc_B, resp_start, W=10)
    fai_B = compute_fai(A_glob_B, resp_start, H_lo=10, H_hi=50)
    gamma_B, stats_B = compute_gamma(waad_B, fai_B, q=0.4, gamma_amp=1.5, alpha=0.5, k=3)

    # Decode tokens for x-axis
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0, resp_start:].tolist())
    n_plot = min(150, len(waad_A), len(waad_B))
    x = np.arange(n_plot)

    fig, axes = plt.subplots(3, 1, figsize=(20, 12), sharex=True)

    # WAAD
    axes[0].plot(x, waad_A[:n_plot].cpu().numpy(), label="Method A (prompt-only)", alpha=0.8)
    axes[0].plot(x, waad_B[:n_plot].cpu().numpy(), label="Method B (prompt+resp)", alpha=0.8)
    axes[0].set_ylabel("WAAD")
    axes[0].set_title("Windowed Average Attention Distance")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # FAI
    axes[1].plot(x, fai_A[:n_plot].cpu().numpy(), label="Method A (prompt-only)", alpha=0.8)
    axes[1].plot(x, fai_B[:n_plot].cpu().numpy(), label="Method B (prompt+resp)", alpha=0.8)
    axes[1].set_ylabel("FAI")
    axes[1].set_title("Future Attention Influence")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Gamma
    axes[2].bar(x - 0.15, gamma_A[:n_plot].cpu().numpy(), width=0.3, label="Method A (prompt-only)", alpha=0.7)
    axes[2].bar(x + 0.15, gamma_B[:n_plot].cpu().numpy(), width=0.3, label="Method B (prompt+resp)", alpha=0.7)
    axes[2].axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    axes[2].set_ylabel("Gamma")
    axes[2].set_title("Per-token gamma (amplification factor)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    # Add token labels (every 5th token)
    tick_positions = list(range(0, n_plot, 5))
    tick_labels = [tokens[i][:10] if i < len(tokens) else "" for i in tick_positions]
    axes[2].set_xticks(tick_positions)
    axes[2].set_xticklabels(tick_labels, rotation=90, fontsize=6)
    axes[2].set_xlabel("Response tokens")

    plt.suptitle(f"WAAD/FAI/Gamma comparison\nMethod A stats: {stats_A}\nMethod B stats: {stats_B}", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/waad_fai_gamma_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved WAAD/FAI/gamma comparison to {OUT_DIR}/waad_fai_gamma_comparison.png")

    # Print stats
    print(f"\nMethod A gamma stats: {stats_A}")
    print(f"Method B gamma stats: {stats_B}")

    # Print high-gamma tokens
    for method, gamma, name in [(gamma_A, tokens, "A"), (gamma_B, tokens, "B")]:
        top_indices = gamma[:n_plot].topk(min(10, n_plot)).indices.tolist()
        print(f"\nMethod {name} top gamma tokens:")
        for idx in sorted(top_indices):
            tok = tokens[idx] if idx < len(tokens) else "?"
            print(f"  pos={idx} token='{tok}' gamma={gamma[idx].item():.3f}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}, Model: {MODEL_NAME}")

    # Load model with eager attention
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        trust_remote_code=True,
    ).to(device)
    model.eval()

    # Load prompts
    prompts = load_prompts(NUM_PROMPTS)
    logger.info(f"Loaded {len(prompts)} prompts")

    # Method A: prompt-only classification
    H_loc_A, H_glob_A, d_matrix_A = classify_prompt_only(model, tokenizer, prompts)

    # Method B: prompt+response classification
    H_loc_B, H_glob_B, d_matrix_B, full_texts = classify_with_responses(
        model, tokenizer, prompts, device
    )

    # Compare
    compare_classifications(H_loc_A, H_glob_A, H_loc_B, H_glob_B, d_matrix_A, d_matrix_B)

    # Plot d_matrix comparison
    plot_d_matrix_comparison(d_matrix_A, d_matrix_B, H_loc_A, H_glob_A, H_loc_B, H_glob_B)

    # Use a generated response for attention map comparison
    sample_text = full_texts[0]
    sample_prompt = prompts[0]
    prompt_tokens = tokenizer(sample_prompt, return_tensors="pt")["input_ids"].shape[1]

    # Attention maps and WAAD/FAI/gamma
    A_loc_A, A_glob_A, A_loc_B, A_glob_B, resp_start = plot_attention_maps(
        model, tokenizer, sample_text,
        H_loc_A, H_glob_A, H_loc_B, H_glob_B,
        device, prompt_tokens
    )

    inputs = tokenizer(sample_text, return_tensors="pt", truncation=True, max_length=1024).to(device)
    plot_waad_fai_gamma(
        A_loc_A, A_glob_A, A_loc_B, A_glob_B,
        resp_start, tokenizer, inputs["input_ids"], inputs["input_ids"].shape[1]
    )

    logger.info(f"\nAll outputs saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
