#!/usr/bin/env python3
"""Plot activation-based head analysis results and compare with EAP-IG."""

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

BASE_DIR = Path("attention_based_rewards")
N_LAYERS = 28
N_HEADS = 12

def main():
    act = torch.load(BASE_DIR / "data" / "base_model_activation_heads.pt", map_location="cpu", weights_only=False)

    combined = act["head_scores"]  # (28, 12)
    fai_div = act["fai_divergence"]
    ent_div = act["entropy_divergence"]
    mag_div = act["magnitude_divergence"]
    selected = act["selected_heads"]  # list of (l, h, score)

    # Also load EAP-IG for comparison
    eapig = torch.load(BASE_DIR / "data" / "base_model_reasoning_heads.pt", map_location="cpu", weights_only=False)
    eapig_scores = eapig["head_scores"]  # (28, 12)
    eapig_selected = eapig["selected_heads"]

    # --- Figure 1: Combined score heatmap ---
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))

    for ax, data, title, cmap in [
        (axes[0], combined.numpy(), "Combined Score\n(activation divergence)", "YlOrRd"),
        (axes[1], ent_div.numpy(), "Entropy Divergence\n(correct vs incorrect)", "YlOrRd"),
        (axes[2], fai_div.numpy(), "FAI Divergence\n(correct vs incorrect)", "YlOrRd"),
    ]:
        im = ax.imshow(data, cmap=cmap, aspect="auto")
        ax.set_xticks(range(N_HEADS))
        ax.set_xticklabels([f"H{h}" for h in range(N_HEADS)], fontsize=8)
        ax.set_yticks(range(N_LAYERS))
        ax.set_yticklabels([f"L{l}" for l in range(N_LAYERS)], fontsize=8)
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle("Activation-Based Head Importance (Qwen2.5-Math-1.5B base)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(BASE_DIR / "plots" / "activation_head_heatmaps.png", dpi=150, bbox_inches="tight")
    print("Saved activation_head_heatmaps.png")

    # --- Figure 2: Side-by-side comparison with EAP-IG ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Normalize both to [0,1] for visual comparison
    def norm01(t):
        return (t - t.min()) / (t.max() - t.min() + 1e-8)

    act_norm = norm01(combined).numpy()
    eapig_norm = norm01(eapig_scores).numpy()

    for ax, data, title in [
        (axes[0], eapig_norm, "EAP-IG Head Importance"),
        (axes[1], act_norm, "Activation-Based Head Importance"),
    ]:
        im = ax.imshow(data, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(N_HEADS))
        ax.set_xticklabels([f"H{h}" for h in range(N_HEADS)], fontsize=8)
        ax.set_yticks(range(N_LAYERS))
        ax.set_yticklabels([f"L{l}" for l in range(N_LAYERS)], fontsize=8)
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")
        ax.set_title(title, fontsize=12)
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle("EAP-IG vs Activation-Based: Head Importance Comparison", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(BASE_DIR / "plots" / "eapig_vs_activation_comparison.png", dpi=150, bbox_inches="tight")
    print("Saved eapig_vs_activation_comparison.png")

    # --- Figure 3: Top-20 overlap Venn-style bar chart ---
    eapig_top20 = {(l, h) for l, h, _ in eapig_selected[:20]}
    act_top20 = {(l, h) for l, h, _ in selected[:20]}

    only_eapig = eapig_top20 - act_top20
    only_act = act_top20 - eapig_top20
    both = eapig_top20 & act_top20

    fig, ax = plt.subplots(figsize=(14, 6))

    all_heads = sorted(eapig_top20 | act_top20)
    labels = [f"L{l}H{h}" for l, h in all_heads]
    colors = []
    for lh in all_heads:
        if lh in both:
            colors.append("#2ecc71")  # green = both
        elif lh in only_eapig:
            colors.append("#3498db")  # blue = EAP-IG only
        else:
            colors.append("#e74c3c")  # red = activation only

    # Get scores for bar heights
    eapig_vals = []
    act_vals = []
    for l, h in all_heads:
        eapig_vals.append(eapig_norm[l, h])
        act_vals.append(act_norm[l, h])

    x = np.arange(len(all_heads))
    width = 0.35
    ax.bar(x - width/2, eapig_vals, width, label="EAP-IG score", color="#3498db", alpha=0.7)
    ax.bar(x + width/2, act_vals, width, label="Activation score", color="#e74c3c", alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Normalized importance (0-1)")
    ax.set_title(f"Top-20 Heads: EAP-IG vs Activation-Based (overlap: {len(both)}/20)")
    ax.legend()

    # Highlight shared heads
    for i, lh in enumerate(all_heads):
        if lh in both:
            ax.axvspan(i - 0.5, i + 0.5, alpha=0.15, color="green")

    plt.tight_layout()
    plt.savefig(BASE_DIR / "plots" / "top20_eapig_vs_activation.png", dpi=150, bbox_inches="tight")
    print("Saved top20_eapig_vs_activation.png")

    # --- Figure 4: Layer distribution comparison ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    eapig_layers = [l for l, h, _ in eapig_selected[:20]]
    act_layers = [l for l, h, _ in selected[:20]]

    bins = np.arange(-0.5, N_LAYERS + 0.5, 1)
    axes[0].hist(eapig_layers, bins=bins, color="#3498db", alpha=0.7, label="EAP-IG top-20")
    axes[0].hist(act_layers, bins=bins, color="#e74c3c", alpha=0.5, label="Activation top-20")
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Layer Distribution of Top-20 Heads")
    axes[0].legend()

    # Scatter: EAP-IG score vs Activation score for all 336 heads
    eapig_flat = eapig_norm.flatten()
    act_flat = act_norm.flatten()

    axes[1].scatter(eapig_flat, act_flat, alpha=0.4, s=20, c="gray")
    # Highlight top-20 from each
    for l, h, _ in eapig_selected[:20]:
        axes[1].scatter(eapig_norm[l, h], act_norm[l, h], c="#3498db", s=60, zorder=5, edgecolors="black", linewidths=0.5)
    for l, h, _ in selected[:20]:
        axes[1].scatter(eapig_norm[l, h], act_norm[l, h], c="#e74c3c", s=60, zorder=5, edgecolors="black", linewidths=0.5)
    for l, h in both:
        axes[1].scatter(eapig_norm[l, h], act_norm[l, h], c="#2ecc71", s=100, zorder=10, edgecolors="black", linewidths=1, marker="*")

    axes[1].set_xlabel("EAP-IG importance (normalized)")
    axes[1].set_ylabel("Activation importance (normalized)")
    axes[1].set_title("All 336 Heads: EAP-IG vs Activation Score")
    axes[1].plot([0, 1], [0, 1], "k--", alpha=0.3)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#3498db", markersize=8, label="EAP-IG top-20"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#e74c3c", markersize=8, label="Activation top-20"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#2ecc71", markersize=12, label="Both top-20"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=6, label="Other heads"),
    ]
    axes[1].legend(handles=legend_elements, loc="upper left")

    plt.tight_layout()
    plt.savefig(BASE_DIR / "plots" / "head_method_comparison.png", dpi=150, bbox_inches="tight")
    print("Saved head_method_comparison.png")

    # Print summary
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    print(f"EAP-IG top-20: {sorted(eapig_top20)}")
    print(f"Activation top-20: {sorted(act_top20)}")
    print(f"Overlap: {sorted(both)} ({len(both)}/20)")
    print(f"Correlation (all 336 heads): {np.corrcoef(eapig_flat, act_flat)[0,1]:.3f}")


if __name__ == "__main__":
    main()
