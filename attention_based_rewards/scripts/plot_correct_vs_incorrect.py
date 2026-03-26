#!/usr/bin/env python3
"""Scatter plots: per-head activation on correct vs incorrect rollouts.
Highlights the 20 EAP-IG reasoning heads to see if they're the ones that differ."""

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
    act = torch.load(BASE_DIR / "data" / "base_model_activation_heads.pt",
                     map_location="cpu", weights_only=False)
    eapig = torch.load(BASE_DIR / "data" / "base_model_reasoning_heads.pt",
                       map_location="cpu", weights_only=False)

    fai_c = act["fai_correct"].numpy()      # (28, 12) mean FAI on correct
    fai_i = act["fai_incorrect"].numpy()     # (28, 12) mean FAI on incorrect
    ent_c = act["entropy_divergence"].numpy()  # already divergence, not raw
    # We need raw entropy — but it wasn't saved separately. We have fai_correct/incorrect.
    # Let's use what we have: FAI correct vs FAI incorrect

    eapig_top20 = {(l, h) for l, h, _ in eapig["selected_heads"][:20]}
    eapig_top10 = {(l, h) for l, h, _ in eapig["selected_heads"][:10]}

    # Flatten
    all_heads = [(l, h) for l in range(N_LAYERS) for h in range(N_HEADS)]

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))

    # ---- Plot 1: FAI correct vs FAI incorrect ----
    ax = axes[0]
    for l, h in all_heads:
        c = fai_c[l, h]
        ic = fai_i[l, h]
        if (l, h) in eapig_top10:
            ax.scatter(ic, c, c="#e74c3c", s=80, zorder=10, edgecolors="black", linewidths=0.8)
        elif (l, h) in eapig_top20:
            ax.scatter(ic, c, c="#f39c12", s=60, zorder=8, edgecolors="black", linewidths=0.5)
        else:
            ax.scatter(ic, c, c="#95a5a6", s=20, alpha=0.5, zorder=1)

    # Diagonal
    lims = [min(fai_c.min(), fai_i.min()), max(fai_c.max(), fai_i.max())]
    ax.plot(lims, lims, "k--", alpha=0.3, label="y=x")
    ax.set_xlabel("Mean FAI (incorrect rollouts)")
    ax.set_ylabel("Mean FAI (correct rollouts)")
    ax.set_title("FAI: Correct vs Incorrect")

    # Label EAP-IG top-10
    for l, h in eapig_top10:
        ax.annotate(f"L{l}H{h}", (fai_i[l, h], fai_c[l, h]),
                    fontsize=7, ha="left", va="bottom",
                    xytext=(3, 3), textcoords="offset points")

    from matplotlib.lines import Line2D
    legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#e74c3c", markersize=10, label="EAP-IG top-10"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#f39c12", markersize=8, label="EAP-IG top-11-20"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#95a5a6", markersize=6, label="Other heads"),
    ]
    ax.legend(handles=legend, loc="upper left", fontsize=9)

    # ---- Plot 2: FAI divergence (|correct - incorrect|) per head, sorted ----
    ax = axes[1]
    fai_div = np.abs(fai_c - fai_i)
    head_divs = []
    for l in range(N_LAYERS):
        for h in range(N_HEADS):
            head_divs.append((l, h, fai_div[l, h]))
    head_divs.sort(key=lambda x: x[2], reverse=True)

    colors = []
    labels_bar = []
    vals = []
    for rank, (l, h, d) in enumerate(head_divs[:40]):
        vals.append(d)
        labels_bar.append(f"L{l}H{h}")
        if (l, h) in eapig_top10:
            colors.append("#e74c3c")
        elif (l, h) in eapig_top20:
            colors.append("#f39c12")
        else:
            colors.append("#3498db")

    ax.barh(range(len(vals)-1, -1, -1), vals, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(vals)-1, -1, -1))
    ax.set_yticklabels(labels_bar, fontsize=7)
    ax.set_xlabel("|FAI_correct - FAI_incorrect|")
    ax.set_title("Top-40 Heads by FAI Divergence")

    legend2 = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#e74c3c", markersize=10, label="EAP-IG top-10"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#f39c12", markersize=8, label="EAP-IG top-11-20"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#3498db", markersize=8, label="Not in EAP-IG top-20"),
    ]
    ax.legend(handles=legend2, loc="lower right", fontsize=8)

    # ---- Plot 3: Signed divergence (correct - incorrect) for EAP-IG top-20 ----
    ax = axes[2]
    eapig_heads_sorted = [(l, h, s) for l, h, s in eapig["selected_heads"][:20]]
    eapig_labels = [f"L{l}H{h}" for l, h, _ in eapig_heads_sorted]
    eapig_signed_div = [fai_c[l, h] - fai_i[l, h] for l, h, _ in eapig_heads_sorted]

    bar_colors = ["#2ecc71" if d > 0 else "#e74c3c" for d in eapig_signed_div]
    ax.barh(range(len(eapig_signed_div)-1, -1, -1), eapig_signed_div, color=bar_colors, edgecolor="white")
    ax.set_yticks(range(len(eapig_signed_div)-1, -1, -1))
    ax.set_yticklabels(eapig_labels, fontsize=8)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("FAI_correct - FAI_incorrect (signed)")
    ax.set_title("EAP-IG Top-20: Are They More Active on Correct?")

    n_positive = sum(1 for d in eapig_signed_div if d > 0)
    ax.text(0.95, 0.05, f"{n_positive}/20 more active\non correct",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.suptitle(f"Head Activation: Correct vs Incorrect Rollouts\n"
                 f"(n_correct={act['n_correct']}, n_incorrect={act['n_incorrect']})",
                 fontsize=13, y=1.03)
    plt.tight_layout()
    plt.savefig(BASE_DIR / "plots" / "correct_vs_incorrect_scatter.png", dpi=150, bbox_inches="tight")
    print("Saved correct_vs_incorrect_scatter.png")

    # ---- Figure 2: Per-layer summary ----
    fig, ax = plt.subplots(figsize=(12, 5))

    # Mean FAI per layer, correct vs incorrect
    layer_fai_c = fai_c.mean(axis=1)  # (28,)
    layer_fai_i = fai_i.mean(axis=1)

    x = np.arange(N_LAYERS)
    width = 0.35
    ax.bar(x - width/2, layer_fai_c, width, label="Correct rollouts", color="#2ecc71", alpha=0.8)
    ax.bar(x + width/2, layer_fai_i, width, label="Incorrect rollouts", color="#e74c3c", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{l}" for l in range(N_LAYERS)], fontsize=8, rotation=45)
    ax.set_ylabel("Mean FAI")
    ax.set_title("Per-Layer Mean FAI: Correct vs Incorrect Rollouts")
    ax.legend()

    # Mark layers with EAP-IG heads
    eapig_layers = {l for l, h in eapig_top20}
    for l in eapig_layers:
        ax.axvspan(l - 0.5, l + 0.5, alpha=0.1, color="blue")

    plt.tight_layout()
    plt.savefig(BASE_DIR / "plots" / "per_layer_fai_correct_incorrect.png", dpi=150, bbox_inches="tight")
    print("Saved per_layer_fai_correct_incorrect.png")

    # Print summary stats
    print(f"\nEAP-IG top-20 heads — FAI on correct vs incorrect:")
    for l, h, s in eapig_heads_sorted:
        diff = fai_c[l, h] - fai_i[l, h]
        direction = "+" if diff > 0 else "-"
        print(f"  L{l}H{h}: correct={fai_c[l,h]:.6f}, incorrect={fai_i[l,h]:.6f}, "
              f"diff={diff:+.6f} ({direction})")


if __name__ == "__main__":
    main()
