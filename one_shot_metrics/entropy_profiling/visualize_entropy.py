#!/usr/bin/env python3
"""Generate all entropy profiling visualizations."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def interpolate_entropy(entropy_array: np.ndarray, num_bins: int = 100) -> np.ndarray:
    """Interpolate entropy array to fixed number of bins (normalized position)."""
    if len(entropy_array) <= 1:
        return np.zeros(num_bins)
    x_orig = np.linspace(0, 1, len(entropy_array))
    x_new = np.linspace(0, 1, num_bins)
    return np.interp(x_new, x_orig, entropy_array)


# ---- Per-example plots ----

def plot_entropy_curves(rollouts: list[dict], name: str, score: float, save_dir: Path, score_label: str = "MATH500"):
    """Normalized entropy curves: 32 faint lines + bold mean + shaded std."""
    num_bins = 100
    curves = np.array([interpolate_entropy(r["entropy_array"], num_bins) for r in rollouts])

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.linspace(0, 1, num_bins)

    # Individual rollouts (faint)
    for i in range(len(curves)):
        color = "tab:green" if rollouts[i].get("is_correct", False) else "tab:red"
        ax.plot(x, curves[i], color=color, alpha=0.15, linewidth=0.5)

    # Mean + std band
    mean_curve = np.mean(curves, axis=0)
    std_curve = np.std(curves, axis=0)
    ax.plot(x, mean_curve, color="black", linewidth=2, label="Mean")
    ax.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2, color="steelblue", label="\u00b11 std")

    ax.set_xlabel("Normalized position")
    ax.set_ylabel("Entropy (nats)")
    score_str = f"{score_label}={score:.1f}" if score is not None else "sampled"
    ax.set_title(f"{name} | {score_str}")
    ax.legend(loc="upper right")
    ax.set_xlim(0, 1)

    save_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_dir / f"entropy_curves_{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_entropy_heatmap(rollouts: list[dict], name: str, score: float, save_dir: Path, score_label: str = "MATH500"):
    """Heatmap: rows = rollouts sorted by length, columns = normalized position."""
    num_bins = 100
    curves = np.array([interpolate_entropy(r["entropy_array"], num_bins) for r in rollouts])
    lengths = [r["num_tokens"] for r in rollouts]

    # Sort by length
    order = np.argsort(lengths)
    curves_sorted = curves[order]

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(curves_sorted, aspect="auto", cmap="viridis",
                   extent=[0, 1, 0, len(rollouts)], origin="lower")
    ax.set_xlabel("Normalized position")
    ax.set_ylabel("Rollout (sorted by length)")
    score_str = f"{score_label}={score:.1f}" if score is not None else "sampled"
    ax.set_title(f"{name} entropy heatmap | {score_str}")
    plt.colorbar(im, ax=ax, label="Entropy (nats)")

    save_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_dir / f"entropy_heatmap_{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---- Comparison plots ----

def plot_comparison_grid(all_results: dict, save_dir: Path, score_key: str = "math500_score", score_label: str = "MATH500", suffix: str = ""):
    """Grid of subplots: examples sorted by score, showing top 16."""
    examples = sorted(all_results.values(), key=lambda r: r["example"].get(score_key, 0) or 0, reverse=True)
    num_bins = 100
    n_show = min(16, len(examples))
    ncols = 4
    nrows = (n_show + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 4 * nrows))
    axes = axes.flatten()

    for i, result in enumerate(examples):
        if i >= n_show:
            break
        ax = axes[i]
        rollouts = result["rollouts"]
        name = result["example"]["name"]
        score = result["example"].get(score_key, 0) or 0

        curves = np.array([interpolate_entropy(r["entropy_array"], num_bins) for r in rollouts])
        x = np.linspace(0, 1, num_bins)
        mean_curve = np.mean(curves, axis=0)
        std_curve = np.std(curves, axis=0)

        ax.plot(x, mean_curve, color="steelblue", linewidth=1.5)
        ax.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2, color="steelblue")
        ax.set_title(f"{name} ({score:.3f})" if score < 1 else f"{name} ({score:.1f})", fontsize=10)
        ax.set_xlim(0, 1)
        ax.tick_params(labelsize=8)

    for i in range(n_show, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(f"Per-Token Entropy Profiles (sorted by {score_label})", fontsize=14, y=1.02)
    fig.tight_layout()
    save_dir.mkdir(parents=True, exist_ok=True)
    fname = f"entropy_comparison_grid{suffix}.png"
    fig.savefig(save_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_violin(all_results: dict, save_dir: Path, score_key: str = "math500_score", score_label: str = "MATH500", suffix: str = ""):
    """Violin/box plot: pool all token entropies per example, ordered by score."""
    examples = sorted(all_results.values(), key=lambda r: r["example"].get(score_key, 0) or 0, reverse=True)

    # Limit to top/bottom 20 for readability when there are many examples
    if len(examples) > 40:
        top = examples[:20]
        bottom = examples[-20:]
        examples = top + bottom

    data = []
    labels = []
    for result in examples:
        name = result["example"]["name"]
        score = result["example"].get(score_key, 0) or 0
        all_ent = np.concatenate([r["entropy_array"] for r in result["rollouts"]])
        if len(all_ent) > 10000:
            all_ent = np.random.choice(all_ent, 10000, replace=False)
        data.append(all_ent)
        score_fmt = f"{score:.3f}" if score < 1 else f"{score:.1f}"
        labels.append(f"{name}\n({score_fmt})")

    fig, ax = plt.subplots(figsize=(max(16, len(examples) * 0.6), 6))
    parts = ax.violinplot(data, showmeans=True, showmedians=True)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Entropy (nats)")
    ax.set_title(f"Token Entropy Distribution per Example (sorted by {score_label})")

    save_dir.mkdir(parents=True, exist_ok=True)
    fname = f"entropy_violins{suffix}.png"
    fig.savefig(save_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_entropy_histograms(all_results: dict, save_dir: Path, score_key: str = "math500_score", score_label: str = "MATH500", suffix: str = ""):
    """Overlay histograms for best vs worst examples."""
    examples = sorted(all_results.values(), key=lambda r: r["example"].get(score_key, 0) or 0)
    worst = examples[0]
    best = examples[-1]

    fig, ax = plt.subplots(figsize=(10, 5))
    for result, color, label_suffix in [(worst, "red", "worst"), (best, "blue", "best")]:
        all_ent = np.concatenate([r["entropy_array"] for r in result["rollouts"]])
        name = result["example"]["name"]
        score = result["example"].get(score_key, 0) or 0
        score_fmt = f"{score:.3f}" if score < 1 else f"{score:.1f}"
        ax.hist(all_ent, bins=100, alpha=0.5, color=color, density=True,
                label=f"{name} ({score_label}={score_fmt}, {label_suffix})")

    ax.set_xlabel("Entropy (nats)")
    ax.set_ylabel("Density")
    ax.set_title(f"Token Entropy Distribution: Highest vs Lowest {score_label}")
    ax.legend()

    save_dir.mkdir(parents=True, exist_ok=True)
    fname = f"entropy_hist_best_vs_worst{suffix}.png"
    fig.savefig(save_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_spike_analysis(features: dict, all_results: dict, save_dir: Path, score_key: str = "math500_score", score_label: str = "MATH500", suffix: str = ""):
    """4-panel scatter: spike features vs target score."""
    names = list(features.keys())
    scores = np.array([all_results[n]["example"].get(score_key, 0) or 0 for n in names])

    # Identify Wang examples
    wang_mask = np.array([all_results[n]["example"].get("math500_score") is not None for n in names])
    math500_scores = {n: all_results[n]["example"].get("math500_score") for n in names}

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    spike_features = [
        ("num_spikes_mean", "Num Spikes (mean)"),
        ("mean_spike_magnitude_mean", "Spike Magnitude (mean)"),
        ("mean_spike_position_mean", "Spike Position (mean)"),
        ("high_entropy_ratio_mean", "High Entropy Ratio"),
    ]

    has_wang = wang_mask.any()

    for idx, (ax, (feat_key, feat_label)) in enumerate(zip(axes.flatten(), spike_features)):
        values = np.array([features[n][feat_key] for n in names])

        if has_wang:
            non_wang = ~wang_mask
            ax.scatter(values[non_wang], scores[non_wang], c="grey", s=15, alpha=0.4, edgecolors="none", label="sampled")
            ax.scatter(values[wang_mask], scores[wang_mask], c="tab:red", s=50, edgecolors="black",
                       linewidth=0.8, alpha=0.9, zorder=5, label="Wang (MATH500)")

            # Annotate Wang examples with MATH500 score
            for i, name in enumerate(names):
                if wang_mask[i]:
                    m500 = math500_scores[name]
                    ax.annotate(f"{m500:.0f}", (values[i], scores[i]),
                                fontsize=6, fontweight="bold", color="tab:red",
                                ha="left", va="bottom", xytext=(3, 3),
                                textcoords="offset points")
        else:
            # No Wang examples (e.g. Li-only run): plot all as colored dots
            ax.scatter(values, scores, c="tab:blue", s=50, edgecolors="black",
                       linewidth=0.8, alpha=0.9, zorder=5)

        if len(names) <= 20:
            for i, name in enumerate(names):
                ax.annotate(name, (values[i], scores[i]), fontsize=7, ha="left", va="bottom")

        ax.set_xlabel(feat_label)
        ax.set_ylabel(score_label)

        if len(set(values)) > 1:
            r, p = stats.pearsonr(values, scores)
            ax.set_title(f"{feat_label} vs {score_label} (r={r:.2f}, p={p:.3f})")
        else:
            ax.set_title(f"{feat_label} vs {score_label}")

        if idx == 0 and has_wang:
            ax.legend(fontsize=7, loc="best")

    fig.tight_layout()
    save_dir.mkdir(parents=True, exist_ok=True)
    fname = f"spike_analysis{suffix}.png"
    fig.savefig(save_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_feature_scatters(features: dict, all_results: dict, save_dir: Path, score_key: str = "math500_score", score_label: str = "MATH500", suffix: str = ""):
    """One scatter per summary feature vs target score, with correlations annotated."""
    names = list(features.keys())
    scores = np.array([all_results[n]["example"].get(score_key, 0) or 0 for n in names])

    # Identify Wang examples for highlighting
    wang_mask = np.array([all_results[n]["example"].get("math500_score") is not None for n in names])
    math500_scores = {n: all_results[n]["example"].get("math500_score") for n in names}

    feat_keys = [
        "mean_entropy_mean", "std_entropy_mean", "max_entropy_mean",
        "num_spikes_mean", "mean_spike_magnitude_mean", "mean_spike_position_mean",
        "high_entropy_ratio_mean", "entropy_trend_mean", "cross_rollout_entropy_var",
        "num_tokens_mean", "pass_rate",
        "num_unique_answers", "answer_entropy",
        "mean_common_prefix_len", "mean_pairwise_bleu",
        "num_entropy_clusters", "positional_entropy_variance",
        "num_unique_structures", "mean_num_steps", "std_num_steps",
    ]

    ncols = 4
    nrows = (len(feat_keys) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    has_wang = wang_mask.any()

    for i, feat_key in enumerate(feat_keys):
        ax = axes[i]
        values = np.array([features[n].get(feat_key, 0) for n in names])

        if has_wang:
            # Plot non-Wang as grey, Wang as colored
            non_wang = ~wang_mask
            ax.scatter(values[non_wang], scores[non_wang], s=15, c="grey", alpha=0.4, edgecolors="none", label="sampled")
            ax.scatter(values[wang_mask], scores[wang_mask], s=50, c="tab:red", edgecolors="black",
                       linewidth=0.8, alpha=0.9, zorder=5, label="Wang (MATH500)")

            # Annotate Wang examples with MATH500 score
            for j, name in enumerate(names):
                if wang_mask[j]:
                    m500 = math500_scores[name]
                    ax.annotate(f"{m500:.0f}", (values[j], scores[j]),
                                fontsize=6, fontweight="bold", color="tab:red",
                                ha="left", va="bottom", xytext=(3, 3),
                                textcoords="offset points")
        else:
            # No Wang examples (e.g. Li-only run): plot all as colored dots
            ax.scatter(values, scores, s=50, c="tab:blue", edgecolors="black",
                       linewidth=0.8, alpha=0.9, zorder=5)

        if len(names) <= 20:
            for j, name in enumerate(names):
                ax.annotate(name, (values[j], scores[j]), fontsize=6, ha="left", va="bottom")

        if len(set(values)) > 1:
            r_pearson, p_pearson = stats.pearsonr(values, scores)
            r_spearman, p_spearman = stats.spearmanr(values, scores)
            ax.set_title(f"{feat_key}\nPearson r={r_pearson:.2f} | Spearman \u03c1={r_spearman:.2f}", fontsize=9)
        else:
            ax.set_title(feat_key, fontsize=9)

        ax.set_xlabel(feat_key, fontsize=8)
        ax.set_ylabel(score_label, fontsize=8)

        if i == 0 and has_wang:
            ax.legend(fontsize=7, loc="best")

    for i in range(len(feat_keys), len(axes)):
        axes[i].set_visible(False)

    fig.tight_layout()
    save_dir.mkdir(parents=True, exist_ok=True)
    fname = f"feature_scatters{suffix}.png"
    fig.savefig(save_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_matrix(features: dict, all_results: dict, save_dir: Path, score_key: str = "math500_score", score_label: str = "MATH500", suffix: str = ""):
    """Correlation matrix heatmap: all entropy features + target score."""
    names = list(features.keys())

    feat_keys = [k for k in sorted(features[names[0]].keys()) if not k.endswith("_std")]
    feat_keys.append(score_key)

    matrix = []
    for name in names:
        row = [features[name].get(k, 0) for k in feat_keys[:-1]]
        row.append(all_results[name]["example"].get(score_key, 0) or 0)
        matrix.append(row)

    matrix = np.array(matrix)

    n_feats = matrix.shape[1]
    corr = np.corrcoef(matrix.T)

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n_feats))
    ax.set_yticks(range(n_feats))
    short_labels = [k.replace("_mean", "").replace("_", "\n") for k in feat_keys]
    ax.set_xticklabels(short_labels, rotation=90, fontsize=7)
    ax.set_yticklabels(short_labels, fontsize=7)

    for i in range(n_feats):
        for j in range(n_feats):
            ax.text(j, i, f"{corr[i,j]:.2f}", ha="center", va="center", fontsize=5)

    plt.colorbar(im, ax=ax)
    ax.set_title(f"Feature Correlation Matrix (incl. {score_label})")

    save_dir.mkdir(parents=True, exist_ok=True)
    fname = f"correlation_matrix{suffix}.png"
    fig.savefig(save_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_hvar_vs_math500(all_results: dict, save_dir: Path):
    """Scatter plot: historical variance vs MATH500 score for the 14 Wang examples."""
    wang = {k: v for k, v in all_results.items() if v["example"].get("math500_score") is not None}
    if not wang:
        return

    names = list(wang.keys())
    hvars = [wang[n]["example"].get("historical_variance", 0) or 0 for n in names]
    m500s = [wang[n]["example"]["math500_score"] for n in names]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(hvars, m500s, s=80, c="tab:red", edgecolors="black", linewidth=1, zorder=5)

    for i, name in enumerate(names):
        ax.annotate(f"{name}\n({m500s[i]:.1f})", (hvars[i], m500s[i]),
                    fontsize=8, ha="left", va="bottom", xytext=(5, 5),
                    textcoords="offset points")

    # Correlation
    if len(set(hvars)) > 1:
        r_s, p_s = stats.spearmanr(hvars, m500s)
        r_p, p_p = stats.pearsonr(hvars, m500s)
        ax.set_title(f"Historical Variance vs MATH500 (n={len(names)})\n"
                     f"Pearson r={r_p:.3f} (p={p_p:.3f}) | Spearman ρ={r_s:.3f} (p={p_s:.3f})")
    else:
        ax.set_title(f"Historical Variance vs MATH500 (n={len(names)})")

    ax.set_xlabel("Historical Variance (std of training accuracy trajectory)")
    ax.set_ylabel("MATH500 Score")

    save_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_dir / "hvar_vs_math500.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_diversity_dashboard(features: dict, all_results: dict, save_dir: Path, score_key: str = "math500_score", score_label: str = "MATH500", suffix: str = ""):
    """2x3 grid of diversity metrics vs score."""
    names = list(features.keys())
    scores = np.array([all_results[n]["example"].get(score_key, 0) or 0 for n in names])

    panel_keys = [
        ("answer_entropy", "Answer Entropy"),
        ("mean_pairwise_bleu", "Mean Pairwise BLEU"),
        ("num_entropy_clusters", "Num Entropy Clusters"),
        ("positional_entropy_variance", "Positional Entropy Var"),
        ("mean_num_steps", "Mean Num Steps"),
        ("pass_rate", "Pass Rate"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    wang_mask = np.array([all_results[n]["example"].get("math500_score") is not None for n in names])
    has_wang = wang_mask.any()

    for i, (feat_key, feat_label) in enumerate(panel_keys):
        ax = axes[i]
        values = np.array([features[n].get(feat_key, 0) for n in names])

        if has_wang:
            non_wang = ~wang_mask
            ax.scatter(values[non_wang], scores[non_wang], s=15, c="grey", alpha=0.4, edgecolors="none", label="sampled")
            ax.scatter(values[wang_mask], scores[wang_mask], s=50, c="tab:red", edgecolors="black",
                       linewidth=0.8, alpha=0.9, zorder=5, label="Wang")
        else:
            ax.scatter(values, scores, s=50, c="tab:blue", edgecolors="black",
                       linewidth=0.8, alpha=0.9, zorder=5)

        if len(names) <= 20:
            for j, name in enumerate(names):
                ax.annotate(name, (values[j], scores[j]), fontsize=6, ha="left", va="bottom")

        if len(set(values)) > 1:
            r_p, p_p = stats.pearsonr(values, scores)
            r_s, p_s = stats.spearmanr(values, scores)
            ax.set_title(f"{feat_label}\nPearson r={r_p:.2f} | Spearman \u03c1={r_s:.2f}", fontsize=9)
        else:
            ax.set_title(feat_label, fontsize=9)

        ax.set_xlabel(feat_label, fontsize=8)
        ax.set_ylabel(score_label, fontsize=8)

        if i == 0 and has_wang:
            ax.legend(fontsize=7, loc="best")

    fig.suptitle(f"Rollout Diversity Dashboard vs {score_label}", fontsize=12)
    fig.tight_layout()
    save_dir.mkdir(parents=True, exist_ok=True)
    fname = f"diversity_dashboard{suffix}.png"
    fig.savefig(save_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _generate_comparison_suite(all_results, features, fig_dir, score_key, score_label, suffix):
    """Generate the full comparison plot suite for a given score metric."""
    plot_comparison_grid(all_results, fig_dir, score_key=score_key, score_label=score_label, suffix=suffix)
    plot_violin(all_results, fig_dir, score_key=score_key, score_label=score_label, suffix=suffix)
    plot_entropy_histograms(all_results, fig_dir, score_key=score_key, score_label=score_label, suffix=suffix)
    plot_spike_analysis(features, all_results, fig_dir, score_key=score_key, score_label=score_label, suffix=suffix)
    plot_feature_scatters(features, all_results, fig_dir, score_key=score_key, score_label=score_label, suffix=suffix)
    plot_correlation_matrix(features, all_results, fig_dir, score_key=score_key, score_label=score_label, suffix=suffix)
    plot_diversity_dashboard(features, all_results, fig_dir, score_key=score_key, score_label=score_label, suffix=suffix)


def generate_all_plots(all_results: dict, features: dict, output_dir: Path,
                       score_key: str = "math500_score", score_label: str = "MATH500"):
    """Generate all visualizations."""
    fig_dir = output_dir / "figures"
    per_example_dir = fig_dir / "per_example"

    print("Generating per-example plots...")
    for name, result in all_results.items():
        score = result["example"].get(score_key)
        plot_entropy_curves(result["rollouts"], name, score, per_example_dir, score_label=score_label)
        plot_entropy_heatmap(result["rollouts"], name, score, per_example_dir, score_label=score_label)

    # Primary score metric plots
    print(f"Generating comparison plots vs {score_label} (n={len(all_results)})...")
    suffix = f"_{score_key}"
    _generate_comparison_suite(all_results, features, fig_dir,
                               score_key=score_key, score_label=score_label, suffix=suffix)

    # Historical variance plots for all examples (if available)
    has_hvar = any(v["example"].get("historical_variance") is not None for v in all_results.values())
    if has_hvar and score_key != "historical_variance":
        print(f"Generating comparison plots vs historical variance (n={len(all_results)})...")
        _generate_comparison_suite(all_results, features, fig_dir,
                                   score_key="historical_variance", score_label="Historical Variance", suffix="_hvar")

    # MATH500 plots for Wang examples only (if not already the primary metric)
    if score_key != "math500_score":
        wang_results = {k: v for k, v in all_results.items() if v["example"].get("math500_score") is not None}
        wang_features = {k: v for k, v in features.items() if k in wang_results}

        if wang_results:
            print(f"Generating comparison plots vs MATH500 (n={len(wang_results)})...")
            _generate_comparison_suite(wang_results, wang_features, fig_dir,
                                       score_key="math500_score", score_label="MATH500", suffix="_math500")

    # Historical variance vs MATH500 scatter for Wang examples
    wang_results = {k: v for k, v in all_results.items() if v["example"].get("math500_score") is not None}
    if has_hvar and wang_results:
        print("Generating historical variance vs MATH500 scatter...")
        plot_hvar_vs_math500(all_results, fig_dir)

    print(f"All plots saved to {fig_dir}")
