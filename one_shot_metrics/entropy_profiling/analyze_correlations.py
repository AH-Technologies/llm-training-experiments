#!/usr/bin/env python3
"""Correlation analysis: entropy features vs historical variance and MATH500 scores.

Run after entropy profiling completes:
    python analyze_correlations.py --results-dir results/large_run
"""

import argparse
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from extract_examples import WANG_EXAMPLES, MATH500_SCORES, ACC_STEP_500_PATH
from entropy_features import extract_all_features


# Features to analyze (mean-aggregated across rollouts)
FEATURE_NAMES = [
    "mean_entropy_mean",
    "median_entropy_mean",
    "std_entropy_mean",
    "max_entropy_mean",
    "num_spikes_mean",
    "mean_spike_magnitude_mean",
    "high_entropy_ratio_mean",
    "early_mean_mean",
    "late_mean_mean",
    "entropy_trend_mean",
    "cross_rollout_entropy_var",
    "pass_rate",
    "num_tokens_mean",
    "num_unique_answers",
    "answer_entropy",
    "mean_common_prefix_len",
    "mean_pairwise_bleu",
    "num_entropy_clusters",
    "positional_entropy_variance",
    "num_unique_structures",
    "mean_num_steps",
    "std_num_steps",
]


def load_results(results_dir: Path) -> dict:
    """Load all entropy profile pickles from a results directory."""
    profiles_dir = results_dir / "entropy_profiles"
    all_results = {}
    for pkl_path in sorted(profiles_dir.glob("entropy_*.pkl")):
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        name = pkl_path.stem.replace("entropy_", "")
        all_results[name] = data
    return all_results


def compute_historical_variance() -> dict[int, float]:
    """Load acc_step_500.json and compute std per example."""
    with open(ACC_STEP_500_PATH) as f:
        acc_data = json.load(f)
    return {int(k): float(np.std(v)) for k, v in acc_data.items()}


def correlate_features(df: pd.DataFrame, target_col: str, feature_cols: list[str]) -> pd.DataFrame:
    """Compute Spearman and Pearson correlations for each feature vs target."""
    rows = []
    for feat in feature_cols:
        mask = df[[feat, target_col]].dropna().index
        x = df.loc[mask, feat].values
        y = df.loc[mask, target_col].values
        if len(x) < 5:
            continue

        spearman_r, spearman_p = stats.spearmanr(x, y)
        pearson_r, pearson_p = stats.pearsonr(x, y)

        rows.append({
            "feature": feat,
            "spearman_r": spearman_r,
            "spearman_p": spearman_p,
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
            "n": len(x),
        })

    return pd.DataFrame(rows).sort_values("spearman_p")


def nonlinear_transforms(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Add log and squared transforms of features."""
    for feat in feature_cols:
        vals = df[feat].values
        # Log transform (shift to positive)
        min_val = np.nanmin(vals)
        shifted = vals - min_val + 1e-6
        df[f"{feat}_log"] = np.log(shifted)
        # Squared
        df[f"{feat}_sq"] = vals ** 2
    return df


def leave_one_out_stability(df: pd.DataFrame, target_col: str, feature_col: str) -> dict:
    """Compute LOO stability: how much Spearman r changes when dropping each point."""
    mask = df[[feature_col, target_col]].dropna().index
    x = df.loc[mask, feature_col].values
    y = df.loc[mask, target_col].values
    n = len(x)
    if n < 6:
        return {"mean_r": np.nan, "std_r": np.nan, "min_r": np.nan, "max_r": np.nan}

    full_r, _ = stats.spearmanr(x, y)
    loo_rs = []
    for i in range(n):
        x_loo = np.delete(x, i)
        y_loo = np.delete(y, i)
        r, _ = stats.spearmanr(x_loo, y_loo)
        loo_rs.append(r)

    return {
        "full_r": full_r,
        "mean_r": np.mean(loo_rs),
        "std_r": np.std(loo_rs),
        "min_r": np.min(loo_rs),
        "max_r": np.max(loo_rs),
    }


def sweet_spot_model(x: np.ndarray, y: np.ndarray) -> dict:
    """Test quadratic (sweet-spot) relationship: y = a*x^2 + b*x + c."""
    if len(x) < 10:
        return {"r_squared": np.nan}
    # Fit quadratic
    coeffs = np.polyfit(x, y, 2)
    y_pred = np.polyval(coeffs, x)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Compare with linear
    coeffs_lin = np.polyfit(x, y, 1)
    y_pred_lin = np.polyval(coeffs_lin, x)
    ss_res_lin = np.sum((y - y_pred_lin) ** 2)
    r_squared_lin = 1 - ss_res_lin / ss_tot if ss_tot > 0 else 0.0

    return {
        "r_squared_quadratic": r_squared,
        "r_squared_linear": r_squared_lin,
        "quadratic_improvement": r_squared - r_squared_lin,
        "coeffs": coeffs.tolist(),
    }


def plot_scatter(df: pd.DataFrame, feature: str, target: str, ax, title: str = None):
    """Scatter plot with regression line."""
    mask = df[[feature, target]].dropna().index
    x = df.loc[mask, feature].values
    y = df.loc[mask, target].values

    ax.scatter(x, y, alpha=0.3, s=15, edgecolors="none")

    # Linear fit
    if len(x) > 2:
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, p(x_line), "r-", linewidth=1.5, alpha=0.7)

    r, pval = stats.spearmanr(x, y)
    ax.set_xlabel(feature.replace("_mean", "").replace("_", " "), fontsize=8)
    ax.set_ylabel(target.replace("_", " "), fontsize=8)
    ax.set_title(title or f"r={r:.3f}, p={pval:.2e}", fontsize=9)
    ax.tick_params(labelsize=7)


def plot_correlation_bars(corr_df: pd.DataFrame, target_name: str, ax):
    """Horizontal bar chart of Spearman correlations."""
    df = corr_df.head(15).sort_values("spearman_r")
    colors = ["#d73027" if r < 0 else "#4575b4" for r in df["spearman_r"]]
    bars = ax.barh(range(len(df)), df["spearman_r"], color=colors, alpha=0.8)

    # Add significance markers
    for i, (_, row) in enumerate(df.iterrows()):
        marker = "***" if row["spearman_p"] < 0.001 else "**" if row["spearman_p"] < 0.01 else "*" if row["spearman_p"] < 0.05 else ""
        x_pos = row["spearman_r"] + 0.02 * np.sign(row["spearman_r"])
        ax.text(x_pos, i, marker, va="center", fontsize=8)

    labels = [f.replace("_mean", "").replace("_", " ") for f in df["feature"]]
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Spearman r", fontsize=9)
    ax.set_title(f"Correlations with {target_name}", fontsize=10)
    ax.axvline(0, color="black", linewidth=0.5)


def main():
    parser = argparse.ArgumentParser(description="Correlation analysis for entropy features")
    parser.add_argument("--results-dir", type=str, default="results/large_run", help="Directory with entropy profiling results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading entropy profiles...")
    all_results = load_results(results_dir)
    print(f"  Loaded {len(all_results)} examples")

    if not all_results:
        print("ERROR: No results found. Run entropy profiling first.")
        return

    # Extract features
    print("Extracting features...")
    features = extract_all_features(all_results)

    # Compute historical variance
    hist_var = compute_historical_variance()

    # Build unified dataframe
    rows = []
    wang_name_by_idx = {v: k for k, v in WANG_EXAMPLES.items()}
    for name, feat in features.items():
        ex = all_results[name]["example"]
        row = {"name": name, "index": ex["index"]}
        row["historical_variance"] = ex.get("historical_variance", hist_var.get(ex["index"]))
        row["math500_score"] = ex.get("math500_score")
        row.update(feat)
        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"  DataFrame: {len(df)} examples, {len(df.columns)} columns")

    # Add non-linear transforms
    transform_features = [f for f in FEATURE_NAMES if f in df.columns]
    df = nonlinear_transforms(df, transform_features)
    all_feature_cols = transform_features + [f"{f}_log" for f in transform_features] + [f"{f}_sq" for f in transform_features]
    all_feature_cols = [f for f in all_feature_cols if f in df.columns]

    # ============================================================
    # PRIMARY ANALYSIS: All examples vs historical variance (n≈400)
    # ============================================================
    print("\n" + "=" * 60)
    print(f"PRIMARY ANALYSIS: Entropy features vs Historical Variance (n={len(df)})")
    print("=" * 60)

    corr_hvar = correlate_features(df, "historical_variance", all_feature_cols)
    print("\nTop correlations with historical variance:")
    print(corr_hvar.head(15).to_string(index=False))

    # Sweet-spot analysis for top features
    print("\nSweet-spot (quadratic) analysis for top linear features:")
    for _, row in corr_hvar.head(5).iterrows():
        feat = row["feature"]
        mask = df[[feat, "historical_variance"]].dropna().index
        x = df.loc[mask, feat].values
        y = df.loc[mask, "historical_variance"].values
        ss = sweet_spot_model(x, y)
        print(f"  {feat}: R²_linear={ss['r_squared_linear']:.4f}, "
              f"R²_quad={ss['r_squared_quadratic']:.4f}, "
              f"improvement={ss['quadratic_improvement']:.4f}")

    # LOO stability for top features
    print("\nLeave-one-out stability for top features:")
    loo_results = {}
    for _, row in corr_hvar.head(5).iterrows():
        feat = row["feature"]
        loo = leave_one_out_stability(df, "historical_variance", feat)
        loo_results[feat] = loo
        print(f"  {feat}: full_r={loo.get('full_r', np.nan):.4f}, "
              f"LOO mean={loo['mean_r']:.4f} ± {loo['std_r']:.4f}, "
              f"range=[{loo['min_r']:.4f}, {loo['max_r']:.4f}]")

    # ============================================================
    # SECONDARY ANALYSIS: Wang examples vs MATH500 (n=14)
    # ============================================================
    wang_df = df[df["math500_score"].notna()].copy()
    print(f"\n{'=' * 60}")
    print(f"SECONDARY ANALYSIS: Entropy features vs MATH500 (n={len(wang_df)})")
    print("=" * 60)

    corr_math = correlate_features(wang_df, "math500_score", transform_features)
    print("\nTop correlations with MATH500:")
    print(corr_math.head(10).to_string(index=False))

    # ============================================================
    # COMPARISON: Do the same features predict both metrics?
    # ============================================================
    print(f"\n{'=' * 60}")
    print("COMPARISON: Feature rankings across metrics")
    print("=" * 60)

    # Use only base features for comparison
    hvar_ranks = corr_hvar[corr_hvar["feature"].isin(transform_features)].reset_index(drop=True)
    math_ranks = corr_math[corr_math["feature"].isin(transform_features)].reset_index(drop=True)

    if len(hvar_ranks) > 0 and len(math_ranks) > 0:
        comparison = pd.merge(
            hvar_ranks[["feature", "spearman_r", "spearman_p"]].rename(
                columns={"spearman_r": "hvar_r", "spearman_p": "hvar_p"}),
            math_ranks[["feature", "spearman_r", "spearman_p"]].rename(
                columns={"spearman_r": "math500_r", "spearman_p": "math500_p"}),
            on="feature", how="outer"
        )
        comparison["same_sign"] = np.sign(comparison["hvar_r"]) == np.sign(comparison["math500_r"])
        print(comparison.to_string(index=False))

    # ============================================================
    # COMPOSITE FEATURES: Combine top predictors
    # ============================================================
    print(f"\n{'=' * 60}")
    print("COMPOSITE FEATURES: Combining top MATH500 predictors")
    print("=" * 60)

    if len(wang_df) >= 5:
        from itertools import combinations

        top_math_features = [r["feature"] for _, r in corr_math.head(6).iterrows()
                             if r["spearman_p"] < 0.1]

        print(f"\nBase features (p<0.1 for MATH500): {top_math_features}")

        # Standardize features on Wang examples (manual z-score)
        wang_std = wang_df.copy()
        for f in top_math_features:
            mu, sigma = wang_df[f].mean(), wang_df[f].std()
            wang_std[f] = (wang_df[f] - mu) / (sigma if sigma > 0 else 1)

        # Try all pairs and triples
        composite_results = []
        for k in range(2, min(len(top_math_features) + 1, 5)):
            for combo in combinations(top_math_features, k):
                # Sign-align: flip features with positive correlation so all contribute negatively
                signs = []
                for feat in combo:
                    row = corr_math[corr_math["feature"] == feat].iloc[0]
                    signs.append(-1 if row["spearman_r"] > 0 else 1)

                composite = sum(s * wang_std[f].values for s, f in zip(signs, combo))
                r_s, p_s = stats.spearmanr(composite, wang_df["math500_score"].values)
                r_p, p_p = stats.pearsonr(composite, wang_df["math500_score"].values)

                # LOO stability
                loo_rs = []
                for i in range(len(composite)):
                    c_loo = np.delete(composite, i)
                    y_loo = np.delete(wang_df["math500_score"].values, i)
                    r_loo, _ = stats.spearmanr(c_loo, y_loo)
                    loo_rs.append(r_loo)

                composite_results.append({
                    "features": " + ".join(combo),
                    "n_features": k,
                    "spearman_r": r_s,
                    "spearman_p": p_s,
                    "pearson_r": r_p,
                    "pearson_p": p_p,
                    "loo_mean": np.mean(loo_rs),
                    "loo_std": np.std(loo_rs),
                    "loo_min": np.min(loo_rs),
                })

        comp_df = pd.DataFrame(composite_results).sort_values("spearman_p")
        print("\nTop 15 composite features (sorted by Spearman p-value):")
        print(comp_df.head(15)[["features", "spearman_r", "spearman_p", "pearson_r", "loo_mean", "loo_std"]].to_string(index=False))

        # Best composite: plot it
        best = comp_df.iloc[0]
        best_feats = best["features"].split(" + ")
        signs = []
        for feat in best_feats:
            row = corr_math[corr_math["feature"] == feat].iloc[0]
            signs.append(-1 if row["spearman_r"] > 0 else 1)
        best_composite = sum(s * wang_std[f].values for s, f in zip(signs, best_feats))

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(best_composite, wang_df["math500_score"].values, s=80, c="tab:red",
                   edgecolors="black", linewidth=1, zorder=5)
        for i, name in enumerate(wang_df["name"].values):
            ax.annotate(name, (best_composite[i], wang_df["math500_score"].values[i]),
                        fontsize=8, ha="left", va="bottom", xytext=(5, 3),
                        textcoords="offset points")
        # Regression line
        z = np.polyfit(best_composite, wang_df["math500_score"].values, 1)
        x_line = np.linspace(best_composite.min(), best_composite.max(), 100)
        ax.plot(x_line, np.polyval(z, x_line), "r--", alpha=0.7)
        ax.set_xlabel(f"Composite: {best['features']}", fontsize=9)
        ax.set_ylabel("MATH500 Score")
        ax.set_title(f"Best Composite Feature vs MATH500\n"
                     f"Spearman ρ={best['spearman_r']:.3f} (p={best['spearman_p']:.4f}) | "
                     f"LOO: {best['loo_mean']:.3f} ± {best['loo_std']:.3f}")
        plt.tight_layout()
        figures_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(figures_dir / "composite_best_vs_math500.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Also plot top 5 composites as bar chart
        fig, ax = plt.subplots(figsize=(10, 5))
        top5 = comp_df.head(5)
        y_pos = range(len(top5))
        bars = ax.barh(y_pos, top5["spearman_r"].values, alpha=0.7,
                       xerr=top5["loo_std"].values, capsize=3)
        ax.set_yticks(y_pos)
        labels = [f.replace("_mean", "").replace("_", " ") for f in top5["features"]]
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("Spearman ρ with MATH500")
        ax.set_title("Top 5 Composite Features vs MATH500 (with LOO error bars)")
        for i, (_, row) in enumerate(top5.iterrows()):
            ax.text(row["spearman_r"] + 0.02, i, f"p={row['spearman_p']:.4f}", va="center", fontsize=7)
        plt.tight_layout()
        fig.savefig(figures_dir / "composite_top5_bars.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Apply best composite to ALL examples and correlate with historical variance
        print("\nApplying best composite to all examples:")
        all_std = df.copy()
        for f in top_math_features:
            mu, sigma = df[f].mean(), df[f].std()
            all_std[f] = (df[f] - mu) / (sigma if sigma > 0 else 1)
        all_composite = sum(s * all_std[f].values for s, f in zip(signs, best_feats))
        df["best_composite"] = all_composite
        hvar_valid = df["historical_variance"].notna()
        r_s_all, p_s_all = stats.spearmanr(all_composite[hvar_valid], df.loc[hvar_valid, "historical_variance"].values)
        print(f"  Best composite vs historical variance (n={hvar_valid.sum()}): Spearman ρ={r_s_all:.4f}, p={p_s_all:.4e}")

    # ============================================================
    # NON-LINEAR INTERACTIONS: Products, ratios, and cross-terms
    # ============================================================
    print(f"\n{'=' * 60}")
    print("NON-LINEAR INTERACTIONS: Products & ratios of features")
    print("=" * 60)

    # Features to combine non-linearly
    interaction_features = ["pass_rate", "mean_entropy_mean", "std_entropy_mean",
                            "early_mean_mean", "median_entropy_mean", "entropy_trend_mean",
                            "cross_rollout_entropy_var", "mean_spike_magnitude_mean"]
    interaction_features = [f for f in interaction_features if f in df.columns]

    interaction_results = []

    # Filter to rows with valid historical variance for hvar correlations
    hvar_mask = df["historical_variance"].notna()
    df_hvar = df[hvar_mask]

    for i, f1 in enumerate(interaction_features):
        for f2 in interaction_features[i+1:]:
            v1_hvar = df_hvar[f1].values.astype(float)
            v2_hvar = df_hvar[f2].values.astype(float)

            # Product
            product_hvar = v1_hvar * v2_hvar
            r_hvar, p_hvar = stats.spearmanr(product_hvar, df_hvar["historical_variance"].values)

            # Wang subset
            if len(wang_df) >= 5:
                v1_w = wang_df[f1].values.astype(float)
                v2_w = wang_df[f2].values.astype(float)
                product_w = v1_w * v2_w
                r_math, p_math = stats.spearmanr(product_w, wang_df["math500_score"].values)
            else:
                r_math, p_math = np.nan, np.nan

            interaction_results.append({
                "interaction": f"{f1} × {f2}",
                "type": "product",
                "hvar_r": r_hvar, "hvar_p": p_hvar,
                "math500_r": r_math, "math500_p": p_math,
            })

            # Ratio (f1 / (f2 + epsilon)) — only if f2 has consistent sign
            if (v2_hvar > 0).all() or (v2_hvar < 0).all():
                ratio_hvar = v1_hvar / (v2_hvar + 1e-12)
                r_hvar_ratio, p_hvar_ratio = stats.spearmanr(ratio_hvar, df_hvar["historical_variance"].values)

                if len(wang_df) >= 5:
                    ratio_w = v1_w / (v2_w + 1e-12)
                    r_math_ratio, p_math_ratio = stats.spearmanr(ratio_w, wang_df["math500_score"].values)
                else:
                    r_math_ratio, p_math_ratio = np.nan, np.nan

                interaction_results.append({
                    "interaction": f"{f1} / {f2}",
                    "type": "ratio",
                    "hvar_r": r_hvar_ratio, "hvar_p": p_hvar_ratio,
                    "math500_r": r_math_ratio, "math500_p": p_math_ratio,
                })

    inter_df = pd.DataFrame(interaction_results)

    # Show top interactions for MATH500
    print("\nTop 15 interactions vs MATH500 (by |Spearman ρ|):")
    inter_math = inter_df.dropna(subset=["math500_r"]).copy()
    inter_math["abs_math500_r"] = inter_math["math500_r"].abs()
    inter_math = inter_math.sort_values("abs_math500_r", ascending=False)
    print(inter_math.head(15)[["interaction", "type", "math500_r", "math500_p", "hvar_r", "hvar_p"]].to_string(index=False))

    # Show top interactions for historical variance
    print("\nTop 15 interactions vs Historical Variance (by |Spearman ρ|):")
    inter_hvar = inter_df.copy()
    inter_hvar["abs_hvar_r"] = inter_hvar["hvar_r"].abs()
    inter_hvar = inter_hvar.sort_values("abs_hvar_r", ascending=False)
    print(inter_hvar.head(15)[["interaction", "type", "hvar_r", "hvar_p", "math500_r", "math500_p"]].to_string(index=False))

    # Highlight interactions that are good for BOTH metrics
    print("\nInteractions significant for BOTH (hvar p<0.05 AND math500 p<0.05):")
    both_sig = inter_df[(inter_df["hvar_p"] < 0.05) & (inter_df["math500_p"] < 0.05)]
    if len(both_sig) > 0:
        both_sig = both_sig.copy()
        both_sig["combined_r"] = both_sig["hvar_r"].abs() + both_sig["math500_r"].abs()
        both_sig = both_sig.sort_values("combined_r", ascending=False)
        print(both_sig[["interaction", "type", "hvar_r", "hvar_p", "math500_r", "math500_p"]].to_string(index=False))
    else:
        print("  None found.")

    # Plot best interaction for each target
    if len(inter_math) > 0 and len(wang_df) >= 5:
        best_math_inter = inter_math.iloc[0]
        parts = best_math_inter["interaction"].split(" × " if best_math_inter["type"] == "product" else " / ")
        f1, f2 = parts[0].strip(), parts[1].strip()
        v1_w = wang_df[f1].values.astype(float)
        v2_w = wang_df[f2].values.astype(float)
        if best_math_inter["type"] == "product":
            inter_vals = v1_w * v2_w
        else:
            inter_vals = v1_w / (v2_w + 1e-12)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(inter_vals, wang_df["math500_score"].values, s=80, c="tab:red",
                   edgecolors="black", linewidth=1, zorder=5)
        for j, name in enumerate(wang_df["name"].values):
            ax.annotate(name, (inter_vals[j], wang_df["math500_score"].values[j]),
                        fontsize=8, ha="left", va="bottom", xytext=(5, 3),
                        textcoords="offset points")
        z = np.polyfit(inter_vals, wang_df["math500_score"].values, 1)
        x_line = np.linspace(inter_vals.min(), inter_vals.max(), 100)
        ax.plot(x_line, np.polyval(z, x_line), "r--", alpha=0.7)
        ax.set_xlabel(best_math_inter["interaction"], fontsize=10)
        ax.set_ylabel("MATH500 Score")
        ax.set_title(f"Best Non-Linear Interaction vs MATH500\n"
                     f"Spearman ρ={best_math_inter['math500_r']:.3f} (p={best_math_inter['math500_p']:.4f})")
        plt.tight_layout()
        fig.savefig(figures_dir / "interaction_best_math500.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    if len(inter_hvar) > 0:
        best_hvar_inter = inter_hvar.iloc[0]
        parts = best_hvar_inter["interaction"].split(" × " if best_hvar_inter["type"] == "product" else " / ")
        f1, f2 = parts[0].strip(), parts[1].strip()
        v1_a = df[f1].values.astype(float)
        v2_a = df[f2].values.astype(float)
        if best_hvar_inter["type"] == "product":
            inter_vals_all = v1_a * v2_a
        else:
            inter_vals_all = v1_a / (v2_a + 1e-12)

        fig, ax = plt.subplots(figsize=(8, 6))
        # Grey for sampled, red for Wang
        is_wang = df["math500_score"].notna()
        ax.scatter(inter_vals_all[~is_wang], df.loc[~is_wang, "historical_variance"],
                   s=30, c="grey", alpha=0.4, label="Sampled")
        ax.scatter(inter_vals_all[is_wang], df.loc[is_wang, "historical_variance"],
                   s=80, c="tab:red", edgecolors="black", linewidth=1, zorder=5, label="Wang")
        ax.set_xlabel(best_hvar_inter["interaction"], fontsize=10)
        ax.set_ylabel("Historical Variance")
        ax.set_title(f"Best Non-Linear Interaction vs Historical Variance\n"
                     f"Spearman ρ={best_hvar_inter['hvar_r']:.3f} (p={best_hvar_inter['hvar_p']:.4e})")
        ax.legend()
        plt.tight_layout()
        fig.savefig(figures_dir / "interaction_best_hvar.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ============================================================
    # PLOTS
    # ============================================================
    print(f"\nGenerating plots to {figures_dir}/...")

    # 1. Correlation bar chart for historical variance
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_correlation_bars(corr_hvar[corr_hvar["feature"].isin(transform_features)], "Historical Variance", ax)
    plt.tight_layout()
    fig.savefig(figures_dir / "correlation_bars_hvar.png", dpi=150)
    plt.close(fig)

    # 2. Top scatter plots vs historical variance
    top_features = corr_hvar["feature"].head(6).tolist()
    top_features = [f for f in top_features if f in df.columns][:6]
    if top_features:
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        for i, feat in enumerate(top_features):
            ax = axes[i // 3, i % 3]
            plot_scatter(df, feat, "historical_variance", ax)
        for j in range(len(top_features), 6):
            axes[j // 3, j % 3].set_visible(False)
        fig.suptitle("Top Features vs Historical Variance", fontsize=12)
        plt.tight_layout()
        fig.savefig(figures_dir / "scatter_top_hvar.png", dpi=150)
        plt.close(fig)

    # 3. MATH500 scatter plots (Wang examples only)
    if len(wang_df) >= 5:
        math_top = corr_math["feature"].head(4).tolist()
        math_top = [f for f in math_top if f in wang_df.columns][:4]
        if math_top:
            fig, axes = plt.subplots(1, len(math_top), figsize=(4 * len(math_top), 4))
            if len(math_top) == 1:
                axes = [axes]
            for i, feat in enumerate(math_top):
                plot_scatter(wang_df, feat, "math500_score", axes[i])
            fig.suptitle("Top Features vs MATH500 Score (Wang examples)", fontsize=12)
            plt.tight_layout()
            fig.savefig(figures_dir / "scatter_top_math500.png", dpi=150)
            plt.close(fig)

    # 4. LOO stability plot
    if loo_results:
        fig, ax = plt.subplots(figsize=(8, 4))
        feats = list(loo_results.keys())
        means = [loo_results[f]["mean_r"] for f in feats]
        stds = [loo_results[f]["std_r"] for f in feats]
        labels = [f.replace("_mean", "").replace("_", " ") for f in feats]
        ax.barh(range(len(feats)), means, xerr=stds, alpha=0.7, capsize=3)
        ax.set_yticks(range(len(feats)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Spearman r (LOO)", fontsize=9)
        ax.set_title("Leave-One-Out Stability of Top Features", fontsize=10)
        ax.axvline(0, color="black", linewidth=0.5)
        plt.tight_layout()
        fig.savefig(figures_dir / "loo_stability.png", dpi=150)
        plt.close(fig)

    # ============================================================
    # SAVE RESULTS
    # ============================================================
    output = {
        "primary_analysis": {
            "target": "historical_variance",
            "n": len(df),
            "correlations": corr_hvar.to_dict(orient="records"),
            "loo_stability": {k: {kk: float(vv) if not np.isnan(vv) else None for kk, vv in v.items()} for k, v in loo_results.items()},
        },
        "secondary_analysis": {
            "target": "math500_score",
            "n": len(wang_df),
            "correlations": corr_math.to_dict(orient="records"),
        },
    }

    with open(results_dir / "correlation_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved correlation_results.json to {results_dir}")

    # Save features CSV
    df.to_csv(results_dir / "features_full.csv", index=False)
    print(f"Saved features_full.csv to {results_dir}")

    print("\nDone!")


if __name__ == "__main__":
    main()
