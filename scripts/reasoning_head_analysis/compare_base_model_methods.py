#!/usr/bin/env python3
"""Compare head identification methods on the base model."""
import torch
from scipy.stats import spearmanr
import itertools
from collections import Counter

SEP = "=" * 70

methods = {
    "EAP-IG": "results/reasoning_head_analysis/identification/eap_ig/base_model/head_importance.pt",
    "math_vs_reading": "results/reasoning_head_analysis/identification/neurosurgery_math_vs_reading/base_model/head_importance.pt",
    "reasoning_vs_easy": "results/reasoning_head_analysis/identification/neurosurgery_reasoning_vs_easy/base_model/head_importance.pt",
    "cot_vs_direct": "results/reasoning_head_analysis/identification/neurosurgery_cot_vs_direct/base_model/head_importance.pt",
}


def top_k(scores, k):
    flat = scores.flatten()
    n_h = scores.shape[1]
    idx = flat.argsort(descending=True)[:k]
    return [(ii.item() // n_h, ii.item() % n_h) for ii in idx]


def top_k_set(scores, k):
    return set(top_k(scores, k))


data = {}
for name, path in methods.items():
    d = torch.load(path, map_location="cpu", weights_only=True)
    data[name] = d
    scores = d["head_scores"]
    n_layers, n_heads = scores.shape
    t20 = top_k(scores, 20)
    active = [tuple(h) for h in d.get("active_heads", [])]
    print(f"\n{SEP}")
    print(f"  {name}")
    cfg = d.get("config", {})
    print(f"  method={cfg.get('method','?')}, contrast={cfg.get('contrast','N/A')}")
    print(SEP)
    print(f"  Top 8:  {t20[:8]}")
    print(f"  Top 20: {t20}")
    if active:
        print(f"  Active ({len(active)}): {active}")

# Pairwise comparisons
names = list(methods.keys())
print(f"\n\n{SEP}")
print("  PAIRWISE COMPARISONS (base model)")
print(SEP)

for i, j in itertools.combinations(range(len(names)), 2):
    n1, n2 = names[i], names[j]
    s1, s2 = data[n1]["head_scores"], data[n2]["head_scores"]

    top8_1, top8_2 = top_k_set(s1, 8), top_k_set(s2, 8)
    top20_1, top20_2 = top_k_set(s1, 20), top_k_set(s2, 20)
    overlap8 = top8_1 & top8_2
    overlap20 = top20_1 & top20_2

    rho, p = spearmanr(s1.flatten().numpy(), s2.flatten().numpy())

    print(f"\n  {n1} vs {n2}:")
    print(f"    Spearman rho = {rho:.4f} (p={p:.2e})")
    print(f"    Top-8 overlap:  {len(overlap8)}/8  {sorted(overlap8) if overlap8 else '[]'}")
    print(f"    Top-20 overlap: {len(overlap20)}/20 {sorted(overlap20) if overlap20 else '[]'}")

# Universal heads
print(f"\n\n{SEP}")
print("  HEADS ACROSS METHODS (top-20)")
print(SEP)

all_top20 = {n: top_k_set(data[n]["head_scores"], 20) for n in names}

universal = set.intersection(*all_top20.values())
print(f"  In ALL 4 methods: {len(universal)} heads: {sorted(universal)}")

head_freq = Counter()
for s in all_top20.values():
    for h in s:
        head_freq[h] += 1

in_3_plus = sorted(h for h, c in head_freq.items() if c >= 3)
print(f"  In >=3 methods:   {len(in_3_plus)} heads:")
for h in in_3_plus:
    which = [n for n in names if h in all_top20[n]]
    print(f"    L{h[0]}H{h[1]}: {which}")

in_2_plus = sorted(h for h, c in head_freq.items() if c >= 2)
print(f"  In >=2 methods:   {len(in_2_plus)} heads:")
for h in in_2_plus:
    which = [n for n in names if h in all_top20[n]]
    print(f"    L{h[0]}H{h[1]}: {which}")

# Layer distribution
print(f"\n\n{SEP}")
print("  LAYER DISTRIBUTION OF TOP-20 HEADS")
print(SEP)
for name in names:
    scores = data[name]["head_scores"]
    t20 = top_k(scores, 20)
    layers = [h[0] for h in t20]
    layer_range = f"L{min(layers)}-L{max(layers)}"
    early = sum(1 for l in layers if l < 7)
    mid = sum(1 for l in layers if 7 <= l < 21)
    late = sum(1 for l in layers if l >= 21)
    print(f"  {name:25s}: {layer_range:8s}  early(0-6)={early}  mid(7-20)={mid}  late(21-27)={late}")
