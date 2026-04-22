#!/usr/bin/env python3
"""Analyze attention patterns of specific (layer, head) positions during CoT.

For each target head, runs a forward pass over (prompt + generated CoT) with
`output_attentions=True`, extracts the attention matrix for that head, and
computes:

  - mean_distance    : average |q - k| weighted by attention weights
                        (small = local head, large = long-range retrieval)
  - entropy          : per-query-position entropy of attention distribution
                        (low = focused on a specific key, high = diffuse)
  - bos_weight       : how much probability mass is absorbed by position 0
                        (high = 'attention sink' head, doing nothing useful)
  - top_attended     : per query position, the top-3 most-attended tokens
  - pattern_class    : heuristic label — local / bos_sink / retrieval /
                        aggregation / unclassified

Saves per-head metrics JSON plus one attention heatmap PNG per head per problem.

Usage:
  # Analyze explicit heads on one model:
  python -m reasoning_head_analysis.analyze_attention_patterns \\
      --model Qwen/Qwen2.5-Math-1.5B \\
      --heads 11.8,15.7,18.8,19.6,19.11,23.9 \\
      --output reasoning_head_analysis/results/attn_base_overlap

  # Use 'active_heads' from a head_importance.pt (paper's circuit set):
  python -m reasoning_head_analysis.analyze_attention_patterns \\
      --model <checkpoint_path> \\
      --importance_path .../head_importance.pt \\
      --exclude_layer0 \\
      --output reasoning_head_analysis/results/attn_ckpt700
"""
import argparse
import json
import math
import os
import time

import matplotlib
matplotlib.use("Agg")
# Disable mathtext parsing so tick labels with '$' (math problems) don't error.
matplotlib.rcParams["text.usetex"] = False
matplotlib.rcParams["text.parse_math"] = False
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


# Same system + reasoning prefix we use for EAP-IG, so results are comparable.
REASONING_PREFIX = "Let me solve this step by step.\nFirst,"

DEFAULT_QUESTIONS = [
    "Find the positive integer n such that 1 + 2 + 3 + ... + n = 210.",
    "A rectangle has area 48 and perimeter 28. What are its dimensions?",
    "If f(x) = 2x + 3, what is f(f(5))?",
]


def parse_heads(spec):
    """Parse 'L.H<sep>L.H<sep>...' into [(l, h), ...].

    Accepts any of ',', '|', ';', whitespace as separators. Using '|' avoids
    sbatch --export consuming commas as variable separators.
    """
    import re as _re
    out = []
    for tok in _re.split(r"[,|;\s]+", spec.strip()):
        if not tok:
            continue
        l, h = tok.split(".")
        out.append((int(l), int(h)))
    return out


def build_prompt(tokenizer, question, prefix=REASONING_PREFIX):
    msgs = [{"role": "user", "content": question}]
    chat = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return chat + prefix


def generate_cot(model, tokenizer, prompt, max_new_tokens, temperature, device):
    """Generate a continuation; return the full (prompt + generation) as token IDs."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=1.0,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    return out[0]  # [seq_len]


def run_attention_forward(model, tokenizer, full_ids, device):
    """Forward pass with output_attentions=True. Returns list of attention
    tensors, one per layer, each [n_heads, seq, seq] (batch removed)."""
    input_ids = full_ids.unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(input_ids, output_attentions=True, return_dict=True)
    # out.attentions is a tuple: layer -> [1, n_heads, seq, seq]
    return [a[0].float().cpu() for a in out.attentions]


def compute_metrics(attn_matrix, token_strs, top_k_tokens=3):
    """Given a single head's attention [seq, seq], compute summary metrics.
    Averaged over query positions in the *generated* region only (we pass all
    query positions here and let caller restrict if desired)."""
    seq = attn_matrix.shape[0]
    # Mean distance weighted by attention weights (absolute distance)
    positions = torch.arange(seq).float()
    distances_per_q = []
    entropies_per_q = []
    bos_weights_per_q = []
    top_attended_per_q = []

    for q in range(seq):
        w = attn_matrix[q]
        dist = (torch.abs(positions - q) * w).sum().item()
        ent = -(w * (w.clamp(min=1e-10)).log()).sum().item()
        bos = w[0].item()
        # Top-k attended keys
        topk = torch.topk(w, min(top_k_tokens, q + 1))
        tk = []
        for idx, wt in zip(topk.indices.tolist(), topk.values.tolist()):
            tok_str = token_strs[idx] if idx < len(token_strs) else "?"
            tk.append({"pos": idx, "token": tok_str, "w": round(wt, 3)})

        distances_per_q.append(dist)
        entropies_per_q.append(ent)
        bos_weights_per_q.append(bos)
        top_attended_per_q.append(tk)

    return {
        "distances": distances_per_q,
        "entropies": entropies_per_q,
        "bos_weights": bos_weights_per_q,
        "top_attended": top_attended_per_q,
    }


def classify_head(distances, entropies, bos_weights):
    """Heuristic pattern classifier. Uses the *mean* over query positions in
    the generated region (skips first 10% of prompt to avoid prompt-filling
    artifacts)."""
    n = len(distances)
    skip = max(1, n // 10)
    d = float(np.mean(distances[skip:]))
    e = float(np.mean(entropies[skip:]))
    b = float(np.mean(bos_weights[skip:]))

    # Classification heuristics — tuned for small models like Qwen2.5-Math-1.5B.
    if b > 0.5:
        label = "bos_sink"
    elif d < 3.0:
        label = "local"
    elif e < 2.0:
        label = "retrieval"
    elif e > 4.0:
        label = "aggregation"
    else:
        label = "mixed"
    return {"label": label, "mean_distance": d, "mean_entropy": e, "mean_bos_weight": b}


def plot_attention(attn_matrix, token_strs, head_label, question_idx, output_dir,
                   max_seq=200):
    """Save an attention heatmap (truncated to max_seq if needed)."""
    n = min(attn_matrix.shape[0], max_seq)
    m = attn_matrix[:n, :n].numpy()
    fig, ax = plt.subplots(figsize=(min(18, max(10, n * 0.08)),
                                    min(18, max(10, n * 0.08))))
    im = ax.imshow(m, cmap="viridis", aspect="auto",
                   norm=matplotlib.colors.LogNorm(vmin=max(1e-4, m[m > 0].min()),
                                                   vmax=m.max()))
    # Short token labels — escape $ (mathtext), newlines, spaces; truncate.
    def _safe(s):
        s = s.replace("\n", "⏎").replace(" ", "·").replace("$", "\\$")
        return s[:10]
    short_labels = [_safe(s) for s in token_strs[:n]]
    step = max(1, n // 40)
    ax.set_xticks(range(0, n, step))
    ax.set_yticks(range(0, n, step))
    ax.set_xticklabels([short_labels[i] for i in range(0, n, step)], rotation=90, fontsize=6)
    ax.set_yticklabels([short_labels[i] for i in range(0, n, step)], fontsize=6)
    ax.set_xlabel("Key position")
    ax.set_ylabel("Query position")
    ax.set_title(f"{head_label}  —  question {question_idx}")
    fig.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout()
    path = os.path.join(output_dir, f"{head_label}_q{question_idx}.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF id or local checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--heads", type=str, default=None,
                        help="Explicit head list 'L.H,L.H,...'")
    parser.add_argument("--importance_path", type=str, default=None,
                        help="Path to head_importance.pt; uses active_heads from it "
                             "if --heads not given")
    parser.add_argument("--exclude_layer0", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_plots", action="store_true",
                        help="Skip per-head heatmap PNGs (only save JSON)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # bfloat16 on CUDA: numerically stable for eager attention (fp16 softmax
    # can produce NaN/Inf, which crashes multinomial sampling).
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    # ─── Resolve target head list ──────────────────────────────────────
    if args.heads:
        target_heads = parse_heads(args.heads)
    elif args.importance_path:
        d = torch.load(args.importance_path, map_location="cpu", weights_only=True)
        active = d.get("active_heads") if isinstance(d, dict) else None
        if not active:
            raise ValueError(f"No 'active_heads' in {args.importance_path}; "
                             "pass --heads explicitly or re-run identify_heads.")
        target_heads = [tuple(h) for h in active]
    else:
        raise ValueError("Must pass --heads or --importance_path")

    if args.exclude_layer0:
        target_heads = [(l, h) for (l, h) in target_heads if l != 0]

    print(f"Target heads ({len(target_heads)}): {target_heads}")
    os.makedirs(args.output, exist_ok=True)

    # ─── Load model + tokenizer ────────────────────────────────────────
    print(f"Loading {args.model} on {device} ({dtype})...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, trust_remote_code=True,
        attn_implementation="eager",  # needed to get attention weights back
    ).to(device).eval()
    print(f"  loaded in {time.time() - t0:.1f}s")

    # ─── For each question: generate CoT, forward w/ attentions ────────
    results = []
    for q_idx, question in enumerate(DEFAULT_QUESTIONS):
        print(f"\n── Question {q_idx}: {question[:60]}... ──")
        prompt = build_prompt(tokenizer, question)

        t = time.time()
        full_ids = generate_cot(model, tokenizer, prompt, args.max_new_tokens,
                                args.temperature, device)
        full_text = tokenizer.decode(full_ids, skip_special_tokens=False)
        token_strs = [tokenizer.decode([int(t)]) for t in full_ids]
        print(f"  generated {len(full_ids)} tokens in {time.time() - t:.1f}s")

        t = time.time()
        attns = run_attention_forward(model, tokenizer, full_ids, device)
        print(f"  attention forward in {time.time() - t:.1f}s")

        # Restrict metric computation to the generated region (skip prompt)
        prompt_len = len(tokenizer(prompt)["input_ids"])
        for (l, h) in target_heads:
            if l >= len(attns) or h >= attns[l].shape[0]:
                print(f"  skipping L{l}H{h} (out of range)")
                continue
            attn_matrix = attns[l][h]  # [seq, seq]
            metrics = compute_metrics(attn_matrix, token_strs)

            # Summary on generated region only
            gen_dist = metrics["distances"][prompt_len:]
            gen_ent = metrics["entropies"][prompt_len:]
            gen_bos = metrics["bos_weights"][prompt_len:]
            if gen_dist:
                cls = classify_head(gen_dist, gen_ent, gen_bos)
            else:
                cls = {"label": "unknown", "mean_distance": 0,
                       "mean_entropy": 0, "mean_bos_weight": 0}

            head_label = f"L{l}H{h}"
            if not args.no_plots:
                plot_attention(attn_matrix, token_strs, head_label, q_idx,
                               args.output)

            # Store compact metrics (don't dump the full attn matrix to JSON)
            results.append({
                "model": args.model,
                "question_idx": q_idx,
                "question": question,
                "head_layer": l,
                "head_index": h,
                "prompt_len": prompt_len,
                "seq_len": len(full_ids),
                "summary": cls,
                "top_attended_gen": metrics["top_attended"][prompt_len::max(1, (len(full_ids)-prompt_len)//10)][:10],
            })

    # ─── Save summary ──────────────────────────────────────────────────
    summary_path = os.path.join(args.output, "summary.json")
    with open(summary_path, "w") as f:
        json.dump({"model": args.model, "heads_analyzed": target_heads,
                   "results": results}, f, indent=2)
    print(f"\nSaved summary to {summary_path}")

    # ─── Print overview ────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print(f"CLASSIFICATION OVERVIEW — {args.model}")
    print("=" * 78)
    # Per-head: aggregate labels across questions
    by_head = {}
    for r in results:
        key = (r["head_layer"], r["head_index"])
        by_head.setdefault(key, []).append(r["summary"])
    for (l, h), sums in sorted(by_head.items()):
        labels = [s["label"] for s in sums]
        d = np.mean([s["mean_distance"] for s in sums])
        e = np.mean([s["mean_entropy"] for s in sums])
        b = np.mean([s["mean_bos_weight"] for s in sums])
        print(f"  L{l}H{h}:  {labels}   dist={d:.1f}  ent={e:.2f}  bos={b:.2f}")


if __name__ == "__main__":
    main()
