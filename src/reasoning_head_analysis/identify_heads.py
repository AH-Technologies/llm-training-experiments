#!/usr/bin/env python3
"""Identify reasoning heads via EAP-IG circuit discovery.

Follows the Thinking Sparks (Park et al., 2025) methodology:
1. Build clean/corrupted prompt pairs (reasoning vs direct prefix)
2. Run EAP-IG attribution across all attention heads
3. Rank heads by importance and save results + heatmap

Supports GPU (fast) and CPU (slower, for when no GPUs available).

Requires: transformer_lens, eap (from hannamw/eap-ig), datasets, seaborn

Usage:
  python -m reasoning_head_analysis.identify_heads --model Qwen/Qwen2.5-1.5B-Instruct
  python -m reasoning_head_analysis.identify_heads --model path/to/checkpoint --device cpu --n_pairs 50
"""
import argparse
import logging
import os
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoModelForCausalLM
from transformer_lens import HookedTransformer
from transformer_lens.loading_from_pretrained import OFFICIAL_MODEL_NAMES
from eap.graph import Graph
from eap.attribute import attribute

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Prompt construction (Thinking Sparks A.2)
# ═══════════════════════════════════════════════════════════════════════

# System prompt matches Qwen2.5-Math's tokenizer-default chat template — this
# is the distribution the base and GRPO checkpoints were trained on.
SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."

# Length-matched (clean, corrupted) assistant-prefix pairs. The clean prefix
# steers the model into CoT mode; the corrupted prefix steers it into
# direct-answer mode. Chosen empirically via the prefix-steering diagnostic
# (reasoning_head_analysis/test_reasoning_prefix.py): all three pairs produce
# 8–17 nats of KL divergence at the prefix boundary on both base and
# GRPO-trained Qwen2.5-Math-1.5B.
PREFIX_PAIRS = [
    # 6 tokens — strongest signal (~17 nats)
    ("To solve this problem, we",        "The answer is \\boxed{"),
    # 7 tokens
    ("Let's think step by step.",        "The final answer is \\boxed{"),
    # 8 tokens
    ("Let me solve this step by step.",  "The answer to this is \\boxed{"),
]


# Map from model architectures to known transformer_lens names
ARCH_TO_TL_NAME = {
    "Qwen2ForCausalLM": "Qwen/Qwen2.5-1.5B",
    "Qwen2_5_ForCausalLM": "Qwen/Qwen2.5-1.5B",
}


def load_hooked_model(model_name, device="cpu", dtype=torch.float32):
    """Load a model into HookedTransformer, handling local checkpoints.

    If model_name is in transformer_lens's registry, load directly.
    Otherwise, load via HuggingFace and pass as hf_model, using the
    architecture to find a compatible transformer_lens name.
    """
    # Check if it's a known transformer_lens model
    known_names = set(OFFICIAL_MODEL_NAMES)
    if model_name in known_names:
        logger.info(f"Loading {model_name} directly via transformer_lens")
        return HookedTransformer.from_pretrained(model_name, device=device, dtype=dtype)

    # Local checkpoint or unknown HF model — load via AutoModel
    logger.info(f"Loading {model_name} via HuggingFace AutoModel...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, trust_remote_code=True,
    )
    arch = type(hf_model).__name__
    logger.info(f"Architecture: {arch}")

    # Find a matching transformer_lens template
    tl_name = ARCH_TO_TL_NAME.get(arch)
    if tl_name is None:
        # Try to match by model config
        n_layers = hf_model.config.num_hidden_layers
        n_heads = hf_model.config.num_attention_heads
        hidden = hf_model.config.hidden_size
        logger.warning(f"Unknown arch {arch} ({n_layers}L/{n_heads}H/{hidden}D), "
                       f"trying Qwen/Qwen2.5-1.5B as template")
        tl_name = "Qwen/Qwen2.5-1.5B"

    logger.info(f"Using transformer_lens template: {tl_name}")
    model = HookedTransformer.from_pretrained(
        tl_name, hf_model=hf_model, device=device, dtype=dtype,
    )
    return model


def make_chat_prompt(question):
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def find_matched_prefixes(model):
    """Return the curated (clean, corrupted) prefix pairs, verifying equal
    token length under this model's tokenizer."""
    matched = []
    for clean, corrupt in PREFIX_PAIRS:
        cl = model.to_tokens(clean, prepend_bos=False).shape[1]
        dl = model.to_tokens(corrupt, prepend_bos=False).shape[1]
        if cl == dl:
            matched.append((clean, corrupt))
        else:
            logger.warning(
                f"Dropping length-mismatched pair ({cl} vs {dl} tokens): "
                f"{clean!r} / {corrupt!r}"
            )
    return matched


def build_pairs(model, n_pairs=300, dataset="aime"):
    """Build clean/corrupted prompt pairs from math questions.

    Args:
        dataset: "aime" (AI-MO/aimo-validation-aime, as in Thinking Sparks)
                 or "gsm8k" (openai/gsm8k)
    """
    matched = find_matched_prefixes(model)
    assert matched, "No matched-length prefix pairs found!"
    logger.info(f"Found {len(matched)} matched prefix pairs")

    if dataset == "aime":
        logger.info("Loading AIME (AI-MO/aimo-validation-aime)...")
        ds = load_dataset("AI-MO/aimo-validation-aime", split="train")
        questions = [ds[i]["problem"] for i in range(len(ds))]
    else:
        logger.info("Loading GSM8K...")
        ds = load_dataset("openai/gsm8k", "main", split="train")
        indices = list(range(len(ds)))
        random.shuffle(indices)
        questions = [ds[i]["question"] for i in indices[:500]]

    logger.info(f"Loaded {len(questions)} questions")

    pairs = []
    for q in questions:
        if len(pairs) >= n_pairs:
            break
        r, d = random.choice(matched)
        chat = make_chat_prompt(q)
        clean, corrupt = chat + r, chat + d
        if model.to_tokens(clean).shape[1] != model.to_tokens(corrupt).shape[1]:
            continue
        pairs.append((clean, corrupt, 0))

    logger.info(f"Built {len(pairs)} prompt pairs")
    return pairs


# ═══════════════════════════════════════════════════════════════════════
# EAP-IG attribution
# ═══════════════════════════════════════════════════════════════════════

class EAPDataset(Dataset):
    def __init__(self, pairs):
        self.data = pairs
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    return [b[0] for b in batch], [b[1] for b in batch], [b[2] for b in batch]


def kl_divergence_metric(logits, clean_logits, input_lengths, labels):
    """KL(clean || model) at the last non-padded position."""
    batch_size = logits.shape[0]
    kl_total = torch.tensor(0.0, device=logits.device, dtype=torch.float32)
    for i in range(batch_size):
        last_pos = input_lengths[i] - 1
        clean_lp = F.log_softmax(clean_logits[i, last_pos].float(), dim=-1)
        model_lp = F.log_softmax(logits[i, last_pos].float(), dim=-1)
        kl = (clean_lp.exp() * (clean_lp - model_lp)).sum()
        kl_total = kl_total + kl
    return kl_total / batch_size


def worker(rank, n_gpus, model_name, all_pairs, ig_steps, result_dict, device_override=None):
    """Run EAP-IG on a shard of the data."""
    if device_override == "cpu":
        device = "cpu"
        dtype = torch.float32
    else:
        device = f"cuda:{rank}"
        torch.cuda.set_device(device)
        dtype = torch.float16

    shard_size = len(all_pairs) // n_gpus
    start = rank * shard_size
    end = start + shard_size if rank < n_gpus - 1 else len(all_pairs)
    shard = all_pairs[start:end]
    logger.info(f"[{device}] Processing pairs {start}-{end} ({len(shard)} pairs)")

    logger.info(f"[{device}] Loading model ({dtype})...")
    t_load = time.time()
    model = load_hooked_model(model_name, device=device, dtype=dtype)
    model.cfg.use_attn_result = True
    model.cfg.use_split_qkv_input = True
    model.cfg.use_hook_mlp_in = True
    model.cfg.ungroup_grouped_query_attention = True
    model.setup()
    logger.info(f"[{device}] Model loaded in {time.time() - t_load:.1f}s")

    graph = Graph.from_model(model)
    dataloader = DataLoader(EAPDataset(shard), batch_size=1, shuffle=False, collate_fn=collate_fn)

    logger.info(f"[{device}] Starting EAP-IG: {len(shard)} pairs x {ig_steps} IG steps")
    t0 = time.time()
    attribute(
        model=model, graph=graph, dataloader=dataloader,
        metric=kl_divergence_metric, method="EAP-IG-inputs",
        ig_steps=ig_steps, quiet=False,
    )
    elapsed = time.time() - t0
    logger.info(f"[{device}] EAP-IG done in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    result_dict[rank] = graph.scores.cpu() * len(shard)


# ═══════════════════════════════════════════════════════════════════════
# Head scoring and visualization
# ═══════════════════════════════════════════════════════════════════════

def extract_head_importance(graph, n_layers, n_heads):
    """Compute per-head importance from EAP-IG edge scores."""
    head_scores = torch.zeros(n_layers, n_heads)
    for layer in range(n_layers):
        for head in range(n_heads):
            node = graph.nodes[f"a{layer}.h{head}"]
            fwd_idx = graph.forward_index(node, attn_slice=False)
            outgoing = graph.scores[fwd_idx, :].abs().sum().item()
            incoming = 0
            for letter in "qkv":
                bwd_idx = graph.backward_index(node, qkv=letter, attn_slice=False)
                incoming += graph.scores[:, bwd_idx].abs().sum().item()
            head_scores[layer, head] = outgoing + incoming
    return head_scores


def plot_heatmap(head_scores, output_path):
    """Save head importance heatmap."""
    n_layers, n_heads = head_scores.shape
    try:
        import seaborn as sns
        fig, ax = plt.subplots(figsize=(max(10, n_heads * 0.6), max(6, n_layers * 0.3)))
        sns.heatmap(
            head_scores.numpy(), ax=ax,
            xticklabels=[f"H{h}" for h in range(n_heads)],
            yticklabels=[f"L{l}" for l in range(n_layers)],
            cmap="Reds",
        )
    except ImportError:
        fig, ax = plt.subplots(figsize=(max(10, n_heads * 0.6), max(6, n_layers * 0.3)))
        im = ax.imshow(head_scores.numpy(), cmap="Reds", aspect="auto")
        ax.set_xticks(range(n_heads), [f"H{h}" for h in range(n_heads)])
        ax.set_yticks(range(n_layers), [f"L{l}" for l in range(n_layers)])
        fig.colorbar(im)

    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_title("EAP-IG: Attention Head Importance (reasoning vs direct)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved heatmap to {output_path}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Identify reasoning heads via EAP-IG")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--n_pairs", type=int, default=300, help="Number of prompt pairs")
    parser.add_argument("--ig_steps", type=int, default=100, help="Integrated gradients steps")
    parser.add_argument("--top_n", type=int, default=5000, help="Top-n edges to keep")
    parser.add_argument("--threshold", type=float, default=0.1,
                        help="Edge-score threshold tau for circuit simplification "
                             "(Thinking Sparks A.2). Applied after top-n, with isolated-node pruning.")
    parser.add_argument("--dataset", type=str, default="aime", choices=["aime", "gsm8k"],
                        help="Dataset for prompt pairs: 'aime' (default, as in paper) or 'gsm8k'")
    parser.add_argument("--device", type=str, default=None,
                        help="Force device: 'cpu' or 'cuda' (default: auto-detect)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: reasoning_head_analysis/results/<model>)")
    args = parser.parse_args()

    random.seed(42)
    torch.manual_seed(42)

    # Determine device
    if args.device:
        use_cpu = args.device == "cpu"
    else:
        use_cpu = not torch.cuda.is_available()

    n_gpus = 0 if use_cpu else torch.cuda.device_count()
    device_str = "cpu" if use_cpu else f"{n_gpus} GPU(s)"

    safe_model = args.model.replace("/", "_").replace(".", "_")
    output_dir = args.output_dir or os.path.join("reasoning_head_analysis", "results", safe_model)
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Model: {args.model}")
    logger.info(f"Device: {device_str}")
    logger.info(f"Config: {args.n_pairs} pairs, {args.ig_steps} IG steps, top-{args.top_n} edges")
    logger.info(f"Output: {output_dir}")

    if use_cpu:
        n_threads = torch.get_num_threads()
        logger.info(f"CPU threads: {n_threads}")

    # Build dataset on CPU
    logger.info("Building prompt pairs...")
    t_build = time.time()
    model = load_hooked_model(args.model, device="cpu", dtype=torch.float32)
    all_pairs = build_pairs(model, n_pairs=args.n_pairs, dataset=args.dataset)
    del model
    logger.info(f"Pairs built in {time.time() - t_build:.1f}s")

    # Run EAP-IG
    logger.info(f"Running EAP-IG ({len(all_pairs)} pairs x {args.ig_steps} steps on {device_str})...")
    t0 = time.time()

    if use_cpu:
        result_dict = {}
        worker(0, 1, args.model, all_pairs, args.ig_steps, result_dict, device_override="cpu")
    elif n_gpus > 1:
        result_dict = mp.Manager().dict()
        mp.spawn(worker, args=(n_gpus, args.model, all_pairs, args.ig_steps, result_dict, None),
                 nprocs=n_gpus, join=True)
    else:
        result_dict = {}
        worker(0, 1, args.model, all_pairs, args.ig_steps, result_dict)

    elapsed = time.time() - t0
    logger.info(f"EAP-IG completed in {elapsed / 60:.1f} min")

    # Aggregate scores
    logger.info("Aggregating scores...")
    n_workers = 1 if use_cpu else n_gpus
    agg_scores = sum(result_dict[r] for r in range(n_workers)) / len(all_pairs)

    logger.info("Loading model for graph construction...")
    load_dtype = torch.float32 if use_cpu else torch.float16
    model = load_hooked_model(args.model, device="cpu", dtype=load_dtype)
    graph = Graph.from_model(model)
    graph.scores[:] = agg_scores

    logger.info(f"Applying top-{args.top_n} edge selection...")
    graph.apply_topn(n=args.top_n)
    logger.info(f"Edges in circuit: {graph.count_included_edges()}")

    # Extract head importance
    n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads
    logger.info(f"Extracting head importance ({n_layers} layers x {n_heads} heads)...")
    head_scores = extract_head_importance(graph, n_layers, n_heads)

    # Rank heads
    flat = head_scores.flatten()
    sorted_idx = flat.argsort(descending=True)
    logger.info("Top 20 reasoning heads:")
    for rank, idx in enumerate(sorted_idx[:20]):
        l, h = idx.item() // n_heads, idx.item() % n_heads
        score = flat[idx].item()
        logger.info(f"  #{rank+1}: L{l}H{h} = {score:.4f}")

    # Save full top-n circuit first (scores for all edges preserved, so threshold
    # can be re-applied offline from circuit.json).
    graph.to_json(os.path.join(output_dir, "circuit.json"))

    # Paper pipeline (Thinking Sparks A.2): simplify with threshold tau and prune
    # isolated nodes. This gives the set of "emergent" attention heads.
    logger.info(f"Applying threshold tau={args.threshold} with isolated-node pruning...")
    graph.apply_threshold(args.threshold, absolute=True, reset=True,
                          level="edge", prune=True)
    active_heads = sorted(
        (l, h) for l in range(n_layers) for h in range(n_heads)
        if graph.nodes[f"a{l}.h{h}"].in_graph
    )
    logger.info(f"Active heads after threshold+prune: {len(active_heads)}")

    importance_path = os.path.join(output_dir, "head_importance.pt")
    torch.save({
        "head_scores": head_scores,
        "active_heads": active_heads,
        "config": {
            "model": args.model,
            "method": "EAP-IG-inputs",
            "ig_steps": args.ig_steps,
            "top_n": args.top_n,
            "threshold": args.threshold,
            "n_pairs": len(all_pairs),
            "device": device_str,
            "elapsed_minutes": elapsed / 60,
        },
    }, importance_path)
    logger.info(f"Saved head importance to {importance_path}")

    # Heatmap
    plot_heatmap(head_scores, os.path.join(output_dir, "head_importance_heatmap.png"))

    logger.info(f"Total time: {(time.time() - t0) / 60:.1f} min")
    logger.info("Done.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
