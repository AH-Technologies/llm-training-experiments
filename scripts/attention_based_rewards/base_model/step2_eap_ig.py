"""Step 2: EAP-IG circuit discovery on Qwen2.5-Math-1.5B (base model).

Multi-GPU: each GPU processes a shard of pairs, scores are aggregated.
Compares results with Instruct model heads from reasoning_heads.pt.

EAP-IG config (same as Instruct analysis):
  method: EAP-IG-inputs, ig_steps=100, top_n=5000, threshold tau=0.1
  metric: KL divergence
"""

import random
import time
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM
from transformer_lens import HookedTransformer
from eap.graph import Graph
from eap.attribute import attribute

BASE_DIR = "/cluster/projects/nn12068k/haaklau/llm-training-experiments/attention_based_rewards"
MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B"
TL_MODEL_NAME = "Qwen/Qwen2.5-1.5B"  # TL architecture name (Math variant loaded via hf_model)
IG_STEPS = 100
TOP_N = 5000


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


def worker(rank, n_gpus, all_pairs, result_dict):
    """Run EAP-IG on a shard of the data on GPU `rank`."""
    device = f"cuda:{rank}"
    torch.cuda.set_device(device)

    shard_size = len(all_pairs) // n_gpus
    start = rank * shard_size
    end = start + shard_size if rank < n_gpus - 1 else len(all_pairs)
    shard = all_pairs[start:end]
    print(f"[GPU {rank}] Processing pairs {start}-{end} ({len(shard)} pairs)", flush=True)

    hf_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
    model = HookedTransformer.from_pretrained(
        TL_MODEL_NAME, hf_model=hf_model, device=device, dtype=torch.float16,
    )
    del hf_model
    model.cfg.use_attn_result = True
    model.cfg.use_split_qkv_input = True
    model.cfg.use_hook_mlp_in = True
    model.cfg.ungroup_grouped_query_attention = True
    model.setup()

    graph = Graph.from_model(model)
    dataset = EAPDataset(shard)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    t0 = time.time()
    attribute(
        model=model,
        graph=graph,
        dataloader=dataloader,
        metric=kl_divergence_metric,
        method="EAP-IG-inputs",
        ig_steps=IG_STEPS,
        quiet=(rank != 0),
    )
    elapsed = time.time() - t0
    print(f"[GPU {rank}] Done in {elapsed/60:.1f} min", flush=True)

    result_dict[rank] = graph.scores.cpu() * len(shard)


def main():
    random.seed(42)
    torch.manual_seed(42)
    n_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {n_gpus}")

    # Build matched-length pairs for EAP-IG.
    # EAP-IG requires clean and corrupt to have the SAME token length.
    # Since our clean suffix ("Let's solve this step by step.\nStep 1:") is
    # much longer than alt3 ("\n"), we build pairs here with suffix matching,
    # similar to the original Instruct step2's find_matched_prefixes approach.
    from transformers import AutoTokenizer
    from datasets import load_dataset

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Use suffixes that are closer in length for easier matching
    clean_suffixes = [
        "\nLet's solve this step by step.\nStep 1:",
        "\nLet me work through this step by step.",
        "\nI'll solve this step by step. First,",
        "\nLet's think about this carefully.",
        "\nTo solve this, let me think step by step.",
    ]
    corrupt_suffixes = [
        "\nThe answer is:",
        "\nThe answer to this problem is simply:",
        "\nI'll write the final answer directly now:",
        "\nThe result of this calculation is:",
        "\nDirectly computing the answer gives us:",
    ]

    # Find suffix pairs that produce the same token length
    matched_suffix_pairs = []
    for cs in clean_suffixes:
        cs_len = len(tokenizer.encode(cs)) - 1  # subtract BOS
        for ds in corrupt_suffixes:
            ds_len = len(tokenizer.encode(ds)) - 1
            if cs_len == ds_len:
                matched_suffix_pairs.append((cs, ds))
            elif abs(cs_len - ds_len) <= 3:
                # Try padding shorter one with periods
                short, long_len = (cs, ds_len) if cs_len < ds_len else (ds, cs_len)
                padded = short
                for _ in range(20):
                    padded += "."
                    pl = len(tokenizer.encode(padded)) - 1
                    if pl == long_len:
                        if cs_len < ds_len:
                            matched_suffix_pairs.append((padded, ds))
                        else:
                            matched_suffix_pairs.append((cs, padded))
                        break
                    elif pl > long_len:
                        break

    print(f"  Found {len(matched_suffix_pairs)} matched suffix pairs")
    for cs, ds in matched_suffix_pairs[:5]:
        print(f"    clean: '{cs.strip()[:50]}' | corrupt: '{ds.strip()[:50]}'")

    if not matched_suffix_pairs:
        raise RuntimeError("No matched suffix pairs found! Adjust suffix lists.")

    # Load GSM8K questions (same as step 1)
    print("  Loading GSM8K questions...")
    gsm = load_dataset("openai/gsm8k", "main", split="train")
    indices = list(range(len(gsm)))
    random.seed(42)
    random.shuffle(indices)
    questions = [gsm[i]["question"] for i in indices[:500]]

    N_PAIRS = 300
    all_pairs = []
    skipped = 0
    for q in questions:
        if len(all_pairs) >= N_PAIRS:
            break
        cs, ds = random.choice(matched_suffix_pairs)
        clean_text = f"Question: {q}{cs}"
        corrupt_text = f"Question: {q}{ds}"
        cl = len(tokenizer.encode(clean_text))
        dl = len(tokenizer.encode(corrupt_text))
        if cl != dl:
            skipped += 1
            continue
        if cl > 512:
            skipped += 1
            continue
        all_pairs.append((clean_text, corrupt_text, 0))

    print(f"  {len(all_pairs)} pairs ready ({skipped} skipped)")
    del tokenizer

    # Run EAP-IG
    print(f"\nLaunching EAP-IG on {n_gpus} GPU(s)...")
    print(f"  ig_steps={IG_STEPS}, pairs={len(all_pairs)}")
    print(f"  ~{len(all_pairs) // max(n_gpus, 1)} pairs per GPU")

    t0 = time.time()
    if n_gpus > 1:
        result_dict = mp.Manager().dict()
        mp.spawn(worker, args=(n_gpus, all_pairs, result_dict), nprocs=n_gpus, join=True)
    else:
        result_dict = {}
        worker(0, 1, all_pairs, result_dict)

    total_elapsed = time.time() - t0
    print(f"\nAll GPUs done in {total_elapsed/60:.1f} min")

    # Aggregate scores
    print("Aggregating scores...")
    agg_scores = sum(result_dict[r] for r in range(n_gpus)) / len(all_pairs)

    # Apply to fresh graph
    hf_model_agg = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
    model = HookedTransformer.from_pretrained(TL_MODEL_NAME, hf_model=hf_model_agg, device="cpu", dtype=torch.float16)
    del hf_model_agg
    graph = Graph.from_model(model)
    graph.scores[:] = agg_scores

    print(f"\nApplying top-{TOP_N} edge selection...")
    graph.apply_topn(n=TOP_N)
    print(f"Edges in circuit: {graph.count_included_edges()}")
    print(f"Nodes in circuit: {graph.count_included_nodes()}")

    # Extract per-head importance
    print("\nExtracting per-head importance scores...")
    n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads
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

    # Rank and print top heads
    flat = head_scores.flatten()
    sorted_idx = flat.argsort(descending=True)
    print(f"\nHead scores range: [{head_scores.min():.4f}, {head_scores.max():.4f}]")

    selected_heads = []
    print("\nTop 20 attention heads (BASE model):")
    for rank, idx in enumerate(sorted_idx[:20]):
        l, h = idx.item() // n_heads, idx.item() % n_heads
        score = flat[idx].item()
        in_circ = graph.nodes[f"a{l}.h{h}"].in_graph
        selected_heads.append((l, h, score))
        print(f"  {'*' if in_circ else ' '} #{rank+1}: L{l}H{h} = {score:.4f}")

    # ── Cross-model comparison with Instruct model ────────────────────
    print(f"\n{'='*60}")
    print("CROSS-MODEL COMPARISON: Base vs Instruct")
    print(f"{'='*60}")

    instruct_path = f"{BASE_DIR}/results/reasoning_heads.pt"
    try:
        instruct_data = torch.load(instruct_path, weights_only=False)
        instruct_scores = instruct_data["head_scores"]
        instruct_selected = instruct_data["selected_heads"][:20]

        # Build lookup tables
        base_rank_map = {}
        for r, idx in enumerate(sorted_idx[:336]):
            l, h = idx.item() // n_heads, idx.item() % n_heads
            base_rank_map[(l, h)] = (r + 1, flat[idx].item())

        instruct_flat = instruct_scores.flatten()
        instruct_sorted = instruct_flat.argsort(descending=True)
        instruct_rank_map = {}
        for r, idx in enumerate(instruct_sorted[:336]):
            l, h = idx.item() // n_heads, idx.item() % n_heads
            instruct_rank_map[(l, h)] = (r + 1, instruct_flat[idx].item())

        print(f"\n{'Head':<10} {'Inst Rank':>10} {'Inst Score':>12} {'Base Rank':>10} {'Base Score':>12}")
        print("-" * 60)

        instruct_top20 = set()
        base_top20 = set()
        for l, h, _ in instruct_selected[:20]:
            instruct_top20.add((l, h))
        for l, h, _ in selected_heads[:20]:
            base_top20.add((l, h))

        # Show all heads that are in either top-20
        all_important = instruct_top20 | base_top20
        rows = []
        for (l, h) in sorted(all_important):
            i_rank, i_score = instruct_rank_map.get((l, h), (">20", 0))
            b_rank, b_score = base_rank_map.get((l, h), (">20", 0))
            marker = ""
            if (l, h) in instruct_top20 and (l, h) in base_top20:
                marker = " <-- BOTH"
            rows.append((l, h, i_rank, i_score, b_rank, b_score, marker))

        for l, h, ir, isc, br, bsc, marker in rows:
            ir_str = f"#{ir}" if isinstance(ir, int) else ir
            br_str = f"#{br}" if isinstance(br, int) else br
            print(f"L{l}H{h:<6} {ir_str:>10} {isc:>12.4f} {br_str:>10} {bsc:>12.4f}{marker}")

        overlap = instruct_top20 & base_top20
        print(f"\nTop-20 overlap: {len(overlap)}/20 heads match")
        if overlap:
            print(f"  Shared heads: {sorted(overlap)}")

        # Top-10 overlap
        instruct_top10 = {(l, h) for l, h, _ in instruct_selected[:10]}
        base_top10 = {(l, h) for l, h, _ in selected_heads[:10]}
        overlap10 = instruct_top10 & base_top10
        print(f"Top-10 overlap: {len(overlap10)}/10 heads match")
        if overlap10:
            print(f"  Shared heads: {sorted(overlap10)}")

    except FileNotFoundError:
        print(f"  Instruct model results not found at {instruct_path}")
        print("  Skipping cross-model comparison.")

    # ── Save results ──────────────────────────────────────────────────
    save_data = {
        "head_scores": head_scores,
        "selected_heads": selected_heads,
        "scores_matrix": graph.scores.cpu(),
        "in_graph": graph.in_graph.cpu(),
        "nodes_in_graph": graph.nodes_in_graph.cpu(),
        "config": {
            "model": MODEL_NAME,
            "method": "EAP-IG-inputs",
            "ig_steps": IG_STEPS,
            "top_n": TOP_N,
            "n_pairs": len(all_pairs),
            "n_gpus": n_gpus,
            "metric": "KL_divergence",
            "elapsed_minutes": total_elapsed / 60,
            "clean_template": "Question: {problem}\\nLet's solve this step by step.\\nStep 1:",
            "corrupted_suffixes": [ds for _, ds in matched_suffix_pairs],
        },
    }
    save_path = f"{BASE_DIR}/data/base_model_reasoning_heads.pt"
    torch.save(save_data, save_path)
    graph.to_json(f"{BASE_DIR}/data/base_model_circuit.json")
    print(f"\nSaved {save_path}")

    # ── Heatmap ───────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        import seaborn as sns
        has_seaborn = True
    except ImportError:
        has_seaborn = False

    fig, ax = plt.subplots(figsize=(14, 8))
    if has_seaborn:
        sns.heatmap(
            head_scores.numpy(), ax=ax,
            xticklabels=[f"H{h}" for h in range(n_heads)],
            yticklabels=[f"L{l}" for l in range(n_layers)],
            cmap="Reds",
        )
    else:
        im = ax.imshow(head_scores.numpy(), cmap="Reds", aspect="auto")
        ax.set_xticks(range(n_heads))
        ax.set_xticklabels([f"H{h}" for h in range(n_heads)])
        ax.set_yticks(range(n_layers))
        ax.set_yticklabels([f"L{l}" for l in range(n_layers)])
        plt.colorbar(im, ax=ax)
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_title(f"EAP-IG Head Importance: {MODEL_NAME} (base model)")
    plt.tight_layout()

    plot_path = f"{BASE_DIR}/plots/base_model_head_importance_heatmap.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Saved {plot_path}")

    print(f"\n{'='*60}")
    print("STEP 2 COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
