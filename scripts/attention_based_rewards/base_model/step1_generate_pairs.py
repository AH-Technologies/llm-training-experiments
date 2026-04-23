"""Step 1: Build diagnostic dataset for BASE model (Qwen2.5-Math-1.5B).

Unlike the Instruct model analysis which used <think> tags, the base model
uses natural text prefixes to contrast reasoning vs direct-answer modes:

Clean (reasoning-inducing):
    "Question: {problem}\nLet's solve this step by step.\nStep 1:"

Corrupted (direct answer):
    "Question: {problem}\nThe answer is:"

Includes KL divergence sanity check with alternative corrupted prefixes.
"""

import random
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformer_lens import HookedTransformer

random.seed(42)
torch.manual_seed(42)

BASE_DIR = "/cluster/projects/nn12068k/haaklau/llm-training-experiments/attention_based_rewards"
MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B"
# TL doesn't have Math variant in its model list, so we load architecture
# from the base Qwen2.5-1.5B and swap in Math weights via hf_model param
TL_MODEL_NAME = "Qwen/Qwen2.5-1.5B"
N_PAIRS = 300
KL_CHECK_N = 10

CLEAN_TEMPLATE = "Question: {problem}\nLet's solve this step by step.\nStep 1:"

CORRUPTED_TEMPLATES = {
    "primary": "Question: {problem}\nThe answer is:",
    "alt1": "Question: {problem}\nAnswer:",
    "alt2": "Question: {problem}\n=",
    "alt3": "Question: {problem}\n",
}


def compute_kl(model, text_a, text_b):
    """KL(P_a || P_b) at the last token position."""
    tok_a = model.to_tokens(text_a)
    tok_b = model.to_tokens(text_b)
    with torch.no_grad():
        logits_a = model(tok_a)
        logits_b = model(tok_b)
    lp_a = F.log_softmax(logits_a[0, -1].float(), dim=-1)
    lp_b = F.log_softmax(logits_b[0, -1].float(), dim=-1)
    kl = (lp_a.exp() * (lp_a - lp_b)).sum().item()
    return kl


def main():
    print(f"Loading {MODEL_NAME} (via TL name {TL_MODEL_NAME})...")
    from transformers import AutoModelForCausalLM
    hf_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
    model = HookedTransformer.from_pretrained(
        TL_MODEL_NAME, hf_model=hf_model, device="cuda", dtype=torch.float16,
    )
    del hf_model
    print(f"Model: {model.cfg.n_layers}L, {model.cfg.n_heads}H, d={model.cfg.d_model}")

    print("\nLoading GSM8K...")
    ds = load_dataset("openai/gsm8k", "main", split="train")
    indices = list(range(len(ds)))
    random.shuffle(indices)
    questions = [ds[i]["question"] for i in indices[:500]]

    # ── KL divergence sanity check ────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"KL DIVERGENCE SANITY CHECK (first {KL_CHECK_N} problems)")
    print(f"{'='*60}")

    check_questions = questions[:KL_CHECK_N]
    best_template_name = None
    best_kl = -1

    for name, template in CORRUPTED_TEMPLATES.items():
        kls = []
        for q in check_questions:
            clean = CLEAN_TEMPLATE.format(problem=q)
            corrupt = template.format(problem=q)
            kl = compute_kl(model, clean, corrupt)
            kls.append(kl)
        mean_kl = sum(kls) / len(kls)
        print(f"  {name:>10}: mean KL = {mean_kl:.4f}  (range: {min(kls):.2f} - {max(kls):.2f})")
        if mean_kl > best_kl:
            best_kl = mean_kl
            best_template_name = name

    print(f"\nBest corrupted template: '{best_template_name}' (KL={best_kl:.4f})")

    if best_kl < 3:
        print(f"\nWARNING: KL < 3 — contrast may be too weak.")
        print("All KL values printed above. Review before proceeding to step 2.")
        print("Proceeding anyway to save pairs (can re-run with different templates).")

    if best_kl > 5:
        print(f"\nGood signal (KL > 5). Proceeding with all {N_PAIRS} problems.")

    # Use the best template
    corrupted_template = CORRUPTED_TEMPLATES[best_template_name]
    print(f"\nUsing corrupted template: '{best_template_name}'")

    # ── Build pairs ───────────────────────────────────────────────────
    print(f"\nBuilding {N_PAIRS} diagnostic pairs...")
    pairs = []
    skipped = 0

    for q in questions:
        if len(pairs) >= N_PAIRS:
            break

        clean_text = CLEAN_TEMPLATE.format(problem=q)
        corrupt_text = corrupted_template.format(problem=q)

        clean_toks = model.to_tokens(clean_text)
        corrupt_toks = model.to_tokens(corrupt_text)

        max_len = max(clean_toks.shape[1], corrupt_toks.shape[1])
        if max_len > 512:
            skipped += 1
            continue

        # Pad shorter sequence (EAP-IG needs same length)
        if clean_toks.shape[1] != corrupt_toks.shape[1]:
            target_len = max_len
            pad_id = model.tokenizer.pad_token_id or model.tokenizer.eos_token_id
            device = clean_toks.device
            if clean_toks.shape[1] < target_len:
                pad = torch.full((1, target_len - clean_toks.shape[1]), pad_id, dtype=clean_toks.dtype, device=device)
                clean_toks = torch.cat([pad, clean_toks], dim=1)
            if corrupt_toks.shape[1] < target_len:
                pad = torch.full((1, target_len - corrupt_toks.shape[1]), pad_id, dtype=corrupt_toks.dtype, device=device)
                corrupt_toks = torch.cat([pad, corrupt_toks], dim=1)

        pairs.append({
            "question": q,
            "clean_text": clean_text,
            "corrupt_text": corrupt_text,
            "clean_tokens": clean_toks.cpu(),
            "corrupt_tokens": corrupt_toks.cpu(),
        })

    print(f"Generated {len(pairs)} pairs (skipped {skipped} too-long)")

    # ── Samples ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SAMPLE PAIRS")
    print(f"{'='*60}")
    for p in pairs[:3]:
        print(f"\nQ: {p['question'][:80]}...")
        print(f"Clean:   ...{p['clean_text'][-60:]}")
        print(f"Corrupt: ...{p['corrupt_text'][-60:]}")
        print(f"Token length: {p['clean_tokens'].shape[1]}")

    # ── Stats ─────────────────────────────────────────────────────────
    tok_lens = [p["clean_tokens"].shape[1] for p in pairs]
    print(f"\nToken lengths: min={min(tok_lens)}, max={max(tok_lens)}, mean={sum(tok_lens)/len(tok_lens):.0f}")

    # ── Save ──────────────────────────────────────────────────────────
    save_path = f"{BASE_DIR}/data/base_model_diagnostic_pairs.pt"
    torch.save({
        "pairs": pairs,
        "n_pairs": len(pairs),
        "model_name": MODEL_NAME,
        "clean_template": CLEAN_TEMPLATE,
        "corrupted_template": corrupted_template,
        "corrupted_template_name": best_template_name,
        "kl_divergence": best_kl,
        "all_kl_scores": {name: None for name in CORRUPTED_TEMPLATES},  # filled above in print
        "methodology": "Base model: reasoning prefix vs direct-answer prefix",
    }, save_path)
    print(f"\nSaved {len(pairs)} pairs to {save_path}")
    print("STEP 1 COMPLETE")


if __name__ == "__main__":
    main()
