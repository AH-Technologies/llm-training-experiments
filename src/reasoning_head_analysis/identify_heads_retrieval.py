#!/usr/bin/env python3
"""Identify retrieval heads using needle-in-a-haystack probing.

Based on: "Retrieval Head Mechanistically Explains Long-Context Factual Failures"
(Wu et al., 2024, arXiv:2404.15574)

Method:
1. Insert a "needle" (known fact) into a "haystack" (long background text) at various depths
2. Ask the model to retrieve the needle via a question
3. During autoregressive decoding, track which attention heads attend maximally to
   needle tokens AND generate matching tokens (copy-paste detection)
4. R_h = |positions_copied_by_h| / |needle_length|
5. Only count trials where model successfully retrieves (ROUGE-L recall > threshold)
6. Average across successful trials -> per-head retrieval score

Output is compatible with the ablation pipeline (head_importance.pt with head_scores
and active_heads keys).

Usage:
  python -m reasoning_head_analysis.identify_heads_retrieval \
      --model Qwen/Qwen2.5-Math-1.5B

  python -m reasoning_head_analysis.identify_heads_retrieval \
      --model /path/to/checkpoint --rouge_threshold 0.3
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
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Needles and haystack
# ═══════════════════════════════════════════════════════════════════════

NEEDLES = [
    # Original 3
    {
        "needle": "The best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.",
        "question": "What is the best thing to do in San Francisco?",
        "answer": "eat a sandwich and sit in Dolores Park on a sunny day",
    },
    {
        "needle": "According to the WMO report, the global average temperature in 2023 was approximately 1.45 degrees Celsius above the pre-industrial baseline.",
        "question": "What was the global average temperature in 2023 according to the WMO report?",
        "answer": "approximately 1.45 degrees Celsius above the pre-industrial baseline",
    },
    {
        "needle": "The annual Dragon Boat Festival in Beijing features traditional zongzi made with sticky rice wrapped in bamboo leaves.",
        "question": "What food is featured at the annual Dragon Boat Festival in Beijing?",
        "answer": "traditional zongzi made with sticky rice wrapped in bamboo leaves",
    },
    # Science
    {
        "needle": "The speed of light in a vacuum is exactly 299,792,458 meters per second, a fundamental constant in physics.",
        "question": "What is the speed of light in a vacuum?",
        "answer": "exactly 299,792,458 meters per second",
    },
    {
        "needle": "Water boils at 100 degrees Celsius at standard atmospheric pressure, which is defined as 101.325 kilopascals.",
        "question": "At what temperature does water boil at standard atmospheric pressure?",
        "answer": "100 degrees Celsius at standard atmospheric pressure",
    },
    # Geography
    {
        "needle": "Mount Everest stands at 8,849 meters above sea level, making it the tallest mountain on Earth measured from sea level.",
        "question": "How tall is Mount Everest?",
        "answer": "8,849 meters above sea level",
    },
    {
        "needle": "The Nile River stretches approximately 6,650 kilometers from its source in Burundi to the Mediterranean Sea in Egypt.",
        "question": "How long is the Nile River?",
        "answer": "approximately 6,650 kilometers",
    },
    # History
    {
        "needle": "The Berlin Wall fell on November 9, 1989, marking the beginning of German reunification after decades of division.",
        "question": "When did the Berlin Wall fall?",
        "answer": "November 9, 1989",
    },
    {
        "needle": "The first successful powered airplane flight was achieved by the Wright brothers on December 17, 1903, at Kitty Hawk, North Carolina.",
        "question": "When and where did the Wright brothers achieve the first powered flight?",
        "answer": "December 17, 1903, at Kitty Hawk, North Carolina",
    },
    # Technology
    {
        "needle": "The first message sent over ARPANET was 'lo', intended to be 'login', but the system crashed after the first two letters on October 29, 1969.",
        "question": "What was the first message sent over ARPANET?",
        "answer": "'lo', intended to be 'login', but the system crashed after the first two letters",
    },
]


def load_haystack_text(source="wikitext", max_chars=50000, seed=42):
    """Load background text from wikitext-103 or PG19.

    Args:
        source: "wikitext" (default) or "pg19"
        max_chars: Maximum characters of haystack text to load
        seed: Random seed for paragraph shuffling and book selection (PG19)
    """
    rng = random.Random(seed)

    if source == "pg19":
        logger.info("Loading haystack text from PG19 (emozilla/pg19)...")
        ds = load_dataset("emozilla/pg19", split="train")
        # Pick a random book controlled by seed
        book_idx = rng.randint(0, len(ds) - 1)
        book = ds[book_idx]
        logger.info(f"  Selected book: {book.get('short_book_title', 'unknown')} (idx={book_idx})")
        # Split into paragraphs
        paragraphs = [p.strip() for p in book["text"].split("\n\n") if len(p.strip()) > 50]
    else:
        logger.info("Loading haystack text from wikitext-103-raw-v1...")
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
        paragraphs = []
        total = 0
        for item in ds:
            text = item["text"].strip()
            if len(text) > 50:
                paragraphs.append(text)
                total += len(text)
                if total >= max_chars * 2:  # collect extra for shuffling
                    break

    # Shuffle paragraphs with the seed (different seeds give different orderings)
    rng.shuffle(paragraphs)

    # Take up to max_chars
    selected = []
    total = 0
    for p in paragraphs:
        selected.append(p)
        total += len(p)
        if total >= max_chars:
            break

    haystack = "\n\n".join(selected)
    logger.info(f"  Loaded {len(haystack)} chars of haystack text ({len(selected)} paragraphs)")
    return haystack


def build_prompt(haystack_text, needle_info, context_length, depth_percent, tokenizer):
    """Build a prompt with needle inserted at the specified depth.

    Returns:
        prompt_text: str
        needle_start_char: int — character offset of needle in the prompt
    """
    needle = needle_info["needle"]
    question = needle_info["question"]

    suffix = f"\n\nBased on the content above, Question: {question}\nAnswer:"

    # Estimate how many chars of haystack to use to hit target context_length tokens
    # Rough estimate: 4 chars per token
    target_chars = context_length * 4
    # Reserve space for needle and suffix
    haystack_budget = max(200, target_chars - len(needle) - len(suffix) - 50)
    haystack_chunk = haystack_text[:haystack_budget]

    # Insert needle at depth_percent position
    insert_pos = int(len(haystack_chunk) * depth_percent / 100)
    # Find a paragraph boundary near insert_pos
    newline_pos = haystack_chunk.find("\n", insert_pos)
    if newline_pos == -1 or newline_pos > insert_pos + 200:
        newline_pos = insert_pos

    before = haystack_chunk[:newline_pos]
    after = haystack_chunk[newline_pos:]
    prompt_text = before + "\n" + needle + "\n" + after + suffix

    # Trim to approximate target length by tokenizing and truncating
    tokens = tokenizer.encode(prompt_text)
    if len(tokens) > context_length:
        # Truncate haystack from the end (before suffix)
        # Re-estimate
        ratio = context_length / len(tokens)
        haystack_budget = int(haystack_budget * ratio * 0.95)
        haystack_chunk = haystack_text[:haystack_budget]
        insert_pos = int(len(haystack_chunk) * depth_percent / 100)
        newline_pos = haystack_chunk.find("\n", insert_pos)
        if newline_pos == -1 or newline_pos > insert_pos + 200:
            newline_pos = insert_pos
        before = haystack_chunk[:newline_pos]
        after = haystack_chunk[newline_pos:]
        prompt_text = before + "\n" + needle + "\n" + after + suffix

    needle_start_char = len(before) + 1  # +1 for the newline
    return prompt_text, needle_start_char


# ═══════════════════════════════════════════════════════════════════════
# ROUGE-L computation (no external dependency)
# ═══════════════════════════════════════════════════════════════════════

def _lcs_length(x, y):
    """Compute length of longest common subsequence."""
    m, n = len(x), len(y)
    if m == 0 or n == 0:
        return 0
    # Use 1D DP for memory efficiency
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(curr[j - 1], prev[j])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


def rouge_l_recall(prediction, reference):
    """Compute ROUGE-L recall: LCS(pred, ref) / len(ref)."""
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()
    if not ref_tokens:
        return 0.0
    lcs = _lcs_length(pred_tokens, ref_tokens)
    return lcs / len(ref_tokens)


# ═══════════════════════════════════════════════════════════════════════
# Needle span detection in token space
# ═══════════════════════════════════════════════════════════════════════

def find_needle_token_span(input_ids, needle_text, tokenizer):
    """Find the token indices corresponding to the needle text.

    Uses fuzzy subsequence matching: encode the needle separately, then find
    the best matching contiguous span in input_ids by token overlap.

    Returns:
        (start_idx, end_idx) — token indices (inclusive start, exclusive end)
        or None if no good match found.
    """
    needle_ids = tokenizer.encode(needle_text, add_special_tokens=False)
    needle_len = len(needle_ids)
    if needle_len == 0:
        return None

    input_list = input_ids.tolist() if torch.is_tensor(input_ids) else list(input_ids)
    seq_len = len(input_list)

    best_overlap = 0
    best_start = 0

    # Slide a window of needle_len over input_ids
    for start in range(seq_len - needle_len + 1):
        window = input_list[start:start + needle_len]
        overlap = sum(1 for a, b in zip(window, needle_ids) if a == b)
        if overlap > best_overlap:
            best_overlap = overlap
            best_start = start

    # Require at least 50% token overlap
    if best_overlap < needle_len * 0.5:
        return None

    return (best_start, best_start + needle_len)


# ═══════════════════════════════════════════════════════════════════════
# Core retrieval head detection
# ═══════════════════════════════════════════════════════════════════════

def run_trial(model, tokenizer, prompt_text, needle_info, device,
              max_new_tokens, rouge_threshold):
    """Run one needle-in-haystack trial.

    Returns:
        head_scores: Tensor[n_layers, n_heads] — copy-paste ratio for this trial
        success: bool — whether model successfully retrieved the needle
        rouge: float — ROUGE-L recall score
        generated_text: str — model's generated text
    """
    config = model.config
    n_layers = config.num_hidden_layers
    n_heads = config.num_attention_heads

    # Tokenize prompt
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    prompt_len = input_ids.shape[1]

    # Find needle span in token space
    needle_span = find_needle_token_span(
        input_ids[0], needle_info["needle"], tokenizer)
    if needle_span is None:
        logger.warning("  Could not locate needle tokens in prompt")
        return torch.zeros(n_layers, n_heads), False, 0.0, ""

    needle_start, needle_end = needle_span

    # Prefill: run without output_attentions (saves memory)
    with torch.no_grad():
        prefill_out = model(
            input_ids=input_ids,
            output_attentions=False,
            use_cache=True,
        )
    past_key_values = prefill_out.past_key_values

    # Decode token-by-token with output_attentions=True
    head_copy_counts = torch.zeros(n_layers, n_heads)
    generated_ids = []
    next_token_id = prefill_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    for step in range(max_new_tokens):
        generated_ids.append(next_token_id.item())

        # Check for EOS
        if next_token_id.item() == tokenizer.eos_token_id:
            break

        with torch.no_grad():
            out = model(
                input_ids=next_token_id,
                past_key_values=past_key_values,
                output_attentions=True,
                use_cache=True,
            )

        past_key_values = out.past_key_values
        # out.attentions: tuple of (n_layers,) each [batch, n_heads, 1, seq_len_so_far]
        attentions = out.attentions

        # For each head, check if argmax attention points to needle span
        # AND the generated token matches the attended token
        current_pos = prompt_len + step  # position of the token we just fed
        gen_token = next_token_id.item()

        for layer_idx in range(n_layers):
            # attn shape: [batch, n_heads, 1, seq_len_so_far]
            attn = attentions[layer_idx][0, :, 0, :]  # [n_heads, seq_len_so_far]
            # Get argmax attention position for each head
            argmax_pos = attn.argmax(dim=-1)  # [n_heads]

            for head_idx in range(n_heads):
                pos = argmax_pos[head_idx].item()
                # Check: is the max-attended position in the needle span?
                if needle_start <= pos < needle_end:
                    # Check: does the generated token match the attended token?
                    # The attended position's token in the full sequence
                    if pos < prompt_len:
                        attended_token = input_ids[0, pos].item()
                    else:
                        gen_offset = pos - prompt_len
                        if gen_offset < len(generated_ids):
                            attended_token = generated_ids[gen_offset]
                        else:
                            continue
                    if gen_token == attended_token:
                        head_copy_counts[layer_idx, head_idx] += 1

        next_token_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    # Compute ROUGE-L recall
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    rouge = rouge_l_recall(generated_text, needle_info["answer"])
    success = rouge >= rouge_threshold

    # Normalize: R_h = copy_count / needle_length
    needle_token_len = needle_end - needle_start
    head_scores = head_copy_counts / max(needle_token_len, 1)

    return head_scores, success, rouge, generated_text


# ═══════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════

def plot_heatmap(head_scores, output_path, title="Retrieval Head Importance"):
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
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved heatmap to {output_path}")


def select_active_heads(head_scores, top_k=None):
    """Select top-K heads by retrieval score."""
    flat = head_scores.flatten()
    sorted_idx = flat.argsort(descending=True)
    n_heads_total = flat.numel()
    k = min(top_k or 20, n_heads_total)

    n_cols = head_scores.shape[1]
    active = []
    for i in range(k):
        idx = sorted_idx[i].item()
        l, h = idx // n_cols, idx % n_cols
        active.append((l, h))

    return sorted(active)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Identify retrieval heads via needle-in-a-haystack probing")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--context_lengths", type=int, nargs="+",
                        default=[1000, 2000, 3000, 4000],
                        help="Context lengths to test (in tokens)")
    parser.add_argument("--depth_percents", type=int, nargs="+",
                        default=[0, 25, 50, 75, 100],
                        help="Needle insertion depths (percent)")
    parser.add_argument("--max_new_tokens", type=int, default=50,
                        help="Max tokens to generate per trial")
    parser.add_argument("--rouge_threshold", type=float, default=0.5,
                        help="Min ROUGE-L recall for a trial to count as successful")
    parser.add_argument("--haystack_source", type=str, default="wikitext",
                        choices=["wikitext", "pg19"],
                        help="Haystack text source: 'wikitext' (default) or 'pg19'")
    parser.add_argument("--top_k_heads", type=int, default=None,
                        help="Number of top heads to mark as active (default: 20)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    safe_model = args.model.replace("/", "_").replace(".", "_")
    output_dir = args.output_dir or os.path.join(
        "results", "reasoning_head_analysis", "identification", "retrieval", safe_model)
    os.makedirs(output_dir, exist_ok=True)

    total_trials = len(NEEDLES) * len(args.context_lengths) * len(args.depth_percents)

    logger.info(f"Model: {args.model}")
    logger.info(f"Device: {device}")
    logger.info(f"Context lengths: {args.context_lengths}")
    logger.info(f"Depth percents: {args.depth_percents}")
    logger.info(f"Total trials: {total_trials} ({len(NEEDLES)} needles x "
                f"{len(args.context_lengths)} lengths x {len(args.depth_percents)} depths)")
    logger.info(f"ROUGE threshold: {args.rouge_threshold}")
    logger.info(f"Output: {output_dir}")

    # Load model with eager attention (need attention weights)
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    logger.info("Loading model (eager attention for attention weight access)...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="eager",
    ).to(device)
    model.eval()
    logger.info(f"Model loaded in {time.time() - t0:.1f}s")

    config = model.config
    n_layers = config.num_hidden_layers
    n_heads = config.num_attention_heads

    # Load haystack text
    haystack_text = load_haystack_text(source=args.haystack_source, seed=args.seed)

    # Run trials
    logger.info("Running needle-in-a-haystack trials...")
    t0 = time.time()
    all_scores = []
    n_success = 0
    n_total = 0
    trial_log = []

    for needle_info in NEEDLES:
        for ctx_len in args.context_lengths:
            for depth in args.depth_percents:
                n_total += 1
                label = (f"needle={needle_info['answer'][:30]}..., "
                         f"ctx={ctx_len}, depth={depth}%")

                prompt_text, _ = build_prompt(
                    haystack_text, needle_info, ctx_len, depth, tokenizer)

                prompt_tokens = len(tokenizer.encode(prompt_text))
                logger.info(f"Trial {n_total}/{total_trials}: {label} "
                            f"({prompt_tokens} tokens)")

                scores, success, rouge, gen_text = run_trial(
                    model, tokenizer, prompt_text, needle_info, device,
                    args.max_new_tokens, args.rouge_threshold)

                status = "SUCCESS" if success else "FAIL"
                logger.info(f"  {status} — ROUGE-L={rouge:.3f}, "
                            f"generated: {gen_text[:80]}...")

                trial_log.append({
                    "needle": needle_info["answer"][:50],
                    "context_length": ctx_len,
                    "depth_percent": depth,
                    "rouge": rouge,
                    "success": success,
                    "generated": gen_text[:200],
                })

                if success:
                    all_scores.append(scores)
                    n_success += 1

    elapsed = time.time() - t0
    logger.info(f"\nTrials completed in {elapsed / 60:.1f} min")
    logger.info(f"Success rate: {n_success}/{n_total} "
                f"({n_success / max(n_total, 1) * 100:.1f}%)")

    if n_success == 0:
        logger.warning("No successful trials! Try lowering --rouge_threshold. "
                       "Saving zero scores.")
        avg_scores = torch.zeros(n_layers, n_heads)
    else:
        avg_scores = torch.stack(all_scores).mean(dim=0)

    # Select active heads
    active_heads = select_active_heads(avg_scores, top_k=args.top_k_heads)
    logger.info(f"Active heads ({len(active_heads)}): {active_heads}")

    # Rank and display top heads
    flat = avg_scores.flatten()
    sorted_idx = flat.argsort(descending=True)
    logger.info("Top 20 retrieval heads:")
    for rank, idx in enumerate(sorted_idx[:20]):
        l, h = idx.item() // n_heads, idx.item() % n_heads
        score = flat[idx].item()
        logger.info(f"  #{rank+1}: L{l}H{h} = {score:.4f}")

    # Save results (compatible with ablation pipeline)
    importance_path = os.path.join(output_dir, "head_importance.pt")
    torch.save({
        "head_scores": avg_scores,
        "active_heads": active_heads,
        "config": {
            "model": args.model,
            "method": "retrieval_heads",
            "seed": args.seed,
            "haystack_source": args.haystack_source,
            "context_lengths": args.context_lengths,
            "depth_percents": args.depth_percents,
            "max_new_tokens": args.max_new_tokens,
            "rouge_threshold": args.rouge_threshold,
            "n_trials": n_total,
            "n_successful": n_success,
            "success_rate": n_success / max(n_total, 1),
            "elapsed_minutes": elapsed / 60,
        },
        "trial_log": trial_log,
    }, importance_path)
    logger.info(f"Saved head importance to {importance_path}")

    # Heatmap
    plot_heatmap(avg_scores, os.path.join(output_dir, "head_importance_heatmap.png"),
                 title=f"Retrieval Head Importance ({n_success}/{n_total} successful trials)")

    logger.info(f"Total time: {(time.time() - t0) / 60:.1f} min")
    logger.info("Done.")


if __name__ == "__main__":
    main()
