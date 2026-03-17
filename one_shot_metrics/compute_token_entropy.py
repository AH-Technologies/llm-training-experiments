"""Compute per-token entropy for saved rollouts using HuggingFace.

Loads rollout jsonl files, reconstructs prompt+completion sequences,
runs a forward pass through the model to get full-vocabulary logits,
and computes Shannon entropy at each token position.

Saves per-rollout entropy profiles to jsonl files alongside the rollouts.

Usage:
    python compute_token_entropy.py \
        --rollout_dir results/wang_benchmark \
        --model Qwen/Qwen2.5-Math-1.5B
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_rollout_file(rollout_path: Path) -> tuple[dict, list[str]]:
    """Load metadata and completions from a rollout jsonl file."""
    metadata = None
    completions = []

    with open(rollout_path) as f:
        for line in f:
            record = json.loads(line)
            if record["type"] == "metadata":
                metadata = record
            elif record["type"] == "rollout":
                completions.append(record["completion"])

    return metadata, completions


def compute_entropy_profile(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    completion: str,
    device: torch.device,
    batch_size: int = 1,
) -> list[float]:
    """Compute per-token entropy over the full vocabulary for a completion.

    Tokenizes prompt+completion, runs a forward pass, and computes
    Shannon entropy (in nats) from the logits at each completion token position.
    """
    # Tokenize prompt and full sequence separately to find the boundary
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    full_ids = tokenizer.encode(prompt + completion, add_special_tokens=False)
    completion_start = len(prompt_ids)

    if completion_start >= len(full_ids):
        return []

    input_ids = torch.tensor([full_ids], device=device)

    with torch.no_grad():
        outputs = model(input_ids)
        # logits shape: (1, seq_len, vocab_size)
        logits = outputs.logits[0]  # (seq_len, vocab_size)

    # For each completion token at position t, the entropy comes from
    # the logits at position t-1 (the distribution that generated token t)
    entropies = []
    for t in range(completion_start, len(full_ids)):
        token_logits = logits[t - 1]  # distribution over vocab for position t
        probs = F.softmax(token_logits, dim=-1)
        # Shannon entropy: H = -sum(p * log(p)), skip zeros
        log_probs = torch.log(probs + 1e-12)
        entropy = -(probs * log_probs).sum().item()
        entropies.append(entropy)

    return entropies


def process_rollout_file(
    rollout_path: Path,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    output_dir: Path,
):
    """Compute token entropy profiles for all rollouts in a file."""
    metadata, completions = load_rollout_file(rollout_path)
    if metadata is None:
        print(f"  Warning: no metadata in {rollout_path}, skipping", file=sys.stderr)
        return

    prompt = metadata["prompt"]
    example_name = metadata["example_name"]
    out_path = output_dir / f"{example_name}_entropy.jsonl"

    with open(out_path, "w") as f:
        # Write metadata
        meta_record = {
            "type": "metadata",
            "example_name": example_name,
            "model": metadata.get("model", "unknown"),
            "k": len(completions),
        }
        f.write(json.dumps(meta_record) + "\n")

        for i, completion in enumerate(completions):
            entropies = compute_entropy_profile(model, tokenizer, prompt, completion, device)

            record = {
                "type": "entropy_profile",
                "rollout_index": i,
                "token_entropies": entropies,
                "num_tokens": len(entropies),
                "mean_entropy": sum(entropies) / len(entropies) if entropies else 0.0,
                "max_entropy": max(entropies) if entropies else 0.0,
            }
            f.write(json.dumps(record) + "\n")

            if (i + 1) % 16 == 0:
                print(f"    {example_name}: {i + 1}/{len(completions)} rollouts")

    print(f"  {example_name}: {len(completions)} rollouts -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Compute per-token entropy using HuggingFace")
    parser.add_argument("--rollout_dir", type=str, required=True,
                        help="Directory containing rollout jsonl files")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Math-1.5B",
                        help="HuggingFace model name (must match rollout model)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: <rollout_dir>/entropy)")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16", "float32"],
                        help="Model dtype")
    parser.add_argument("--file_list", type=str, default=None,
                        help="Path to a text file listing rollout files to process (one per line, for sharding)")
    args = parser.parse_args()

    rollout_dir = Path(args.rollout_dir)
    output_dir = Path(args.output_dir) if args.output_dir else rollout_dir / "entropy"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.file_list:
        with open(args.file_list) as f:
            rollout_files = [Path(line.strip()) for line in f if line.strip()]
    else:
        rollout_files = sorted(rollout_dir.glob("pi_*.jsonl"))

    if not rollout_files:
        print(f"No rollout files to process")
        return

    print(f"Found {len(rollout_files)} rollout files")

    # Load model
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model {args.model} ({args.dtype}) on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map=device,
    )
    model.eval()
    print("Model loaded.")

    for rollout_path in rollout_files:
        print(f"Processing {rollout_path.name}...")
        process_rollout_file(rollout_path, model, tokenizer, device, output_dir)

    print(f"\nDone. Entropy profiles saved to {output_dir}")


if __name__ == "__main__":
    main()
