#!/usr/bin/env python3
"""Two-phase entropy profiling: vLLM generation + HF entropy scoring.

Phase 1: vLLM generates all rollouts across 4 GPUs (fast, tensor-parallel).
Phase 2: HF forward pass on single GPU computes full-vocab Shannon entropy.
"""

import gc
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

# Add project root to path for reward imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rlvr_grokking.rewards.deepscaler_reward import compute_score, extract_answer


def compute_shannon_entropy(logits: torch.Tensor) -> np.ndarray:
    """Compute Shannon entropy from logits tensor.

    Args:
        logits: (batch_size, vocab_size) raw logits

    Returns:
        (batch_size,) entropy in nats
    """
    # Use float32 for numerical stability
    probs = torch.softmax(logits.float(), dim=-1)
    log_probs = torch.log(probs + 1e-12)
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy.cpu().numpy()


def vllm_generate_all(
    examples: list[dict],
    model_name: str,
    num_rollouts: int = 32,
    temperature: float = 0.6,
    max_new_tokens: int = 3072,
    seed: int = 42,
    tensor_parallel_size: int = 4,
) -> dict[str, list[dict]]:
    """Phase 1: Generate all rollouts with vLLM (fast, multi-GPU).

    Returns dict mapping example name -> list of rollout dicts with:
        {text, token_ids, is_correct, extracted_answer, num_tokens}
    """
    # Build all prompts: each example repeated num_rollouts times
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    all_prompts = []
    prompt_map = []  # Track which example each prompt belongs to
    for example in examples:
        messages = [{"role": "user", "content": example["prompt_text"]}]
        prompt_text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        for _ in range(num_rollouts):
            all_prompts.append(prompt_text)
            prompt_map.append(example["name"])

    del tokenizer

    print(f"Loading vLLM model: {model_name} (tensor_parallel_size={tensor_parallel_size})")
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=4096,
        seed=seed,
    )

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=0.95 if temperature > 0 else 1.0,
        max_tokens=max_new_tokens,
    )

    print(f"Generating {len(all_prompts)} rollouts ({len(examples)} examples x {num_rollouts})...")
    t0 = time.time()
    outputs = llm.generate(all_prompts, sampling_params)
    elapsed = time.time() - t0
    print(f"vLLM generation done in {elapsed:.1f}s ({len(all_prompts)/elapsed:.1f} samples/s)")

    # Organize results by example and grade correctness
    results_by_example: dict[str, list[dict]] = {ex["name"]: [] for ex in examples}
    gt_by_name = {ex["name"]: ex["ground_truth"] for ex in examples}

    for i, output in enumerate(outputs):
        name = prompt_map[i]
        text = output.outputs[0].text
        token_ids = list(output.outputs[0].token_ids)

        score = compute_score(
            data_source="math",
            solution_str=text,
            ground_truth=gt_by_name[name],
        )

        results_by_example[name].append({
            "text": text,
            "token_ids": token_ids,
            "num_tokens": len(token_ids),
            "is_correct": score > 0.5,
            "extracted_answer": extract_answer(text),
        })

    # Print per-example stats
    for ex in examples:
        name = ex["name"]
        rollouts = results_by_example[name]
        correct = sum(1 for r in rollouts if r["is_correct"])
        avg_len = np.mean([r["num_tokens"] for r in rollouts])
        print(f"  {name}: pass@{num_rollouts}={correct/len(rollouts):.3f}, avg_tokens={avg_len:.0f}")

    # Free vLLM resources
    print("Freeing vLLM resources...")
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    return results_by_example


def hf_score_entropy(
    results_by_example: dict[str, list[dict]],
    examples: list[dict],
    model_name: str,
    batch_size: int = 8,
) -> dict[str, list[dict]]:
    """Phase 2: Compute per-token entropy via HF forward pass.

    For each rollout, concatenates prompt_ids + generated_ids and runs a single
    forward pass. Extracts logits at generation positions for entropy computation.

    Updates rollout dicts in-place with 'entropy_array' key.
    """
    print(f"Loading HF model on cuda:0: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

    # Pre-tokenize all prompts
    prompt_ids_by_name = {}
    for example in examples:
        messages = [{"role": "user", "content": example["prompt_text"]}]
        ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
        prompt_ids_by_name[example["name"]] = ids.squeeze(0)  # (prompt_len,)

    total_rollouts = sum(len(v) for v in results_by_example.values())
    processed = 0
    t0 = time.time()

    for example in examples:
        name = example["name"]
        rollouts = results_by_example[name]
        prompt_ids = prompt_ids_by_name[name]
        prompt_len = len(prompt_ids)

        # Process rollouts in batches
        for batch_start in range(0, len(rollouts), batch_size):
            batch_rollouts = rollouts[batch_start:batch_start + batch_size]

            # Build full sequences: prompt + generated tokens
            sequences = []
            gen_lengths = []
            for rollout in batch_rollouts:
                gen_ids = torch.tensor(rollout["token_ids"], dtype=torch.long)
                full_seq = torch.cat([prompt_ids, gen_ids])
                sequences.append(full_seq)
                gen_lengths.append(len(gen_ids))

            # Pad sequences to same length
            max_len = max(len(s) for s in sequences)
            padded = torch.full((len(sequences), max_len), pad_id, dtype=torch.long)
            attention_mask = torch.zeros((len(sequences), max_len), dtype=torch.long)
            for i, seq in enumerate(sequences):
                padded[i, :len(seq)] = seq
                attention_mask[i, :len(seq)] = 1

            padded = padded.to(model.device)
            attention_mask = attention_mask.to(model.device)

            with torch.no_grad():
                out = model(input_ids=padded, attention_mask=attention_mask)

            # Extract entropy at generation positions
            # logits[:, prompt_len-1:-1, :] gives next-token predictions at generated positions
            logits = out.logits

            for i, rollout in enumerate(batch_rollouts):
                gen_len = gen_lengths[i]
                # Logits at positions [prompt_len-1, prompt_len, ..., prompt_len+gen_len-2]
                # predict tokens at positions [prompt_len, prompt_len+1, ..., prompt_len+gen_len-1]
                gen_logits = logits[i, prompt_len - 1:prompt_len - 1 + gen_len, :]
                rollout["entropy_array"] = compute_shannon_entropy(gen_logits)

            processed += len(batch_rollouts)

            # Free GPU memory
            del out, logits, padded, attention_mask
            torch.cuda.empty_cache()

        elapsed = time.time() - t0
        print(f"  {name}: scored {len(rollouts)} rollouts ({processed}/{total_rollouts} total, {elapsed:.1f}s)")

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return results_by_example


def run_entropy_profiling(
    examples: list[dict],
    model_name: str = "Qwen/Qwen2.5-Math-1.5B",
    num_rollouts: int = 32,
    batch_size: int = 8,
    temperature: float = 0.6,
    max_new_tokens: int = 3072,
    seed: int = 42,
    tensor_parallel_size: int = 4,
    output_dir: Path = Path("results"),
) -> dict:
    """Run two-phase entropy profiling on all examples.

    Phase 1: vLLM generation (all GPUs, tensor parallel)
    Phase 2: HF entropy scoring (single GPU forward passes)

    Returns:
        Dict mapping example name -> {example, rollouts, pass_rate}
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check which examples are already cached
    uncached_examples = []
    cached_results = {}
    for example in examples:
        save_path = output_dir / f"entropy_{example['name']}.pkl"
        if save_path.exists():
            print(f"  {example['name']}: loading cached result")
            with open(save_path, "rb") as f:
                cached_results[example["name"]] = pickle.load(f)
        else:
            uncached_examples.append(example)

    if not uncached_examples:
        print("All examples already cached!")
        return cached_results

    print(f"{len(uncached_examples)} examples to process, {len(cached_results)} cached")
    print()

    # Check for Phase 1 checkpoint
    phase1_checkpoint = output_dir / "_phase1_checkpoint.pkl"

    if phase1_checkpoint.exists():
        print("=" * 60)
        print("Phase 1: Loading from checkpoint")
        print("=" * 60)
        with open(phase1_checkpoint, "rb") as f:
            results_by_example = pickle.load(f)
        print(f"Loaded Phase 1 checkpoint with {len(results_by_example)} examples")
    else:
        # ── Phase 1: vLLM generation ──
        print("=" * 60)
        print("Phase 1: vLLM Generation")
        print("=" * 60)
        results_by_example = vllm_generate_all(
            examples=uncached_examples,
            model_name=model_name,
            num_rollouts=num_rollouts,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            seed=seed,
            tensor_parallel_size=tensor_parallel_size,
        )

        # Save Phase 1 checkpoint
        with open(phase1_checkpoint, "wb") as f:
            pickle.dump(results_by_example, f)
        print(f"Saved Phase 1 checkpoint to {phase1_checkpoint}")
    print()

    # ── Phase 2: HF entropy scoring ──
    print("=" * 60)
    print("Phase 2: HF Entropy Scoring")
    print("=" * 60)
    results_by_example = hf_score_entropy(
        results_by_example=results_by_example,
        examples=uncached_examples,
        model_name=model_name,
        batch_size=batch_size,
    )
    print()

    # ── Save results ──
    all_results = dict(cached_results)

    for example in uncached_examples:
        name = example["name"]
        rollouts = results_by_example[name]
        correct_count = sum(1 for r in rollouts if r["is_correct"])
        pass_rate = correct_count / len(rollouts)

        result = {
            "example": example,
            "rollouts": rollouts,
            "pass_rate": pass_rate,
        }

        # Save incrementally
        save_path = output_dir / f"entropy_{name}.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(result, f)
        print(f"  Saved {save_path}")

        all_results[name] = result

    # Clean up Phase 1 checkpoint
    if phase1_checkpoint.exists():
        phase1_checkpoint.unlink()
        print("Removed Phase 1 checkpoint (all results saved successfully)")

    return all_results
