#!/usr/bin/env python3
"""Generate rollouts for Wang 14 with no-code suffix.

Adds instruction telling the model not to use Python/code.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from extract_examples import extract_examples
from compute_entropy import run_entropy_profiling

NOCODE_SUFFIX = " You do not have access to a Python interpreter or any code execution environment. Solve this problem using mathematical reasoning only."

# Get Wang 14 examples and modify their prompts
wang_examples = extract_examples()

nocode_examples = []
for ex in wang_examples:
    # The prompt already ends with "Let's think step by step and output the final answer within \boxed{}."
    # Insert the no-code instruction before that
    prompt = ex["prompt_text"]
    boxed_marker = "Let's think step by step"
    if boxed_marker in prompt:
        idx = prompt.index(boxed_marker)
        new_prompt = prompt[:idx].rstrip() + NOCODE_SUFFIX + " " + prompt[idx:]
    else:
        new_prompt = prompt + NOCODE_SUFFIX

    nocode_examples.append({
        "name": ex["name"] + "_nocode",
        "prompt_text": new_prompt,
        "ground_truth": ex["ground_truth"],
        "math500_score": ex.get("math500_score"),
    })

if __name__ == "__main__":
    print(f"Processing {len(nocode_examples)} examples with no-code suffix")
    print(f"\nExample prompt (first 300 chars):")
    print(nocode_examples[0]["prompt_text"][:300])
    print("...")
    print()

    output_dir = Path("results/nocode/entropy_profiles")

    results = run_entropy_profiling(
        examples=nocode_examples,
        model_name="Qwen/Qwen2.5-Math-1.5B",
        num_rollouts=256,
        batch_size=8,
        temperature=0.6,
        max_new_tokens=3072,
        seed=42,
        tensor_parallel_size=4,
        output_dir=output_dir,
    )

    print("\nDone! Results saved to", output_dir)
