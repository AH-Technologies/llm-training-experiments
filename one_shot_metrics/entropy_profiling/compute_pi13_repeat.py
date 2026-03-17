#!/usr/bin/env python3
"""Generate rollouts and compute entropy+RSR for pi13 with repeat-question suffix.

Compares the original pi13 prompt with the 'repeat the question' variant.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from compute_entropy import run_entropy_profiling

PI13_ORIGINAL = """Given that circle $C$ passes through points $P(0,-4)$, $Q(2,0)$, and $R(3,-1)$.
$(1)$ Find the equation of circle $C$;
$(2)$ If line $l: mx+y-1=0$ intersects circle $C$ at points $A$ and $B$, and $|AB|=4$, find the value of $m$. Let's think step by step and output the final answer within \\boxed{}."""

PI13_REPEAT = """Given that circle $C$ passes through points $P(0,-4)$, $Q(2,0)$, and $R(3,-1)$.
$(1)$ Find the equation of circle $C$;
$(2)$ If line $l: mx+y-1=0$ intersects circle $C$ at points $A$ and $B$, and $|AB|=4$, find the value of $m$. First, repeat the question word by word, and then let's think step by step to solve the problem. Output the final answer within \\boxed{}."""

GROUND_TRUTH = "4/3"

examples = [
    {
        "name": "pi_13_repeat",
        "prompt_text": PI13_REPEAT,
        "ground_truth": GROUND_TRUTH,
    },
]

if __name__ == "__main__":
    output_dir = Path("results/pi13_repeat/entropy_profiles")

    results = run_entropy_profiling(
        examples=examples,
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
