#!/usr/bin/env python3
"""Quick test: does pi13 step700 produce gibberish at temperature 0?"""
import os
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

import re
from vllm import LLM, SamplingParams

MODEL = "models/grpo_pi13_math500_v2_lr_increased_multi_val/global_step_700"
CJK_RE = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf]')

PROBLEMS = [
    "What is $7 \\times 8$?",
    "Convert the point $(0,3)$ from rectangular coordinates to polar coordinates.",
    "Find the sum $1 + 2 + 3 + \\cdots + 100$.",
    "What is the remainder when $2^{10}$ is divided by 7?",
    "Solve for $x$: $2x + 5 = 17$.",
]

SYSTEM = ("Please reason step by step, and put your final answer within \\boxed{}.")

def build_prompt(q):
    return (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{q}<|im_end|>\n"
            f"<|im_start|>assistant\n")

def main():
    llm = LLM(
        model=MODEL,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.5,
        max_model_len=4096,
        trust_remote_code=True,
        dtype="bfloat16",
        seed=42,
    )

    prompts = [build_prompt(q) for q in PROBLEMS]

    for temp in [0.0, 0.6]:
        print(f"\n{'='*70}")
        print(f"  Temperature = {temp}")
        print(f"{'='*70}")
        params = SamplingParams(
            temperature=temp,
            top_p=0.95 if temp > 0 else 1.0,
            max_tokens=2048,
            n=2 if temp > 0 else 1,
        )
        outputs = llm.generate(prompts, params)
        for i, out in enumerate(outputs):
            for j, o in enumerate(out.outputs):
                text = o.text
                n_cjk = len(CJK_RE.findall(text))
                has_gib = "GIBBERISH" if n_cjk > 5 else "clean"
                print(f"\n--- Problem {i}, sample {j} [{has_gib}, {len(text)} chars, {n_cjk} CJK] ---")
                # Print first 500 chars and last 300 chars
                if len(text) <= 800:
                    print(text)
                else:
                    print(text[:500])
                    print(f"\n  [...{len(text)-800} chars omitted...]\n")
                    print(text[-300:])

if __name__ == "__main__":
    main()
