#!/usr/bin/env python3
"""Prepare multi-prompt validation dataset for evaluating with different CoT strategies.

Takes an existing validation parquet (e.g., math500.parquet) and creates a new parquet
with 3 versions of each question, each with a different prompting strategy:

1. "math_no_cot"    — raw question only, no CoT instruction at all.
2. "math_train_cot" — training-style suffix appended to user message:
                       "Let's think step by step and output the final answer within \\boxed{}."
3. "math_qwen_cot"  — Qwen CoT as system message (raw question as user message):
                       system: "Please reason step by step, and put your final answer within \\boxed{}."
                       This matches the PROMPT_TYPE="qwen25-math-cot" from One-Shot-RLVR eval.

Each version gets a different `data_source` tag so verl's validation metrics are
automatically logged separately per prompt style in wandb.
"""

import pandas as pd
from pathlib import Path
import copy

# Qwen CoT system message (matches Qwen2.5-Eval PROMPT_TYPE="qwen25-math-cot")
QWEN_COT_SYSTEM_MSG = r"Please reason step by step, and put your final answer within \boxed{}."

# Training-style CoT suffix (appended to user message, same as in training data)
TRAIN_COT_SUFFIX = r" Let's think step by step and output the final answer within \boxed{}."

# data_source tags for each variant
NO_COT_SOURCE = "math_no_cot"
TRAIN_COT_SOURCE = "math_train_cot"
QWEN_COT_SOURCE = "math_qwen_cot"


def get_user_content(prompt_msgs) -> str:
    """Extract user message content from a prompt message list.

    NOTE: pandas iterrows() can return nested list-of-dicts from parquet as
    numpy arrays or pyarrow structs, not plain Python lists. We convert first.
    """
    # Force conversion to plain Python list of dicts
    if not isinstance(prompt_msgs, list):
        prompt_msgs = [dict(m) for m in prompt_msgs]
    user_msgs = [m for m in prompt_msgs if m.get("role") == "user"]
    return user_msgs[-1]["content"] if user_msgs else prompt_msgs[-1]["content"]


def make_prompt_no_cot(original_content: str) -> list[dict]:
    """Raw question only, no CoT instruction."""
    return [{"role": "user", "content": original_content}]


def make_prompt_train_cot(original_content: str) -> list[dict]:
    """Training-style: CoT suffix appended to user message."""
    return [{"role": "user", "content": original_content + TRAIN_COT_SUFFIX}]


def make_prompt_qwen_cot(original_content: str) -> list[dict]:
    """Qwen-style: CoT as system message, raw question as user message.

    Matches the Qwen2.5-Eval format (PROMPT_TYPE="qwen25-math-cot"):
        <|im_start|>system
        Please reason step by step, and put your final answer within \\boxed{}.
        <|im_end|>
        <|im_start|>user
        {question}
        <|im_end|>
        <|im_start|>assistant
    """
    return [
        {"role": "system", "content": QWEN_COT_SYSTEM_MSG},
        {"role": "user", "content": original_content},
    ]


PROMPT_BUILDERS = [
    (NO_COT_SOURCE, make_prompt_no_cot),
    (TRAIN_COT_SOURCE, make_prompt_train_cot),
    (QWEN_COT_SOURCE, make_prompt_qwen_cot),
]


def create_multi_prompt_val(input_path: Path, output_path: Path):
    """Create multi-prompt validation dataset from an existing val parquet.

    Args:
        input_path: Path to original validation parquet (e.g., math500.parquet)
        output_path: Path for the output multi-prompt parquet
    """
    df = pd.read_parquet(input_path)
    print(f"Input: {input_path} ({len(df)} rows)")

    # Sanity check: verify we can extract content correctly from first row
    first_prompt = df.iloc[0]["prompt"]
    print(f"  prompt type: {type(first_prompt)}")
    if hasattr(first_prompt, '__len__') and len(first_prompt) > 0:
        print(f"  prompt[0] type: {type(first_prompt[0])}")
    test_content = get_user_content(first_prompt)
    assert not test_content.startswith("[{"), (
        f"Bug: content looks like a stringified list: {test_content[:100]}..."
    )
    print(f"  first question preview: {test_content[:100]}...")

    rows = []
    for _, row in df.iterrows():
        original_content = get_user_content(row["prompt"])

        for data_source, build_prompt in PROMPT_BUILDERS:
            # Copy all columns, converting pyarrow/numpy types to plain Python
            new_row = {}
            for col in df.columns:
                val = row[col]
                if hasattr(val, 'as_py'):
                    # pyarrow scalar
                    new_row[col] = val.as_py()
                elif hasattr(val, 'tolist'):
                    # numpy array
                    new_row[col] = val.tolist()
                elif isinstance(val, (dict, list)):
                    new_row[col] = copy.deepcopy(val)
                else:
                    new_row[col] = val

            new_row["prompt"] = build_prompt(original_content)
            new_row["data_source"] = data_source
            rows.append(new_row)

    out_df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(output_path)

    print(f"Output: {output_path} ({len(out_df)} rows)")
    for src in [NO_COT_SOURCE, TRAIN_COT_SOURCE, QWEN_COT_SOURCE]:
        print(f"  {src}: {len(out_df[out_df['data_source'] == src])} rows")

    # Preview prompt structure for each variant
    print("\nSample prompt structures:")
    for src in [NO_COT_SOURCE, TRAIN_COT_SOURCE, QWEN_COT_SOURCE]:
        sample = out_df[out_df["data_source"] == src].iloc[0]
        msgs = sample["prompt"]
        print(f"\n  [{src}]")
        for msg in msgs:
            content_preview = msg["content"][:80] + "..." if len(msg["content"]) > 80 else msg["content"]
            print(f"    {msg['role']}: {content_preview}")

    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare multi-prompt validation dataset")
    parser.add_argument(
        "--input",
        type=str,
        default="./data/math500.parquet",
        help="Input validation parquet file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output parquet file (default: input stem + _multi_prompt.parquet)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}_multi_prompt.parquet"

    create_multi_prompt_val(input_path, output_path)
