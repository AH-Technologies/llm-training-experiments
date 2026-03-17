#!/usr/bin/env python3
"""Verify the mapping between pi names, parquet rows, and extra_info indices."""
import json
import numpy as np
import pandas as pd

PARQUET = "../../one_shot_metrics/One-Shot-RLVR/data/train/one_shot_rlvr/dsr_sub.parquet"
PARQUET = "/cluster/projects/nn12068k/haaklau/llm-training-experiments/one_shot_metrics/One-Shot-RLVR/data/train/one_shot_rlvr/dsr_sub.parquet"
ACC_JSON = "/cluster/projects/nn12068k/haaklau/llm-training-experiments/one_shot_metrics/One-Shot-RLVR/data/acc_step_500.json"

df = pd.read_parquet(PARQUET)

# Get extra_info index for each row
print(f"Parquet has {len(df)} rows")
print()

# Show first few rows: row position vs extra_info index
print("Row#  extra_info['index']  prompt_start")
print("-" * 80)
for i in range(20):
    row = df.iloc[i]
    ei = row["extra_info"]["index"]
    prompt = row["prompt"]
    if hasattr(prompt, "tolist"):
        prompt = prompt.tolist()
    p = prompt[0]["content"][:70].replace("\n", " ")
    print(f"{i:<6} {ei:<20} {p}")

# Now check: are row positions == extra_info indices?
print()
mismatches = 0
for i in range(len(df)):
    ei = df.iloc[i]["extra_info"]["index"]
    if i != ei:
        mismatches += 1
        if mismatches <= 5:
            print(f"MISMATCH: row {i} has extra_info index {ei}")

if mismatches == 0:
    print("All row positions match extra_info indices!")
else:
    print(f"Total mismatches: {mismatches} out of {len(df)}")

# Now sort acc_step_500 by std to get the pi ranking
with open(ACC_JSON) as f:
    acc = json.load(f)

ranked = sorted(acc.items(), key=lambda x: np.std(x[1]), reverse=True)

# The WANG_EXAMPLES mapping says pi_1 -> 124, pi_2 -> 267, etc.
# These should be extra_info indices, not row positions.
# Let's check if the data_selection.py from Wang's code uses the same indexing.
print()
print("=== Pi ranking by std (top 20) ===")
print("pi_rank  acc_step_key  std      parquet_row_prompt")
for i, (idx_str, vals) in enumerate(ranked[:20]):
    idx = int(idx_str)
    # Find the parquet row with this extra_info index
    for j in range(len(df)):
        if df.iloc[j]["extra_info"]["index"] == idx:
            prompt = df.iloc[j]["prompt"]
            if hasattr(prompt, "tolist"):
                prompt = prompt.tolist()
            p = prompt[0]["content"][:60].replace("\n", " ")
            print(f"pi_{i+1:<5} {idx:<13} {np.std(vals):.4f}  row={j}: {p}")
            break

# Key question: what does the CURRENT extract_examples produce?
print()
print("=== What extract_examples() currently returns ===")
import sys
sys.path.insert(0, "/cluster/projects/nn12068k/haaklau/llm-training-experiments/one_shot_metrics/entropy_profiling")
from extract_examples import extract_examples, WANG_EXAMPLES

examples = extract_examples()
for ex in examples:
    print(f"  {ex['name']:<10} idx={ex['index']:<5} gt={str(ex['ground_truth'])[:25]:<25} prompt: {ex['prompt_text'][:70].replace(chr(10), ' ')}...")
