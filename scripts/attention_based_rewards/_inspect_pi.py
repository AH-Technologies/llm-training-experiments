#!/usr/bin/env python3
"""Quick inspection of pi example parquets."""
import pandas as pd
import glob

# Check what pi parquets we have
files = glob.glob("one_shot_metrics/One-Shot-RLVR/data/train/one_shot_rlvr/pi*.parquet")
for f in sorted(files):
    df = pd.read_parquet(f)
    prompt = df.iloc[0]["prompt"]
    gt = df.iloc[0]["reward_model"]
    for msg in prompt:
        if msg["role"] == "user":
            text = msg["content"][:120].replace("\n", " ")
            print(f"{f}: GT={gt}, n={len(df)}")
            print(f"  prompt: {text}...")
            break

# We need all 14 pi examples. Check if there's a bigger dataset
print("\n--- Checking entropy profiling data source ---")
import json
with open("one_shot_metrics/entropy_profiling/results/entropy_features.json") as f:
    data = json.load(f)
print(f"Examples in entropy_features.json: {list(data.keys())}")

# Check the large_run for pi example data
large_features = "one_shot_metrics/entropy_profiling/results/large_run/features_full.csv"
try:
    df_large = pd.read_csv(large_features)
    pi_examples = df_large[df_large["example"].str.startswith("pi_")]
    print(f"\nPi examples in large_run: {pi_examples['example'].tolist()}")
except Exception as e:
    print(f"No large_run features: {e}")

# Check the DAPO training data for the pi problems
print("\n--- Checking DAPO data ---")
df_dapo = pd.read_parquet("attention_based_rewards/data/dapo_math_17k.parquet")
print(f"DAPO columns: {df_dapo.columns.tolist()}")
print(f"DAPO shape: {df_dapo.shape}")
