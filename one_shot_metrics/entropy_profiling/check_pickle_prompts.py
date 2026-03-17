#!/usr/bin/env python3
"""Check all pickle prompts vs what's in the parquet."""
import pickle
import glob

print("=== All Wang 14 pickle files ===")
for pkl in sorted(glob.glob("results/entropy_profiles/entropy_pi_*.pkl")):
    with open(pkl, "rb") as f:
        d = pickle.load(f)
    e = d["example"]
    name = e["name"]
    idx = e.get("index", "?")
    gt = str(e["ground_truth"])[:25]
    prompt_start = e["prompt_text"][:100].replace("\n", " ")
    print(f"  {name:<10} idx={idx:<5} gt={gt:<25} prompt: {prompt_start}...")
