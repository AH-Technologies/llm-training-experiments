#!/usr/bin/env python3
"""Analyze linguistic patterns across all ablation runs."""
import json
import re
import os
import sys
from collections import defaultdict
from pathlib import Path

RESULTS_BASE = Path("/cluster/projects/nn12068k/haaklau/llm-training-experiments/results/reasoning_head_analysis")

RUNS = {
    "pi13_step700": RESULTS_BASE / "ablation" / "pi13_step700" / "math500",
    "pi1_step700": RESULTS_BASE / "ablation" / "pi1_step700" / "math500",
    "base_common": RESULTS_BASE / "ablation" / "base_common_heads" / "math500",
    "base_own": RESULTS_BASE / "ablation" / "base_own_heads" / "math500",
}

# Conditions to analyze per run
TARGET_CONDITIONS = {
    "pi13_step700": [
        "baseline", "topk_ablate_0.0", "topk_scale_1.5", "topk_scale_2.0",
        "topk_scale_3.0", "topk_scale_4.0",
        "indiv_L15H7_0.0", "indiv_L17H11_0.0", "indiv_L10H5_0.0",
        "indiv_L18H8_0.0", "indiv_L19H11_0.0",
    ],
    "pi1_step700": [
        "baseline", "topk_ablate_0.0", "topk_scale_1.5", "topk_scale_2.0",
        "topk_scale_3.0", "topk_scale_4.0",
        "indiv_L15H7_0.0", "indiv_L15H9_0.0", "indiv_L18H8_0.0",
    ],
    "base_common": [
        "baseline", "topk_ablate_0.0", "topk_scale_1.5", "topk_scale_3.0",
        "indiv_L15H11_0.0", "indiv_L15H7_0.0",
    ],
    "base_own": [
        "baseline", "topk_ablate_0.0", "topk_scale_1.5", "topk_scale_3.0",
        "indiv_L2H4_0.0", "indiv_L19H6_0.0",
    ],
}

REASONING_WORDS = [
    "therefore", "thus", "hence", "so ", "because", "since ",
    "note that", "recall that", "observe that", "notice that",
    "it follows", "we have", "we get", "we find", "we know",
    "we need", "we can", "we see", "this means", "this gives",
    "this implies", "which means", "which gives",
    "let's", "let us", "first", "next", "then ", "finally",
    "step", "now ",
]

SELF_CORRECTION = [
    "wait", "actually", "no,", "sorry", "mistake", "error",
    "wrong", "correct this", "let me reconsider", "let me recalculate",
    "I made", "oops", "but wait", "hold on",
]

THINKING_PHRASES = [
    "let me think", "I need to", "the key", "the idea",
    "approach", "strategy", "method", "technique",
]

VERIFICATION_PHRASES = [
    "verify", "check", "confirm", "substitute", "plug in",
    "test", "validate",
]

HESITATION = ["hmm", "alternatively", "but ", "however", "actually"]

CJK_RE = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf\uac00-\ud7af\u3040-\u309f\u30a0-\u30ff]')
NON_ASCII_RE = re.compile(r'[^\x00-\x7f]')


def count_phrase_occurrences(text, phrases):
    """Count total occurrences of phrases in text (case-insensitive)."""
    text_lower = text.lower()
    total = 0
    for phrase in phrases:
        total += text_lower.count(phrase.lower())
    return total


def analyze_response(text):
    """Compute all metrics for a single response."""
    text_lower = text.lower()
    n_chars = len(text)

    metrics = {}
    metrics["length_chars"] = n_chars
    metrics["reasoning_words"] = count_phrase_occurrences(text, REASONING_WORDS)
    metrics["self_correction"] = count_phrase_occurrences(text, SELF_CORRECTION)
    metrics["thinking"] = count_phrase_occurrences(text, THINKING_PHRASES)
    metrics["verification"] = count_phrase_occurrences(text, VERIFICATION_PHRASES)
    metrics["hesitation"] = count_phrase_occurrences(text, HESITATION)
    metrics["has_code"] = 1 if ("```python" in text_lower or "```py" in text_lower or "import " in text_lower) else 0
    metrics["has_boxed"] = 1 if "\\boxed" in text else 0
    metrics["has_cjk"] = 1 if CJK_RE.search(text) else 0
    metrics["n_cjk_chars"] = len(CJK_RE.findall(text))

    # Non-ASCII ratio
    non_ascii = len(NON_ASCII_RE.findall(text))
    metrics["non_ascii_ratio"] = non_ascii / max(n_chars, 1)

    # Count specific reasoning word categories
    metrics["causal"] = sum(text_lower.count(w.lower()) for w in ["therefore", "thus", "hence", "because", "since ", "it follows"])
    metrics["sequential"] = sum(text_lower.count(w.lower()) for w in ["first", "next", "then ", "finally", "step"])
    metrics["connective"] = sum(text_lower.count(w.lower()) for w in ["so ", "now ", "we have", "we get", "we find", "we know", "we need", "we can", "we see"])

    # Normalize per 1000 chars
    if n_chars > 0:
        metrics["reasoning_per_1k"] = metrics["reasoning_words"] / n_chars * 1000
        metrics["self_correction_per_1k"] = metrics["self_correction"] / n_chars * 1000
        metrics["verification_per_1k"] = metrics["verification"] / n_chars * 1000
        metrics["causal_per_1k"] = metrics["causal"] / n_chars * 1000
        metrics["sequential_per_1k"] = metrics["sequential"] / n_chars * 1000
    else:
        metrics["reasoning_per_1k"] = 0
        metrics["self_correction_per_1k"] = 0
        metrics["verification_per_1k"] = 0
        metrics["causal_per_1k"] = 0
        metrics["sequential_per_1k"] = 0

    return metrics


def load_responses(run_dir, target_conditions):
    """Load all responses from sharded JSONL files, filtered by conditions."""
    responses = defaultdict(list)
    for jsonl_file in sorted(run_dir.glob("full_responses_shard*.jsonl")):
        with open(jsonl_file, encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                cond = d.get("condition", "")
                if cond in target_conditions:
                    responses[cond].append(d)
    return responses


def aggregate_metrics(responses_list):
    """Aggregate metrics across all responses for a condition."""
    if not responses_list:
        return {}

    all_metrics = [analyze_response(r["response"]) for r in responses_list]
    n = len(all_metrics)

    agg = {}
    agg["n_responses"] = n
    agg["n_correct"] = sum(1 for r in responses_list if r.get("correct", False))
    agg["accuracy"] = agg["n_correct"] / n

    for key in ["length_chars", "reasoning_words", "self_correction", "thinking",
                 "verification", "hesitation", "reasoning_per_1k", "self_correction_per_1k",
                 "verification_per_1k", "causal_per_1k", "sequential_per_1k",
                 "non_ascii_ratio", "causal", "sequential", "connective", "n_cjk_chars"]:
        vals = [m[key] for m in all_metrics]
        agg[f"mean_{key}"] = sum(vals) / n

    for key in ["has_code", "has_boxed", "has_cjk"]:
        agg[f"pct_{key}"] = sum(m[key] for m in all_metrics) / n * 100

    return agg


def main():
    all_results = {}

    for run_name, run_dir in RUNS.items():
        print(f"\n{'='*80}")
        print(f"  {run_name}")
        print(f"{'='*80}")

        if not run_dir.exists():
            print(f"  Directory not found: {run_dir}")
            continue

        targets = TARGET_CONDITIONS.get(run_name, [])
        responses = load_responses(run_dir, targets)

        run_results = {}
        for cond in targets:
            if cond not in responses:
                print(f"  {cond}: NOT FOUND")
                continue
            agg = aggregate_metrics(responses[cond])
            run_results[cond] = agg

        all_results[run_name] = run_results

        # Print table
        print(f"\n{'Condition':<25} {'Acc':>5} {'Len':>6} {'Reason/1k':>9} {'SelfCorr/1k':>11} {'Verif/1k':>8} {'Causal/1k':>9} {'Seq/1k':>6} {'Code%':>5} {'Boxed%':>6} {'CJK%':>5} {'NonASCII':>8}")
        print("-" * 130)
        for cond in targets:
            if cond not in run_results:
                continue
            a = run_results[cond]
            print(f"{cond:<25} {a['accuracy']:>5.1%} {a['mean_length_chars']:>6.0f} {a['mean_reasoning_per_1k']:>9.2f} {a['mean_self_correction_per_1k']:>11.3f} {a['mean_verification_per_1k']:>8.3f} {a['mean_causal_per_1k']:>9.3f} {a['mean_sequential_per_1k']:>6.3f} {a['pct_has_code']:>5.1f} {a['pct_has_boxed']:>6.1f} {a['pct_has_cjk']:>5.1f} {a['mean_non_ascii_ratio']:>8.3f}")

    # Cross-run comparison for baselines
    print(f"\n\n{'='*80}")
    print("  CROSS-RUN BASELINE COMPARISON")
    print(f"{'='*80}")
    print(f"\n{'Run':<20} {'Acc':>5} {'Len':>6} {'Reason/1k':>9} {'SelfCorr/1k':>11} {'Verif/1k':>8} {'Code%':>5} {'Boxed%':>6} {'CJK%':>5}")
    print("-" * 90)
    for run_name in RUNS:
        if run_name in all_results and "baseline" in all_results[run_name]:
            a = all_results[run_name]["baseline"]
            print(f"{run_name:<20} {a['accuracy']:>5.1%} {a['mean_length_chars']:>6.0f} {a['mean_reasoning_per_1k']:>9.2f} {a['mean_self_correction_per_1k']:>11.3f} {a['mean_verification_per_1k']:>8.3f} {a['pct_has_code']:>5.1f} {a['pct_has_boxed']:>6.1f} {a['pct_has_cjk']:>5.1f}")

    # Dump raw numbers as JSON for the document
    output_path = RESULTS_BASE / "linguistic_analysis.json"
    # Convert for JSON serialization
    json_results = {}
    for run_name, conds in all_results.items():
        json_results[run_name] = {}
        for cond, metrics in conds.items():
            json_results[run_name][cond] = {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()}

    with open(output_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\nSaved raw data to {output_path}")


if __name__ == "__main__":
    main()
