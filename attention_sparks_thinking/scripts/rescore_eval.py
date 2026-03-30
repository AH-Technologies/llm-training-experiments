#!/usr/bin/env python3
"""Re-score eval_extraction.jsonl with improved answer matching."""
import json
import re
import sys
from fractions import Fraction
import math
from typing import Optional, Tuple

sys.path.insert(0, ".")


def normalize_math_answer_v2(answer: str) -> str:
    """Improved normalization that handles symbolic expressions better."""
    if answer is None:
        return ""

    s = answer.strip()

    # Remove \left and \right but preserve delimiters
    s = re.sub(r'\\left\s*([(\[{])', r'\1', s)
    s = re.sub(r'\\right\s*([)\]}])', r'\1', s)
    # Handle \left. and \right. (invisible delimiters)
    s = re.sub(r'\\left\s*\.', '', s)
    s = re.sub(r'\\right\s*\.', '', s)

    # Remove \displaystyle
    s = re.sub(r'\\displaystyle\s*', '', s)

    # Handle \dfrac, \tfrac same as \frac
    s = re.sub(r'\\[dt]frac', r'\\frac', s)

    # Remove spaces (but keep structure)
    s = re.sub(r'\s+', '', s)

    # Normalize \cdot, \times, \div
    s = s.replace('\\cdot', '*')
    s = s.replace('\\times', '*')
    s = s.replace('\\div', '/')
    s = s.replace('\\pm', '±')

    # Remove \text{} wrapper but keep content
    s = re.sub(r'\\text\{([^}]*)\}', r'\1', s)

    # Normalize degree symbol
    s = s.replace('^\\circ', '°')
    s = s.replace('^{\\circ}', '°')

    return s


def try_parse_number_v2(s):
    """Parse number, handling fractions and common expressions."""
    if not s:
        return None
    s = s.strip().replace('−', '-')

    # Remove wrapping parens
    if s.startswith('(') and s.endswith(')'):
        inner = s[1:-1]
        if inner.count('(') == inner.count(')'):
            s = inner

    try:
        return float(s)
    except ValueError:
        pass

    # Fraction a/b
    if '/' in s and '\\' not in s:
        parts = s.split('/')
        if len(parts) == 2:
            try:
                return float(parts[0]) / float(parts[1])
            except (ValueError, ZeroDivisionError):
                pass

    # Handle \frac{a}{b}
    frac_match = re.match(r'^\\frac\{([^{}]+)\}\{([^{}]+)\}$', s)
    if frac_match:
        try:
            num = float(frac_match.group(1))
            den = float(frac_match.group(2))
            if den != 0:
                return num / den
        except ValueError:
            pass

    return None


def answers_match_v2(extracted, ground_truth):
    """Improved answer matching with better symbolic handling."""
    norm_ext = normalize_math_answer_v2(extracted)
    norm_truth = normalize_math_answer_v2(ground_truth)

    # Exact string match after normalization
    if norm_ext == norm_truth:
        return True, "exact_string_v2"

    # Try numeric comparison
    ext_val = try_parse_number_v2(norm_ext)
    truth_val = try_parse_number_v2(norm_truth)

    if ext_val is not None and truth_val is not None:
        if abs(ext_val - truth_val) < 1e-6:
            return True, "numeric_v2"
        if truth_val != 0 and abs((ext_val - truth_val) / truth_val) < 1e-6:
            return True, "numeric_relative_v2"

    # Try matching with \frac converted to decimal
    # e.g., \frac{14}{3} should match 4.666...
    frac_ext = re.match(r'^\\frac\{([^{}]+)\}\{([^{}]+)\}$', norm_ext)
    frac_truth = re.match(r'^\\frac\{([^{}]+)\}\{([^{}]+)\}$', norm_truth)

    if frac_truth and ext_val is not None:
        try:
            truth_num = float(frac_truth.group(1))
            truth_den = float(frac_truth.group(2))
            if abs(ext_val - truth_num / truth_den) < 1e-6:
                return True, "frac_to_decimal"
        except ValueError:
            pass

    if frac_ext and truth_val is not None:
        try:
            ext_num = float(frac_ext.group(1))
            ext_den = float(frac_ext.group(2))
            if abs(ext_num / ext_den - truth_val) < 1e-6:
                return True, "decimal_to_frac"
        except ValueError:
            pass

    return False, "no_match_v2"


# Load and rescore
INPUT = sys.argv[1] if len(sys.argv) > 1 else "attention_sparks_thinking/logs/eval_extraction.jsonl"

results = []
with open(INPUT) as f:
    for line in f:
        results.append(json.loads(line))

old_correct = sum(1 for r in results if r["correct"])

for r in results:
    extracted = r["extracted"]
    gt = r["ground_truth"]

    # Also try re-extracting from response (look for last \boxed{})
    response = r["response"]
    # Find ALL boxed answers and take the last one
    all_boxed = []
    pattern = r'\\boxed\s*\{'
    for m in re.finditer(pattern, response):
        start = m.end()
        brace_count = 1
        pos = start
        while pos < len(response) and brace_count > 0:
            if response[pos] == '{':
                brace_count += 1
            elif response[pos] == '}':
                brace_count -= 1
            pos += 1
        if brace_count == 0:
            all_boxed.append(response[start:pos-1].strip())

    last_boxed = all_boxed[-1] if all_boxed else None

    # Score with original extraction
    match_orig, type_orig = answers_match_v2(extracted or "", gt)

    # Score with last boxed
    match_last_boxed, type_last_boxed = answers_match_v2(last_boxed or "", gt) if last_boxed else (False, "no_boxed")

    best_match = match_orig or match_last_boxed
    best_type = type_orig if match_orig else type_last_boxed

    r["v2_correct"] = best_match
    r["v2_match_type"] = best_type
    r["v2_extracted"] = extracted
    r["v2_last_boxed"] = last_boxed
    r["v2_norm_gt"] = normalize_math_answer_v2(gt)
    r["v2_norm_ext"] = normalize_math_answer_v2(extracted) if extracted else ""
    r["v2_norm_last_boxed"] = normalize_math_answer_v2(last_boxed) if last_boxed else ""

new_correct = sum(1 for r in results if r["v2_correct"])

print(f"Old scoring: {old_correct}/{len(results)} correct ({100*old_correct/len(results):.1f}%)")
print(f"New scoring: {new_correct}/{len(results)} correct ({100*new_correct/len(results):.1f}%)")
print()

for r in results:
    changed = r["correct"] != r["v2_correct"]
    marker = " ** CHANGED **" if changed else ""
    status = "CORRECT" if r["v2_correct"] else "WRONG"
    print(f"[{status}] idx={r['idx']}{marker}")
    print(f"  GT:             {r['ground_truth']}")
    print(f"  Norm GT (v2):   {r['v2_norm_gt']}")
    print(f"  Extracted:      {r['extracted']}")
    print(f"  Last \\boxed:    {r['v2_last_boxed']}")
    print(f"  Old match:      {r['match_type']} -> {'CORRECT' if r['correct'] else 'WRONG'}")
    print(f"  New match:      {r['v2_match_type']} -> {'CORRECT' if r['v2_correct'] else 'WRONG'}")
    print()
