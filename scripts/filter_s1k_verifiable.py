"""Filter simplescaling/s1K dataset to keep only verifiable answers.

Phase 1: Keep bare verifiable values (numeric, boolean, MC, number lists).
Phase 2: Extract final answers from free-text solutions via \boxed{} and
         "Answer: X" patterns, then check if the extracted answer is verifiable.

Outputs:
  - data/s1K/s1k_verifiable.parquet  (filtered dataset)
  - data/s1K/filter_audit.json       (extraction audit log)
"""

import json
import re
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path


# ── Verifiability checks ──────────────────────────────────────────────────

def is_numeric(s: str) -> bool:
    s = s.strip().replace(",", "").replace("$", "").replace("%", "")
    try:
        float(s)
        return True
    except ValueError:
        return False


def is_boolean(s: str) -> bool:
    return s.strip().lower() in ("true", "false", "yes", "no")


def is_multiple_choice(s: str) -> bool:
    return bool(re.fullmatch(r"[A-E]{1,5}", s.strip()))


def is_number_list(s: str) -> bool:
    return bool(re.fullmatch(r"\[[\d.,\s-]+\]", s.strip()))


def is_verifiable(s: str) -> bool:
    s = s.strip()
    return is_numeric(s) or is_boolean(s) or is_multiple_choice(s) or is_number_list(s)


def classify_answer(s: str) -> str | None:
    s = s.strip()
    if is_numeric(s):
        return "numerical"
    if is_boolean(s):
        return "boolean"
    if is_multiple_choice(s):
        return "multiple_choice"
    if is_number_list(s):
        return "number_list"
    return None


# ── Answer extraction from free-text solutions ────────────────────────────

def extract_last_boxed(text: str) -> str | None:
    """Extract content from the last \\boxed{...} in text, handling nested braces."""
    last = None
    idx = 0
    while idx < len(text):
        pos = text.find("\\boxed{", idx)
        if pos < 0:
            break
        start = pos + len("\\boxed{") - 1  # points to '{'
        depth = 0
        i = start
        while i < len(text):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    last = text[start + 1:i]
                    break
            i += 1
        idx = i + 1 if i < len(text) else len(text)
    return last.strip() if last is not None else None


def extract_answer_label(text: str) -> str | None:
    """Extract answer from 'Answer: X' or 'Answer is X' patterns."""
    m = re.search(r"(?:^|\n)\s*[Aa]nswer(?:\s+is)?[:\s]+([^\n]+)", text)
    if m:
        ans = m.group(1).strip().rstrip(".")
        if len(ans) < 50:
            return ans
    return None


def clean_extracted(s: str) -> str:
    """Strip LaTeX wrappers to get a bare value for verifiability check."""
    s = s.strip()
    # Remove surrounding $...$
    s = re.sub(r"^\$+|\$+$", "", s).strip()
    # Remove \text{...} wrapper
    s = re.sub(r"^\\text\{(.+)\}$", r"\1", s).strip()
    # Remove trailing periods
    s = s.rstrip(".")
    # Remove dollar signs and percent
    s = s.replace("$", "").replace("\\$", "")
    return s.strip()


def try_extract_verifiable(solution: str) -> tuple[str | None, str | None]:
    """Try to extract a verifiable answer from a free-text solution.

    Returns (extracted_answer, extraction_method) or (None, None).
    """
    # Strategy 1: Last \boxed{...}
    boxed = extract_last_boxed(solution)
    if boxed is not None:
        cleaned = clean_extracted(boxed)
        if is_verifiable(cleaned):
            return cleaned, "boxed"

    # Strategy 2: "Answer: X" label (common for multiple choice)
    label = extract_answer_label(solution)
    if label is not None:
        cleaned = clean_extracted(label)
        if is_verifiable(cleaned):
            return cleaned, "answer_label"

    return None, None


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    input_path = Path("data/s1K/s1k.parquet")
    output_path = Path("data/s1K/s1k_verifiable.parquet")
    audit_path = Path("data/s1K/filter_audit.json")

    table = pq.read_table(input_path)
    solutions = table.column("solution").to_pylist()
    total = len(solutions)

    keep_indices = []
    extracted_answers = []  # parallel to keep_indices
    audit_entries = []

    stats = {
        "bare_numerical": 0,
        "bare_boolean": 0,
        "bare_multiple_choice": 0,
        "bare_number_list": 0,
        "extracted_boxed": 0,
        "extracted_answer_label": 0,
        "removed": 0,
    }

    for i, sol in enumerate(solutions):
        raw = sol.strip()

        # Phase 1: bare verifiable value
        cat = classify_answer(raw)
        if cat is not None:
            keep_indices.append(i)
            extracted_answers.append(raw)
            stats[f"bare_{cat}"] += 1
            audit_entries.append({
                "index": i,
                "method": "bare",
                "category": cat,
                "answer": raw[:100],
            })
            continue

        # Phase 2: extract from free-text
        extracted, method = try_extract_verifiable(sol)
        if extracted is not None:
            keep_indices.append(i)
            extracted_answers.append(extracted)
            stats[f"extracted_{method}"] += 1
            audit_entries.append({
                "index": i,
                "method": method,
                "category": classify_answer(extracted),
                "answer": extracted[:100],
                "original_length": len(raw),
            })
            continue

        stats["removed"] += 1

    # Build filtered table, replacing solution column with extracted answers
    filtered = table.take(keep_indices)
    # Replace the solution column with the clean extracted answers
    col_idx = filtered.schema.get_field_index("solution")
    filtered = filtered.set_column(
        col_idx, "solution", pa.array(extracted_answers, type=pa.string())
    )

    pq.write_table(filtered, output_path)

    # Print stats
    kept = len(keep_indices)
    bare = sum(v for k, v in stats.items() if k.startswith("bare_"))
    extracted_total = sum(v for k, v in stats.items() if k.startswith("extracted_"))

    print(f"Total: {total}")
    print(f"Kept:  {kept} ({kept/total*100:.1f}%)")
    print(f"  Bare values:       {bare}")
    for k, v in stats.items():
        if k.startswith("bare_") and v > 0:
            print(f"    {k}: {v}")
    print(f"  Extracted answers: {extracted_total}")
    for k, v in stats.items():
        if k.startswith("extracted_") and v > 0:
            print(f"    {k}: {v}")
    print(f"Removed: {stats['removed']} (truly free-text)")
    print(f"Saved to {output_path}")

    # Save audit log
    audit = {
        "total": total,
        "kept": kept,
        "stats": stats,
        "entries": audit_entries,
    }
    with open(audit_path, "w") as f:
        json.dump(audit, f, indent=2)
    print(f"Audit log saved to {audit_path}")


if __name__ == "__main__":
    main()
