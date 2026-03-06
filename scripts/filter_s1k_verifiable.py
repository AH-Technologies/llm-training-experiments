"""Filter simplescaling/s1K dataset to keep only verifiable answers.

Keeps: numerical, boolean (true/false), multiple-choice (letter combos), number lists.
Removes: free-text proofs, derivations, explanations, chemical formulas.
"""

import re
import pyarrow.parquet as pq
from pathlib import Path


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


def is_verifiable(solution: str) -> bool:
    s = solution.strip()
    return is_numeric(s) or is_boolean(s) or is_multiple_choice(s) or is_number_list(s)


def main():
    input_path = Path("data/s1K/train-00000-of-00001.parquet")
    output_path = Path("data/s1K/s1k_verifiable.parquet")

    table = pq.read_table(input_path)
    solutions = table.column("solution").to_pylist()

    keep = [i for i, s in enumerate(solutions) if is_verifiable(s)]
    filtered = table.take(keep)

    pq.write_table(filtered, output_path)

    # Print stats
    total = table.num_rows
    kept = filtered.num_rows
    cats = {"numerical": 0, "boolean": 0, "multiple_choice": 0, "number_list": 0}
    for i in keep:
        s = solutions[i].strip()
        if is_numeric(s):
            cats["numerical"] += 1
        elif is_boolean(s):
            cats["boolean"] += 1
        elif is_multiple_choice(s):
            cats["multiple_choice"] += 1
        elif is_number_list(s):
            cats["number_list"] += 1

    print(f"Total: {total} -> Kept: {kept} ({kept/total*100:.1f}%)")
    for cat, count in cats.items():
        print(f"  {cat}: {count}")
    print(f"Removed: {total - kept} free-text answers")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
