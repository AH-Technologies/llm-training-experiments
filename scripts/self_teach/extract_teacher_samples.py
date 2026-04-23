"""Extract a few teacher log entries for manual inspection."""
import json
import sys

path = "checkpoints/rlvr-grokking/self_teach_teacher_only_conditioned_dapo_qwen3/teacher_logs/teacher_log_20260324_221014.jsonl"
n = 10

with open(path) as f:
    for i, line in enumerate(f):
        if i >= n:
            break
        entry = json.loads(line)
        out = {
            "step": entry.get("step"),
            "ground_truth": entry.get("ground_truth"),
            "a1_correct": entry.get("a1_correct"),
            "teacher_reward": entry.get("teacher_reward"),
            "question": entry.get("question", "")[:200],
            "feedback": entry.get("feedback", "")[:500],
        }
        print(json.dumps(out, indent=2))
        print("---")
