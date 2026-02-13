"""Generation-based evaluation for A* grokking experiments.

Generates model completions and evaluates path correctness,
measuring the metrics that matter for grokking:
  - optimal_accuracy (main signal)
  - valid_path_rate
  - format_rate
  - avg_reward
  - avg_gen_tokens
"""

import re
import time

import pandas as pd
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from .reward import compute_metrics


def load_eval_instances(parquet_path: str, max_samples: int = 50, seed: int = 42) -> list[dict]:
    """Load evaluation instances from parquet, returning raw row dicts."""
    df = pd.read_parquet(parquet_path)
    if max_samples > 0 and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=seed)
    return df.to_dict("records")


def build_prompt(messages: list[dict], tokenizer: PreTrainedTokenizer) -> str:
    """Build a generation prompt from system + user messages."""
    prompt_messages = messages[:2]  # system + user only
    return tokenizer.apply_chat_template(
        prompt_messages,
        add_generation_prompt=True,
        tokenize=False,
    )


def _parse_coord_str(s: str) -> tuple[int, int]:
    m = re.match(r"\((\d+),\s*(\d+)\)", s)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def run_generation_eval(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    parquet_path: str,
    max_samples: int = 50,
    max_new_tokens: int = 2048,
    prefix: str = "eval",
) -> dict:
    """Run generation eval on a set of instances.

    Args:
        model: Unwrapped HuggingFace model (not FSDP-wrapped).
        tokenizer: The tokenizer.
        parquet_path: Path to parquet with messages, optimal_path, etc.
        max_samples: How many samples to evaluate.
        max_new_tokens: Max tokens per generation.
        prefix: Metric prefix (e.g. "eval" or "train_gen").

    Returns:
        Dict of metrics like {"{prefix}/optimal_accuracy": 0.42, ...}
    """
    instances = load_eval_instances(parquet_path, max_samples)
    if not instances:
        return {}

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    device = next(model.parameters()).device
    model.eval()

    all_metrics = []
    total_gen_tokens = 0
    t0 = time.time()

    with torch.no_grad():
        for inst in instances:
            messages = inst["messages"]
            prompt = build_prompt(messages, tokenizer)
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

            output = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

            gen_ids = output[0][input_ids.shape[1]:]
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            gen_tokens = len(gen_ids)
            total_gen_tokens += gen_tokens

            start = inst["start"]
            goal = inst["goal"]
            if isinstance(start, str):
                start = _parse_coord_str(start)
            if isinstance(goal, str):
                goal = _parse_coord_str(goal)

            m = compute_metrics(
                model_output=gen_text,
                grid_string=inst["grid_string"],
                start=start,
                goal=goal,
                optimal_path=inst["optimal_path"],
                optimal_path_length=inst["optimal_path_length"],
            )
            m["gen_tokens"] = gen_tokens
            all_metrics.append(m)

    elapsed = time.time() - t0
    n = len(all_metrics)

    return {
        f"{prefix}/optimal_accuracy": sum(m["path_optimal"] for m in all_metrics) / n,
        f"{prefix}/valid_path_rate": sum(m["path_valid"] for m in all_metrics) / n,
        f"{prefix}/format_rate": sum(m["format_valid"] for m in all_metrics) / n,
        f"{prefix}/avg_reward": sum(m["reward"] for m in all_metrics) / n,
        f"{prefix}/avg_gen_tokens": total_gen_tokens / n,
        f"{prefix}/eval_time_s": elapsed,
        f"{prefix}/n_samples": n,
    }
