"""Generate a correct response cache for BER (Bidirectional Experience Replay).

Loads the base model and training prompt, then rejection-samples at temperature 0.6
until a correct response is found. Saves the tokenized response for BER injection.

Usage:
    python scripts/generate_correct_cache.py \
        --model Qwen/Qwen2.5-Math-1.5B \
        --train_file data/pi13_r128.parquet \
        --output data/ber_correct_cache_pi13.pt \
        --max_attempts 64 \
        --temperature 0.6
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add project root to path for reward function import
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.rlvr_grokking.rewards.deepscaler_reward import compute_score


def extract_prompt_and_ground_truth(train_file: str):
    """Load the training parquet and extract the first prompt + ground truth."""
    df = pd.read_parquet(train_file)
    row = df.iloc[0]

    # Get prompt messages (list of dicts with role/content)
    prompt_msgs = row["prompt"]
    if not isinstance(prompt_msgs, list):
        prompt_msgs = [dict(m) for m in prompt_msgs]

    # Get ground truth answer
    ground_truth = row["reward_model"]["ground_truth"]
    if not isinstance(ground_truth, str):
        ground_truth = str(ground_truth)

    return prompt_msgs, ground_truth


def generate_correct_response(
    model,
    tokenizer,
    prompt_msgs: list[dict],
    ground_truth: str,
    max_attempts: int = 64,
    temperature: float = 0.6,
    max_new_tokens: int = 3072,
) -> torch.Tensor:
    """Rejection-sample until we get a correct response.

    Returns:
        1D tensor of response token IDs (without prompt tokens, no padding).
    """
    # Apply chat template to get prompt tokens
    prompt_text = tokenizer.apply_chat_template(
        prompt_msgs, tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(model.device)
    prompt_len = prompt_ids.shape[1]

    print(f"Prompt length: {prompt_len} tokens")
    print(f"Ground truth: {ground_truth}")
    print(f"Sampling at temperature={temperature}, max_new_tokens={max_new_tokens}")

    for attempt in range(1, max_attempts + 1):
        with torch.no_grad():
            output = model.generate(
                prompt_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=1.0,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        # Extract response tokens (everything after prompt)
        response_ids = output[0, prompt_len:]
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True)

        # Grade using the same reward function as training
        score = compute_score(
            data_source="",
            solution_str=response_text,
            ground_truth=ground_truth,
        )

        print(f"  Attempt {attempt}: score={score:.0f}, response_len={len(response_ids)} tokens")

        if score >= 1.0:
            print(f"  Found correct response on attempt {attempt}!")
            # Preview the answer
            boxed_idx = response_text.rfind("\\boxed")
            if boxed_idx >= 0:
                print(f"  Answer snippet: ...{response_text[max(0, boxed_idx-20):boxed_idx+80]}")
            return response_ids.cpu()

    raise RuntimeError(
        f"Failed to find a correct response after {max_attempts} attempts. "
        "Try increasing --max_attempts or using a stronger model."
    )


def main():
    parser = argparse.ArgumentParser(description="Generate correct cache for BER")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--max_attempts", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max_new_tokens", type=int, default=3072)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    args = parser.parse_args()

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map="auto", trust_remote_code=True
    )
    model.eval()

    print(f"Loading training data: {args.train_file}")
    prompt_msgs, ground_truth = extract_prompt_and_ground_truth(args.train_file)

    print("Generating correct response via rejection sampling...")
    response_tokens = generate_correct_response(
        model=model,
        tokenizer=tokenizer,
        prompt_msgs=prompt_msgs,
        ground_truth=ground_truth,
        max_attempts=args.max_attempts,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
    )

    # Save cache
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"response_tokens": response_tokens}, str(output_path))
    print(f"Saved correct cache ({len(response_tokens)} tokens) to {output_path}")


if __name__ == "__main__":
    main()
