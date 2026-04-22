"""Prepare s1K dataset for SFT training with the torchtune worker.

Converts the s1K parquet (question, solution, thinking_trajectories) into
prompt/response format matching the s1 paper's Qwen chat template (§D):

    <|im_start|>user
    {question}<|im_end|>
    <|im_start|>assistant
    <|im_start|>think
    {thinking}
    <|im_start|>answer
    {solution}<|im_end|>

The paper uses <|im_start|>think and <|im_start|>answer as delimiters
to separate the thinking stage from the answering stage.

The torchtune SFT worker expects columns: prompt, response.

Usage:
    python scripts/prepare_s1k_sft.py
    python scripts/prepare_s1k_sft.py --input data/s1K/s1k.parquet --output data/s1K/s1k_sft.parquet
"""

import argparse

import pyarrow as pa
import pyarrow.parquet as pq


def format_s1k_for_sft(input_path: str, output_path: str) -> None:
    table = pq.read_table(input_path)
    d = table.to_pydict()

    prompts = []
    responses = []

    for i in range(table.num_rows):
        question = d["question"][i]
        solution = d["solution"][i]

        # thinking_trajectories is a list; use the first one if available
        thinking_trajs = d.get("thinking_trajectories", [None] * table.num_rows)[i]
        if thinking_trajs and isinstance(thinking_trajs, list) and len(thinking_trajs) > 0:
            thinking = thinking_trajs[0]
        elif thinking_trajs and isinstance(thinking_trajs, str):
            thinking = thinking_trajs
        else:
            thinking = ""

        # Match the s1 paper's Qwen chat format (§D):
        # Thinking delimiters: <|im_start|>think and <|im_start|>answer
        # "both preceded and followed by a newline"
        prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        response = f"<|im_start|>think\n{thinking}\n<|im_start|>answer\n{solution}<|im_end|>"

        prompts.append(prompt)
        responses.append(response)

    out_table = pa.table({"prompt": prompts, "response": responses})
    pq.write_table(out_table, output_path)
    print(f"Wrote {table.num_rows} examples to {output_path}")

    # Print stats
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-32B-Instruct", trust_remote_code=True)
    lengths = []
    for p, r in zip(prompts, responses):
        toks = tokenizer.encode(p + r, add_special_tokens=False)
        lengths.append(len(toks))
    lengths.sort()
    print(f"Token length stats: min={lengths[0]}, median={lengths[len(lengths)//2]}, "
          f"max={lengths[-1]}, mean={sum(lengths)/len(lengths):.0f}")
    print(f"Samples > 4096 tokens: {sum(1 for l in lengths if l > 4096)}")
    print(f"Samples > 8192 tokens: {sum(1 for l in lengths if l > 8192)}")
    print(f"Samples > 16384 tokens: {sum(1 for l in lengths if l > 16384)}")
    print(f"Samples > 32768 tokens: {sum(1 for l in lengths if l > 32768)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/s1K/s1k.parquet")
    parser.add_argument("--output", default="data/s1K/s1k_sft.parquet")
    args = parser.parse_args()

    format_s1k_for_sft(args.input, args.output)


if __name__ == "__main__":
    main()
