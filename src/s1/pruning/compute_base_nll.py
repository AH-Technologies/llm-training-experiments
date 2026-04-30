"""Score s1K responses under the untrained base model — produces signals for
the `base_loss` and `base_logprob_mean` screening strategies.

For each s1K row we build the SFT-format `(prompt, response)` exactly as
`scripts/prepare_s1k_sft.py` does, then run the *untrained* Qwen2.5-32B-Instruct
over `prompt + response` once and record the per-token log-probabilities of
the actually-presented response tokens. Two scalars per row:

  - total_nll: -sum(logP) over response tokens. Larger = the response is more
    surprising under the base model. The "informativeness" signal — how much
    the model has to update from its base state to produce this trace.
  - mean_nll: total_nll / response_token_count. Length-normalised. Larger =
    each token is intrinsically hard to predict; smaller = the trace is
    smoothly extending from a base-model-natural distribution. Inversely
    tracks what we call "coherence": top of the screening partition (highest
    mean log-prob = lowest mean_nll) means most coherent / sustained
    confidence.

Special tokens (`<|im_start|>think`, `<|im_start|>answer`, `<|im_end|>`) are
included in both reductions — matches what the SFT trainer would optimise
toward.

Usage:
    python -m s1.pruning.compute_base_nll \\
        --input data/s1K/s1k.parquet \\
        --output data/s1K/s1k_base_nll.parquet
"""

from __future__ import annotations

import argparse

import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def _build_pair(question: str, thinking: str, solution: str) -> tuple[str, str]:
    """Mirror `scripts/prepare_s1k_sft.py::format_s1k_for_sft` byte-for-byte.
    The base-model score must reflect the exact target the SFT trainer would
    optimise toward, otherwise the signal is incomparable to training.
    """
    prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    response = f"<|im_start|>think\n{thinking}\n<|im_start|>answer\n{solution}<|im_end|>"
    return prompt, response


def _extract_thinking(traj_field) -> str:
    if traj_field and isinstance(traj_field, list) and traj_field:
        return traj_field[0]
    if traj_field and isinstance(traj_field, str):
        return traj_field
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-example base-model NLL on s1K (signals for screening)."
    )
    parser.add_argument("--input", default="data/s1K/s1k.parquet")
    parser.add_argument("--output", default="data/s1K/s1k_base_nll.parquet")
    parser.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct")
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=16384,
        help=(
            "Cap combined prompt+response length. s1K traces median ~3k tokens "
            "but tail to ~7k; 16k covers every example with headroom."
        ),
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    table = pq.read_table(args.input)
    questions = table.column("question").to_pylist()
    solutions = table.column("solution").to_pylist()
    trajectories = table.column("thinking_trajectories").to_pylist()
    n = len(questions)

    print(f"Tokenising {n} examples ...")
    prompt_lens: list[int] = []
    combined_token_ids: list[list[int]] = []
    for i in range(n):
        thinking = _extract_thinking(trajectories[i])
        prompt, response = _build_pair(questions[i], thinking, solutions[i])
        p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        r_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
        # Keep at least one response token; truncate from the response tail
        # so the combined fits in max_model_len.
        max_resp = max(1, args.max_model_len - len(p_ids))
        if len(r_ids) > max_resp:
            r_ids = r_ids[:max_resp]
        prompt_lens.append(len(p_ids))
        combined_token_ids.append(p_ids + r_ids)

    print(f"Loading base model {args.model} ...")
    llm = LLM(
        args.model,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        # vLLM 0.12 renamed enable_custom_all_reduce → disable_custom_all_reduce.
        # Custom AR crashes on this cluster's multi-GPU topology (see s1.eval).
        disable_custom_all_reduce=True,
    )

    # max_tokens=1 (vLLM rejects 0) — we ignore the generated token; the only
    # output we care about is `prompt_logprobs`, which scores every token of
    # the supplied input under the model.
    sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=1)
    print(f"Scoring {n} examples (prompt_logprobs=1) ...")
    outputs = llm.generate(
        prompts=[{"prompt_token_ids": ids} for ids in combined_token_ids],
        sampling_params=sp,
    )

    indices: list[int] = []
    total_nlls: list[float] = []
    mean_nlls: list[float] = []
    response_lens: list[int] = []

    for i, out in enumerate(outputs):
        plp = out.prompt_logprobs
        prompt_len = prompt_lens[i]
        combined_ids = combined_token_ids[i]
        # vLLM returns one entry per token (first is None). Score every
        # response position by looking up the actually-presented token's
        # logprob in the dict at that position.
        response_logprobs: list[float] = []
        if plp is not None:
            for pos in range(prompt_len, len(combined_ids)):
                if pos >= len(plp) or plp[pos] is None:
                    continue
                tok_id = combined_ids[pos]
                lp = plp[pos].get(tok_id)
                if lp is None:
                    # Token wasn't surfaced in the top-k entries vLLM returns.
                    # With prompt_logprobs=1 the actually-presented token is
                    # always one of the entries, so this should be rare.
                    continue
                response_logprobs.append(lp.logprob)

        if not response_logprobs:
            total = 0.0
            mean = 0.0
            length = 0
        else:
            total = -float(sum(response_logprobs))
            length = len(response_logprobs)
            mean = total / length

        indices.append(i)
        total_nlls.append(total)
        mean_nlls.append(mean)
        response_lens.append(length)

    out_table = pa.table({
        "index": indices,
        "total_nll": total_nlls,
        "mean_nll": mean_nlls,
        "response_tokens": response_lens,
    })
    pq.write_table(out_table, args.output)
    print(f"Wrote per-example base-model NLL to {args.output} ({n} rows)")


if __name__ == "__main__":
    main()
