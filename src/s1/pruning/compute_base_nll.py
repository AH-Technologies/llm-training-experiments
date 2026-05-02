"""Score s1K responses under the untrained base model — produces signals for
the `base_loss`, `base_logprob_mean`, and `ifd` screening strategies.

Two passes are supported:

  - conditional (default): score the response *given the question* — runs the
    model on `prompt + response` and records per-token logprobs of the
    response. Output: data/s1K/s1k_base_nll.parquet. Drives `base_loss` and
    `base_logprob_mean`.
  - response_only: score the response *without the question* — runs the model
    on the response body alone. Output: data/s1K/s1k_base_nll_response_only.parquet.
    Combined with the conditional pass, gives IFD = exp(L_mean(A|Q) - L_mean(A))
    (Cherry-LLM, NAACL 2024).

Each pass writes a parquet with `{index, total_nll, mean_nll, response_tokens}`:

  - total_nll: -sum(logP) over response tokens.
  - mean_nll: total_nll / response_token_count.
  - response_tokens: number of scored tokens.

For the conditional pass, `total_nll` is the "informativeness" signal — how
much the model has to update from its base state to produce this trace.
For both passes, `mean_nll` is length-normalised; their difference is the
log-IFD score used by the `ifd` strategy.

Special tokens (`<|im_start|>think`, `<|im_start|>answer`, `<|im_end|>`) are
included in both reductions — matches what the SFT trainer would optimise
toward.

Usage:
    # default conditional pass (backward-compatible)
    python -m s1.pruning.compute_base_nll

    # response-only pass (writes the IFD denominator)
    python -m s1.pruning.compute_base_nll --mode response_only

    # both passes under one model load (~12 min instead of 2×10 min)
    python -m s1.pruning.compute_base_nll --mode both
"""

from __future__ import annotations

import argparse
import os

import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def _build_pair(question: str, thinking: str, solution: str) -> tuple[str, str]:
    """Mirror `scripts/s1/prepare_s1k_sft.py::format_s1k_for_sft` byte-for-byte.
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


def _tokenize(
    rows: list[tuple[str, str, str]],
    tokenizer,
    max_model_len: int,
    include_question: bool,
) -> tuple[list[list[int]], list[int]]:
    """Tokenise `(question, thinking, solution)` triples into model-input ids.

    Returns (combined_token_ids, prompt_lens) where `prompt_lens[i]` is the
    boundary up to which tokens are excluded from scoring. For the conditional
    pass this is the question-prefix length; for response_only it is 0.
    """
    combined: list[list[int]] = []
    prompt_lens: list[int] = []
    for question, thinking, solution in rows:
        prompt, response = _build_pair(question, thinking, solution)
        r_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
        if include_question:
            p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            max_resp = max(1, max_model_len - len(p_ids))
            if len(r_ids) > max_resp:
                r_ids = r_ids[:max_resp]
            combined.append(p_ids + r_ids)
            prompt_lens.append(len(p_ids))
        else:
            if len(r_ids) > max_model_len:
                r_ids = r_ids[:max_model_len]
            combined.append(r_ids)
            prompt_lens.append(0)
    return combined, prompt_lens


def _score(
    outputs,
    combined_token_ids: list[list[int]],
    prompt_lens: list[int],
) -> pa.Table:
    indices: list[int] = []
    total_nlls: list[float] = []
    mean_nlls: list[float] = []
    response_lens: list[int] = []

    for i, out in enumerate(outputs):
        plp = out.prompt_logprobs
        prompt_len = prompt_lens[i]
        combined_ids = combined_token_ids[i]
        response_logprobs: list[float] = []
        if plp is not None:
            for pos in range(prompt_len, len(combined_ids)):
                if pos >= len(plp) or plp[pos] is None:
                    # First position has no context to score against, and the
                    # tail can be cut off by max_model_len.
                    continue
                tok_id = combined_ids[pos]
                lp = plp[pos].get(tok_id)
                if lp is None:
                    # With prompt_logprobs=1 the presented token is always
                    # surfaced, so this should be rare.
                    continue
                response_logprobs.append(lp.logprob)

        if not response_logprobs:
            total, mean, length = 0.0, 0.0, 0
        else:
            total = -float(sum(response_logprobs))
            length = len(response_logprobs)
            mean = total / length

        indices.append(i)
        total_nlls.append(total)
        mean_nlls.append(mean)
        response_lens.append(length)

    return pa.table({
        "index": indices,
        "total_nll": total_nlls,
        "mean_nll": mean_nlls,
        "response_tokens": response_lens,
    })


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-example base-model NLL on s1K (signals for screening)."
    )
    parser.add_argument("--input", default="data/s1K/s1k.parquet")
    parser.add_argument(
        "--mode",
        choices=("conditional", "response_only", "both"),
        default="conditional",
        help=(
            "conditional: score response given the question (drives base_loss / "
            "base_logprob_mean). response_only: score response alone (drives the "
            "ifd denominator). both: run both passes under one model load."
        ),
    )
    parser.add_argument(
        "--output",
        default="data/s1K/s1k_base_nll.parquet",
        help="Output parquet for the conditional pass.",
    )
    parser.add_argument(
        "--output-response-only",
        default="data/s1K/s1k_base_nll_response_only.parquet",
        help="Output parquet for the response-only pass.",
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct")
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.7,
        help=(
            "Lower than the eval default (0.9) because prompt_logprobs=1 "
            "materialises full-vocab logits at every position (Qwen vocab "
            "~152K × max_model_len × 4 bytes ≈ 10 GiB transient). With 0.9 "
            "the KV cache reservation leaves no headroom for that spike and "
            "we OOM on the first batch."
        ),
    )
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

    do_cond = args.mode in ("conditional", "both")
    do_resp = args.mode in ("response_only", "both")

    # Idempotent skip: if the requested output(s) already exist, nothing to do.
    cond_done = (not do_cond) or os.path.exists(args.output)
    resp_done = (not do_resp) or os.path.exists(args.output_response_only)
    if cond_done and resp_done:
        if do_cond:
            print(f"Conditional parquet already exists at {args.output}; skipping.")
        if do_resp:
            print(f"Response-only parquet already exists at {args.output_response_only}; skipping.")
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    table = pq.read_table(args.input)
    questions = table.column("question").to_pylist()
    solutions = table.column("solution").to_pylist()
    trajectories = table.column("thinking_trajectories").to_pylist()
    rows = [
        (questions[i], _extract_thinking(trajectories[i]), solutions[i])
        for i in range(len(questions))
    ]
    n = len(rows)

    # Tokenise only the passes we actually need to run.
    cond_inputs = None
    resp_inputs = None
    if do_cond and not cond_done:
        print(f"Tokenising {n} examples for conditional pass ...")
        cond_inputs = _tokenize(rows, tokenizer, args.max_model_len, include_question=True)
    if do_resp and not resp_done:
        print(f"Tokenising {n} examples for response-only pass ...")
        resp_inputs = _tokenize(rows, tokenizer, args.max_model_len, include_question=False)

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

    # max_tokens=1 (vLLM rejects 0) — the generated token is ignored; only
    # prompt_logprobs matters.
    sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=1)

    if cond_inputs is not None:
        combined, prompt_lens = cond_inputs
        print(f"Scoring {n} examples (conditional) ...")
        outputs = llm.generate(
            prompts=[{"prompt_token_ids": ids} for ids in combined],
            sampling_params=sp,
        )
        out_table = _score(outputs, combined, prompt_lens)
        pq.write_table(out_table, args.output)
        print(f"Wrote conditional NLL to {args.output} ({n} rows)")

    if resp_inputs is not None:
        combined, prompt_lens = resp_inputs
        print(f"Scoring {n} examples (response_only) ...")
        outputs = llm.generate(
            prompts=[{"prompt_token_ids": ids} for ids in combined],
            sampling_params=sp,
        )
        out_table = _score(outputs, combined, prompt_lens)
        pq.write_table(out_table, args.output_response_only)
        print(f"Wrote response-only NLL to {args.output_response_only} ({n} rows)")


if __name__ == "__main__":
    main()
