#!/usr/bin/env python3
"""Main entry point for attention-rhythm-guided GRPO training.

Thin wrapper around verl's main_ppo that:
  1. Parses --run_type (A/B/C) and rhythm hyperparams
  2. For B/C: runs head classification before training starts
  3. Applies monkey-patches via rhythm_trainer.apply_patches()
  4. Delegates to verl's Hydra-based main_ppo

Usage:
  python -m attention_sparks_thinking.scripts.train \
    --run_type A \
    --model_name Qwen/Qwen2.5-Math-1.5B \
    [verl hydra overrides...]

  python -m attention_sparks_thinking.scripts.train \
    --run_type B \
    --model_name Qwen/Qwen2.5-Math-1.5B \
    --dry_run \
    [verl hydra overrides...]
"""

import argparse
import json
import logging
import sys

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_custom_args():
    """Parse our custom args, returning (our_args, verl_hydra_overrides)."""
    custom_args = []
    verl_args = []

    i = 0
    argv = sys.argv[1:]
    while i < len(argv):
        arg = argv[i]
        if arg.startswith("--"):
            custom_args.append(arg)
            if i + 1 < len(argv) and not argv[i + 1].startswith("--") and "=" not in argv[i + 1]:
                custom_args.append(argv[i + 1])
                i += 2
            else:
                i += 1
        else:
            verl_args.append(arg)
            i += 1

    parser = argparse.ArgumentParser(description="Attention-rhythm-guided GRPO training")
    parser.add_argument("--run_type", type=str, required=True,
                        choices=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
                        help="Run type: A=baseline, B=static rhythm, C=adaptive rhythm, "
                             "D=attention, E=FAI, F=FAI-allheads, G=FAI-asymmetric, H=anchor-credit, "
                             "I=FAI-discrete, J=FAI-allheads-discrete")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Math-1.5B",
                        help="HF model path")
    parser.add_argument("--dry_run", action="store_true",
                        help="Run 2 steps then print diagnostics and exit")
    parser.add_argument("--waad_W", type=int, default=10, help="WAAD clipping window")
    parser.add_argument("--fai_H_lo", type=int, default=10, help="FAI horizon lower bound")
    parser.add_argument("--fai_H_hi", type=int, default=50, help="FAI horizon upper bound")
    parser.add_argument("--quantile_q", type=float, default=0.4, help="Quantile for T_loc/T_glob")
    parser.add_argument("--gamma_amp", type=float, default=1.5, help="Amplification factor")
    parser.add_argument("--gamma_mode", type=str, default="raw", choices=["raw", "normalized"],
                        help="'raw': direct gamma multiply, 'normalized': rescale to preserve mean advantage")
    parser.add_argument("--alpha", type=float, default=0.5, help="Back-allocation fraction")
    parser.add_argument("--neighborhood_k", type=int, default=3, help="Dominated anchor lookback")
    parser.add_argument("--reclassify_K", type=int, default=20, help="Reclassification interval (Run C)")
    parser.add_argument("--head_quantile", type=float, default=0.3, help="Head classification quantile")
    parser.add_argument("--num_class_prompts", type=int, default=20, help="Prompts for head classification (generates responses, so keep modest)")
    parser.add_argument("--reasoning_heads_path", type=str, default=None,
                        help="Path to head_importance_qwen3.pt from EAP-IG (required for D/E/G/H)")
    parser.add_argument("--num_reasoning_heads", type=int, default=200,
                        help="Number of top EAP-IG heads to use (default 200)")
    parser.add_argument("--eapig_rediscovery_K", type=int, default=0,
                        help="Re-run EAP-IG every K steps (0=disabled, e.g. 50)")
    parser.add_argument("--eapig_rediscovery_problems", type=int, default=50,
                        help="Number of problems for EAP-IG rediscovery")
    parser.add_argument("--positive_rollouts_only", action="store_true",
                        help="Only apply gamma weighting to correct (reward > 0) rollouts")

    args = parser.parse_args(custom_args)
    return args, verl_args


def _head_classification_subprocess(model_name: str, prompts: list[str], head_quantile: float, output_path: str):
    """Run head classification on GPU in a subprocess.

    This runs in a forked process so the CUDA context is fully released
    when the subprocess exits, avoiding conflicts with Ray workers.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from attention_sparks_thinking.src.head_classifier import classify_heads

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[HeadClassify] Loading model on {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        trust_remote_code=True,
    ).to(device)
    model.eval()

    # Paper's method: generate responses with T=0.7 then classify on
    # full prompt+response text.
    print(f"[HeadClassify] Generating responses for {len(prompts)} prompts (T=0.7)...")
    full_texts = []
    for i, prompt in enumerate(prompts):
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                top_p=1.0,
            )
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        full_texts.append(full_text)
        if i < 3:
            resp = full_text[len(prompt):].strip()
            print(f"[HeadClassify]   Sample {i}: {len(prompt)} prompt chars + {len(resp)} response chars")

    print(f"[HeadClassify] Classifying heads on {len(full_texts)} full texts...")
    H_loc, H_glob, d_matrix = classify_heads(
        model, tokenizer, full_texts, head_quantile=head_quantile
    )
    print(f"[HeadClassify] Done: {len(H_loc)} local, {len(H_glob)} global heads")

    # Save results to disk for parent process to read
    torch.save({"H_loc": H_loc, "H_glob": H_glob}, output_path)


def run_head_classification(model_name: str, num_prompts: int, head_quantile: float):
    """Run head classification on the base model before training.

    Runs in a subprocess so the CUDA context is fully released before
    veRL spawns Ray workers (avoids 'CUDA devices busy' errors).

    Returns (H_loc, H_glob, classification_prompts).
    """
    import multiprocessing as mp
    import pandas as pd

    logger.info(f"Running head classification on {model_name} with {num_prompts} prompts")

    # Load classification prompts from training data
    data_path = "attention_sparks_thinking/data/dapo_math_17k.parquet"
    try:
        df = pd.read_parquet(data_path)
    except FileNotFoundError:
        data_path = "attention_based_rewards/data/dapo_math_17k.parquet"
        df = pd.read_parquet(data_path)

    prompts = []
    for _, row in df.head(num_prompts).iterrows():
        prompt_msgs = row["prompt"]
        if isinstance(prompt_msgs, str):
            prompt_msgs = json.loads(prompt_msgs)
        text = "\n".join(m["content"] for m in prompt_msgs if m["role"] in ("user", "system"))
        prompts.append(text)

    logger.info(f"Loaded {len(prompts)} classification prompts from {data_path}")

    # Run in subprocess to isolate CUDA context
    import os
    output_path = f"/tmp/head_classify_{os.environ.get('SLURM_JOB_ID', 'default')}.pt"

    logger.info("Launching head classification subprocess (GPU)...")
    ctx = mp.get_context("spawn")  # spawn to get clean CUDA state
    p = ctx.Process(
        target=_head_classification_subprocess,
        args=(model_name, prompts, head_quantile, output_path),
    )
    p.start()
    p.join()

    if p.exitcode != 0:
        raise RuntimeError(f"Head classification subprocess failed with exit code {p.exitcode}")

    results = torch.load(output_path, map_location="cpu", weights_only=False)
    H_loc = results["H_loc"]
    H_glob = results["H_glob"]

    logger.info(f"Classification complete: {len(H_loc)} local, {len(H_glob)} global heads")
    logger.info(f"H_loc: {H_loc}")
    logger.info(f"H_glob: {H_glob}")

    return H_loc, H_glob, prompts


def main():
    args, verl_args = parse_custom_args()

    logger.info(f"Run type: {args.run_type}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Verl overrides: {verl_args}")

    # Head classification for B/C
    H_loc = None
    H_glob = None
    classification_prompts = None

    if args.run_type in ("B", "C"):
        H_loc, H_glob, classification_prompts = run_head_classification(
            args.model_name, args.num_class_prompts, args.head_quantile
        )

    # EAP-IG reasoning heads for D-J
    EAPIG_METHODS = {"D": "attention", "E": "fai", "F": "fai_allheads",
                     "G": "fai_asymmetric", "H": "anchor_credit",
                     "I": "fai_discrete", "J": "fai_allheads_discrete"}
    reasoning_heads = None
    head_scores = None
    weighting_method = "rhythm"

    if args.run_type in EAPIG_METHODS:
        weighting_method = EAPIG_METHODS[args.run_type]

        if args.run_type in ("D", "E", "G", "H", "I"):
            # These methods need reasoning heads from EAP-IG
            heads_path = args.reasoning_heads_path
            if heads_path is None:
                heads_path = "attention_sparks_thinking/logs/head_importance_qwen3.pt"
            logger.info(f"Loading EAP-IG reasoning heads from {heads_path}")
            eapig_data = torch.load(heads_path, map_location="cpu", weights_only=False)
            head_scores = eapig_data["importance"]  # (n_layers, n_heads)

            # Select top N heads from full importance matrix
            n_heads = min(args.num_reasoning_heads, head_scores.numel())
            flat_scores = head_scores.flatten()
            topk_indices = flat_scores.topk(n_heads).indices
            n_heads_per_layer = head_scores.shape[1]
            reasoning_heads = [
                (int(idx // n_heads_per_layer), int(idx % n_heads_per_layer))
                for idx in topk_indices
            ]
            min_score = flat_scores[topk_indices[-1]].item()
            logger.info(f"Selected top {n_heads} reasoning heads from importance matrix "
                        f"(shape {head_scores.shape}), min score: {min_score:.6f}")
        # F (fai_allheads) needs no reasoning heads — uses all heads equally

    # Apply patches
    from attention_sparks_thinking.src.rhythm_trainer import apply_patches

    apply_patches(
        run_type=args.run_type,
        model_name=args.model_name,
        H_loc=H_loc,
        H_glob=H_glob,
        classification_prompts=classification_prompts if args.run_type == "C" else None,
        waad_W=args.waad_W,
        fai_H_lo=args.fai_H_lo,
        fai_H_hi=args.fai_H_hi,
        quantile_q=args.quantile_q,
        gamma_amp=args.gamma_amp,
        gamma_mode=args.gamma_mode,
        alpha=args.alpha,
        neighborhood_k=args.neighborhood_k,
        reclassify_every_K=args.reclassify_K,
        head_quantile=args.head_quantile,
        dry_run=args.dry_run,
        reasoning_heads=reasoning_heads,
        head_scores=head_scores,
        weighting_method=weighting_method,
        num_reasoning_heads=args.num_reasoning_heads,
        eapig_rediscovery_K=args.eapig_rediscovery_K,
        eapig_rediscovery_problems=args.eapig_rediscovery_problems,
        positive_rollouts_only=args.positive_rollouts_only,
    )

    # Rewrite sys.argv for Hydra
    sys.argv = [sys.argv[0]] + verl_args

    # Import and run verl
    from verl.trainer.main_ppo import main as verl_main
    verl_main()


if __name__ == "__main__":
    main()
