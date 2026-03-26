#!/usr/bin/env python3
"""Main entry point for circuit-guided GRPO training.

Thin wrapper around verl's main_ppo that:
  1. Parses --condition flag (uniform/attention/entropy/combined)
  2. Loads reasoning heads from Phase 1 if needed
  3. Applies monkey-patches for token weighting
  4. Delegates to verl's Hydra-based main_ppo

Usage:
  python -m attention_based_rewards.scripts.train \
    --condition uniform \
    [verl hydra overrides...]

  python -m attention_based_rewards.scripts.train \
    --condition attention \
    --reasoning_heads attention_based_rewards/results/reasoning_heads.pt \
    --model_name Qwen/Qwen2.5-Math-1.5B \
    algorithm.adv_estimator=grpo \
    data.train_files=attention_based_rewards/data/gsm8k_train.parquet \
    ...
"""

import argparse
import logging
import sys

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_custom_args():
    """Parse our custom args, returning (our_args, verl_hydra_overrides)."""
    # Find where our args end and verl's hydra overrides begin
    # Our args use -- prefix, verl args use key=value format
    custom_args = []
    verl_args = []

    i = 0
    argv = sys.argv[1:]
    while i < len(argv):
        arg = argv[i]
        if arg.startswith("--"):
            custom_args.append(arg)
            # Check if next arg is a value (not another flag or hydra override)
            if i + 1 < len(argv) and not argv[i + 1].startswith("--") and "=" not in argv[i + 1]:
                custom_args.append(argv[i + 1])
                i += 2
            else:
                i += 1
        else:
            verl_args.append(arg)
            i += 1

    parser = argparse.ArgumentParser(description="Circuit-guided GRPO training")
    parser.add_argument(
        "--condition",
        type=str,
        required=True,
        choices=["uniform", "attention", "entropy", "combined", "fai", "asymmetric", "attention_top5", "fai_allheads", "fai_asymmetric", "anchor_credit", "circuit_reward", "activation_entropy", "mlp_circuit_reward", "layerwise_slope", "likelihood_bonus"],
        help="Token weighting condition",
    )
    parser.add_argument(
        "--reasoning_heads",
        type=str,
        default=None,
        help="Path to reasoning_heads.pt from Phase 1 (required for attention/combined)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-Math-1.5B",
        help="HF model name for attention extraction",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Mixing coefficient for combined condition (1=attention, 0=entropy)",
    )
    parser.add_argument(
        "--top_k_heads",
        type=int,
        default=10,
        help="Number of top reasoning heads to use",
    )
    parser.add_argument(
        "--anchor_percentile",
        type=float,
        default=90.0,
        help="FAI percentile threshold for anchor tokens (default 90 = top 10%%)",
    )
    parser.add_argument(
        "--likelihood_lambda",
        type=float,
        default=0.2,
        help="Weight for likelihood bonus (must be < 0.5 to keep correct > incorrect)",
    )
    parser.add_argument(
        "--likelihood_beta",
        type=float,
        default=5.0,
        help="Sigmoid sharpness for likelihood bonus centering",
    )
    parser.add_argument(
        "--suffix_mode",
        type=str,
        default="none",
        choices=["none", "step", "random"],
        help="Prompt suffix mode: none (default), step (baked into parquet), random (monkey-patch)",
    )

    args = parser.parse_args(custom_args)
    return args, verl_args


def load_reasoning_heads(path: str, top_k: int = 10):
    """Load reasoning heads from Phase 1 results.

    The .pt file from step2_eap_ig.py contains:
      - "selected_heads": list of (layer, head, score) 3-tuples
      - "head_scores": (n_layers, n_heads) tensor

    Returns:
        reasoning_heads: list of (layer, head) tuples
        head_scores: (n_layers, n_heads) tensor
    """
    data = torch.load(path, map_location="cpu", weights_only=False)
    head_scores = data["head_scores"]

    # selected_heads is a list of (layer, head, score) 3-tuples
    selected = data["selected_heads"][:top_k]
    reasoning_heads = [(layer, head) for layer, head, _score in selected]

    logger.info(f"Loaded {len(reasoning_heads)} reasoning heads from {path}")
    for i, (layer, head) in enumerate(reasoning_heads):
        score = head_scores[layer, head].item()
        logger.info(f"  #{i+1}: L{layer}H{head} (score={score:.2f})")

    return reasoning_heads, head_scores


def main():
    args, verl_args = parse_custom_args()

    logger.info(f"Condition: {args.condition}")
    logger.info(f"Verl overrides: {verl_args}")

    # Load reasoning heads if needed
    reasoning_heads = None
    head_scores = None
    _attn_conditions = ("attention", "combined", "fai", "asymmetric", "attention_top5", "fai_allheads", "fai_asymmetric", "anchor_credit", "circuit_reward", "activation_entropy", "mlp_circuit_reward", "layerwise_slope")
    _needs_reasoning_heads = ("attention", "combined", "fai", "asymmetric", "attention_top5", "fai_asymmetric", "anchor_credit", "circuit_reward", "mlp_circuit_reward")
    if args.condition in _needs_reasoning_heads:
        if args.reasoning_heads is None:
            logger.error(f"--reasoning_heads is required for {args.condition} condition")
            sys.exit(1)
        reasoning_heads, head_scores = load_reasoning_heads(args.reasoning_heads, args.top_k_heads)

    # Apply random suffix patch if requested (must be before verl import)
    if args.suffix_mode == "random":
        from attention_based_rewards.scripts.suffix_patch import apply_random_suffix_patch
        apply_random_suffix_patch()
        logger.info("Random suffix mode: patched RLHFDataset.__getitem__")
    elif args.suffix_mode == "step":
        logger.info("Step suffix mode: expecting step_suffix parquet as input")

    # Apply monkey-patches BEFORE importing verl's main
    from attention_based_rewards.scripts.weighted_trainer import apply_patches

    apply_patches(
        condition=args.condition,
        model_name=args.model_name if args.condition in _attn_conditions else None,
        reasoning_heads=reasoning_heads,
        head_scores=head_scores,
        alpha=args.alpha,
        anchor_percentile=args.anchor_percentile,
        likelihood_lambda=args.likelihood_lambda,
        likelihood_beta=args.likelihood_beta,
    )

    # Now rewrite sys.argv for Hydra (verl uses @hydra.main)
    # Hydra expects sys.argv[0] to be the script name
    sys.argv = [sys.argv[0]] + verl_args

    # Import and run verl's main_ppo
    from verl.trainer.main_ppo import main as verl_main

    verl_main()


if __name__ == "__main__":
    main()
