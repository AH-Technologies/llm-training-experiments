#!/bin/bash
# Submit likelihood bonus + uniform baseline on BASE model (no suffix).
#
# Submits 2 jobs:
#   1. uniform         — regular GRPO baseline
#   2. likelihood_bonus — GRPO + base-model likelihood bonus (lambda=0.2, beta=5.0)
#
# Both use:
#   Model: Qwen2.5-Math-1.5B (base, NOT Instruct)
#   LR=1e-5, batch=64, temp=0.7, 500 steps

set -e

echo "Submitting likelihood bonus experiment (BASE model, no suffix)..."
echo ""

BASE_MODEL="/cluster/projects/nn12068k/haaklau/.cache/huggingface/hub/models--Qwen--Qwen2.5-Math-1.5B/snapshots/4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2"
OVERRIDES="actor_rollout_ref.actor.optim.lr=1e-5 data.train_batch_size=64 actor_rollout_ref.actor.ppo_mini_batch_size=64 actor_rollout_ref.rollout.temperature=0.7"

# Likelihood bonus hyperparameters
LIKELIHOOD_LAMBDA=${LIKELIHOOD_LAMBDA:-0.2}
LIKELIHOOD_BETA=${LIKELIHOOD_BETA:-5.0}

echo "1. Submitting: uniform (baseline)"
sbatch --export=ALL,CONDITION=uniform,TOTAL_STEPS=500,MODEL=${BASE_MODEL},EXTRA_OVERRIDES="${OVERRIDES}",EXP_SUFFIX=_base_llbonus \
    attention_based_rewards/slurm/train_dapo.slurm

echo "2. Submitting: likelihood_bonus (lambda=${LIKELIHOOD_LAMBDA}, beta=${LIKELIHOOD_BETA})"
sbatch --export=ALL,CONDITION=likelihood_bonus,TOTAL_STEPS=500,MODEL=${BASE_MODEL},LIKELIHOOD_LAMBDA=${LIKELIHOOD_LAMBDA},LIKELIHOOD_BETA=${LIKELIHOOD_BETA},EXTRA_OVERRIDES="${OVERRIDES}",EXP_SUFFIX=_base_llbonus \
    attention_based_rewards/slurm/train_dapo.slurm

echo ""
echo "Both jobs submitted. Monitor with: squeue -u \$USER"
echo "Logs: attention_based_rewards/logs/dapo_*_base_llbonus_*.log"
echo "Weight logs: attention_based_rewards/logs/weight_check_likelihood_bonus_*.log"
