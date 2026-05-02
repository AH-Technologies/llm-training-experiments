#!/bin/bash
# Submit EAP-IG token weighting runs to SLURM (2-node, positive rollouts only).
#
# Prerequisite: head_importance_qwen3.pt must exist at
#   attention_sparks_thinking/logs/head_importance_qwen3.pt
# Run analyze_heads_qwen3.py first if it doesn't.
#
# E. FAI             — smooth FAI on reasoning heads (positive rollouts only)
# F. FAI-AllHeads    — smooth FAI on all heads (positive rollouts only, control)
# H. Anchor-Credit   — discrete anchor/dependent weights (positive rollouts only)
# I. FAI-Discrete    — top 20% tokens by FAI get 1.5, rest 1.0 (positive rollouts only)
# J. FAI-AllHeads-Discrete — same but all heads (positive rollouts only, control)

set -e

SLURM_SCRIPT=scripts/attention_sparks_thinking/slurm/train_2node.slurm
EXCLUDE="--exclude=gpu-1-83,gpu-1-88"
MODEL="Qwen/Qwen3-4B-Base"
HEADS_PATH="attention_sparks_thinking/logs/head_importance_qwen3.pt"

# Verify prerequisite
if [ ! -f "$HEADS_PATH" ]; then
    echo "ERROR: $HEADS_PATH not found."
    echo "Run the EAP-IG analysis first: sbatch scripts/attention_sparks_thinking/slurm/analyze_heads.slurm"
    exit 1
fi

# Common overrides
COMMON="actor_rollout_ref.actor.optim.lr=1e-6 data.train_batch_size=512 actor_rollout_ref.actor.ppo_mini_batch_size=32 actor_rollout_ref.rollout.n=8 actor_rollout_ref.rollout.temperature=1.0 actor_rollout_ref.actor.entropy_coeff=0 actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16000 actor_rollout_ref.actor.optim.weight_decay=0 actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean data.max_response_length=1024 trainer.project_name=attention-illuminates-qwen3"

echo "Submitting 5 EAP-IG runs (2-node, positive rollouts only)..."

# E. FAI (smooth, reasoning heads)
echo "  E. FAI (smooth, positive rollouts only)"
sbatch $EXCLUDE \
  --export=ALL,RUN_TYPE=E,MODEL=$MODEL,REASONING_HEADS_PATH=$HEADS_PATH,POSITIVE_ROLLOUTS_ONLY=1,EXP_SUFFIX=_qwen3_fai_posonly,EXTRA_OVERRIDES="$COMMON" \
  $SLURM_SCRIPT

# F. FAI-AllHeads (smooth, all heads, control)
echo "  F. FAI-AllHeads (smooth, positive rollouts only)"
sbatch $EXCLUDE \
  --export=ALL,RUN_TYPE=F,MODEL=$MODEL,POSITIVE_ROLLOUTS_ONLY=1,EXP_SUFFIX=_qwen3_fai_allheads_posonly,EXTRA_OVERRIDES="$COMMON" \
  $SLURM_SCRIPT

# H. Anchor-Credit (discrete anchors/dependents)
echo "  H. Anchor-Credit (positive rollouts only)"
sbatch $EXCLUDE \
  --export=ALL,RUN_TYPE=H,MODEL=$MODEL,REASONING_HEADS_PATH=$HEADS_PATH,POSITIVE_ROLLOUTS_ONLY=1,EXP_SUFFIX=_qwen3_anchor_credit_posonly,EXTRA_OVERRIDES="$COMMON" \
  $SLURM_SCRIPT

# I. FAI-Discrete (top 20% get 1.5, reasoning heads)
echo "  I. FAI-Discrete (top 20% -> 1.5, positive rollouts only)"
sbatch $EXCLUDE \
  --export=ALL,RUN_TYPE=I,MODEL=$MODEL,REASONING_HEADS_PATH=$HEADS_PATH,POSITIVE_ROLLOUTS_ONLY=1,EXP_SUFFIX=_qwen3_fai_discrete_posonly,EXTRA_OVERRIDES="$COMMON" \
  $SLURM_SCRIPT

# J. FAI-AllHeads-Discrete (top 20% get 1.5, all heads, control)
echo "  J. FAI-AllHeads-Discrete (top 20% -> 1.5, positive rollouts only)"
sbatch $EXCLUDE \
  --export=ALL,RUN_TYPE=J,MODEL=$MODEL,POSITIVE_ROLLOUTS_ONLY=1,EXP_SUFFIX=_qwen3_fai_allheads_discrete_posonly,EXTRA_OVERRIDES="$COMMON" \
  $SLURM_SCRIPT

echo ""
echo "All 5 runs submitted. Monitor with: squeue -u \$USER"
echo "Wandb project: attention-illuminates-qwen3"
