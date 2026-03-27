#!/bin/bash
# Submit 5 EAP-IG token weighting runs to SLURM (methods D-H).
#
# Prerequisite: head_importance_qwen3.pt must exist at
#   attention_sparks_thinking/logs/head_importance_qwen3.pt
# Run analyze_heads_qwen3.py first if it doesn't.
#
# D. Attention       — weight tokens by reasoning head attention received
# E. FAI             — Future Attention Influence on reasoning heads
# F. FAI-AllHeads    — FAI on all 1152 heads equally (control)
# G. FAI-Asymmetric  — FAI for correct, inverted FAI for incorrect
# H. Anchor-Credit   — discrete anchor/dependent weights

set -e

SLURM_SCRIPT=attention_sparks_thinking/slurm/train.slurm
EXCLUDE=""
MODEL="Qwen/Qwen3-4B-Base"
HEADS_PATH="attention_sparks_thinking/logs/head_importance_qwen3.pt"

# Verify prerequisite
if [ ! -f "$HEADS_PATH" ]; then
    echo "ERROR: $HEADS_PATH not found."
    echo "Run the EAP-IG analysis first: sbatch attention_sparks_thinking/slurm/analyze_heads.slurm"
    exit 1
fi

# Common overrides (same as submit_qwen3_runs.sh v6)
COMMON="actor_rollout_ref.actor.optim.lr=1e-6 data.train_batch_size=512 actor_rollout_ref.actor.ppo_mini_batch_size=32 actor_rollout_ref.rollout.n=8 actor_rollout_ref.rollout.temperature=1.0 actor_rollout_ref.actor.entropy_coeff=0 actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16000 actor_rollout_ref.actor.optim.weight_decay=0 actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean data.max_response_length=1024 +algorithm.advantage_clip=2.0 +algorithm.reward_clip=10 +algorithm.max_len_mask=True trainer.project_name=attention-illuminates-qwen3"

echo "Submitting 5 EAP-IG token weighting runs (Qwen3-4B-Base)..."

# D. Attention — weight by reasoning head attention received
echo "  D. Attention weighting"
sbatch $EXCLUDE \
  --export=ALL,RUN_TYPE=D,MODEL=$MODEL,REASONING_HEADS_PATH=$HEADS_PATH,EXP_SUFFIX=_qwen3_attention,EXTRA_OVERRIDES="$COMMON" \
  $SLURM_SCRIPT

# E. FAI — Future Attention Influence on reasoning heads
echo "  E. FAI weighting"
sbatch $EXCLUDE \
  --export=ALL,RUN_TYPE=E,MODEL=$MODEL,REASONING_HEADS_PATH=$HEADS_PATH,EXP_SUFFIX=_qwen3_fai,EXTRA_OVERRIDES="$COMMON" \
  $SLURM_SCRIPT

# F. FAI-AllHeads — control (all 1152 heads equally)
echo "  F. FAI-AllHeads (control)"
sbatch $EXCLUDE \
  --export=ALL,RUN_TYPE=F,MODEL=$MODEL,EXP_SUFFIX=_qwen3_fai_allheads,EXTRA_OVERRIDES="$COMMON" \
  $SLURM_SCRIPT

# G. FAI-Asymmetric — FAI for correct, inverted for incorrect
echo "  G. FAI-Asymmetric"
sbatch $EXCLUDE \
  --export=ALL,RUN_TYPE=G,MODEL=$MODEL,REASONING_HEADS_PATH=$HEADS_PATH,EXP_SUFFIX=_qwen3_fai_asymmetric,EXTRA_OVERRIDES="$COMMON" \
  $SLURM_SCRIPT

# H. Anchor-Credit — discrete anchor/dependent weights
echo "  H. Anchor-Credit"
sbatch $EXCLUDE \
  --export=ALL,RUN_TYPE=H,MODEL=$MODEL,REASONING_HEADS_PATH=$HEADS_PATH,EXP_SUFFIX=_qwen3_anchor_credit,EXTRA_OVERRIDES="$COMMON" \
  $SLURM_SCRIPT

echo ""
echo "All 5 EAP-IG runs submitted. Monitor with: squeue -u \$USER"
echo "Wandb project: attention-illuminates-qwen3"
