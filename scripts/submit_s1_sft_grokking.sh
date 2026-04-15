#!/bin/bash
# Submit S1 Grokking job: Qwen2.5-32B-Instruct on s1K — 25 epochs (5x paper run)
# Tests whether extended training produces a grokking-like effect.
# Saves 5 checkpoints (every 5 epochs) for post-hoc eval.
# Uses SEQ_LENGTH=30000 to avoid truncating any s1K samples (paper §D.1).
#
# Usage:
#   bash scripts/submit_s1_sft_grokking.sh
#   TOTAL_GPUS=8 bash scripts/submit_s1_sft_grokking.sh   # override GPU count

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/cluster_config.sh"

# Compute resources
TOTAL_GPUS=${TOTAL_GPUS:-16}
GPUS_PER_NODE=${GPUS_PER_NODE:-$DEFAULT_GPUS_PER_NODE}
NODES=$(( TOTAL_GPUS / GPUS_PER_NODE ))
TIME=${TIME:-"12:00:00"}
ACCOUNT=${ACCOUNT:-$SLURM_ACCOUNT_DEFAULT}
PARTITION=${PARTITION:-$SLURM_PARTITION_DEFAULT}

echo "Submitting S1 Grokking job on $CLUSTER_NAME"
echo "  Nodes: $NODES x $GPUS_PER_NODE GPUs = $TOTAL_GPUS total"
echo "  Account: $ACCOUNT, Partition: $PARTITION"
[ -n "$CONSTRAINT" ] && echo "  Constraint: $CONSTRAINT"

CONSTRAINT=${CONSTRAINT:-${DEFAULT_CONSTRAINT:-}}
CONSTRAINT_FLAG=""
[ -n "$CONSTRAINT" ] && CONSTRAINT_FLAG="--constraint=$CONSTRAINT"

sbatch \
    --job-name=s1-grok-32b \
    --account="$ACCOUNT" \
    --partition="$PARTITION" \
    --time="$TIME" \
    --nodes="$NODES" \
    --ntasks-per-node=1 \
    --gpus-per-node="$GPUS_PER_NODE" \
    --cpus-per-task="$DEFAULT_CPUS_PER_TASK" \
    --mem="$DEFAULT_MEM" \
    --output=logs/s1-grok-%j.out \
    --error=logs/s1-grok-%j.err \
    $CONSTRAINT_FLAG \
    --export=ALL,NUM_EPOCHS=25,SAVE_EVERY=5,SEQ_LENGTH=30000,WEIGHT_DECAY=1e-3,OUTPUT_DIR=checkpoints/s1_grokking_qwen32b,JOB_DESCRIPTION="S1 Grokking: Qwen2.5-32B on s1K (25 epochs, wd=1e-3)",MONITOR_INTERVAL=120 \
    "$SCRIPT_DIR/s1_sft_job.slurm" \
    --wandb-run-name "s1_grokking_25ep"
