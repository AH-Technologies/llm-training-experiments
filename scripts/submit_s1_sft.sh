#!/bin/bash
# Submit S1 SFT job: Qwen2.5-32B-Instruct on s1K (standard 5-epoch run)
#
# Usage:
#   bash scripts/submit_s1_sft.sh
#   NUM_EPOCHS=10 bash scripts/submit_s1_sft.sh   # override epochs
#   TOTAL_GPUS=8 bash scripts/submit_s1_sft.sh    # override GPU count

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/cluster_config.sh"

# Compute resources
TOTAL_GPUS=${TOTAL_GPUS:-16}
GPUS_PER_NODE=${GPUS_PER_NODE:-$DEFAULT_GPUS_PER_NODE}
NODES=$(( TOTAL_GPUS / GPUS_PER_NODE ))
TIME=${TIME:-"02:00:00"}
ACCOUNT=${ACCOUNT:-$SLURM_ACCOUNT_DEFAULT}
PARTITION=${PARTITION:-$SLURM_PARTITION_DEFAULT}

# Training params (passthrough via env)
NUM_EPOCHS=${NUM_EPOCHS:-5}
SAVE_EVERY=${SAVE_EVERY:-1}
OUTPUT_DIR=${OUTPUT_DIR:-"checkpoints/s1_sft_qwen32b"}

echo "Submitting S1 SFT job on $CLUSTER_NAME"
echo "  Nodes: $NODES x $GPUS_PER_NODE GPUs = $TOTAL_GPUS total"
echo "  Account: $ACCOUNT, Partition: $PARTITION"
echo "  Epochs: $NUM_EPOCHS"
[ -n "$CONSTRAINT" ] && echo "  Constraint: $CONSTRAINT"

CONSTRAINT=${CONSTRAINT:-${DEFAULT_CONSTRAINT:-}}
CONSTRAINT_FLAG=""
[ -n "$CONSTRAINT" ] && CONSTRAINT_FLAG="--constraint=$CONSTRAINT"

sbatch \
    --job-name=s1-sft-32b \
    --account="$ACCOUNT" \
    --partition="$PARTITION" \
    --time="$TIME" \
    --nodes="$NODES" \
    --ntasks-per-node=1 \
    --gpus-per-node="$GPUS_PER_NODE" \
    --cpus-per-task="$DEFAULT_CPUS_PER_TASK" \
    --mem="$DEFAULT_MEM" \
    --output=logs/s1-sft-%j.out \
    --error=logs/s1-sft-%j.err \
    $CONSTRAINT_FLAG \
    --export=ALL,NUM_EPOCHS=$NUM_EPOCHS,SAVE_EVERY=$SAVE_EVERY,OUTPUT_DIR=$OUTPUT_DIR,MONITOR_INTERVAL=60 \
    "$SCRIPT_DIR/s1_sft_job.slurm"
