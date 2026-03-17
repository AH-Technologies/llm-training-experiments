#!/bin/bash
#SBATCH --job-name=osm-rollouts
#SBATCH --account=nn12068k
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=100G
#SBATCH --partition=accel
#SBATCH --output=logs/osm-rollouts-%A_%a.out
#SBATCH --error=logs/osm-rollouts-%A_%a.err
#SBATCH --export=NONE

# One-Shot Metrics: Generate rollouts + compute metrics
#
# Usage:
#   # All 14 Wang benchmark examples on 1 GPU (no array)
#   sbatch one_shot_metrics/submit_rollouts.slurm
#
#   # All 14 Wang examples sharded across 4 GPUs
#   sbatch --array=0-3 one_shot_metrics/submit_rollouts.slurm
#
#   # All 1209 examples across 8 GPUs
#   sbatch --array=0-7 one_shot_metrics/submit_rollouts.slurm --all
#
#   # Specific examples, custom k
#   sbatch one_shot_metrics/submit_rollouts.slurm --examples 1 13 1207 1208 --k 64
#
# When using --array, each task gets a shard of examples. Metrics are
# computed by array task 0 after all shards finish.

set -e

echo "=========================================="
echo "One-Shot Metrics: Rollout Generation"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "Args: $@"
echo "=========================================="

# --- Configuration ---
PROJECT_DIR=/cluster/projects/nn12068k/alexaau/llm-training-experiments
OSM_DIR=$PROJECT_DIR/one_shot_metrics
cd $PROJECT_DIR

mkdir -p logs

# Load modules
module load NRIS/GPU
module load Python/3.12.3-GCCcore-13.3.0

# Activate virtual environment
source $PROJECT_DIR/venv/bin/activate

# Load .env file
if [ -f "$PROJECT_DIR/.env" ]; then
    set -a
    source "$PROJECT_DIR/.env"
    set +a
fi

# HuggingFace cache
export HF_HOME=/cluster/projects/nn12068k/alexaau/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export TOKENIZERS_PARALLELISM=false

# vLLM
export VLLM_USE_V1=1

# Verify environment
python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA: {torch.cuda.is_available()}')
print(f'  GPUs: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'    GPU {i}: {torch.cuda.get_device_name(i)}')
import vllm
print(f'  vLLM: {vllm.__version__}')
"

# --- Parse arguments ---
MODEL="Qwen/Qwen2.5-Math-1.5B"
K=128
OUTPUT_DIR="$OSM_DIR/results/wang_benchmark"
EXAMPLES=""
RUN_ALL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2 ;;
        --k) K="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --all) RUN_ALL=true; shift ;;
        --examples)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                EXAMPLES="$EXAMPLES $1"
                shift
            done
            ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Default: Wang benchmark 14 examples
if [ "$RUN_ALL" = false ] && [ -z "$EXAMPLES" ]; then
    EXAMPLES="1 2 4 7 11 13 16 17 605 606 1201 1207 1208 1209"
fi

# Convert examples string to array
read -ra EXAMPLE_ARRAY <<< "$EXAMPLES"

# --- Shard examples for array jobs ---
NUM_TASKS=${SLURM_ARRAY_TASK_COUNT:-1}
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

if [ "$NUM_TASKS" -gt 1 ]; then
    TOTAL=${#EXAMPLE_ARRAY[@]}
    # Compute this shard's start and count (distribute remainder across first tasks)
    BASE_SIZE=$((TOTAL / NUM_TASKS))
    REMAINDER=$((TOTAL % NUM_TASKS))

    if [ "$TASK_ID" -lt "$REMAINDER" ]; then
        SHARD_SIZE=$((BASE_SIZE + 1))
        SHARD_START=$((TASK_ID * SHARD_SIZE))
    else
        SHARD_SIZE=$BASE_SIZE
        SHARD_START=$((REMAINDER * (BASE_SIZE + 1) + (TASK_ID - REMAINDER) * BASE_SIZE))
    fi

    SHARD_EXAMPLES="${EXAMPLE_ARRAY[@]:$SHARD_START:$SHARD_SIZE}"
    echo "Array task $TASK_ID/$NUM_TASKS: examples [$SHARD_EXAMPLES] ($SHARD_SIZE of $TOTAL)"

    if [ -z "$SHARD_EXAMPLES" ]; then
        echo "No examples for this shard, exiting."
        exit 0
    fi

    EXAMPLES="$SHARD_EXAMPLES"
fi

# Build rollout command
ROLLOUT_CMD="python $OSM_DIR/compute_rollouts.py \
    --model $MODEL \
    --data_path $OSM_DIR/One-Shot-RLVR/data/train/one_shot_rlvr/dsr_sub.parquet \
    --ranking_path $OSM_DIR/One-Shot-RLVR/data/acc_step_500.json \
    --k $K \
    --output_dir $OUTPUT_DIR"

if [ -n "$EXAMPLES" ]; then
    ROLLOUT_CMD="$ROLLOUT_CMD --examples $EXAMPLES"
fi

echo ""
echo "Running: $ROLLOUT_CMD"
echo ""
eval $ROLLOUT_CMD

# --- Compute metrics (only after all shards are done) ---
if [ "$NUM_TASKS" -gt 1 ]; then
    if [ "$TASK_ID" -eq 0 ]; then
        # Wait for all other array tasks to finish
        echo ""
        echo "Shard 0: waiting for all array tasks to complete..."
        ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID}
        while true; do
            # Count running/pending tasks (excluding this one)
            REMAINING=$(squeue --array -j "$ARRAY_JOB_ID" -h -t running,pending 2>/dev/null | grep -v "${ARRAY_JOB_ID}_0" | wc -l)
            if [ "$REMAINING" -eq 0 ]; then
                break
            fi
            echo "  $REMAINING tasks still running, waiting 30s..."
            sleep 30
        done
        echo "All shards done. Computing metrics..."
        python $OSM_DIR/compute_metrics.py --rollout_dir $OUTPUT_DIR
        echo ""
        echo "Done! Results in $OUTPUT_DIR"
    else
        echo "Shard $TASK_ID done. Task 0 will compute metrics after all shards finish."
    fi
else
    echo ""
    echo "Computing metrics..."
    python $OSM_DIR/compute_metrics.py --rollout_dir $OUTPUT_DIR
    echo ""
    echo "Done! Results in $OUTPUT_DIR"
fi
