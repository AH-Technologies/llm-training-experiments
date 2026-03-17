#!/bin/bash
#SBATCH --job-name=osm-entropy
#SBATCH --account=nn12068k
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=100G
#SBATCH --partition=accel
#SBATCH --output=logs/osm-entropy-%A_%a.out
#SBATCH --error=logs/osm-entropy-%A_%a.err
#SBATCH --export=NONE

# One-Shot Metrics: Compute per-token entropy via HuggingFace forward pass
#
# Reads existing rollout jsonl files and computes full-vocabulary entropy
# at each token position. Does NOT regenerate rollouts.
#
# Usage:
#   # Single GPU
#   sbatch one_shot_metrics/submit_token_entropy.slurm
#
#   # Shard across 4 GPUs
#   sbatch --array=0-3 one_shot_metrics/submit_token_entropy.slurm
#
#   # Custom rollout dir or model
#   sbatch one_shot_metrics/submit_token_entropy.slurm --rollout_dir results/other_run --model Qwen/Qwen2.5-Math-7B

set -e

echo "=========================================="
echo "One-Shot Metrics: Token Entropy"
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

# Verify environment
python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA: {torch.cuda.is_available()}')
print(f'  GPUs: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'    GPU {i}: {torch.cuda.get_device_name(i)}')
"

# --- Parse arguments ---
MODEL="Qwen/Qwen2.5-Math-1.5B"
ROLLOUT_DIR="$OSM_DIR/results/wang_benchmark"
OUTPUT_DIR=""
DTYPE="bfloat16"

while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2 ;;
        --rollout_dir) ROLLOUT_DIR="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --dtype) DTYPE="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="$ROLLOUT_DIR/entropy"
fi

# --- Discover rollout files and shard for array jobs ---
ROLLOUT_FILES=($(ls "$ROLLOUT_DIR"/pi_*.jsonl 2>/dev/null | sort))
TOTAL=${#ROLLOUT_FILES[@]}

if [ "$TOTAL" -eq 0 ]; then
    echo "No rollout files found in $ROLLOUT_DIR"
    exit 1
fi

NUM_TASKS=${SLURM_ARRAY_TASK_COUNT:-1}
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

if [ "$NUM_TASKS" -gt 1 ]; then
    BASE_SIZE=$((TOTAL / NUM_TASKS))
    REMAINDER=$((TOTAL % NUM_TASKS))

    if [ "$TASK_ID" -lt "$REMAINDER" ]; then
        SHARD_SIZE=$((BASE_SIZE + 1))
        SHARD_START=$((TASK_ID * SHARD_SIZE))
    else
        SHARD_SIZE=$BASE_SIZE
        SHARD_START=$((REMAINDER * (BASE_SIZE + 1) + (TASK_ID - REMAINDER) * BASE_SIZE))
    fi

    SHARD_FILES=("${ROLLOUT_FILES[@]:$SHARD_START:$SHARD_SIZE}")
    echo "Array task $TASK_ID/$NUM_TASKS: $SHARD_SIZE of $TOTAL rollout files"

    if [ "${#SHARD_FILES[@]}" -eq 0 ]; then
        echo "No files for this shard, exiting."
        exit 0
    fi
else
    SHARD_FILES=("${ROLLOUT_FILES[@]}")
    echo "Processing all $TOTAL rollout files"
fi

# Write shard file list to a temp file for the python script
SHARD_LIST=$(mktemp)
printf '%s\n' "${SHARD_FILES[@]}" > "$SHARD_LIST"

echo ""
echo "Model: $MODEL"
echo "Rollout dir: $ROLLOUT_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "Dtype: $DTYPE"
echo ""

python $OSM_DIR/compute_token_entropy.py \
    --rollout_dir "$ROLLOUT_DIR" \
    --model "$MODEL" \
    --output_dir "$OUTPUT_DIR" \
    --dtype "$DTYPE" \
    --file_list "$SHARD_LIST"

rm -f "$SHARD_LIST"

echo ""
echo "Shard $TASK_ID done. Entropy profiles saved to $OUTPUT_DIR"
