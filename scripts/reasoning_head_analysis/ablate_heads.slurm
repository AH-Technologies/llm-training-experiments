#!/bin/bash
#SBATCH --job-name=ablate-heads
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/ablate_%j.log
#SBATCH --error=logs/ablate_%j.err

# Head scaling / ablation study (MATH-500).
#
# Usage:
#   sbatch --export=ALL,\
#     MODEL="Qwen/Qwen2.5-Math-1.5B",\
#     IMPORTANCE_PATH="reasoning_head_analysis/results/Qwen_Qwen2.5-Math-1.5B/head_importance.pt",\
#     NUM_PROBLEMS=30,SCALES="0.0,0.5,2.0" \
#     reasoning_head_analysis/slurm/ablate_heads.slurm

set -e

MODEL=${MODEL:-"Qwen/Qwen2.5-Math-1.5B"}
IMPORTANCE_PATH=${IMPORTANCE_PATH:-"reasoning_head_analysis/results/Qwen_Qwen2.5-Math-1.5B/head_importance.pt"}
DATA_PATH=${DATA_PATH:-"data/math500.parquet"}
OUTPUT_DIR=${OUTPUT_DIR:-""}
NUM_PROBLEMS=${NUM_PROBLEMS:-30}
SCALES=${SCALES:-"0.0,0.25,0.5,1.5,2.0,4.0"}
TOP_K=${TOP_K:-10}
USE_ACTIVE_HEADS=${USE_ACTIVE_HEADS:-1}
EXCLUDE_LAYER0=${EXCLUDE_LAYER0:-1}
HEADS=${HEADS:-""}

echo "==========================================="
echo "Head Scaling / Ablation Study"
echo "==========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Model: $MODEL"
echo "Importance: $IMPORTANCE_PATH"
echo "Problems: $NUM_PROBLEMS"
echo "Scales: $SCALES"
echo "Top-k: $TOP_K"
echo "==========================================="

if module avail NRIS/GPU 2>&1 | grep -q NRIS; then
    module load NRIS/GPU
fi
module load Python/3.12.3-GCCcore-13.3.0

# Find venv (same pattern as run_evolution.slurm)
PROJECT_DIR=""
for candidate in \
    /cluster/projects/nn12068k/alexaau/llm-training-experiments \
    /cluster/projects/nn12068k/haaklau/llm-training-experiments \
    /cluster/home/haaklau/code/llm-training-experiments; do
    if [ -d "$candidate/venv" ]; then
        PROJECT_DIR="$candidate"
        break
    fi
done
if [ -z "$PROJECT_DIR" ]; then
    echo "ERROR: No venv found"
    exit 1
fi
cd $PROJECT_DIR
source $PROJECT_DIR/venv/bin/activate
echo "Using venv: $PROJECT_DIR/venv"

if [ -f "$PROJECT_DIR/.env" ]; then
    set -a; source "$PROJECT_DIR/.env"; set +a
fi

if [ -w "/cluster/projects/nn12068k" ]; then
    export HF_HOME=/cluster/projects/nn12068k/alexaau/.cache/huggingface
else
    export HF_HOME=${HOME}/.cache/huggingface
fi
export TRANSFORMERS_CACHE=$HF_HOME
mkdir -p $HF_HOME

export HF_TOKEN=${HF_TOKEN:-""}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH=$PROJECT_DIR/src:$PYTHONPATH

mkdir -p logs

echo ""
echo "Environment check:"
python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA: {torch.cuda.is_available()}')
print(f'  GPUs: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'    GPU {i}: {torch.cuda.get_device_name(i)}')
"
echo ""

EXTRA_FLAGS=""
[ "$USE_ACTIVE_HEADS" = "1" ] && EXTRA_FLAGS="$EXTRA_FLAGS --use_active_heads"
[ "$EXCLUDE_LAYER0" = "1" ] && EXTRA_FLAGS="$EXTRA_FLAGS --exclude_layer0"
[ -n "$OUTPUT_DIR" ] && EXTRA_FLAGS="$EXTRA_FLAGS --output_dir $OUTPUT_DIR"
[ -n "$HEADS" ] && EXTRA_FLAGS="$EXTRA_FLAGS --heads $HEADS"

python -m reasoning_head_analysis.ablate_heads \
    --model "$MODEL" \
    --importance_path "$IMPORTANCE_PATH" \
    --data_path "$DATA_PATH" \
    --num_problems "$NUM_PROBLEMS" \
    --scales "$SCALES" \
    --top_k "$TOP_K" \
    $EXTRA_FLAGS

echo "Done. Job ID: $SLURM_JOB_ID"
