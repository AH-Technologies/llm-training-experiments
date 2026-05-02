#!/bin/bash
#SBATCH --job-name=mathneuro-evo
#SBATCH --account=nn12068k
#SBATCH --partition=accel
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/mathneuro_evo_%j.log
#SBATCH --error=logs/mathneuro_evo_%j.err

# Neurosurgery evolution: identify task-specific heads at base model + each checkpoint.
#
# Runs identify_heads_mathneuro.py sequentially for each step, skipping any
# that already have results. Uses 4 GPUs per run (data parallelism).
#
# Supports different contrasts via CONTRAST env var:
#   math_vs_reading  — GSM8K vs RACE (original MathNeuro, default)
#   reasoning_vs_easy — ARC-Challenge vs ARC-Easy
#   cot_vs_direct    — GSM8K full CoT vs direct answer only
#
# Usage:
#   # Pi13, default contrast (math_vs_reading):
#   sbatch scripts/reasoning_head_analysis/run_evolution_mathneuro.slurm
#
#   # Pi1:
#   sbatch --export=ALL,\
#     RUN_NAME="pi1",\
#     LOCAL_MODEL_DIR="models/grpo_pi1_math500_lr_increased_multi_val" \
#     scripts/reasoning_head_analysis/run_evolution_mathneuro.slurm
#
#   # Reasoning vs easy contrast:
#   sbatch --export=ALL,CONTRAST="reasoning_vs_easy" \
#     scripts/reasoning_head_analysis/run_evolution_mathneuro.slurm
#
#   # CoT vs direct contrast:
#   sbatch --export=ALL,CONTRAST="cot_vs_direct" \
#     scripts/reasoning_head_analysis/run_evolution_mathneuro.slurm

set -e

RUN_NAME=${RUN_NAME:-"pi13"}
BASE_MODEL=${BASE_MODEL:-"Qwen/Qwen2.5-Math-1.5B"}
STEPS=${STEPS:-"100 200 300 400 500 600 700"}
N_SAMPLES=${N_SAMPLES:-500}
KEEP_RATIO=${KEEP_RATIO:-0.05}
MAX_LENGTH=${MAX_LENGTH:-512}
CONTRAST=${CONTRAST:-"math_vs_reading"}

# Default local model dirs by run name
if [ -z "$LOCAL_MODEL_DIR" ]; then
    if [ "$RUN_NAME" = "pi13" ]; then
        LOCAL_MODEL_DIR="models/grpo_pi13_math500_v2_lr_increased_multi_val"
    elif [ "$RUN_NAME" = "pi1" ]; then
        LOCAL_MODEL_DIR="models/grpo_pi1_math500_lr_increased_multi_val"
    else
        echo "ERROR: Set LOCAL_MODEL_DIR for custom run name '$RUN_NAME'"
        exit 1
    fi
fi

echo "==========================================="
echo "MathNeuro Evolution Analysis"
echo "==========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Architecture: $(uname -m)"
echo "Run name: $RUN_NAME"
echo "Base model: $BASE_MODEL"
echo "Local models: $LOCAL_MODEL_DIR"
echo "Steps: $STEPS"
echo "Contrast: $CONTRAST"
echo "Samples/domain: $N_SAMPLES, keep_ratio: $KEEP_RATIO"
echo "==========================================="

module load NRIS/GPU
module load Python/3.12.3-GCCcore-13.3.0

PROJECT_DIR=/cluster/projects/nn12068k/haaklau/llm-training-experiments
cd $PROJECT_DIR
source $PROJECT_DIR/venv/bin/activate

if [ -f "$PROJECT_DIR/.env" ]; then
    set -a
    source "$PROJECT_DIR/.env"
    set +a
fi

export HF_HOME=/cluster/projects/nn12068k/haaklau/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME/datasets
mkdir -p $HF_DATASETS_CACHE

export HF_TOKEN=${HF_TOKEN:-""}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH=$PROJECT_DIR/src:$PYTHONPATH
export NCCL_DEBUG=WARN

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

RESULTS_BASE="results/reasoning_head_analysis/identification/neurosurgery_${CONTRAST}"

# ── Step 1: Base model ───────────────────────────────────────────────
BASE_OUTPUT="${RESULTS_BASE}/base_model"
BASE_IMPORTANCE="${BASE_OUTPUT}/head_importance.pt"

if [ -f "$BASE_IMPORTANCE" ]; then
    echo "Base model already done: $BASE_IMPORTANCE"
else
    echo "==========================================="
    echo "Neurosurgery ($CONTRAST): base model ($BASE_MODEL)"
    echo "==========================================="
    python -m reasoning_head_analysis.identify_heads_mathneuro \
        --model "$BASE_MODEL" \
        --contrast "$CONTRAST" \
        --n_samples "$N_SAMPLES" \
        --keep_ratio "$KEEP_RATIO" \
        --max_length "$MAX_LENGTH" \
        --output_dir "$BASE_OUTPUT"
fi

# ── Step 2: Each checkpoint ──────────────────────────────────────────
for STEP in $STEPS; do
    STEP_OUTPUT="${RESULTS_BASE}/${RUN_NAME}_step${STEP}"
    STEP_IMPORTANCE="${STEP_OUTPUT}/head_importance.pt"

    if [ -f "$STEP_IMPORTANCE" ]; then
        echo "Step $STEP already done: $STEP_IMPORTANCE"
        continue
    fi

    MODEL_PATH="${LOCAL_MODEL_DIR}/global_step_${STEP}"
    if [ ! -d "$MODEL_PATH" ] || [ ! -f "$MODEL_PATH/config.json" ]; then
        echo "WARNING: Checkpoint not found: $MODEL_PATH — skipping step $STEP"
        continue
    fi

    echo ""
    echo "==========================================="
    echo "Neurosurgery ($CONTRAST): ${RUN_NAME} step ${STEP}"
    echo "  Model: $MODEL_PATH"
    echo "==========================================="
    python -m reasoning_head_analysis.identify_heads_mathneuro \
        --model "$MODEL_PATH" \
        --contrast "$CONTRAST" \
        --n_samples "$N_SAMPLES" \
        --keep_ratio "$KEEP_RATIO" \
        --max_length "$MAX_LENGTH" \
        --output_dir "$STEP_OUTPUT"
done

echo ""
echo "==========================================="
echo "DONE — Neurosurgery ($CONTRAST) evolution for $RUN_NAME"
echo "==========================================="
echo "Results:"
ls -la "$RESULTS_BASE/"
echo "Job ID: $SLURM_JOB_ID"
