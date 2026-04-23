#!/bin/bash
#SBATCH --job-name=head-analysis
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/analysis_%j.log
#SBATCH --error=logs/analysis_%j.err

# Full reasoning head analysis pipeline:
#   1. Download checkpoint from HuggingFace
#   2. Identify reasoning heads via EAP-IG + heatmap
#   3. Ablation study (baseline / top-10 / 3x random-10 / incremental curve)
#
# Usage:
#   # Analyze a specific checkpoint step
#   sbatch --export=ALL,HF_REPO="alexauren/grpo_pi13_math500_v2_lr_increased_multi_val",STEP=400 \
#       reasoning_head_analysis/slurm/run_analysis.slurm
#
#   # Analyze a regular HF model (no download needed)
#   sbatch --export=ALL,MODEL="Qwen/Qwen2.5-1.5B-Instruct" \
#       reasoning_head_analysis/slurm/run_analysis.slurm
#
#   # Override number of eval problems
#   sbatch --export=ALL,HF_REPO="alexauren/grpo_pi13_math500_v2_lr_increased_multi_val",STEP=400,NUM_PROBLEMS=100 \
#       reasoning_head_analysis/slurm/run_analysis.slurm

set -e

# Config
HF_REPO=${HF_REPO:-""}
STEP=${STEP:-""}
MODEL=${MODEL:-""}
N_PAIRS=${N_PAIRS:-300}
IG_STEPS=${IG_STEPS:-100}
NUM_PROBLEMS=${NUM_PROBLEMS:-50}
DATA_PATH=${DATA_PATH:-"data/math500.parquet"}

echo "==========================================="
echo "Reasoning Head Analysis Pipeline"
echo "==========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Architecture: $(uname -m)"
echo "HF_REPO: ${HF_REPO:-<none>}"
echo "STEP: ${STEP:-<none>}"
echo "MODEL: ${MODEL:-<auto>}"
echo "==========================================="

# Load modules — NRIS/GPU only exists on ARM (Olivia) nodes
if module avail NRIS/GPU 2>&1 | grep -q NRIS; then
    module load NRIS/GPU
fi
module load Python/3.12.3-GCCcore-13.3.0

# Try project venvs in order of preference
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
    set -a
    source "$PROJECT_DIR/.env"
    set +a
fi

export HF_TOKEN=${HF_TOKEN:-""}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH=$PROJECT_DIR/src:$PYTHONPATH
export NCCL_DEBUG=WARN
export VLLM_USE_V1=1

if [ -d "$PROJECT_DIR/tiktoken_encodings" ]; then
    export TIKTOKEN_ENCODINGS_BASE=$PROJECT_DIR/tiktoken_encodings
fi

mkdir -p logs

# ── Step 0: Resolve MODEL path ──────────────────────────────────────
if [ -n "$HF_REPO" ] && [ -n "$STEP" ]; then
    CHECKPOINT_DIR="reasoning_head_analysis/checkpoints/$(echo $HF_REPO | tr '/' '_')"
    SUBFOLDER="global_step_${STEP}"
    MODEL_PATH="${CHECKPOINT_DIR}/${SUBFOLDER}"

    if [ -d "$MODEL_PATH" ] && [ -f "$MODEL_PATH/config.json" ]; then
        echo "Checkpoint already downloaded: $MODEL_PATH"
    else
        echo ""
        echo "Downloading checkpoint: ${HF_REPO} / ${SUBFOLDER}"
        mkdir -p "$CHECKPOINT_DIR"
        huggingface-cli download "$HF_REPO" \
            --include "${SUBFOLDER}/*" \
            --local-dir "$CHECKPOINT_DIR"
        echo "Downloaded to: $MODEL_PATH"
    fi

    MODEL="$MODEL_PATH"
elif [ -z "$MODEL" ]; then
    echo "ERROR: Set either HF_REPO+STEP or MODEL"
    exit 1
fi

# Set HF cache — use project dir if writable, else fall back to home
MODEL_CACHE_KEY=$(echo "$MODEL" | tr '/' '_')
if [ -w "/cluster/projects/nn12068k" ]; then
    export HF_HOME=/cluster/projects/nn12068k/alexaau/.cache/huggingface_${MODEL_CACHE_KEY}
else
    export HF_HOME=${HOME}/.cache/huggingface
fi
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME/datasets
mkdir -p $HF_DATASETS_CACHE

# Output directory
SAFE_MODEL=$(echo "$MODEL" | tr '/' '_' | tr '.' '_')
OUTPUT_DIR="reasoning_head_analysis/results/${SAFE_MODEL}"

echo ""
echo "Model: $MODEL"
echo "Output: $OUTPUT_DIR"
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

# ── Step 1: Identify reasoning heads ────────────────────────────────
echo "==========================================="
echo "STEP 1: EAP-IG Head Identification"
echo "==========================================="

python -m reasoning_head_analysis.identify_heads \
    --model "$MODEL" \
    --n_pairs "$N_PAIRS" \
    --ig_steps "$IG_STEPS" \
    --output_dir "$OUTPUT_DIR"

IMPORTANCE_PATH="${OUTPUT_DIR}/head_importance.pt"

if [ ! -f "$IMPORTANCE_PATH" ]; then
    echo "ERROR: identify_heads did not produce $IMPORTANCE_PATH"
    exit 1
fi

# ── Step 2: Ablation study ──────────────────────────────────────────
echo ""
echo "==========================================="
echo "STEP 2: Ablation Study"
echo "==========================================="

python -m reasoning_head_analysis.ablate_heads \
    --model "$MODEL" \
    --importance_path "$IMPORTANCE_PATH" \
    --data_path "$DATA_PATH" \
    --num_problems "$NUM_PROBLEMS" \
    --output_dir "${OUTPUT_DIR}/ablation"

echo ""
echo "==========================================="
echo "DONE"
echo "==========================================="
echo "Results: $OUTPUT_DIR"
echo "  - head_importance.pt"
echo "  - head_importance_heatmap.png"
echo "  - ablation/ablation_results.json"
echo "  - ablation/ablation_barplot.png"
echo "  - ablation/ablation_curve.png"
echo "Job ID: $SLURM_JOB_ID"
ls -la "$OUTPUT_DIR" "$OUTPUT_DIR/ablation/"
