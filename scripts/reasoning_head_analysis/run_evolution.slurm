#!/bin/bash
#SBATCH --job-name=head-evolution
#SBATCH --account=nn12068k
#SBATCH --partition=accel
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/evolution_%j.log
#SBATCH --error=logs/evolution_%j.err

# Full evolution analysis: identify heads at each checkpoint, then plot evolution.
#
# Usage:
#   sbatch --export=ALL,HF_REPO="alexauren/grpo_pi13_math500_v2_lr_increased_multi_val",BASE_MODEL="Qwen/Qwen2.5-Math-1.5B",STEPS="100 200 300 400 500 600 700" \
#       reasoning_head_analysis/slurm/run_evolution.slurm
#
#   # With fewer steps (faster):
#   sbatch --export=ALL,HF_REPO="alexauren/grpo_pi13_math500_v2_lr_increased_multi_val",BASE_MODEL="Qwen/Qwen2.5-Math-1.5B",STEPS="100 400 700" \
#       reasoning_head_analysis/slurm/run_evolution.slurm

set -e

HF_REPO=${HF_REPO:-"alexauren/grpo_pi13_math500_v2_lr_increased_multi_val"}
BASE_MODEL=${BASE_MODEL:-"Qwen/Qwen2.5-Math-1.5B"}
STEPS=${STEPS:-"100 200 300 400 500 600 700"}
N_PAIRS=${N_PAIRS:-300}
IG_STEPS=${IG_STEPS:-100}
TOP_K=${TOP_K:-20}

echo "==========================================="
echo "Reasoning Head Evolution Analysis"
echo "==========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Architecture: $(uname -m)"
echo "Base model: $BASE_MODEL"
echo "HF repo: $HF_REPO"
echo "Steps: $STEPS"
echo "==========================================="

# Load modules
if module avail NRIS/GPU 2>&1 | grep -q NRIS; then
    module load NRIS/GPU
fi
module load Python/3.12.3-GCCcore-13.3.0

# Find venv
PROJECT_DIR=""
for candidate in \
    /cluster/projects/nn12068k/haaklau/llm-training-experiments \
    /cluster/projects/nn12068k/alexaau/llm-training-experiments \
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

# HF cache
if [ -w "/cluster/projects/nn12068k" ]; then
    export HF_HOME=/cluster/projects/nn12068k/haaklau/.cache/huggingface
else
    export HF_HOME=${HOME}/.cache/huggingface
fi
export TRANSFORMERS_CACHE=$HF_HOME
mkdir -p $HF_HOME

if [ -d "$PROJECT_DIR/tiktoken_encodings" ]; then
    export TIKTOKEN_ENCODINGS_BASE=$PROJECT_DIR/tiktoken_encodings
fi

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

SAFE_REPO=$(echo "$HF_REPO" | tr '/' '_')
RESULTS_BASE="results/reasoning_head_analysis"

# ── Step 1: Identify heads for base model ────────────────────────────
SAFE_BASE=$(echo "$BASE_MODEL" | tr '/' '_')
BASE_IMPORTANCE="${RESULTS_BASE}/${SAFE_BASE}/head_importance.pt"

if [ -f "$BASE_IMPORTANCE" ]; then
    echo "Base model heads already identified: $BASE_IMPORTANCE"
else
    echo "==========================================="
    echo "Identifying heads for base model: $BASE_MODEL"
    echo "==========================================="
    python -m reasoning_head_analysis.identify_heads \
        --model "$BASE_MODEL" \
        --n_pairs "$N_PAIRS" \
        --ig_steps "$IG_STEPS" \
        --output_dir "${RESULTS_BASE}/${SAFE_BASE}"
fi

# ── Step 2: Identify heads for each checkpoint ───────────────────────
# Use local models from models/ directory, or fall back to downloading
LOCAL_MODEL_DIR=${LOCAL_MODEL_DIR:-"models/${SAFE_REPO}"}

for STEP in $STEPS; do
    SUBFOLDER="global_step_${STEP}"

    # Check local models/ first, then fall back to downloading
    if [ -d "${LOCAL_MODEL_DIR}/${SUBFOLDER}" ] && [ -f "${LOCAL_MODEL_DIR}/${SUBFOLDER}/config.json" ]; then
        MODEL_PATH="${LOCAL_MODEL_DIR}/${SUBFOLDER}"
        echo "Using local model: $MODEL_PATH"
    else
        CHECKPOINT_DIR="reasoning_head_analysis/checkpoints/${SAFE_REPO}"
        MODEL_PATH="${CHECKPOINT_DIR}/${SUBFOLDER}"
        if [ ! -d "$MODEL_PATH" ] || [ ! -f "$MODEL_PATH/config.json" ]; then
            echo ""
            echo "Downloading: ${HF_REPO} / ${SUBFOLDER}"
            mkdir -p "$CHECKPOINT_DIR"
            huggingface-cli download "$HF_REPO" \
                --include "${SUBFOLDER}/*" \
                --local-dir "$CHECKPOINT_DIR" || \
            hf download "$HF_REPO" \
                --include "${SUBFOLDER}/*" \
                --local-dir "$CHECKPOINT_DIR"
        fi
    fi

    # Identify heads
    SAFE_STEP=$(echo "$MODEL_PATH" | tr '/' '_' | tr '.' '_')
    STEP_IMPORTANCE="${RESULTS_BASE}/${SAFE_STEP}/head_importance.pt"

    if [ -f "$STEP_IMPORTANCE" ]; then
        echo "Step $STEP heads already identified: $STEP_IMPORTANCE"
    else
        echo ""
        echo "==========================================="
        echo "Identifying heads for step $STEP"
        echo "==========================================="
        python -m reasoning_head_analysis.identify_heads \
            --model "$MODEL_PATH" \
            --n_pairs "$N_PAIRS" \
            --ig_steps "$IG_STEPS" \
            --output_dir "${RESULTS_BASE}/${SAFE_STEP}"
    fi
done

# ── Step 3: Plot evolution ───────────────────────────────────────────
echo ""
echo "==========================================="
echo "Plotting head evolution"
echo "==========================================="

EVOLUTION_DIR="${RESULTS_BASE}/evolution_${SAFE_REPO}"
BASE_CIRCUIT="${RESULTS_BASE}/${SAFE_BASE}/circuit.json"

python -m reasoning_head_analysis.plot_head_evolution \
    --base "$BASE_CIRCUIT" \
    --results_dir "$RESULTS_BASE" \
    --filter "$SAFE_REPO" \
    --selection circuit \
    --threshold 0.1 \
    --output_dir "$EVOLUTION_DIR"

echo ""
echo "==========================================="
echo "DONE"
echo "==========================================="
echo "Results: $EVOLUTION_DIR"
ls -la "$EVOLUTION_DIR/"
echo "Job ID: $SLURM_JOB_ID"
