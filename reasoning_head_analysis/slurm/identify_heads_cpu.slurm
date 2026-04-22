#!/bin/bash
#SBATCH --job-name=eap-ig-cpu
#SBATCH --account=share-ie-idi
#SBATCH --partition=CPUQ
#SBATCH --nodes=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=reasoning_head_analysis/logs/identify_cpu_%j.log
#SBATCH --error=reasoning_head_analysis/logs/identify_cpu_%j.err

# EAP-IG on CPU — slower but doesn't need a GPU.
# Uses all 56 cores for PyTorch MKL parallelism.
#
# Usage:
#   sbatch reasoning_head_analysis/slurm/identify_heads_cpu.slurm
#   sbatch --export=ALL,HF_REPO="alexauren/grpo_pi13_math500_v2_lr_increased_multi_val",STEP=400,N_PAIRS=50 \
#       reasoning_head_analysis/slurm/identify_heads_cpu.slurm

set -e

HF_REPO=${HF_REPO:-""}
STEP=${STEP:-""}
MODEL=${MODEL:-"Qwen/Qwen2.5-Math-1.5B"}
N_PAIRS=${N_PAIRS:-300}
IG_STEPS=${IG_STEPS:-100}

echo "==========================================="
echo "EAP-IG Head Identification (CPU)"
echo "==========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "HF_REPO: ${HF_REPO:-<none>}"
echo "STEP: ${STEP:-<none>}"
echo "MODEL: ${MODEL}"
echo "N_PAIRS: $N_PAIRS, IG_STEPS: $IG_STEPS"
echo "==========================================="

module load Python/3.12.3-GCCcore-13.3.0

# Find venv
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

# HF cache
if [ -w "/cluster/projects/nn12068k" ]; then
    export HF_HOME=/cluster/projects/nn12068k/alexaau/.cache/huggingface
else
    export HF_HOME=${HOME}/.cache/huggingface
fi
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME/datasets
mkdir -p $HF_DATASETS_CACHE

export HF_TOKEN=${HF_TOKEN:-""}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH

# Maximize CPU parallelism
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

mkdir -p reasoning_head_analysis/logs

# Download checkpoint if needed
if [ -n "$HF_REPO" ] && [ -n "$STEP" ]; then
    SAFE_REPO=$(echo "$HF_REPO" | tr '/' '_')
    CHECKPOINT_DIR="reasoning_head_analysis/checkpoints/${SAFE_REPO}"
    SUBFOLDER="global_step_${STEP}"
    MODEL_PATH="${CHECKPOINT_DIR}/${SUBFOLDER}"

    if [ -d "$MODEL_PATH" ] && [ -f "$MODEL_PATH/config.json" ]; then
        echo "Checkpoint already downloaded: $MODEL_PATH"
    else
        echo "Downloading: ${HF_REPO} / ${SUBFOLDER}"
        mkdir -p "$CHECKPOINT_DIR"
        huggingface-cli download "$HF_REPO" \
            --include "${SUBFOLDER}/*" \
            --local-dir "$CHECKPOINT_DIR" || \
        hf download "$HF_REPO" \
            --include "${SUBFOLDER}/*" \
            --local-dir "$CHECKPOINT_DIR"
    fi
    MODEL="$MODEL_PATH"
fi

SAFE_MODEL=$(echo "$MODEL" | tr '/' '_' | tr '.' '_')
OUTPUT_DIR="reasoning_head_analysis/results/${SAFE_MODEL}"

echo ""
echo "Model: $MODEL"
echo "Output: $OUTPUT_DIR"
echo "Threads: OMP=$OMP_NUM_THREADS MKL=$MKL_NUM_THREADS"
echo ""

python -m reasoning_head_analysis.identify_heads \
    --model "$MODEL" \
    --n_pairs "$N_PAIRS" \
    --ig_steps "$IG_STEPS" \
    --device cpu \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "Done. Job ID: $SLURM_JOB_ID"
echo "Results: $OUTPUT_DIR"
ls -la "$OUTPUT_DIR/"
