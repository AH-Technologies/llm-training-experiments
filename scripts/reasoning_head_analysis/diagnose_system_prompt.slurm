#!/bin/bash
#SBATCH --job-name=sysdiag
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/sysdiag_%j.log
#SBATCH --error=logs/sysdiag_%j.err

# System-prompt diagnostic: does Qwen2.5-Math-1.5B condition on system messages?
#
# Usage:
#   sbatch reasoning_head_analysis/slurm/diagnose_system_prompt.slurm
#   sbatch --export=ALL,MODEL="Qwen/Qwen2.5-Math-1.5B-Instruct" \
#       reasoning_head_analysis/slurm/diagnose_system_prompt.slurm

set -e

MODEL=${MODEL:-"Qwen/Qwen2.5-Math-1.5B"}
SAFE_MODEL=$(echo "$MODEL" | tr '/' '_')
OUTPUT=${OUTPUT:-"reasoning_head_analysis/results/system_prompt_diagnosis_${SAFE_MODEL}"}
N_SAMPLES=${N_SAMPLES:-2}

echo "==========================================="
echo "System-prompt diagnostic"
echo "==========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Architecture: $(uname -m)"
echo "Model: $MODEL"
echo "Output: $OUTPUT"
echo "Samples per cell: $N_SAMPLES"
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
    set -a
    source "$PROJECT_DIR/.env"
    set +a
fi

export HF_TOKEN=${HF_TOKEN:-""}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH=$PROJECT_DIR/src:$PYTHONPATH

# HF cache
if [ -w "/cluster/projects/nn12068k" ]; then
    export HF_HOME=/cluster/projects/nn12068k/alexaau/.cache/huggingface
else
    export HF_HOME=${HOME}/.cache/huggingface
fi
export TRANSFORMERS_CACHE=$HF_HOME
mkdir -p $HF_HOME

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

python -m reasoning_head_analysis.diagnose_system_prompt \
    --model "$MODEL" \
    --output "$OUTPUT" \
    --n_samples "$N_SAMPLES"

echo "Done. Job ID: $SLURM_JOB_ID"
