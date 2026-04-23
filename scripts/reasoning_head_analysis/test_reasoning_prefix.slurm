#!/bin/bash
#SBATCH --job-name=pfxtest
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/pfxtest_%j.log
#SBATCH --error=logs/pfxtest_%j.err

# Assistant-prefix steering test.
#
# Usage:
#   sbatch reasoning_head_analysis/slurm/test_reasoning_prefix.slurm
#   sbatch --export=ALL,MODEL="Qwen/Qwen2.5-Math-1.5B-Instruct" \
#       reasoning_head_analysis/slurm/test_reasoning_prefix.slurm
#   sbatch --export=ALL,MODEL="reasoning_head_analysis/checkpoints/alexauren_grpo_pi13_math500_v2_lr_increased_multi_val/global_step_600",OUTPUT_TAG="ckpt600" \
#       reasoning_head_analysis/slurm/test_reasoning_prefix.slurm

set -e

MODEL=${MODEL:-"Qwen/Qwen2.5-Math-1.5B"}
# OUTPUT_TAG: optional label for checkpoint paths (since they don't flatten nicely)
if [ -n "$OUTPUT_TAG" ]; then
    SAFE_MODEL="$OUTPUT_TAG"
else
    SAFE_MODEL=$(echo "$MODEL" | tr '/' '_')
fi
OUTPUT=${OUTPUT:-"reasoning_head_analysis/results/prefix_test_${SAFE_MODEL}"}

echo "==========================================="
echo "Prefix-steering test"
echo "==========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Model: $MODEL"
echo "Output: $OUTPUT"
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

if [ -f "$PROJECT_DIR/.env" ]; then
    set -a; source "$PROJECT_DIR/.env"; set +a
fi

if [ -w "/cluster/projects/nn12068k" ]; then
    export HF_HOME=/cluster/projects/nn12068k/alexaau/.cache/huggingface
else
    export HF_HOME=${HOME}/.cache/huggingface
fi
export TRANSFORMERS_CACHE=$HF_HOME
export HF_TOKEN=${HF_TOKEN:-""}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH=$PROJECT_DIR/src:$PYTHONPATH

mkdir -p logs

python -m reasoning_head_analysis.test_reasoning_prefix \
    --model "$MODEL" \
    --output "$OUTPUT"

echo "Done. Job ID: $SLURM_JOB_ID"
