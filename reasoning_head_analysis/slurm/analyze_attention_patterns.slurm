#!/bin/bash
#SBATCH --job-name=attn-patterns
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=reasoning_head_analysis/logs/attn_%j.log
#SBATCH --error=reasoning_head_analysis/logs/attn_%j.err

# Attention-pattern analysis for a set of heads in a model.
#
# Usage — explicit heads on the base model (e.g. the 6 overlapping ones):
#   sbatch --export=ALL,\
#     MODEL="Qwen/Qwen2.5-Math-1.5B",\
#     HEADS="11.8,15.7,18.8,19.6,19.11,23.9",\
#     OUTPUT="reasoning_head_analysis/results/attn_base_overlap" \
#     reasoning_head_analysis/slurm/analyze_attention_patterns.slurm
#
# Usage — active_heads from a checkpoint's head_importance.pt:
#   sbatch --export=ALL,\
#     MODEL="reasoning_head_analysis/checkpoints/alexauren_grpo_pi13_.../global_step_700",\
#     IMPORTANCE_PATH="reasoning_head_analysis/results/...global_step_700/head_importance.pt",\
#     EXCLUDE_LAYER0=1,\
#     OUTPUT="reasoning_head_analysis/results/attn_pi13_step700" \
#     reasoning_head_analysis/slurm/analyze_attention_patterns.slurm

set -e

MODEL=${MODEL:?"MODEL is required"}
OUTPUT=${OUTPUT:?"OUTPUT is required"}
HEADS=${HEADS:-""}
IMPORTANCE_PATH=${IMPORTANCE_PATH:-""}
EXCLUDE_LAYER0=${EXCLUDE_LAYER0:-0}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-400}

echo "==========================================="
echo "Attention-pattern analysis"
echo "==========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Model: $MODEL"
echo "Heads: ${HEADS:-(from $IMPORTANCE_PATH)}"
echo "Output: $OUTPUT"
echo "==========================================="

if module avail NRIS/GPU 2>&1 | grep -q NRIS; then
    module load NRIS/GPU
fi
module load Python/3.12.3-GCCcore-13.3.0

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
mkdir -p $HF_HOME
export HF_TOKEN=${HF_TOKEN:-""}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
mkdir -p reasoning_head_analysis/logs

EXTRA=""
[ -n "$HEADS" ] && EXTRA="$EXTRA --heads $HEADS"
[ -n "$IMPORTANCE_PATH" ] && EXTRA="$EXTRA --importance_path $IMPORTANCE_PATH"
[ "$EXCLUDE_LAYER0" = "1" ] && EXTRA="$EXTRA --exclude_layer0"

python -m reasoning_head_analysis.analyze_attention_patterns \
    --model "$MODEL" \
    --output "$OUTPUT" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    $EXTRA

echo "Done. Job ID: $SLURM_JOB_ID"
