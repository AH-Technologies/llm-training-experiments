#!/bin/bash
#SBATCH --job-name=dl-ckpts
#SBATCH --account=nn12068k
#SBATCH --partition=accel
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/cluster/projects/nn12068k/haaklau/llm-training-experiments/logs/download_%j.log
#SBATCH --error=/cluster/projects/nn12068k/haaklau/llm-training-experiments/logs/download_%j.err
#SBATCH --chdir=/cluster/projects/nn12068k/haaklau/llm-training-experiments

# Download training checkpoints (steps 100-700) from HuggingFace.

set -euo pipefail

echo "PWD at start: $(pwd)"
echo "Hostname: $(hostname)"

module load NRIS/GPU 2>&1 && echo "NRIS/GPU loaded" || { echo "NRIS/GPU failed"; exit 1; }
module load Python/3.12.3-GCCcore-13.3.0 2>&1 && echo "Python loaded" || { echo "Python failed"; exit 1; }

PROJECT_DIR="/cluster/projects/nn12068k/haaklau/llm-training-experiments"
cd "$PROJECT_DIR"
echo "Working in: $(pwd)"
source "$PROJECT_DIR/venv/bin/activate"
echo "Venv activated"

[ -f "$PROJECT_DIR/.env" ] && { set -a; source "$PROJECT_DIR/.env"; set +a; }

export HF_HOME=/cluster/projects/nn12068k/haaklau/.cache/huggingface
export HF_TOKEN=${HF_TOKEN:-""}
mkdir -p "$HF_HOME"

STEPS="100 200 300 400 500 600 700"

REPOS=(
    "alexauren/grpo_pi13_math500_v2_lr_increased_multi_val"
    "alexauren/grpo_pi1_math500_lr_increased_multi_val"
)

mkdir -p reasoning_head_analysis/checkpoints logs

for REPO in "${REPOS[@]}"; do
    CKPT_DIR="reasoning_head_analysis/checkpoints/$(echo "$REPO" | tr '/' '_')"
    mkdir -p "$CKPT_DIR"

    for STEP in $STEPS; do
        SUBFOLDER="global_step_${STEP}"
        MODEL_PATH="${CKPT_DIR}/${SUBFOLDER}"

        if [ -d "$MODEL_PATH" ] && [ -f "$MODEL_PATH/config.json" ]; then
            echo "SKIP: $REPO / $SUBFOLDER (already exists)"
            continue
        fi

        echo ""
        echo "Downloading: $REPO / $SUBFOLDER"
        huggingface-cli download "$REPO" \
            --include "${SUBFOLDER}/*" \
            --local-dir "$CKPT_DIR"
        echo "Done: $MODEL_PATH"
    done

    echo ""
    echo "=== $REPO complete ==="
done

echo ""
echo "All downloads finished. Job $SLURM_JOB_ID"
