#!/bin/bash
#SBATCH --job-name=prep-dapo
#SBATCH --account=nn12068k
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=small
#SBATCH --output=logs/prep-dapo-%j.out
#SBATCH --error=logs/prep-dapo-%j.err

# Download and prepare DAPO-Math-17K dataset
#
# Usage:
#   sbatch scripts/submit_prepare_dapo.slurm
#   sbatch scripts/submit_prepare_dapo.slurm --max-problems 5000

set -e

echo "=========================================="
echo "Prepare DAPO-Math-17K"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Args: $@"
echo "=========================================="

mkdir -p logs

module load NRIS/CPU
module load Python/3.12.3-GCCcore-13.3.0

PROJECT_DIR=/cluster/projects/nn12068k/alexaau/llm-training-experiments

if [ -d "$PROJECT_DIR/venv" ]; then
    source $PROJECT_DIR/venv/bin/activate
fi

python -m pip install -q datasets

if [ -f "$PROJECT_DIR/.env" ]; then
    set -a
    source "$PROJECT_DIR/.env"
    set +a
fi

export HF_HOME=/cluster/projects/nn12068k/alexaau/.cache/huggingface

cd $PROJECT_DIR

python scripts/prepare_dapo_self_teach.py "$@"

echo "Done!"
