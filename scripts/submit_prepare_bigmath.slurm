#!/bin/bash
#SBATCH --job-name=prep-bigmath
#SBATCH --account=nn12068k
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=small
#SBATCH --output=logs/prep-bigmath-%j.out
#SBATCH --error=logs/prep-bigmath-%j.err

# Download and prepare Big-Math-RL-Verified dataset
#
# Usage:
#   sbatch scripts/submit_prepare_bigmath.slurm
#   sbatch scripts/submit_prepare_bigmath.slurm --min-solve-rate 0.20 --max-solve-rate 0.60

set -e

echo "=========================================="
echo "Prepare Big-Math-RL-Verified"
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

export HF_HOME=/cluster/projects/nn12068k/alexaau/llm-training-experiments/.cache/huggingface

cd $PROJECT_DIR

python scripts/prepare_bigmath_self_teach.py "$@"

echo "Done!"
