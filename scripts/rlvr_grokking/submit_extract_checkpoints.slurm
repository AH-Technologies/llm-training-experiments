#!/bin/bash
#SBATCH --job-name=extract-ckpt
#SBATCH --account=nn12068k
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=16G
#SBATCH --partition=accel
#SBATCH --output=logs/extract-ckpt-%j.out
#SBATCH --error=logs/extract-ckpt-%j.err

# Checkpoint extraction: FSDP → HF → upload
# Needs ARM node (same arch as venv), but no GPUs needed.

set -e

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Architecture: $(uname -m)"

PROJECT_DIR=/cluster/projects/nn12068k/alexaau/llm-training-experiments

# Load modules (same as training)
module load NRIS/GPU
module load Python/3.12.3-GCCcore-13.3.0

# Activate venv
source $PROJECT_DIR/venv/bin/activate

# Run extraction
bash $PROJECT_DIR/scripts/rlvr_grokking/extract_checkpoints.sh
