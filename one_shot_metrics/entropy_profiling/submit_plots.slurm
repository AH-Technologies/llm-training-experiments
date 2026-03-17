#!/bin/bash
#SBATCH --job-name=plot-corr
#SBATCH --account=nn12068k
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=accel
#SBATCH --output=logs/plot-corr-%j.out
#SBATCH --error=logs/plot-corr-%j.err

module load NRIS/GPU
module load Python/3.12.3-GCCcore-13.3.0

PROJECT_DIR=/cluster/projects/nn12068k/haaklau/llm-training-experiments
source $PROJECT_DIR/venv/bin/activate

python $PROJECT_DIR/one_shot_metrics/entropy_profiling/plot_correlations.py
