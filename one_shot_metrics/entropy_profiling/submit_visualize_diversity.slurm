#!/bin/bash
#SBATCH --job-name=viz-diversity
#SBATCH --account=nn12068k
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --partition=accel
#SBATCH --output=logs/viz-diversity-%j.out
#SBATCH --error=logs/viz-diversity-%j.err

module load NRIS/GPU
module load Python/3.12.3-GCCcore-13.3.0

PROJECT_DIR=/cluster/projects/nn12068k/haaklau/llm-training-experiments
source $PROJECT_DIR/venv/bin/activate
cd $PROJECT_DIR/one_shot_metrics/entropy_profiling

echo "=== Wang 14 (visualize-only) ==="
python run_entropy_profiling.py --visualize-only --output-dir results

echo ""
echo "=== Li 9 (visualize-only) ==="
python run_entropy_profiling.py --li --visualize-only --output-dir results
