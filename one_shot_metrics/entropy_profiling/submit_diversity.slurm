#!/bin/bash
#SBATCH --job-name=diversity
#SBATCH --account=nn12068k
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=accel
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/diversity-%j.out
#SBATCH --error=logs/diversity-%j.err

module load NRIS/GPU
module load Python/3.12.3-GCCcore-13.3.0

PROJECT_DIR=/cluster/projects/nn12068k/haaklau/llm-training-experiments
source $PROJECT_DIR/venv/bin/activate
cd $PROJECT_DIR/one_shot_metrics/entropy_profiling

echo "=== Wang 14 ==="
python compute_diversity_standalone.py --results-dir results/entropy_profiles

echo ""
echo "=== Large run (Wang 400) ==="
python compute_diversity_standalone.py --results-dir results/large_run/entropy_profiles

echo ""
echo "=== Li run (9) ==="
python compute_diversity_standalone.py --results-dir results/li_run/entropy_profiles
