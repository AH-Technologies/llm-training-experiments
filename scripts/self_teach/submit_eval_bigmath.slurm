#!/bin/bash
#SBATCH --job-name=eval-bigmath
#SBATCH --account=nn12068k
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --partition=accel
#SBATCH --output=logs/eval-bigmath-%j.out
#SBATCH --error=logs/eval-bigmath-%j.err

# Evaluate Qwen3-8B on bigmath_filtered with pass@k
#
# Single GPU:
#   sbatch scripts/self_teach/submit_eval_bigmath.slurm
#
# Sharded across 4 jobs (submit each separately):
#   sbatch scripts/self_teach/submit_eval_bigmath.slurm --shard 0 --num-shards 4
#   sbatch scripts/self_teach/submit_eval_bigmath.slurm --shard 1 --num-shards 4
#   sbatch scripts/self_teach/submit_eval_bigmath.slurm --shard 2 --num-shards 4
#   sbatch scripts/self_teach/submit_eval_bigmath.slurm --shard 3 --num-shards 4

set -e

echo "=========================================="
echo "BigMath Evaluation"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Args: $@"
echo "=========================================="

mkdir -p logs

module load NRIS/GPU
module load Python/3.12.3-GCCcore-13.3.0

PROJECT_DIR=/cluster/projects/nn12068k/alexaau/llm-training-experiments

if [ -d "$PROJECT_DIR/venv" ]; then
    source $PROJECT_DIR/venv/bin/activate
fi

if [ -f "$PROJECT_DIR/.env" ]; then
    set -a
    source "$PROJECT_DIR/.env"
    set +a
fi

export HF_HOME=/cluster/projects/nn12068k/alexaau/.cache/huggingface
export VLLM_USE_V1=1
export TOKENIZERS_PARALLELISM=false

cd $PROJECT_DIR

echo ""
echo "Environment:"
python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA: {torch.cuda.is_available()}')
print(f'  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')
"
echo ""

python scripts/self_teach/eval_bigmath.py --model Qwen/Qwen3-8B --k 8 "$@"

echo "Done!"
