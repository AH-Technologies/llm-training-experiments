#!/bin/bash
#SBATCH --job-name=eval-s1k
#SBATCH --account=nn12068k
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --partition=accel
#SBATCH --output=logs/eval-s1k-%j.out
#SBATCH --error=logs/eval-s1k-%j.err

# Evaluate model on s1K verifiable dataset
#
# Usage:
#   sbatch scripts/submit_eval_s1k.slurm              # defaults: Qwen3-1.7B, k=1
#   sbatch scripts/submit_eval_s1k.slurm --k 4
#   sbatch scripts/submit_eval_s1k.slurm --model path/to/checkpoint --k 8
#   sbatch scripts/submit_eval_s1k.slurm --enable-thinking

set -e

echo "=========================================="
echo "s1K Evaluation"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Args: $@"
echo "=========================================="

mkdir -p logs

# Load modules
module load NRIS/GPU
module load Python/3.12.3-GCCcore-13.3.0

PROJECT_DIR=/cluster/projects/nn12068k/alexaau/llm-training-experiments

# Activate venv
if [ -d "$PROJECT_DIR/venv" ]; then
    source $PROJECT_DIR/venv/bin/activate
fi

# Load .env
if [ -f "$PROJECT_DIR/.env" ]; then
    set -a
    source "$PROJECT_DIR/.env"
    set +a
fi

export HF_HOME=/cluster/projects/nn12068k/alexaau/llm-training-experiments/.cache/huggingface
export VLLM_USE_V1=1
export TOKENIZERS_PARALLELISM=false

# Tiktoken encodings
if [ -d "$PROJECT_DIR/tiktoken_encodings" ]; then
    export TIKTOKEN_ENCODINGS_BASE=$PROJECT_DIR/tiktoken_encodings
fi

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

python scripts/eval_s1k.py "$@"

echo "Done!"
