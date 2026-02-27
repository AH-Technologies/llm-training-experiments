#!/bin/bash
#SBATCH --job-name=rlvr-grokking
#SBATCH --account=nn12068k
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=400G
#SBATCH --partition=accel
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

# RLVR GRPO Training on Olivia HPC (GH200)
#
# Usage:
#   sbatch scripts/submit_training_olivia.slurm pi13_math500
#   sbatch scripts/submit_training_olivia.slurm pi13_math500_v2
#   sbatch scripts/submit_training_olivia.slurm ber
#   sbatch scripts/submit_training_olivia.slurm ber_amplify
#   sbatch scripts/submit_training_olivia.slurm ber MULTI_PROMPT_VAL=1
#   sbatch scripts/submit_training_olivia.slurm astar_sft
#   sbatch --gpus=8 scripts/submit_training_olivia.slurm pi13_math500

set -e

# Task type passed as argument
TASK=${1:-ber}

echo "=========================================="
echo "RLVR Grokking Training with verl"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Architecture: $(uname -m)"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "Task: $TASK"
echo "=========================================="

# Create directories
mkdir -p logs checkpoints data

# Load modules
module load NRIS/GPU
module load Python/3.12.3-GCCcore-13.3.0

# Unset AMD ROCm variable set by NRIS/GPU — verl crashes if both
# ROCR_VISIBLE_DEVICES and CUDA_VISIBLE_DEVICES are set
unset ROCR_VISIBLE_DEVICES

# Activate virtual environment
PROJECT_DIR=/cluster/projects/nn12068k/haaklau/llm-training-experiments
source $PROJECT_DIR/venv/bin/activate

# Load .env file (contains WANDB_API_KEY, etc.)
if [ -f "$PROJECT_DIR/.env" ]; then
    set -a
    source "$PROJECT_DIR/.env"
    set +a
fi

# Environment variables
# HuggingFace cache
export HF_HOME=/cluster/projects/nn12068k/haaklau/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$PROJECT_DIR/.cache/datasets_$(whoami)
mkdir -p $HF_DATASETS_CACHE
export HF_TOKEN=${HF_TOKEN:-""}
export TOKENIZERS_PARALLELISM=false

# NCCL settings for multi-GPU
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0

# vLLM settings
# Enable V1 engine (required by verl 0.7.0 with vLLM 0.12.0)
export VLLM_USE_V1=1

# Tiktoken encoding files for offline use (openai_harmony/vLLM needs these)
export TIKTOKEN_CACHE_DIR=$PROJECT_DIR/tiktoken_encodings
export TIKTOKEN_ENCODINGS_BASE=$PROJECT_DIR/tiktoken_encodings
export TIKTOKEN_RS_CACHE_DIR=$PROJECT_DIR/tiktoken_encodings

# Verify environment
echo ""
echo "Environment check:"
python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA: {torch.cuda.is_available()}')
print(f'  GPUs: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'    GPU {i}: {torch.cuda.get_device_name(i)}')
"
echo ""

# Step 1: Prepare data if not already done
echo "Checking/preparing data..."
cd $PROJECT_DIR

if [ "$TASK" == "pi13_math500" ]; then
    # Pi13 -> Math500: train on Pi13, evaluate on Math500
    # Always regenerate pi13 data to ensure it has correct format (128 duplicates)
    echo "Regenerating π13 dataset with 128 duplicates..."
    python scripts/prepare_pi13_data.py --output-dir ./data --copies 128

    if [ ! -f "data/math500.parquet" ]; then
        echo "ERROR: Math500 data file must be present in ./data/"
        echo "Expected: data/math500.parquet"
        exit 1
    fi

    TRAIN_SCRIPT="scripts/run_grpo_pi13_math500.sh"
elif [ "$TASK" == "pi13_math500_medium_wd" ]; then
    # Pi13 -> Math500 with medium weight decay
    echo "Regenerating π13 dataset with 128 duplicates..."
    python scripts/prepare_pi13_data.py --output-dir ./data --copies 128
    if [ ! -f "data/math500.parquet" ]; then
        echo "ERROR: Math500 data file must be present in ./data/"
        exit 1
    fi
    TRAIN_SCRIPT="scripts/run_grpo_pi13_math500_medium_wd.sh"
elif [ "$TASK" == "pi13_math500_high_wd" ]; then
    # Pi13 -> Math500 with high weight decay
    echo "Regenerating π13 dataset with 128 duplicates..."
    python scripts/prepare_pi13_data.py --output-dir ./data --copies 128
    if [ ! -f "data/math500.parquet" ]; then
        echo "ERROR: Math500 data file must be present in ./data/"
        exit 1
    fi
    TRAIN_SCRIPT="scripts/run_grpo_pi13_math500_high_wd.sh"
elif [ "$TASK" == "pi13_math500_pos_ent" ]; then
    # Pi13 -> Math500 with positive entropy coefficient
    echo "Regenerating π13 dataset with 128 duplicates..."
    python scripts/prepare_pi13_data.py --output-dir ./data --copies 128
    if [ ! -f "data/math500.parquet" ]; then
        echo "ERROR: Math500 data file must be present in ./data/"
        exit 1
    fi
    TRAIN_SCRIPT="scripts/run_grpo_pi13_math500_pos_ent.sh"
elif [ "$TASK" == "pi13_math500_high_ent" ]; then
    # Pi13 -> Math500 with high (negative) entropy coefficient
    echo "Regenerating π13 dataset with 128 duplicates..."
    python scripts/prepare_pi13_data.py --output-dir ./data --copies 128
    if [ ! -f "data/math500.parquet" ]; then
        echo "ERROR: Math500 data file must be present in ./data/"
        exit 1
    fi
    TRAIN_SCRIPT="scripts/run_grpo_pi13_math500_high_ent.sh"
elif [ "$TASK" == "pi13_math500_v2" ]; then
    # Pi13 -> Math500: paper-matching config (verl v0.2 params)
    # 1.5B model, dynamic batching, entropy_coeff=0.001
    echo "Regenerating π13 dataset with 128 duplicates..."
    python scripts/prepare_pi13_data.py --output-dir ./data --copies 128
    if [ ! -f "data/math500.parquet" ]; then
        echo "ERROR: Math500 data file must be present in ./data/"
        exit 1
    fi

    TRAIN_SCRIPT="scripts/run_grpo_pi13_math500_v2.sh"
elif [ "$TASK" == "pi1_math500_v2" ]; then
    # Pi1 -> Math500: paper-matching config (verl v0.2 params)
    # 1.5B model, dynamic batching, entropy_coeff=0.001
    echo "Regenerating π1 dataset with 128 duplicates..."
    python scripts/prepare_pi1_data.py --output-dir ./data --copies 128
    if [ ! -f "data/math500.parquet" ]; then
        echo "ERROR: Math500 data file must be present in ./data/"
        exit 1
    fi

    TRAIN_SCRIPT="scripts/run_grpo_pi1_math500_v2.sh"
elif [ "$TASK" == "ber" ]; then
    # Pi13 -> Math500 with Bidirectional Experience Replay (BER)
    echo "Regenerating π13 dataset with 128 duplicates..."
    python scripts/prepare_pi13_data.py --output-dir ./data --copies 128
    if [ ! -f "data/math500.parquet" ]; then
        echo "ERROR: Math500 data file must be present in ./data/"
        exit 1
    fi

    # Generate correct cache if it doesn't exist
    BER_CACHE="data/ber_correct_cache_pi13.pt"
    if [ ! -f "$BER_CACHE" ]; then
        echo "Generating BER correct cache via rejection sampling..."
        python scripts/generate_correct_cache.py \
            --model Qwen/Qwen2.5-Math-1.5B \
            --train_file data/pi13_r128.parquet \
            --output "$BER_CACHE" \
            --max_attempts 64 \
            --temperature 0.6
    else
        echo "BER correct cache already exists: $BER_CACHE"
    fi

    TRAIN_SCRIPT="scripts/run_grpo_ber.sh"
elif [ "$TASK" == "ber_amplify" ]; then
    # Pi13 -> Math500 with BER "amplify" mode (within-step error amplification)
    echo "Regenerating π13 dataset with 128 duplicates..."
    python scripts/prepare_pi13_data.py --output-dir ./data --copies 128
    if [ ! -f "data/math500.parquet" ]; then
        echo "ERROR: Math500 data file must be present in ./data/"
        exit 1
    fi

    # Generate correct cache if it doesn't exist
    BER_CACHE="data/ber_correct_cache_pi13.pt"
    if [ ! -f "$BER_CACHE" ]; then
        echo "Generating BER correct cache via rejection sampling..."
        python scripts/generate_correct_cache.py \
            --model Qwen/Qwen2.5-Math-1.5B \
            --train_file data/pi13_r128.parquet \
            --output "$BER_CACHE" \
            --max_attempts 64 \
            --temperature 0.6
    else
        echo "BER correct cache already exists: $BER_CACHE"
    fi

    TRAIN_SCRIPT="scripts/run_grpo_ber_amplify.sh"
elif [ "$TASK" == "spt" ]; then
    # Pi13 -> Math500 with Self-Play Teaching (SPT)
    echo "Regenerating π13 dataset with 128 duplicates..."
    python scripts/prepare_pi13_data.py --output-dir ./data --copies 128
    if [ ! -f "data/math500.parquet" ]; then
        echo "ERROR: Math500 data file must be present in ./data/"
        exit 1
    fi

    TRAIN_SCRIPT="scripts/run_grpo_spt.sh"
elif [ "$TASK" == "astar_sft" ]; then
    # A* search grokking: SFT on search traces
    ASTAR_DATA_DIR="data/astar_grokking_dataset"
    # Always regenerate to pick up config changes (7x7 only, tokenizer filtering)
    echo "Generating A* grokking dataset..."
    rm -rf ${ASTAR_DATA_DIR}
    PYTHONPATH=src:$PYTHONPATH python -m astar_dataset.generate_dataset \
        --output-dir ${ASTAR_DATA_DIR} --seed 42

    TRAIN_SCRIPT="scripts/run_sft_astar.sh"
else
    echo "Unknown task: $TASK"
    echo "Available tasks: pi13_math500, pi13_math500_medium_wd, pi13_math500_high_wd, pi13_math500_pos_ent, pi13_math500_high_ent, pi13_math500_v2, pi1_math500_v2, ber, ber_amplify, spt, astar_sft"
    exit 1
fi

# Step 2: Run training
echo "Starting training for $TASK..."
echo "Using script: $TRAIN_SCRIPT"

chmod +x $TRAIN_SCRIPT
bash $TRAIN_SCRIPT "${@:2}"

echo "Training complete!"
