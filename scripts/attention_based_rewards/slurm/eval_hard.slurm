#!/bin/bash
#SBATCH --job-name=eval-hard
#SBATCH --account=nn12068k
#SBATCH --partition=accel
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --output=attention_based_rewards/logs/eval_hard_%j.log
#SBATCH --error=attention_based_rewards/logs/eval_hard_%j.err

# Evaluate all 4 GRPO conditions on AIME 2025 + AMC 2023 in parallel.
# Each condition runs on its own GPU.
#
# Usage:
#   sbatch scripts/attention_based_rewards/slurm/eval_hard.slurm
#   sbatch --export=ALL,EVAL_STEP=200 scripts/attention_based_rewards/slurm/eval_hard.slurm

set -e

echo "=========================================="
echo "Hard Benchmark Evaluation (4 GPUs parallel)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Eval step: ${EVAL_STEP:-latest available}"
echo "=========================================="

mkdir -p attention_based_rewards/logs results/attention_based_rewards/eval_hard

# Load modules
module load NRIS/GPU
module load Python/3.12.3-GCCcore-13.3.0

PROJECT_DIR=/cluster/projects/nn12068k/haaklau/llm-training-experiments
cd $PROJECT_DIR
source $PROJECT_DIR/venv/bin/activate

# Load .env
if [ -f "$PROJECT_DIR/.env" ]; then
    set -a
    source "$PROJECT_DIR/.env"
    set +a
fi

export HF_HOME=/cluster/projects/nn12068k/haaklau/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH=$PROJECT_DIR/src:$PROJECT_DIR:$PYTHONPATH

# CUDA
export CUDA_HOME=/cluster/software/NRIS.old/neoverse_v2/software/NVHPC/25.3-CUDA-12.8.0/Linux_aarch64/25.3/cuda/12.8
MATH_LIBS=/cluster/software/NRIS.old/neoverse_v2/software/NVHPC/25.3-CUDA-12.8.0/Linux_aarch64/25.3/math_libs/12.8/lib64
export LD_LIBRARY_PATH=${MATH_LIBS}:/opt/cray/libfabric/1.22.0/lib64:/usr/lib64:${LD_LIBRARY_PATH}
unset ROCR_VISIBLE_DEVICES

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

# Step 1: Prepare eval data if needed (run once, sequentially)
if [ ! -f "attention_based_rewards/data/aime_2025.parquet" ] || [ ! -f "attention_based_rewards/data/amc_2023.parquet" ]; then
    echo "Preparing AIME 2025 + AMC 2023 data..."
    python scripts/attention_based_rewards/prepare_eval_data.py
fi

# Step 2: Build step argument
STEP_ARG=""
if [ -n "$EVAL_STEP" ]; then
    STEP_ARG="--step $EVAL_STEP"
fi

# Step 3: Launch all 4 conditions in parallel, one per GPU
CONDITIONS=(uniform attention entropy combined)
PIDS=()

echo "Launching 4 conditions in parallel..."
for i in "${!CONDITIONS[@]}"; do
    COND=${CONDITIONS[$i]}
    echo "  GPU $i -> ${COND}"
    CUDA_VISIBLE_DEVICES=$i python scripts/attention_based_rewards/eval_checkpoints.py \
        --condition $COND \
        $STEP_ARG \
        --datasets aime_2025 amc_2023 \
        > attention_based_rewards/logs/eval_hard_${COND}_${SLURM_JOB_ID}.log 2>&1 &
    PIDS+=($!)
done

echo ""
echo "Waiting for all conditions to finish..."
echo "PIDs: ${PIDS[*]}"

# Wait for all and track failures
FAILED=0
for i in "${!CONDITIONS[@]}"; do
    COND=${CONDITIONS[$i]}
    if wait ${PIDS[$i]}; then
        echo "  [OK] ${COND} finished successfully"
    else
        echo "  [FAIL] ${COND} failed (exit code: $?)"
        FAILED=$((FAILED + 1))
    fi
done

# Step 4: Print summary
echo ""
echo "=========================================="
echo "RESULTS SUMMARY"
echo "=========================================="
python -c "
import json
from pathlib import Path

results_dir = Path('results/attention_based_rewards/eval_hard')
conditions = ['uniform', 'attention', 'entropy', 'combined']
datasets = ['aime_2025', 'amc_2023']

for ds in datasets:
    print(f'\n{ds.upper()}:')
    print(f'  {\"Condition\":<12} {\"Accuracy\":>10} {\"Correct\":>10} {\"Total\":>8}')
    print(f'  {\"-\"*42}')
    for cond in conditions:
        matches = list(results_dir.glob(f'{cond}_step*_{ds}.json'))
        if matches:
            with open(sorted(matches)[-1]) as f:
                r = json.load(f)
            print(f'  {cond:<12} {r[\"accuracy\"]*100:>9.1f}% {r[\"correct\"]:>10} {r[\"total\"]:>8}')
        else:
            print(f'  {cond:<12} {\"N/A\":>10}')
"

echo ""
if [ $FAILED -gt 0 ]; then
    echo "$FAILED condition(s) failed. Check logs in attention_based_rewards/logs/"
    exit 1
fi
echo "All evaluations complete!"
echo "Results: results/attention_based_rewards/eval_hard/"
