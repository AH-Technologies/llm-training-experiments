#!/bin/bash
# Launch all 4 GRPO conditions as separate SLURM jobs.
#
# Usage:
#   bash scripts/attention_based_rewards/run_all_conditions.sh
#   DRY_RUN=1 bash scripts/attention_based_rewards/run_all_conditions.sh  # 2-step test
set -e

cd /cluster/projects/nn12068k/haaklau/llm-training-experiments

echo "=== Launching Circuit-Guided GRPO Training ==="
echo "Conditions: uniform, attention, entropy, combined"
echo ""

# First, prepare the data if not present
if [ ! -f "attention_based_rewards/data/gsm8k_train.parquet" ]; then
    echo "Preparing GSM8K data..."
    srun --account=nn12068k --partition=accel --gpus=1 --cpus-per-task=8 --mem=32G --time=00:10:00 \
        bash -c "module load NRIS/GPU && module load Python/3.12.3-GCCcore-13.3.0 && \
                 source venv/bin/activate && \
                 python scripts/attention_based_rewards/prepare_gsm8k_verl.py"
fi

# Submit all 4 conditions
for COND in uniform attention entropy combined; do
    echo "Submitting: ${COND}"
    JOB_ID=$(sbatch \
        --parsable \
        --export=ALL,CONDITION=${COND} \
        scripts/attention_based_rewards/slurm/train_condition.slurm)
    echo "  Job ID: ${JOB_ID}"
done

echo ""
echo "All jobs submitted. Monitor with: squeue -u \$USER"
echo "Logs: attention_based_rewards/logs/train_<condition>_<jobid>.log"
