#!/bin/bash
# Submit all 5 DAPO-Math-17k GRPO conditions as parallel Slurm jobs.
#
# Usage:
#   bash scripts/attention_based_rewards/run_all_dapo.sh
#
# Conditions:
#   1. uniform       - Standard GRPO baseline
#   2. entropy       - GTPO-style entropy weighting
#   3. fai_allheads  - FAI across all 336 heads (Attention Illuminates replication)
#   4. fai           - FAI on reasoning heads (our method)
#   5. fai_asymmetric - FAI reasoning heads + inverted on incorrect

set -e

echo "Submitting 5 DAPO-Math-17k GRPO conditions..."
echo ""

for COND in uniform entropy fai_allheads fai fai_asymmetric; do
    echo "Submitting condition: ${COND}"
    sbatch --export=ALL,CONDITION=${COND} scripts/attention_based_rewards/slurm/train_dapo.slurm
done

echo ""
echo "All 5 conditions submitted. Monitor with: squeue -u \$USER"
