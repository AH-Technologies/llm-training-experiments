#!/bin/bash
# Submit all 4 single-example 100-step experiments
# Best predicted: ex_907 (rank 3), ex_331 (rank 4)
# Worst predicted: ex_1108 (rank 658), ex_672 (rank 659)

set -e

for EXAMPLE in ex_907 ex_331 ex_1108 ex_672; do
    echo "Submitting ${EXAMPLE}..."
    sbatch --time=06:00:00 \
           --job-name="rlvr-${EXAMPLE}" \
           scripts/submit_training.slurm ${EXAMPLE}
done

echo "All 4 jobs submitted. Check with: squeue -u \$USER"
