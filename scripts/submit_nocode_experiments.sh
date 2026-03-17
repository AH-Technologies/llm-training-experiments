#!/bin/bash
# Submit no-code RLVR experiments for pi1 and pi13
# 500 steps, code penalty reward, multi-prompt validation
set -e

for EXAMPLE in pi1_nocode pi13_nocode; do
    echo "Submitting ${EXAMPLE}..."
    sbatch --time=24:00:00 \
           --job-name="rlvr-${EXAMPLE}" \
           scripts/submit_training.slurm ${EXAMPLE}
done

echo "Both jobs submitted. Check with: squeue -u \$USER"
