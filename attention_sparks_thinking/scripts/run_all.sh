#!/bin/bash
# Submit all 3 attention-rhythm GRPO training runs.
#
# Usage:
#   bash attention_sparks_thinking/scripts/run_all.sh
#   DRY_RUN=1 bash attention_sparks_thinking/scripts/run_all.sh

set -e

for RUN_TYPE in A B C; do
    echo "Submitting Run ${RUN_TYPE}..."
    sbatch --export=ALL,RUN_TYPE=${RUN_TYPE} attention_sparks_thinking/slurm/train.slurm
    echo "  Submitted."
done

echo "All 3 runs submitted."
