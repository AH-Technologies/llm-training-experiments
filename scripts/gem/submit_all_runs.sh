#!/bin/bash
# Submit all 10 SFT 1-shot training runs (5 datasets x 2 losses)
# Base model: Qwen/Qwen2.5-Math-1.5B (NOT instruct)
#
# Usage:
#   bash scripts/gem/submit_all_runs.sh          # submit all
#   bash scripts/gem/submit_all_runs.sh --dry-run # show commands without submitting

set -e

DRY_RUN=false
if [ "$1" = "--dry-run" ]; then
    DRY_RUN=true
    echo "=== DRY RUN — showing commands only ==="
    echo ""
fi

MODEL="Qwen/Qwen2.5-Math-1.5B"

# Dataset name -> parquet path
declare -A DATASETS=(
    ["standard_pi1"]="data/sft_1shot_datasets/standard_pi1/problem_0000.parquet"
    ["standard_pi13"]="data/sft_1shot_datasets/standard_pi13/problem_0000.parquet"
    ["bon_pi1"]="data/sft_1shot_datasets/best_of_n_pi1/problem_0000.parquet"
    ["bon_pi13"]="data/sft_1shot_datasets/best_of_n_pi13/problem_0000.parquet"
    ["gemini_pi13"]="data/sft_1shot_datasets/gemini_pi13/problem_0000.parquet"
)

LOSSES=("gem" "ce")

echo "Submitting 10 training runs:"
echo "  Model: $MODEL"
echo "  Losses: ${LOSSES[*]}"
echo "  Datasets: ${!DATASETS[*]}"
echo ""

for ds_name in "${!DATASETS[@]}"; do
    ds_path="${DATASETS[$ds_name]}"
    for loss in "${LOSSES[@]}"; do
        output_dir="models/sft_1shot_${loss}_${ds_name}"
        wandb_run="${loss}_${ds_name}"
        slurm_script="scripts/gem/submit_${loss}.slurm"

        echo "--- ${loss^^} / ${ds_name} ---"
        echo "  Dataset: $ds_path"
        echo "  Output:  $output_dir"
        echo "  W&B run: $wandb_run"

        if [ "$DRY_RUN" = true ]; then
            echo "  [dry-run] MODEL=$MODEL DATASET=$ds_path OUTPUT_DIR=$output_dir WANDB_RUN=$wandb_run sbatch $slurm_script"
        else
            job_id=$(MODEL=$MODEL \
                DATASET=$ds_path \
                OUTPUT_DIR=$output_dir \
                WANDB_RUN=$wandb_run \
                sbatch --parsable $slurm_script)
            echo "  Submitted: job $job_id"
        fi
        echo ""
    done
done

echo "Done! Monitor with: squeue -u \$USER"
