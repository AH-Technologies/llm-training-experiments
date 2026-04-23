#!/bin/bash
# Extract RLVR grokking checkpoints: merge FSDP shards → HF format → upload to HF Hub
# Run this as late as possible before losing HPC access.
#
# Usage: sbatch scripts/rlvr_grokking/submit_extract_checkpoints.slurm
#        or directly on a GPU node: bash scripts/rlvr_grokking/extract_checkpoints.sh

set -euo pipefail

PROJECT_DIR=/cluster/projects/nn12068k/alexaau/llm-training-experiments
CKPT_BASE="${PROJECT_DIR}/checkpoints/rlvr-grokking"
MERGED_BASE="${PROJECT_DIR}/checkpoints/rlvr-grokking-merged"

# Load env (HF_TOKEN)
if [ -f "$PROJECT_DIR/.env" ]; then
    set -a
    source "$PROJECT_DIR/.env"
    set +a
fi

# HF org/user prefix for uploaded repos
HF_USER="alexaau"

mkdir -p "${MERGED_BASE}"

echo "============================================"
echo "RLVR Grokking Checkpoint Extraction"
echo "Started: $(date)"
echo "============================================"

# Process each run
for run_dir in "${CKPT_BASE}"/grpo_*; do
    run_name=$(basename "${run_dir}")
    hf_repo="${HF_USER}/${run_name}"

    echo ""
    echo ">>> Processing run: ${run_name}"
    echo ">>> HF repo: ${hf_repo}"

    # Process each step
    for step_dir in "${run_dir}"/global_step_*; do
        step_name=$(basename "${step_dir}")
        actor_dir="${step_dir}/actor"
        merged_dir="${MERGED_BASE}/${run_name}/${step_name}"

        # Skip if no actor dir (incomplete checkpoint)
        if [ ! -f "${actor_dir}/fsdp_config.json" ]; then
            echo "  [SKIP] ${step_name} - no fsdp_config.json (incomplete?)"
            continue
        fi

        # Skip if already uploaded (marker file)
        if [ -f "${merged_dir}/.uploaded" ]; then
            echo "  [DONE] ${step_name} - already uploaded"
            continue
        fi

        echo "  [MERGE] ${step_name}..."
        mkdir -p "${merged_dir}"

        # Merge FSDP shards into HF format
        python -m verl.model_merger merge \
            --backend fsdp \
            --local_dir "${actor_dir}" \
            --target_dir "${merged_dir}"

        echo "  [UPLOAD] ${step_name} → ${hf_repo} (path_in_repo: ${step_name})"

        # Upload to HF Hub as a subfolder of the run repo
        python -c "
from huggingface_hub import HfApi
api = HfApi()
api.create_repo('${hf_repo}', private=True, exist_ok=True)
api.upload_folder(
    folder_path='${merged_dir}',
    repo_id='${hf_repo}',
    path_in_repo='${step_name}',
    repo_type='model',
)
print('  Upload complete: ${step_name}')
"

        # Mark as uploaded and clean up merged files to save disk
        touch "${merged_dir}/.uploaded"
        find "${merged_dir}" -name "*.safetensors" -delete
        find "${merged_dir}" -name "*.bin" -delete
        echo "  [CLEAN] Removed merged weight files for ${step_name}"

    done

    echo ">>> Done with ${run_name}"
done

echo ""
echo "============================================"
echo "All checkpoints extracted and uploaded!"
echo "Finished: $(date)"
echo "============================================"
