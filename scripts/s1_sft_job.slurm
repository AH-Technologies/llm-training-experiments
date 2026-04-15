#!/bin/bash
# Shared SLURM job body for S1 SFT training (both standard and grokking).
# SLURM resource parameters are passed via CLI flags from the submit wrapper scripts.
#
# This script is NOT meant to be submitted directly — use:
#   bash scripts/submit_s1_sft.sh
#   bash scripts/submit_s1_sft_grokking.sh

set -e

PROJECT_DIR=${PROJECT_DIR:-"/cluster/home/alexaau/llm-training-experiments"}
source "$PROJECT_DIR/scripts/cluster_config.sh"

echo "=========================================="
echo "S1 SFT: ${JOB_DESCRIPTION:-Qwen2.5-32B on s1K}"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Cluster: $CLUSTER_NAME ($(uname -m))"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "GPUs per node: $SLURM_GPUS_ON_NODE"
echo "Start time: $(date)"
echo "=========================================="

cd "$PROJECT_DIR"
mkdir -p logs checkpoints data

# --- Load modules and configure networking ---
load_cluster_modules
configure_nccl

# --- Activate venv ---
if [ -d "$PROJECT_DIR/venv" ]; then
    source "$PROJECT_DIR/venv/bin/activate"
fi

# --- Load .env (WANDB_API_KEY, HF_TOKEN, etc.) ---
if [ -f "$PROJECT_DIR/.env" ]; then
    set -a
    source "$PROJECT_DIR/.env"
    set +a
fi

# --- HuggingFace cache ---
MODEL_CACHE_KEY="Qwen_Qwen2.5-32B-Instruct"
export HF_HOME=${HF_CACHE_BASE}/huggingface_${MODEL_CACHE_KEY}
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME/datasets
mkdir -p "$HF_DATASETS_CACHE"

# --- Other env ---
export TIKTOKEN_CACHE_DIR=$PROJECT_DIR/tiktoken_encodings
export TIKTOKEN_ENCODINGS_BASE=$PROJECT_DIR/tiktoken_encodings
export TIKTOKEN_RS_CACHE_DIR=$PROJECT_DIR/tiktoken_encodings
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
unset ROCR_VISIBLE_DEVICES

# --- Multi-node setup ---
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "HF cache: $HF_HOME"
nvidia-smi --query-gpu=name,memory.total --format=csv

# --- Background GPU/memory monitor ---
MONITOR_INTERVAL=${MONITOR_INTERVAL:-120}
(while true; do
    echo "--- GPU monitor $(date) ---"
    nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv,noheader
    echo "CPU mem: $(free -g | awk '/^Mem:/{print $3"/"$2" GB"}')"
    sleep "$MONITOR_INTERVAL"
done) &
MONITOR_PID=$!

# --- Prepare data if needed ---
if [ ! -f "data/s1K/s1k_sft.parquet" ]; then
    echo "Preparing s1K SFT data..."
    python scripts/prepare_s1k_sft.py
fi

# --- Run training ---
echo "Starting S1 SFT training..."
bash scripts/run_sft_s1k.sh "$@"

# --- Cleanup ---
kill $MONITOR_PID 2>/dev/null || true
wait

echo "End: $(date)"
echo "=== Done ==="
