#!/bin/bash
#SBATCH --job-name=best-of-n
#SBATCH --account=nn12068k
#SBATCH --partition=accel
#SBATCH --gpus=2
#SBATCH --mem=92G
#SBATCH --time=02:00:00
#SBATCH --output=logs/best_of_n_%j.out
#SBATCH --error=logs/best_of_n_%j.err

MODEL=${MODEL:-"Qwen/Qwen2.5-Math-1.5B"}
DATASET=${DATASET:-"data/pi1_r128.parquet"}
NUM_CORRECT=${NUM_CORRECT:-1024}
NUM_KEEP=${NUM_KEEP:-128}
TEMPERATURE=${TEMPERATURE:-1.0}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2048}
OUTPUT_DIR=${OUTPUT_DIR:-"sft_data_generation/results/generate_best_of_n"}

module load NRIS/GPU
module load Python/3.12.3-GCCcore-13.3.0

source /cluster/projects/nn12068k/haaklau/llm-training-experiments/venv/bin/activate

export HF_HOME=/cluster/projects/nn12068k/haaklau/.cache/huggingface
export PYTHONPATH="${PYTHONPATH}:/cluster/projects/nn12068k/haaklau/llm-training-experiments/src"
export PYTHONUNBUFFERED=1

mkdir -p logs "$OUTPUT_DIR"

echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Num correct: $NUM_CORRECT, Keep top: $NUM_KEEP"
echo "Temperature: $TEMPERATURE"

python src/sft_data_generation/generate_correct_best_of_n.py \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --num_correct "$NUM_CORRECT" \
    --num_keep "$NUM_KEEP" \
    --temperature "$TEMPERATURE" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --output_dir "$OUTPUT_DIR" \
    --problem_idx 0 \
    --tp 2
