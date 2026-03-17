#!/bin/bash
# Download Qwen2.5-Math-1.5B model to HF cache (run on login node with internet)
#
# Usage: bash one_shot_metrics/entropy_profiling/download_model.sh

set -e

# Load modules for Python access
module load NRIS/GPU
module load Python/3.12.3-GCCcore-13.3.0

PROJECT_DIR=/cluster/projects/nn12068k/haaklau/llm-training-experiments

if [ -d "$PROJECT_DIR/venv" ]; then
    source $PROJECT_DIR/venv/bin/activate
elif [ -d "$HOME/venv" ]; then
    source $HOME/venv/bin/activate
fi

export HF_HOME=/cluster/projects/nn12068k/haaklau/.cache/huggingface

echo "Downloading Qwen/Qwen2.5-Math-1.5B to $HF_HOME..."

python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = 'Qwen/Qwen2.5-Math-1.5B'
print(f'Downloading tokenizer for {model_name}...')
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
print(f'Downloading model {model_name}...')
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
print('Done! Model cached.')
"
