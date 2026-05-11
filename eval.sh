#!/bin/bash
set -e

# ==============================================================================
# 1. EVALUATION CONFIGURATION (Edit these for each model)
# ==============================================================================
# The path to your model (relative or absolute)
TARGET_MODEL_DIR="../Models/Mistral/Mistral-4k-homophonic-nosp/"

# The title for the visualization graph
EVAL_TITLE="Mistral 4k Homophonic No Spaces"

# Set to true if this model was trained with the space character inclusion
USE_SPACES=false 
# ==============================================================================

# Navigate to your mounted workspace
cd /work

# 2. Clone the repository and specific branch if it doesn't exist yet
if [ ! -d "evaluate-models" ]; then
    echo "Cloning repository..."
    git clone -b UCloud https://github.com/SW10-Cryptanalysis/evaluate-models.git
fi

cd evaluate-models
mkdir -p logs
LOG_FILE="logs/eval_live_$(date +%s).log"

# Safely resolve the model path to an absolute path
MODEL_PATH="$(realpath "$TARGET_MODEL_DIR")"

if [ -z "$EVAL_TITLE" ] || [ -z "$MODEL_PATH" ]; then
    echo "ERROR: EVAL_TITLE and TARGET_MODEL_DIR must be set at the top of the script." | tee -a $LOG_FILE
    exit 1
fi

NUM_GPUS=$(nvidia-smi --list-gpus | wc -l || echo 0)
echo "Eval Job started on $(hostname) at $(date) with $NUM_GPUS GPU(s)" | tee -a $LOG_FILE
echo "Resolved Model Path: $MODEL_PATH" | tee -a $LOG_FILE

# 2. Environment Setup (uv & venv)
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

echo "Creating and activating virtual environment..." | tee -a $LOG_FILE
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -e .

# 3. Install pre-compiled FlashAttention-2 (CUDA 13.0 / PyTorch 2.10)
echo "Installing Flash Attention 2 for B200..." | tee -a $LOG_FILE
uv pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.0/flash_attn-2.8.3+cu130torch2.10-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl

# 4. vLLM Distributed Setup
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=0
export CUDA_VISIBLE_DEVICES=0,1

# Build the argument array based on the configuration block
EVAL_ARGS=("--model_path" "$MODEL_PATH")
if [ "$USE_SPACES" = true ]; then
    EVAL_ARGS+=("--spaces")
fi

# 5. Execution
echo "Launching vLLM evaluation engine..." | tee -a $LOG_FILE
uv run python -m src.eval "${EVAL_ARGS[@]}" 2>&1 | tee -a $LOG_FILE

# 6. Visualization
EVAL_FILE_PATH="$MODEL_PATH/evaluation_results.jsonl"
echo "Generating evaluation graphs..." | tee -a $LOG_FILE
uv run python -m src.visualize_eval --eval_file_path "$EVAL_FILE_PATH" --title "$EVAL_TITLE" 2>&1 | tee -a $LOG_FILE

echo "Evaluation Job finished at $(date)" | tee -a $LOG_FILE