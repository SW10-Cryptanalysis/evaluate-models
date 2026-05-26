#!/bin/bash
set -e

# ==============================================================================
# 1. EVALUATION CONFIGURATION (Edit these for each model)
# ==============================================================================
# The path to your model directory containing rwkv7_cipher_final.pth
TARGET_MODEL_DIR="../Models/RWKV/RWKV-mono-10k-nosp/"

# The title for the visualization graph
EVAL_TITLE="RWKV-7 10k Monoalphabetic No Spaces"

# Set to true if this model was trained with the space character inclusion
USE_SPACES=false 
# ==============================================================================

# Navigate to your mounted workspace
cd /work

# 2. Clone the repository and specific branch if it doesn't exist yet
if [ ! -d "evaluate-models" ]; then
    echo "Cloning repository..."
    git clone -b eval_rwkv https://github.com/SW10-Cryptanalysis/evaluate-models.git
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

# 3. Environment Setup (uv & venv)
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

echo "Creating and activating virtual environment..." | tee -a $LOG_FILE
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -e .

# 4. Hardware / Environment Setup
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0  # Target the first L4 GPU for native PyTorch evaluation

# Build the argument array based on our updated parameter names
# Main test eval script expects '--model_dir', Z408 expects '--model_path'
COMMON_ARGS=()
if [ "$USE_SPACES" = true ]; then
    COMMON_ARGS+=("--spaces")
fi

# 5. Run Core Dataset Evaluation
echo "Launching native PyTorch evaluation engine..." | tee -a $LOG_FILE
uv run python -m src.eval --model_dir "$MODEL_PATH" "${COMMON_ARGS[@]}" 2>&1 | tee -a $LOG_FILE

# 5.1 Evaluate Z408 Cipher
Z408_FILE_PATH="../Ciphers/z408.json"
echo "Evaluating Z408 cipher natively..." | tee -a $LOG_FILE
uv run python -m src.eval_z408 --model_path "$MODEL_PATH" --z408_path "$Z408_FILE_PATH" "${COMMON_ARGS[@]}" 2>&1 | tee -a $LOG_FILE

# 6. Visualization
EVAL_FILE_PATH="$MODEL_PATH/evaluation_results.jsonl"
echo "Generating evaluation graphs..." | tee -a $LOG_FILE
uv run python -m src.visualize_eval --eval_file_path "$EVAL_FILE_PATH" --title "$EVAL_TITLE" 2>&1 | tee -a $LOG_FILE

echo "Evaluation Job finished at $(date)" | tee -a $LOG_FILE