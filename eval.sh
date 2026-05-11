#!/bin/bash
set -e

# 1. Navigate to your mounted workspace
# Assuming your project is cloned at /work/evaluate-models
cd /work/evaluate-models

mkdir -p logs
LOG_FILE="logs/eval_live_$(date +%s).log"

# 2. Parse Arguments and Standardize Paths
MODEL_PATH=""
TITLE=""
EVAL_ARGS=()

args=("$@")
for ((i=0; i<${#args[@]}; i++)); do
    if [[ "${args[$i]}" == "--model_path" ]]; then
        # Convert the relative path to an absolute path dynamically
        MODEL_PATH="$(realpath "${args[$i+1]}")"
        EVAL_ARGS+=("--model_path" "$MODEL_PATH")
        ((i++))
    elif [[ "${args[$i]}" == "--title" ]]; then
        TITLE="${args[$i+1]}"
        ((i++))
    else
        EVAL_ARGS+=("${args[$i]}")
    fi
done

if [ -z "$TITLE" ] || [ -z "$MODEL_PATH" ]; then
    echo "ERROR: --title and --model_path arguments are mandatory." | tee -a $LOG_FILE
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

# 4. Install pre-compiled FlashAttention-2 (CUDA 13.0 / PyTorch 2.10)
echo "Installing Flash Attention 2 for B200..." | tee -a $LOG_FILE
uv pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.0/flash_attn-2.8.3+cu130torch2.10-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl

# 5. vLLM Distributed Setup
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=0

# 6. Execution
echo "Launching vLLM evaluation engine..." | tee -a $LOG_FILE
uv run python -m src.eval "${EVAL_ARGS[@]}" 2>&1 | tee -a $LOG_FILE

# 7. Visualization
EVAL_FILE_PATH="$MODEL_PATH/evaluation_results.jsonl"
echo "Generating evaluation graphs..." | tee -a $LOG_FILE
uv run python -m src.visualize_eval --eval_file_path "$EVAL_FILE_PATH" --title "$TITLE" 2>&1 | tee -a $LOG_FILE

echo "Evaluation Job finished at $(date)" | tee -a $LOG_FILE