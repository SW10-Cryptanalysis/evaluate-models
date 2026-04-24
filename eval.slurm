#!/bin/bash
#SBATCH --job-name=cipher_eval
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=l4
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G

set -e

# -------------------------------------------------------------------
# 0. Mandatory GPU Check
# -------------------------------------------------------------------
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l || echo 0)

if [ "$NUM_GPUS" -eq 0 ]; then
    echo "====================================================================="
    echo " ERROR: No GPUs allocated!"
    echo " You must specify the number of GPUs when submitting the job."
    echo " Example: sbatch --gres=gpu:l4:3 eval.slurm --model_path <path> --title \"My Title\""
    echo "====================================================================="
    exit 1
fi

PROJECT_DIR=/ceph/project/SW10-CausalLM/evaluate-models
cd $PROJECT_DIR

mkdir -p logs
LOG_FILE="logs/eval_live_${SLURM_JOB_ID}.log"

echo "Cipher Eval Job started on $(hostname) at $(date)" | tee $LOG_FILE
echo "Running with arguments: $@"                        | tee -a $LOG_FILE
echo "GPUs available: $(nvidia-smi --list-gpus | wc -l)" | tee -a $LOG_FILE

export PATH="$HOME/.local/bin:$PATH"

# -------------------------------------------------------------------
# 1. GPU & Distributed Setup
# -------------------------------------------------------------------
export TOKENIZERS_PARALLELISM=false
export NCCL_IGNORE_DISABLED_P2P=1
export NCCL_P2P_DISABLE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1=0

# -------------------------------------------------------------------
# 2. HPC Storage Mitigation: Redirect Temp/Cache to Ceph
# -------------------------------------------------------------------
echo "Setting up isolated cache directories..." | tee -a $LOG_FILE
CACHE_DIR="$PROJECT_DIR/.cache"
mkdir -p "$CACHE_DIR/tmp"
mkdir -p "$CACHE_DIR/torchinductor"
mkdir -p "$CACHE_DIR/triton"
mkdir -p "$CACHE_DIR/vllm"

# Force Linux OS temporary writing to Ceph
export TMPDIR="$CACHE_DIR/tmp"
export TEMP="$CACHE_DIR/tmp"
export TMP="$CACHE_DIR/tmp"

# Force PyTorch, Triton, and vLLM compilation caching to Ceph
export TORCHINDUCTOR_CACHE_DIR="$CACHE_DIR/torchinductor"
export TRITON_CACHE_DIR="$CACHE_DIR/triton"
export VLLM_CACHE_ROOT="$CACHE_DIR/vllm"
export XDG_CACHE_HOME="$CACHE_DIR"

# -------------------------------------------------------------------
# 3. Parse Arguments
# -------------------------------------------------------------------
MODEL_PATH=""
TITLE=""
EVAL_ARGS=()

args=("$@")
for ((i=0; i<${#args[@]}; i++)); do
    if [[ "${args[$i]}" == "--model_path" ]]; then
        MODEL_PATH="${args[$i+1]}"
        EVAL_ARGS+=("${args[$i]}")
    elif [[ "${args[$i]}" == "--title" ]]; then
        TITLE="${args[$i+1]}"
        ((i++)) # Skip the title value so it isn't passed to eval.py
        continue
    else
        EVAL_ARGS+=("${args[$i]}")
    fi
done

if [ -z "$TITLE" ]; then
    echo "ERROR: --title argument is missing. Please provide a title." | tee -a $LOG_FILE
    exit 1
fi

# -------------------------------------------------------------------
# 4. Environment & Execution
# -------------------------------------------------------------------
echo "Syncing environment..." | tee -a $LOG_FILE
uv sync

echo "Installing pre-compiled FlashAttention-2..." | tee -a $LOG_FILE
uv pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

echo "Launching vLLM evaluation engine..." | tee -a $LOG_FILE
# Only passing args meant for eval.py (excludes --title)
uv run python -m src.eval "${EVAL_ARGS[@]}" 2>&1 | tee -a $LOG_FILE

# -------------------------------------------------------------------
# 5. Visualization
# -------------------------------------------------------------------
if [ -n "$MODEL_PATH" ]; then
    EVAL_FILE_PATH="$MODEL_PATH/evaluation_results.jsonl"
    echo "Generating evaluation graphs..." | tee -a $LOG_FILE
    uv run python -m src.visualize_eval --eval_file_path "$EVAL_FILE_PATH" --title "$TITLE" 2>&1 | tee -a $LOG_FILE
else
    echo "WARNING: Could not parse --model_path. Skipping visualization." | tee -a $LOG_FILE
fi

echo "Evaluation Job finished at $(date)" | tee -a $LOG_FILE