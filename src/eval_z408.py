import argparse
import json
import logging
import torch
from pathlib import Path

from src.classes.config import EvalConfig
from src import eval_utils
from model import get_model  # Import your custom RWKV model loader

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# L4 GPU Optimizations
torch.backends.cuda.matmul.fp32_precision = "tf32"


def main() -> None:
    """Evaluate the Z408 cipher using a trained RWKV-7 model natively in PyTorch."""
    parser = argparse.ArgumentParser(
        description="Evaluate the Z408 cipher using a trained model.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model directory containing rwkv7_cipher_final.pth",
    )
    parser.add_argument(
        "--z408_path",
        type=str,
        required=True,
        help="Path to the z408.json file",
    )
    parser.add_argument(
        "--spaces",
        action="store_true",
        help="Use spaces in evaluation",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_path)
    config = EvalConfig.from_model_path(args.model_path, args.spaces)
    
    if not EvalConfig.tokenizer_dir.exists():
        logger.error(f"Global tokenizer not found at {EvalConfig.tokenizer_dir}.")
        return

    z408_file = Path(args.z408_path)
    if not z408_file.exists():
        logger.error(f"Z408 JSON file not found at {args.z408_path}")
        return

    with open(z408_file) as f:
        z408_data = json.load(f)

    # Parse cipher IDs
    raw_encoding_str = z408_data["recurrence_encoding"].replace(".", "")
    cipher_ids = [int(x) for x in raw_encoding_str.split()]
    true_plain = z408_data["plaintext"]

    prompt_ids = [config.bos_token_id] + cipher_ids + [config.sep_token_id]
    target_length = len(cipher_ids)

    # --- NATIVE PYTORCH MODEL SETUP ---
    if not torch.cuda.is_available():
        logger.error("CUDA is required for evaluation.")
        return
    device = torch.device("cuda:0")

    logger.info("Loading Native RWKV-7 Model for Z408 Evaluation...")
    model = get_model().to(device)
    checkpoint_path = model_dir / "rwkv7_cipher_final.pth"
    
    if not checkpoint_path.exists():
        logger.error(f"Could not find model weights at {checkpoint_path}")
        return
        
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()

    # --- ALLOWED TOKENS MASK ---
    vocab_size = config.vocab_size
    allowed_mask = torch.full((vocab_size,), float('-inf'), device=device)
    allowed_token_ids = eval_utils.build_allowed_token_ids(config)
    valid_allowed_ids = [t for t in allowed_token_ids if t < vocab_size]
    allowed_mask[valid_allowed_ids] = 0.0

    # --- AUTOREGRESSIVE GREEDY INFERENCE ---
    logger.info(f"Running autoregressive generation for Z408 ({target_length} tokens)...")
    current_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    pred_ids = []

    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            for _ in range(target_length):
                logits = model(current_ids)
                
                # Get logits for the absolute last token in the sequence
                next_token_logits = logits[0, -1, :]
                
                # Filter out unallowed characters (force vocabulary constraints)
                next_token_logits += allowed_mask
                
                # Greedy choice
                next_token = torch.argmax(next_token_logits, dim=-1)
                pred_ids.append(next_token.item())
                
                # Append to context for the next step loop
                current_ids = torch.cat([current_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=-1)

    # --- PROCESS OUTPUTS ---
    pred_plain = eval_utils.decode_prediction(pred_ids, config)
    compare_plain = true_plain[:target_length]
    ser, wrong_spaces = eval_utils.calculate_ser(compare_plain, pred_plain)

    # Create the result dictionary with matching keys
    z408_result_dict = {
        "index": "Z408",
        "redundancy": z408_data.get("difficulty", 4),  # Mapping difficulty to redundancy
        "ciphertext": eval_utils.decode_ciphertext(cipher_ids, config),
        "plaintext": compare_plain,
        "predicted_plaintext": pred_plain,
        "ser": ser,
        "wrong_spaces": wrong_spaces,
    }

    # Inject results into evaluation_stats.json
    stats_path = model_dir / "evaluation_stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            try:
                stats_data = json.load(f)
            except json.JSONDecodeError:
                stats_data = {}
    else:
        stats_data = {}

    stats_data["z408_result"] = z408_result_dict

    with open(stats_path, "w") as f:
        json.dump(stats_data, f, indent=4)

    logger.info(f"Z408 Results evaluated (SER: {ser:.4f}) and written to {stats_path}")


if __name__ == "__main__":
    main()