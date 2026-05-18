import argparse
import json
import logging
from pathlib import Path
from vllm import LLM, SamplingParams  # type: ignore

from src.classes.config import EvalConfig
from src import eval_utils

# Setup basic logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the Z408 cipher using a trained model."
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the model directory"
    )
    parser.add_argument(
        "--z408_path", type=str, required=True, help="Path to the z408.json file"
    )
    parser.add_argument(
        "--spaces", action="store_true", help="Use spaces in evaluation"
    )
    args = parser.parse_args()

    # 1. Load config and check tokenizer
    config = EvalConfig.from_model_path(args.model_path, args.spaces)
    if not EvalConfig.tokenizer_dir.exists():
        logger.error(f"Global tokenizer not found at {EvalConfig.tokenizer_dir}.")
        return

    # 2. Load and parse the Z408 JSON data
    z408_file = Path(args.z408_path)
    if not z408_file.exists():
        logger.error(f"Z408 JSON file not found at {args.z408_path}")
        return

    with open(z408_file, "r") as f:
        z408_data = json.load(f)

    # Clean the string (removing any trailing periods from the snippet) and split into integer IDs
    raw_encoding_str = z408_data["recurrence_encoding"].replace(".", "")
    cipher_ids = [int(x) for x in raw_encoding_str.split()]
    true_plain = z408_data["plaintext"]

    # 3. Construct the prompt
    # Layout: [BOS] <cipher_ids> [SEP]
    prompt_ids = [config.bos_token_id] + cipher_ids + [config.sep_token_id]

    # Since substitution ciphers are 1:1, target generation length equals cipher length
    target_length = len(cipher_ids)

    logger.info(f"Loaded Z408. Cipher length: {target_length} tokens.")

    # 4. Initialize vLLM
    logger.info("Initializing vLLM...")
    llm = LLM(
        model=args.model_path,
        tokenizer=str(EvalConfig.tokenizer_dir),
        max_model_len=config.max_context,
        enforce_eager=False,
        gpu_memory_utilization=0.90,
    )

    vocab_size = (
        llm.get_tokenizer().vocab_size if llm.get_tokenizer() else config.vocab_size
    )
    allowed_token_ids = eval_utils.build_allowed_token_ids(config)
    valid_allowed_ids = [t for t in allowed_token_ids if t < vocab_size]

    # 5. Define sampling parameters (forcing strict formatting using allowed_token_ids)
    sp = SamplingParams(
        temperature=0.0,
        max_tokens=target_length,
        min_tokens=target_length,
        allowed_token_ids=valid_allowed_ids,
        detokenize=False,
    )

    # 6. Run Inference
    logger.info("Running inference...")
    outputs = llm.generate([{"prompt_token_ids": prompt_ids}], sampling_params=[sp])

    # 7. Process Output
    pred_ids = list(outputs[0].outputs[0].token_ids)[:target_length]
    pred_plain = eval_utils.decode_prediction(pred_ids, config)

    # Truncate true plaintext to target length for fair comparison if lengths mismatch
    compare_plain = true_plain[:target_length]
    ser, wrong_spaces = eval_utils.calculate_ser(compare_plain, pred_plain)

    # 8. Print Results
    print("\n" + "=" * 50)
    print("Z408 EVALUATION RESULTS")
    print("=" * 50)
    print(f"True Plaintext (Truncated to {target_length}):\n{compare_plain}\n")
    print(f"Model Prediction:\n{pred_plain}\n")
    print("-" * 50)
    print(f"Symbol Error Rate (SER): {ser:.4f} ({(ser*100):.2f}%)")
    print(f"Wrong Spaces:            {wrong_spaces}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
