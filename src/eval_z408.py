import argparse
import json
import logging
from pathlib import Path
from vllm import LLM, SamplingParams  # type: ignore

from src.classes.config import EvalConfig
from src import eval_utils

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Evaluate the Z408 cipher using a trained model and vLLM."""
    parser = argparse.ArgumentParser(
        description="Evaluate the Z408 cipher using a trained model.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model directory",
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
    parser.add_argument(
        "--mapping",
        action="store_true",
        help="Use mapping in evaluation",
    )
    args = parser.parse_args()

    config = EvalConfig.from_model_path(args.model_path, args.spaces, args.mapping)
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

    # Inference setup
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

    sp = SamplingParams(
        temperature=0.0,
        max_tokens=target_length,
        min_tokens=target_length,
        allowed_token_ids=valid_allowed_ids,
        detokenize=False,
    )

    # Run inference
    outputs = llm.generate([{"prompt_token_ids": prompt_ids}], sampling_params=[sp])

    # Process outputs
    pred_ids = list(outputs[0].outputs[0].token_ids)[:target_length]
    pred_plain = eval_utils.decode_prediction(pred_ids, config)
    compare_plain = true_plain[:target_length]
    ser, wrong_spaces = eval_utils.calculate_ser(compare_plain, pred_plain)

    # Create the result dictionary with matching keys
    z408_result_dict = {
        "index": "Z408",
        "redundancy": z408_data.get(
            "difficulty",
            4,
        ),  # Mapping difficulty to redundancy for plotting
        "ciphertext": eval_utils.decode_ciphertext(cipher_ids, config),
        "plaintext": compare_plain,
        "predicted_plaintext": pred_plain,
        "ser": ser,
        "wrong_spaces": wrong_spaces,
    }

    # Inject into evaluation_stats.json
    stats_path = Path(args.model_path) / "evaluation_stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            stats_data = json.load(f)
    else:
        stats_data = {}

    stats_data["z408_result"] = z408_result_dict

    with open(stats_path, "w") as f:
        json.dump(stats_data, f, indent=4)

    logger.info(f"Z408 Results evaluated (SER: {ser:.4f}) and written to {stats_path}")


if __name__ == "__main__":
    main()
