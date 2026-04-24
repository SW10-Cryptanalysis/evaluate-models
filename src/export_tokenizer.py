import argparse
import logging
import json
from pathlib import Path
from easy_logging import EasyFormatter
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast

from src.classes.config import EvalConfig, TOKENIZER_DIR

handler = logging.StreamHandler()
handler.setFormatter(EasyFormatter())
logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def create_hf_tokenizer(model_path: str) -> None:
    """Generates a Hugging Face compatible tokenizer for custom integer-based cipher models.
    This satisfies vLLM's initialization requirements without interfering with our
    raw token ID generation logic.
    """
    logger.info(f"Loading configuration from {model_path} to build global tokenizer...")
    save_dir = Path(model_path)

    # Read the actual vocab_size the model was trained with
    model_config_path = save_dir / "config.json"
    with open(model_config_path) as f:
        model_config = json.load(f)
    actual_vocab_size = model_config["vocab_size"]  # Will be 2560

    config = EvalConfig.from_model_path(model_path=model_path, use_spaces=True)

    vocab = {}
    for i in range(actual_vocab_size):
        if i == config.bos_token_id:
            vocab["[BOS]"] = i
        elif i == config.eos_token_id:
            vocab["[EOS]"] = i
        elif i == config.pad_token_id:
            vocab["[PAD]"] = i
        elif i == config.sep_token_id:
            vocab["[SEP]"] = i
        elif config.use_spaces and i == config.space_token_id:
            vocab["_"] = i
        else:
            vocab[f"[unused_{i}]"] = i

    # 2. Initialize the base Tokenizer using the WordLevel model
    # WordLevel is perfectly suited for 1:1 symbol-to-integer cipher models
    tokenizer_model = models.WordLevel(vocab=vocab, unk_token="[UNK]")
    base_tokenizer = Tokenizer(tokenizer_model)

    # 3. Wrap into a Transformers fast tokenizer
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=base_tokenizer,
        bos_token="[BOS]",
        eos_token="[EOS]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        unk_token="[UNK]",
    )

    # 4. Export to the CENTRAL tokenizer directory instead of the specific model directory
    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
    fast_tokenizer.save_pretrained(TOKENIZER_DIR)

    logger.info(f"Successfully generated and saved Global Tokenizer to {TOKENIZER_DIR}")
    logger.info(
        "Files created: tokenizer.json, tokenizer_config.json, special_tokens_map.json",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to a reference trained model checkpoint (e.g., outputs/checkpoint-16750)",
    )
    args = parser.parse_args()

    create_hf_tokenizer(args.model_path)
