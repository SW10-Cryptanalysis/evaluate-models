import argparse
import logging
import json
from pathlib import Path
from easy_logging import EasyFormatter

# Hugging Face dependencies (automatically available via vLLM/transformers)
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast

from src.classes.config import EvalConfig

handler = logging.StreamHandler()
handler.setFormatter(EasyFormatter())
logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def create_hf_tokenizer(model_path: str):
    """
    Generates a Hugging Face compatible tokenizer for custom integer-based cipher models.
    This satisfies vLLM's initialization requirements without interfering with our
    raw token ID generation logic.
    """
    logger.info(f"Loading configuration for tokenizer export...")
    save_dir = Path(model_path)

    # Read the actual vocab_size the model was trained with
    model_config_path = save_dir / "config.json"
    with open(model_config_path) as f:
        model_config = json.load(f)
    actual_vocab_size = model_config["vocab_size"]  # Will be 2560

    config = EvalConfig.from_model_path(model_path=model_path, use_spaces=True)

    vocab = {}
    for i in range(actual_vocab_size):  # Use 2560, not 2535
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

    # 4. Export to the checkpoint directory
    save_dir = Path(model_path)
    if not save_dir.exists():
        logger.error(f"Model path does not exist: {save_dir}")
        return

    fast_tokenizer.save_pretrained(save_dir)
    logger.info(f"Successfully generated and saved SOTA Tokenizer to {save_dir}")
    logger.info(
        f"Files created: tokenizer.json, tokenizer_config.json, special_tokens_map.json"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model checkpoint (e.g., outputs/checkpoint-16750)",
    )
    args = parser.parse_args()

    create_hf_tokenizer(args.model_path)
