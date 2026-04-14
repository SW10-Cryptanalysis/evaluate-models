import argparse
import logging
from pathlib import Path
from easy_logging import EasyFormatter

# Hugging Face dependencies (automatically available via vLLM/transformers)
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast

from src.classes.config import Config

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
    config = Config()

    # We must set data_dir to where metadata.json actually is if necessary,
    # but Config() handles loading homophones if the path is correct.
    config.load_homophones()

    # 1. Build a strict vocabulary dictionary mapping strings to your token integers
    vocab = {}

    # Pre-fill the entire vocabulary space to ensure vLLM dimensional checks pass
    for i in range(config.vocab_size):
        vocab[f"[unused_{i}]"] = i

    # Overwrite known structural tokens based on your mapping logic
    vocab["[PAD]"] = config.pad_token_id
    vocab["[SEP]"] = config.sep_token_id
    vocab["[SPACE]"] = config.space_token_id
    vocab["[BOS]"] = config.bos_token_id
    vocab["[EOS]"] = config.eos_token_id

    # We place the [UNK] token at the very end of the allocated buffer
    unk_token_id = config.vocab_size - 1
    vocab["[UNK]"] = unk_token_id

    # Add latin alphabet mapping for completeness
    for i, char in enumerate("abcdefghijklmnopqrstuvwxyz"):
        vocab[char] = config.char_offset + i

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
