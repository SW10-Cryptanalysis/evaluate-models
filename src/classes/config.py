import json
import os
from dataclasses import dataclass
import logging
from easy_logging import EasyFormatter
from pathlib import Path

MAX_TEXT_LEN = 10_000
UNIQUE_HOMOPHONE_COUNT = 500
UNIQUE_LETTER_COUNT = 26
TOTAL_SEQ = MAX_TEXT_LEN * 2
BUFFER = 8

OUTPUT_DIR = Path(__file__).parent.parent.parent / "outputs"
DATA_DIR = Path(__file__).parent.parent.parent.parent / "Ciphers"

HOMOPHONE_FILE = "metadata.json"

handler = logging.StreamHandler()
handler.setFormatter(EasyFormatter())
logger = logging.getLogger(__name__)
logger.addHandler(handler)


@dataclass
class Config:
    """Config dataclass.

    This dataclass contains the configuration parameters for the model.

    Attributes:
            unique_homophones (int): The number of unique homophones in the dataset.
            unique_letters (int): The number of unique letters in the dataset.
            vocab_size (int): The size of the vocabulary.
            max_context (int): The maximum context length.
            dims (int): The number of dimensions in the model.
            layers (int): The number of layers in the model.
            att_heads (int): The number of attention heads in the model.
            kv_heads (int): The number of key-value heads in the model.
            rope_theta (float): The RoPE theta parameter.
            batch_size (int): The batch size for training.
            grad_accum (int): The number of batches to accumulate gradients over.
            learning_rate (float): The learning rate for training.
            epochs (int): The number of epochs to train for.
            log_steps (int): The number of steps to log the training progress at.
            save_steps (int): The number of steps to save the model at.
            output_dir (str): The output directory for the model.
            data_dir (str): The data directory for the model.

    """

    # ARCHITECTURE

    # Default value is UNIQUE_HOMOPHONE_COUNT unless a HOMOPHONE_FILE exists
    unique_homophones: int = UNIQUE_HOMOPHONE_COUNT
    unique_letters: int = UNIQUE_LETTER_COUNT
    pad_token_id: int = 0

    # Vocab needs to be larger than unique homophone count + unique letter count
    # + buffer (start/end/padding, etc) and maybe spacing "_"
    vocab_size: int = UNIQUE_HOMOPHONE_COUNT + UNIQUE_LETTER_COUNT + BUFFER
    # Input is BOS + ciphertext + SEP + plaintext + EOS
    max_context: int = TOTAL_SEQ + 3
    dims: int = 384
    layers: int = 16
    att_heads: int = 6
    kv_heads: int = 2
    rope_theta: float = 1_000_000.0

    @property
    def final_output_dir(self) -> Path:
        """Dynamic output dir to either outputs/spaces/ or outputs/normal/."""
        suffix = "spaces" if self.use_spaces else "normal"
        return self.output_dir / suffix

    # TOKEN PROPERTIES
    @property
    def sep_token_id(self) -> int:
        """Seperator token."""
        return self.unique_homophones + 1

    @property
    def space_token_id(self) -> int:
        """Space token."""
        return self.sep_token_id + 1

    @property
    def bos_token_id(self) -> int:
        """Beginning of sequence token."""
        return self.space_token_id + 1

    @property
    def eos_token_id(self) -> int:
        """End of sequence token."""
        return self.bos_token_id + 1

    @property
    def char_offset(self) -> int:
        """Character ofset to avoid clashes with defined tokens."""
        return self.eos_token_id + 1

    # TRAINING
    batch_size: int = 16
    grad_accum: int = 1
    learning_rate: float = 3e-4
    epochs: int = 5
    log_steps: int = 10
    save_steps: int = 1000
    use_spaces: bool = False

    # SYSTEM
    output_dir: Path = OUTPUT_DIR
    data_dir: Path = DATA_DIR

    @property
    def tokenized_dir(self) -> Path:
        """Dynamic path based on whether we use spaces or not."""
        suffix = "spaced" if self.use_spaces else "normal"
        return self.data_dir / f"tokenized_{suffix}"

    def load_homophones(self) -> None:
        """Load the homophone metadata file and set the unique homophone count."""
        homophone_path = os.path.join(self.data_dir, HOMOPHONE_FILE)
        if os.path.exists(homophone_path):
            try:
                with open(homophone_path) as f:
                    meta = json.load(f)
                    homophones = int(meta["max_symbol_id"])
                    self.unique_homophones = homophones
            except OSError as e:
                logger.warning(f"Could not read file: {HOMOPHONE_FILE}")
                logger.warning(f"Using default value: {self.unique_homophones}")
                logger.warning(f"Error: {e}")
            except (ValueError, KeyError) as e:
                logger.warning(f"Invalid or missing data in {HOMOPHONE_FILE}")
                logger.warning(f"Using default value: {self.unique_homophones}")
                logger.warning(f"Error: {e}")

        self.vocab_size = self.char_offset + 26 + 1
