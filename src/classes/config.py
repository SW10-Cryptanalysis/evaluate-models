from dataclasses import dataclass
import json
from pathlib import Path


DATA_DIR = Path(__file__).parent.parent.parent.parent / "Ciphers"
TOKENIZER_DIR = Path(__file__).parent.parent.parent / "tokenizer"


@dataclass
class EvalConfig:
    """Derived entirely from the model's saved config.json — no recomputation needed."""

    vocab_size: int
    pad_token_id: int
    bos_token_id: int
    eos_token_id: int
    max_context: int
    use_spaces: bool

    tokenizer_dir: Path = TOKENIZER_DIR

    @property
    def sep_token_id(self) -> int:
        """Seperator token."""
        return self.bos_token_id - 2

    @property
    def space_token_id(self) -> int:
        """Space token."""
        return self.bos_token_id - 1

    @property
    def char_offset(self) -> int:
        """Offset for character token IDs."""
        return self.eos_token_id + 1

    @property
    def tokenized_dir(self) -> Path:
        """Dynamic path based on whether we use spaces or not."""
        suffix = "spaced" if self.use_spaces else "normal"
        return DATA_DIR / f"tokenized_{suffix}"  # type: ignore

    @classmethod
    def from_model_path(cls, model_path: str | Path, use_spaces: bool) -> "EvalConfig":
        """Load configuration directly from the training datas metadata file."""
        config_path = Path(model_path) / "config.json"
        with open(config_path) as f:
            mc = json.load(f)
        return cls(
            vocab_size=mc["vocab_size"],
            pad_token_id=mc["pad_token_id"],
            bos_token_id=mc["bos_token_id"],
            eos_token_id=mc["eos_token_id"],
            max_context=mc["max_position_embeddings"],
            use_spaces=use_spaces,
        )
