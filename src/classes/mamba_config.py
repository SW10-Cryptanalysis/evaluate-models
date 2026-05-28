from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json
from dotenv import load_dotenv
import os
from typing import Literal

load_dotenv()
CIPHERS_DIR = os.environ["CIPHERS_DIR"]

BASE_SEQ_LEN_NORMAL = 10063
BASE_SEQ_LEN_SPACES = 13077


@dataclass
class MambaConfig:
    r"""Configuration class for the Mamba2 model.

    Attributes:
        num_heads (int): Number of SSM heads, analogous to the number of heads in
            multi-head attention.
        head_dim (int): Dimensionality of each individual head.
        vocab_size (int): Total number of unique tokens in the model's vocabulary
            (set during post-initialization of parent config).
        hidden_size (int): The overall model dimension.
        state_size (int): The latent state dimension for the State Space Model.
        num_hidden_layers (int): Total number of stacked Mamba2 blocks
            (depth of the model).
        layer_norm_epsilon (float): Small constant added for numerical stability during
            normalization.
        pad_token_id (int): Index of the padding token in the vocabulary
            (set during post-initialization of parent config).
        sep_token_id (int): Index of the separator token in the vocabulary
            (set during post-initialization of parent config).
        eos_token_id (int): Index of the end-of-sequence token in the vocabulary
            (set during post-initialization of parent config).
        expand (int): The expansion factor for the inner hidden dimension.
        conv_kernel (int): The kernel size for the 1D convolution layer within the
            mixer.
        n_groups (int): Number of groups for the Grouped State Space Diffusion (GSSD)
            mechanism.
        use_bias (bool): Whether to include bias terms in the linear projections.
        use_conv_bias (bool): Whether to include bias terms in the 1D convolution
            layers.
        hidden_act (str): The activation function used (e.g., 'silu' or 'swish').
        initializer_range (float): The standard deviation used for initializing weight
            matrices.
        residual_in_fp32 (bool): If True, computes residual connections in float32 for
            higher precision.
        time_step_rank (str | int): The rank of the $\Delta$ (time-step) projection;
            'auto' calculates it based on hidden_size.
        time_step_min (float): The minimum allowable value for the time-step parameter.
        time_step_max (float): The maximum allowable value for the time-step parameter.
        time_step_floor (float): The floor value for time-step initialization.
        time_step_limit (list[float] | tuple[float, ...]): Bounds used to clip the
            time-step values.
        rescale_prenorm_residual (bool): If True, rescales the output projection weights
            during initialization.
        use_cache (bool): Whether to maintain a state cache for faster incremental
            inference.
        rms_norm (bool): If True, uses Root Mean Square Layer Normalization (RMSNorm)
            instead of standard LayerNorm.
        chunk_size (int): The chunk size used for the parallel associative scan;
            balances memory and speed.
        tie_word_embeddings (bool): Whether to share the same weights between input and
            output embeddings.

    """

    num_heads: int = 16
    head_dim: int = 64
    vocab_size: int = field(init=False)
    hidden_size: int = 1024
    state_size: int = 48
    num_hidden_layers: int = 8
    layer_norm_epsilon: float = 1e-5
    pad_token_id: int = field(init=False)
    sep_token_id: int = field(init=False)
    eos_token_id: int = field(init=False)
    bos_token_id: int = field(init=False)
    expand: int = 1
    conv_kernel: int = 4
    n_groups: int = 8
    use_bias: bool = False
    use_conv_bias: bool = True
    hidden_act: str = "silu"
    initializer_range: float = 0.1
    residual_in_fp32: bool = True
    time_step_rank: str | int = "auto"
    time_step_min: float = 0.001
    time_step_max: float = 0.1
    time_step_floor: float = 1e-4
    time_step_limit: list[float] | tuple[float, ...] = (0.0, float("inf"))
    rescale_prenorm_residual: bool = False
    use_cache: bool = False
    rms_norm: bool = True
    chunk_size: int = 256
    tie_word_embeddings: bool = False


@dataclass
class CosineSchedulerConfig:
    """Hyperparameters for the optimizer and training loop.

    Attributes:
        learning_rate (float): The learning rate.
        epochs (int): The number of epochs.
        patience (int): The patience for early stopping.
        factor (float): The factor for early stopping.
        pct_start (float): Percentage of training for LR warmup.

    """

    learning_rate: float = 5e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    weight_decay: float = 0.1
    grad_accum: int = 2

    epochs: int = 5
    batch_size: int = 4


@dataclass
class Config:
    """Dataclass for MambaCipherSolver configuration.

    Attributes:
        use_spaces (bool, optional): Whether to use spaces or not. Defaults to False.
        device (str, optional): The device to use for training. Defaults to "cuda".
        mamba_config (MambaConfig): Configuration for the Mamba model.
        batch_size (int, optional): The batch size. Defaults to 128.
        save_step (int): Frequency of intermediate checkpoint saves in steps.
            Defaults to 5000.
        save_path (Path): Directory where model outputs and logs are stored.
        data_dir (Path): Base directory for all cipher-related data.
        homophone_file (str, optional): The name of the homophone file.
            Defaults to "metadata.json".
        plain_vocab_size (int, optional): The size of the plain vocabulary.
            Defaults to 26.
        unique_homophones (int, optional): The number of unique homophones.
            Defaults to 500.
        max_len (int, optional): The maximum length of the dataset.
        vocab_size (int, optional): The size of the vocabulary. Defaults to 0.
        buffer (int, optional): The buffer size. Defaults to 1.
        pad_token_id (int): The token ID to use for padding. Defaults to 0.
        sep_token_id (int): The token ID used to separate cipher
            and plain text.
        space_token_id (int): The token ID used to separate cipher
            and plain text.
        bos_token_id (int): The token ID used to signal the beginning of a sequence.
        eos_token_id (int): The token ID used to signal the end of a sequence.
        char_offset (int): The starting index for plaintext character IDs to avoid
            collisions with cipher homophones.
        tokenized_dir (Path): The directory containing the tokenized datasets.

    Methods:
        load_homophones(homophone_file: str = "metadata.json"):
            Load the homophone metadata file and set the unique homophone count.
            Also sets the MambaConfig vocab_size and token IDs.
        __post_init__():
            Post-initialization hook to load homophone metadata.



    """

    task: Literal["causal", "mapping"] = "causal"
    use_spaces: bool = False
    truncated: bool = False
    mono: bool = False
    device: str = "cuda"

    mamba_config: MambaConfig = field(default_factory=MambaConfig)
    scheduler_config: CosineSchedulerConfig = field(
        default_factory=CosineSchedulerConfig,
    )
    save_step: int = 25_000

    outputs_dir: Path = Path(__file__).parent.parent / "outputs"
    data_dir: Path = Path(__file__).parent.parent.parent / CIPHERS_DIR
    homophone_file: str = "metadata.json"

    plain_vocab_size: int = 26
    unique_homophones: int = 2503
    buffer: int = 10
    pad_token_id: int = 0

    @property
    def sep_token_id(self) -> int:
        """Separator token."""
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
        """Character offset to avoid clashes with defined tokens."""
        return self.eos_token_id + 1

    @property
    def tokenized_dir(self) -> Path:
        """Dynamic path based on whether we use spaces or not."""
        suffix = "spaced" if self.use_spaces else "normal"
        suffix += "_mapping" if self.task == "mapping" else ""
        suffix += "_truncated_4000" if self.truncated else ""
        suffix += "_monoalphabetic" if self.mono else ""
        return self.data_dir / f"tokenized_{suffix}"

    @property
    def max_len(self) -> int:
        """Max len based on with or without spaces."""
        base = BASE_SEQ_LEN_SPACES if self.use_spaces else BASE_SEQ_LEN_NORMAL
        if self.task == "mapping":
            return base + 2 + self.buffer
        return base * 2 + 3 + self.buffer

    @property
    def save_path(self) -> Path:
        """Dynamic outputs dir based on timestamp and whether we use spaces or not."""
        mode = "spaces" if self.use_spaces else "normal"
        mode += "_mapping" if self.task == "mapping" else ""
        mode += "_truncated_4000" if self.truncated else ""
        mode += "_mono" if self.mono else ""
        return self.outputs_dir / f"{mode}_{self._timestamp}"

    def load_homophones(self, homophone_file: str = "metadata.json") -> None:
        """Load the homophone metadata file and set the unique homophone count."""
        homophone_path = self.data_dir / homophone_file
        try:
            with open(homophone_path) as f:
                meta = json.load(f)
                self.unique_homophones = int(meta["max_symbol_id"])
        except (OSError, ValueError, KeyError) as e:
            raise RuntimeError(
                "Aborting initialization: Invalid or missing homophone metadata.",
            ) from e
        self.mamba_config.vocab_size = (
            self.char_offset + self.plain_vocab_size + self.buffer
        )
        self.mamba_config.sep_token_id = self.sep_token_id
        self.mamba_config.eos_token_id = self.eos_token_id
        self.mamba_config.bos_token_id = self.bos_token_id
        self.mamba_config.pad_token_id = self.pad_token_id

    def __post_init__(self) -> None:
        """Post-initialization hook to load homophone metadata."""
        self._timestamp = datetime.now().strftime("%d%m_%H%M%S_%Y")
        self.load_homophones()
