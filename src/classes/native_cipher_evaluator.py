from classes.base_cipher_evaluator import BaseCipherEvaluator
import torch
from safetensors.torch import load_file
from pathlib import Path
import time
import logging
from easy_logging import EasyFormatter
from transformers import Mamba2Config

handler = logging.StreamHandler()
handler.setFormatter(EasyFormatter())
logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class NativeMappingEvaluator(BaseCipherEvaluator):
    """Evaluator subclass utilizing native PyTorch for custom pooling architectures."""

    def __init__(self, model_path: str, use_spaces: bool, mapping: bool) -> None:
        """Initialize base properties and verify CUDA availability for PyTorch."""
        super().__init__(model_path, use_spaces, mapping)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for native evaluation.")
        self.device = torch.device("cuda")

    def run(self) -> list[dict]:
        """Load the custom mapping model and execute sequential or batched inference."""
        logger.info("Initializing Native PyTorch Mapping Model...")

        from models.mamba_mapping import Mamba2ForMapping

        mamba_config = Mamba2Config.from_pretrained(self.model_path)

        mamba_config.time_step_limit = (0.0, float("inf"))

        model = Mamba2ForMapping(mamba_config, num_labels=26)
        weights_path = Path(self.model_path) / "model.safetensors"
        model.load_state_dict(load_file(weights_path))

        model.to(self.device)
        model.eval()

        parsed_samples = self.parse_samples()
        clean_outputs = []

        start_time = time.perf_counter()

        with torch.no_grad():
            for sample in parsed_samples:
                input_tensor = torch.tensor([sample["prompt_ids"]], device=self.device)

                outputs = model(input_ids=input_tensor)
                logits = outputs.logits

                pred_ids = torch.argmax(logits, dim=-1).squeeze(0).tolist()

                target_len = sample["target_length"]
                clean_outputs.append(pred_ids[:target_len])

        generation_time = time.perf_counter() - start_time

        return self.process_outputs(parsed_samples, clean_outputs, generation_time)
