import sys
import time
import torch
from vllm import LLM, SamplingParams  # type: ignore

from classes.base_cipher_evaluator import BaseCipherEvaluator
from classes.config import EvalConfig

import logging
from easy_logging import EasyFormatter

handler = logging.StreamHandler()
handler.setFormatter(EasyFormatter())
logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class VLLMCipherEvaluator(BaseCipherEvaluator):
    """Evaluator subclass utilizing vLLM for rapid autoregressive generation."""

    def __init__(self, model_path: str, use_spaces: bool, mapping: bool) -> None:
        """Initialize base properties and verify CUDA availability for vLLM."""
        super().__init__(model_path, use_spaces, mapping)
        self.world_size = torch.cuda.device_count()
        if self.world_size == 0:
            raise RuntimeError("vLLM requires CUDA devices, but none were found.")

    def run(self) -> list[dict]:
        """Execute vLLM engine initialization and batch inference."""
        if not EvalConfig.tokenizer_dir.exists():
            logger.error("Global tokenizer not found.")
            sys.exit(1)

        logger.info(f"Initializing vLLM across {self.world_size} GPUs...")

        llm = LLM(
            model=self.model_path,
            tokenizer=str(EvalConfig.tokenizer_dir),
            tensor_parallel_size=self.world_size,
            dtype="bfloat16",
            max_model_len=self.config.max_context,
            enforce_eager=False,
            gpu_memory_utilization=0.95,
        )

        vocab_size = (
            llm.get_tokenizer().vocab_size
            if llm.get_tokenizer()
            else self.config.vocab_size
        )
        valid_allowed_ids = [t for t in self.allowed_token_ids if t < vocab_size]

        parsed_samples = self.parse_samples()
        parsed_samples.sort(key=lambda x: x["target_length"], reverse=True)

        prompts = [{"prompt_token_ids": s["prompt_ids"]} for s in parsed_samples]
        sampling_params_list = [
            SamplingParams(
                temperature=0.0,
                max_tokens=s["target_length"],
                min_tokens=s["target_length"],
                allowed_token_ids=valid_allowed_ids,
                detokenize=False,
            )
            for s in parsed_samples
        ]

        start_time = time.perf_counter()
        raw_outputs = llm.generate(prompts, sampling_params=sampling_params_list)
        generation_time = time.perf_counter() - start_time

        clean_outputs = [
            list(out.outputs[0].token_ids)[: sample["target_length"]]
            for out, sample in zip(raw_outputs, parsed_samples, strict=False)
        ]

        return self.process_outputs(parsed_samples, clean_outputs, generation_time)
