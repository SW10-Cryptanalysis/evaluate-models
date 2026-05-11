import argparse
import json
import logging
import time
import torch
import sys
from pathlib import Path
from datasets import load_from_disk
from vllm import LLM, SamplingParams, RequestOutput  # type: ignore
from easy_logging import EasyFormatter

from src.classes.config import EvalConfig
from src import eval_utils

handler = logging.StreamHandler()
handler.setFormatter(EasyFormatter())
logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class VLLMCipherEvaluator:
    """Evaluator class that uses vLLM to assess cipher decryption performance."""

    def __init__(self, model_path: str, use_spaces: bool) -> None:
        """Initialize the evaluator with model path and configuration."""
        self.model_path = model_path
        self.config = EvalConfig.from_model_path(model_path, use_spaces)
        self.config.use_spaces = use_spaces

        self.output_log_path = Path(self.model_path) / "evaluation_results.jsonl"
        self.summary_scores_path = Path(self.model_path) / "avg_ser_scores.json"

        self.dataset = load_from_disk(self.config.tokenized_dir / "Test")
        self.allowed_token_ids = eval_utils.build_allowed_token_ids(self.config)

        self.world_size = torch.cuda.device_count()
        if self.world_size == 0:
            raise RuntimeError("vLLM requires CUDA devices, but none were found.")

        self.skipped_count = 0

    def parse_samples(self) -> list[dict]:
        """Extract prompt token IDs and track skipped samples."""
        parsed_data = []
        self.skipped_count = 0  # Reset counter

        for index, item in enumerate(self.dataset):
            all_ids = item["input_ids"]
            try:
                sep_idx = all_ids.index(self.config.sep_token_id)
                prompt_ids = all_ids[: sep_idx + 1]
                raw_cipher_ids = all_ids[1:sep_idx]
            except ValueError:
                continue

            target_length = len(raw_cipher_ids)
            total_required_context = len(prompt_ids) + target_length

            if total_required_context > self.config.max_context:
                self.skipped_count += 1
                continue

            if target_length > 0:
                parsed_data.append(
                    {
                        "index": index,
                        "prompt_ids": prompt_ids,
                        "raw_cipher_ids": raw_cipher_ids,
                        "true_plain": item["raw_plaintext"],
                        "redundancy": int(item["redundancy"]),
                        "target_length": target_length,
                    },
                )

        if self.skipped_count > 0:
            logger.warning(
                f"Skipped {self.skipped_count} samples that exceeded max_context ({self.config.max_context})",
            )

        return parsed_data

    def run(self) -> list[dict]:
        """Main method to execute the evaluation process."""
        if not EvalConfig.tokenizer_dir.exists():
            logger.error(f"Global tokenizer not found at {EvalConfig.tokenizer_dir}.")
            logger.error(
                "Please run `python -m src.export_tokenizer --model_path <any_valid_model>` once.",
            )
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
            enable_prefix_caching=False,
        )

        vocab_size = (
            llm.get_tokenizer().vocab_size
            if llm.get_tokenizer()
            else self.config.vocab_size
        )
        eval_utils.run_preflight_checks(
            self.config,
            self.dataset,
            self.allowed_token_ids,
            self.output_log_path,
            vocab_size,
            logger,
        )

        valid_allowed_ids = [t for t in self.allowed_token_ids if t < vocab_size]
        parsed_samples = self.parse_samples()
        parsed_samples.sort(key=lambda x: x["target_length"], reverse=True)

        prompts = []
        sampling_params_list = []

        for sample in parsed_samples:
            tl = sample["target_length"]
            sp = SamplingParams(
                temperature=0.0,
                max_tokens=tl,
                min_tokens=tl,
                allowed_token_ids=valid_allowed_ids,
                detokenize=False,
            )
            prompts.append({"prompt_token_ids": sample["prompt_ids"]})
            sampling_params_list.append(sp)

        logger.info(f"Launching batched inference for {len(prompts)} sequences...")
        start_time = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params=sampling_params_list)
        generation_time = time.perf_counter() - start_time
        logger.info(f"Inference complete in {generation_time:.2f} seconds.")

        return self.process_outputs(parsed_samples, outputs, generation_time)

    def process_outputs(
        self,
        parsed_samples: list[dict],
        outputs: list[RequestOutput],
        total_time: float,
    ) -> list[dict]:
        """Decode predictions, calculate SER, and save summary results."""
        all_results = []
        group_stats = {}

        for sample, output in zip(parsed_samples, outputs, strict=False):
            pred_ids = list(output.outputs[0].token_ids)[: sample["target_length"]]
            pred_plain = eval_utils.decode_prediction(pred_ids, self.config)
            true_plain = sample["true_plain"]

            ser, wrong_spaces = eval_utils.calculate_ser(true_plain, pred_plain)

            result_dict = {
                "index": sample["index"],
                "redundancy": sample["redundancy"],
                "ser": ser,
                "wrong_spaces": wrong_spaces,
            }
            all_results.append(result_dict)

            cipher_length = len(sample["raw_cipher_ids"])
            bucket = eval_utils.closest_n(cipher_length)
            key = (bucket, sample["redundancy"])
            if key not in group_stats:
                group_stats[key] = {"total_ser": 0.0, "count": 0}
            group_stats[key]["total_ser"] += ser
            group_stats[key]["count"] += 1

        # Prepare summary data for the new file
        summary_data = {
            "model_path": self.model_path,
            "total_skipped_ciphers": self.skipped_count,
            "group_results": [],
        }

        for (n, redundancy), stats in sorted(group_stats.items()):
            count = stats["count"]
            avg_ser = stats["total_ser"] / count

            logger.info(
                f"  N={n:>5}  μ={redundancy:>3}  count={count:>3}  avg_ser={avg_ser:.4f}",
            )

            summary_data["group_results"].append(
                {
                    "n": n,
                    "redundancy": redundancy,
                    "count": count,
                    "avg_ser": round(avg_ser, 4),
                },
            )

        # Save to avg_ser_scores.json
        with open(self.summary_scores_path, "w") as f:
            json.dump(summary_data, f, indent=4)

        logger.info(f"Summary scores saved to {self.summary_scores_path}")
        return all_results


def main() -> None:
    """Entry point for the evaluation script. Parses command-line arguments and initiates the evaluation process."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--spaces", action="store_true")
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()

    evaluator = VLLMCipherEvaluator(model_path=args.model_path, use_spaces=args.spaces)
    evaluator.run()


if __name__ == "__main__":
    main()
