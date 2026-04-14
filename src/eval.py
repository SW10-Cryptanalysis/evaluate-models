import argparse
import json
import logging
import time
from pathlib import Path
import torch
from datasets import load_from_disk
from vllm import LLM, SamplingParams
from easy_logging import EasyFormatter

from src.classes.config import Config
from src import eval_utils

handler = logging.StreamHandler()
handler.setFormatter(EasyFormatter())
logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class VLLMCipherEvaluator:
    """
    Orchestrates the evaluation of a Causal LM using vLLM for high-throughput
    decoding of long homophonic substitutions on Ada Lovelace architecture.
    """

    def __init__(self, model_path: str, use_spaces: bool):
        self.model_path = model_path
        self.config = Config()
        self.config.use_spaces = use_spaces
        self.config.load_homophones()

        self.output_log_path = Path(self.model_path) / "evaluation_results.jsonl"
        self.dataset = load_from_disk(self.config.tokenized_dir / "Test")
        self.allowed_token_ids = eval_utils.build_allowed_token_ids(self.config)

        # Detect available L4 GPUs for Tensor Parallelism
        self.world_size = torch.cuda.device_count()
        if self.world_size == 0:
            raise RuntimeError("vLLM requires CUDA devices, but none were found.")

    def _build_logits_processor(self, vocab_size: int):
        """
        Constructs a highly optimized tensor-based logits processor.
        Forces the model to only output valid plaintext characters, collapsing
        the non-determinism of the homophonic mapping space.
        """
        allowed_tensor = torch.tensor(
            [t for t in self.allowed_token_ids if t < vocab_size], dtype=torch.long
        )

        def restrict_vocab(
            prompt_tokens, output_tokens, logits: torch.Tensor
        ) -> torch.Tensor:
            mask = torch.full_like(logits, float("-inf"))
            # Move tensor to the current logits device to avoid runtime clashes
            local_allowed = allowed_tensor.to(logits.device)
            mask.scatter_(0, local_allowed, 0.0)
            return logits + mask

        return restrict_vocab

    def parse_samples(self):
        """Extract prompt token IDs (up to SEP) and target lengths."""
        parsed_data = []
        for index, item in enumerate(self.dataset):
            all_ids = item["input_ids"]
            try:
                sep_idx = all_ids.index(self.config.sep_token_id)
                # Model receives up to and including [SEP] to trigger Mapping Logic
                prompt_ids = all_ids[: sep_idx + 1]
                raw_cipher_ids = all_ids[1:sep_idx]
            except ValueError:
                continue

            target_length = len(raw_cipher_ids)
            if target_length > 0:
                parsed_data.append(
                    {
                        "index": index,
                        "prompt_ids": prompt_ids,
                        "raw_cipher_ids": raw_cipher_ids,
                        "true_plain": item["raw_plaintext"],
                        "redundancy": int(item["redundancy"]),
                        "target_length": target_length,
                    }
                )
        return parsed_data

    def run(self):
        logger.info(f"Initializing vLLM across {self.world_size} L4 GPUs...")

        llm = LLM(
            model=self.model_path,
            tensor_parallel_size=self.world_size,
            dtype="bfloat16",
            max_model_len=self.config.max_context,
            enforce_eager=False,  # Use CUDA graphs
            gpu_memory_utilization=0.95,  # Maximize memory for PagedAttention
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

        logits_processor_fn = self._build_logits_processor(vocab_size)
        parsed_samples = self.parse_samples()

        # Sort by target length for efficient continuous batching in vLLM
        parsed_samples.sort(key=lambda x: x["target_length"])

        prompts = []
        sampling_params_list = []

        for sample in parsed_samples:
            prompts.append({"prompt_token_ids": sample["prompt_ids"]})
            # Equal Loss Weighting mirror: We force exact target_length generation
            sp = SamplingParams(
                temperature=0.0,
                max_tokens=sample["target_length"],
                min_tokens=sample["target_length"],
                logits_processors=[logits_processor_fn],
            )
            sampling_params_list.append(sp)

        logger.info(f"Launching batched inference for {len(prompts)} sequences...")
        start_time = time.perf_counter()

        # vLLM handles all batching and multiprocessing automatically
        outputs = llm.generate(prompts, sampling_params=sampling_params_list)

        generation_time = time.perf_counter() - start_time
        logger.info(f"Inference complete in {generation_time:.2f} seconds.")

        return self.process_outputs(parsed_samples, outputs, generation_time)

    def process_outputs(self, parsed_samples, outputs, total_time):
        all_results = []
        total_ser = 0.0
        group_stats = {}

        # Ensure we zip correctly, matching inputs to vLLM outputs
        for sample, output in zip(parsed_samples, outputs):
            pred_ids = list(output.outputs[0].token_ids)
            pred_plain = eval_utils.decode_prediction(pred_ids, self.config)
            true_plain = sample["true_plain"]

            ser = eval_utils.calculate_ser(true_plain, pred_plain)

            result_dict = {
                "index": sample["index"],
                "redundancy": sample["redundancy"],
                "ciphertext": eval_utils.decode_ciphertext(
                    sample["raw_cipher_ids"], self.config
                ),
                "plaintext": true_plain,
                "predicted_plaintext": pred_plain,
                "ser": ser,
            }
            all_results.append(result_dict)
            total_ser += ser

            # Bucketing logic for statistical separation
            cipher_length = len(sample["raw_cipher_ids"])
            bucket = eval_utils.closest_n(cipher_length)
            key = (bucket, sample["redundancy"])
            if key not in group_stats:
                group_stats[key] = {"total_ser": 0.0, "count": 0}
            group_stats[key]["total_ser"] += ser
            group_stats[key]["count"] += 1

        all_results.sort(key=lambda r: r["index"])

        processed_count = len(all_results)
        global_avg_ser = total_ser / processed_count if processed_count else 0.0

        with open(self.output_log_path, "w") as f:
            for result in all_results:
                f.write(json.dumps(result) + "\n")
            f.write(
                json.dumps(
                    {
                        "type": "summary_global",
                        "processed_count": processed_count,
                        "avg_ser": round(global_avg_ser, 6),
                        "total_inference_time": round(total_time, 2),
                    }
                )
                + "\n"
            )

            for (n, redundancy), stats in sorted(group_stats.items()):
                count = stats["count"]
                avg = stats["total_ser"] / count
                logger.info(
                    f"  N={n:>5}  μ={redundancy:>3}  count={count:>3}  avg_ser={avg:.4f}"
                )
                f.write(
                    json.dumps(
                        {
                            "type": "summary_group",
                            "n": n,
                            "redundancy": redundancy,
                            "count": count,
                            "avg_ser": round(avg, 6),
                        }
                    )
                    + "\n"
                )

        logger.info(f"Results written to {self.output_log_path}")
        return all_results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spaces", action="store_true")
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()

    evaluator = VLLMCipherEvaluator(model_path=args.model_path, use_spaces=args.spaces)
    evaluator.run()


if __name__ == "__main__":
    main()
