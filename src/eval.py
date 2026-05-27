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
    """Evaluator class that uses vLLM to assess cipher decryption performance via mapping predictions."""

    def __init__(self, model_path: str, use_spaces: bool, mapping: bool) -> None:
        """Initialize the evaluator with model path and configuration."""
        self.model_path = model_path
        self.config = EvalConfig.from_model_path(model_path, use_spaces, mapping)

        self.output_log_path = Path(self.model_path) / "evaluation_results.jsonl"
        self.stats_log_path = Path(self.model_path) / "evaluation_stats.json"
        self.dataset = load_from_disk(self.config.tokenized_dir / "Test")
        self.allowed_token_ids = eval_utils.build_allowed_token_ids(self.config)

        self.skipped_count = 0

        self.world_size = torch.cuda.device_count()
        if self.world_size == 0:
            raise RuntimeError("vLLM requires CUDA devices, but none were found.")

    def parse_samples(self) -> list[dict]:
        """Extract prompt token IDs and target lengths based on the evaluation mode."""
        parsed_data = []
        self.skipped_count = 0
        for index, item in enumerate(self.dataset):
            all_ids = item["input_ids"] # type: ignore
            try:
                sep_idx = all_ids.index(self.config.sep_token_id)
                prompt_ids = all_ids[: sep_idx + 1]
                raw_cipher_ids = all_ids[1:sep_idx]
            except ValueError:
                continue

            if self.config.mapping:
                target_length = len(set(raw_cipher_ids))
            else:
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
                        "true_plain": item["raw_plaintext"], # type: ignore
                        "redundancy": int(item["redundancy"]), # type: ignore
                        "target_length": target_length,
                    },
                )
        return parsed_data

    def derive_plaintext(self, raw_cipher_ids: list[int], pred_ids: list[int]) -> str:
        """Derives the plaintext by constructing a key from the predicted mapping and applying it to the ciphertext."""
        unique_cipher_ids = sorted(list(set(raw_cipher_ids)))

        mapping_key = dict(zip(unique_cipher_ids, pred_ids, strict=False))

        decrypted_ids = [mapping_key.get(cid, cid) for cid in raw_cipher_ids]

        return eval_utils.decode_prediction(decrypted_ids, self.config)

    def run(self) -> list[dict]:
        """Main method to execute the evaluation process initializing vLLM, running inference, and processing outputs."""
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
            self.dataset,  # type: ignore
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

    def _process_entry(self, sample: dict, output: RequestOutput) -> dict:
        """Parse data and calculates SMER for a individual line entry."""
        pred_ids = list(output.outputs[0].token_ids)[: sample["target_length"]]
        if self.config.mapping:
            pred_plain = self.derive_plaintext(sample["raw_cipher_ids"], pred_ids)
        else:
            pred_plain = eval_utils.decode_prediction(pred_ids, self.config)
        true_plain = sample["true_plain"]

        ser, wrong_spaces = eval_utils.calculate_ser(true_plain, pred_plain)

        return {
            "index": sample["index"],
            "redundancy": sample["redundancy"],
            "ciphertext": eval_utils.decode_ciphertext(
                sample["raw_cipher_ids"],
                self.config,
            ),
            "plaintext": true_plain,
            "predicted_plaintext": pred_plain,
            "ser": ser,
            "wrong_spaces": wrong_spaces,
        }

    def _write_log(
        self,
        all_results: list[dict],
        evaluation_stats: dict,
        group_stats: dict,
        total_stats: dict,
    ) -> None:
        """Write results to output log file."""
        total_ser = total_stats["total_ser"]
        total_wrong_spaces = total_stats["wrong_spaces"]
        total_time = total_stats["total_time"]

        processed_count = len(all_results)
        global_avg_ser = total_ser / processed_count if processed_count else 0.0
        global_avg_wrong_spaces = (
            total_wrong_spaces / processed_count if processed_count else 0
        )
        with open(self.output_log_path, "w") as f:
            for result in all_results:
                f.write(json.dumps(result) + "\n")
            f.write(
                json.dumps(
                    {
                        "type": "summary_global",
                        "processed_count": processed_count,
                        "global_avg_ser": round(global_avg_ser, 4),
                        "global_avg_wrong_spaces": round(global_avg_wrong_spaces, 4),
                        "total_inference_time": round(total_time, 2),
                    },
                )
                + "\n",
            )

            for (n, redundancy), stats in sorted(group_stats.items()):
                count = stats["count"]
                avg = stats["total_ser"] / count
                avg_wrong_spaces = stats["wrong_spaces"] / count

                log_str = f"  N={n:>5}  μ={redundancy:>3}  count={count:>3}  avg_ser={avg:.4f}"
                logger.info(log_str)

                evaluation_stats["group_logs"].append(log_str)

                f.write(
                    json.dumps(
                        {
                            "type": "summary_group",
                            "n": n,
                            "redundancy": redundancy,
                            "count": count,
                            "avg_ser": round(avg, 4),
                            "avg_wrong_spaces": round(avg_wrong_spaces, 4),
                        },
                    )
                    + "\n",
                )

    def process_outputs(
        self,
        parsed_samples: list[dict],
        outputs: list[RequestOutput],
        total_time: float,
    ) -> list[dict]:
        """Decode predicted keys, apply to ciphertexts, calculate SER, and aggregate results with statistical bucketing."""
        all_results = []
        total_ser = 0.0
        total_wrong_spaces = 0
        group_stats = {}

        for sample, output in zip(parsed_samples, outputs, strict=False):
            result_dict = self._process_entry(sample, output)
            all_results.append(result_dict)
            total_ser += result_dict["ser"]
            total_wrong_spaces += result_dict["wrong_spaces"]

            cipher_length = len(sample["raw_cipher_ids"])
            bucket = eval_utils.closest_n(cipher_length)
            key = (bucket, sample["redundancy"])
            if key not in group_stats:
                group_stats[key] = {"total_ser": 0.0, "wrong_spaces": 0, "count": 0}
            group_stats[key]["total_ser"] += result_dict["ser"]
            group_stats[key]["wrong_spaces"] += result_dict["wrong_spaces"]
            group_stats[key]["count"] += 1

        all_results.sort(key=lambda r: r["index"])

        evaluation_stats = {"skipped_count": self.skipped_count, "group_logs": []}

        total_stats = {
            "total_ser": total_ser,
            "wrong_spaces": total_wrong_spaces,
            "total_time": total_time,
        }

        self._write_log(all_results, evaluation_stats, group_stats, total_stats)
        logger.info(f"Results written to {self.output_log_path}")

        with open(self.stats_log_path, "w") as sf:
            json.dump(evaluation_stats, sf, indent=4)
        logger.info(f"Stats written to {self.stats_log_path}")

        return all_results


def main() -> None:
    """Entry point for the evaluation script."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--spaces", action="store_true")
    parser.add_argument("--mapping", action="store_true")
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()

    evaluator = VLLMCipherEvaluator(
        model_path=args.model_path,
        use_spaces=args.spaces,
        mapping=args.mapping,
    )
    evaluator.run()


if __name__ == "__main__":
    main()
