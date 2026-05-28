from classes.config import EvalConfig
import eval_utils
from pathlib import Path
from datasets import load_from_disk
import json
import logging
from easy_logging import EasyFormatter

handler = logging.StreamHandler()
handler.setFormatter(EasyFormatter())
logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class BaseCipherEvaluator:
    """Base class containing shared logic for data parsing and metric calculation."""

    def __init__(self, model_path: str, use_spaces: bool, mapping: bool) -> None:
        """Initialize common evaluation properties and load the dataset."""
        self.model_path = model_path
        self.config = EvalConfig.from_model_path(model_path, use_spaces, mapping)

        self.output_log_path = Path(self.model_path) / "evaluation_results.jsonl"
        self.stats_log_path = Path(self.model_path) / "evaluation_stats.json"
        self.dataset = load_from_disk(self.config.tokenized_dir / "Test")
        self.allowed_token_ids = eval_utils.build_allowed_token_ids(self.config)

        self.skipped_count = 0

    def parse_samples(self) -> list[dict]:
        """Extract prompt token IDs and target lengths based on the evaluation mode."""
        parsed_data = []
        self.skipped_count = 0
        for index, item in enumerate(self.dataset):
            all_ids = item["input_ids"]  # type: ignore
            try:
                sep_idx = all_ids.index(self.config.sep_token_id)
                prompt_ids = all_ids[: sep_idx + 1]
                raw_cipher_ids = all_ids[1:sep_idx]
            except ValueError:
                continue

            target_length = (
                len(set(raw_cipher_ids)) if self.config.mapping else len(raw_cipher_ids)
            )
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
                        "true_plain": item["raw_plaintext"],  # type: ignore
                        "redundancy": int(item["redundancy"]),  # type: ignore
                        "target_length": target_length,
                    },
                )
        return parsed_data

    def derive_plaintext(self, raw_cipher_ids: list[int], pred_ids: list[int]) -> str:
        """Constructs a key from the predicted mapping and applies it to the ciphertext."""
        unique_cipher_ids = sorted(set(raw_cipher_ids))
        mapping_key = dict(zip(unique_cipher_ids, pred_ids, strict=False))
        decrypted_ids = [mapping_key.get(cid, cid) for cid in raw_cipher_ids]
        return eval_utils.decode_prediction(decrypted_ids, self.config)

    def _process_entry(self, sample: dict, pred_ids: list[int]) -> dict:
        """Calculates Sequence Error Rate (SER) for an individual line entry."""
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

    def process_outputs(
        self,
        parsed_samples: list[dict],
        outputs_list: list[list[int]],
        total_time: float,
    ) -> list[dict]:
        """Aggregate results with statistical bucketing and trigger log writing."""
        all_results = []
        total_ser = 0.0
        total_wrong_spaces = 0
        group_stats = {}

        for sample, pred_ids in zip(parsed_samples, outputs_list, strict=False):
            result_dict = self._process_entry(sample, pred_ids)
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

    def _write_log(
        self,
        all_results: list[dict],
        evaluation_stats: dict,
        group_stats: dict,
        total_stats: dict,
    ) -> None:
        """Write JSONL and JSON stat files to disk. Include your existing logic here."""
        pass
