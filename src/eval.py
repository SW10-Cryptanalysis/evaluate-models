import argparse
import json
import logging
import os
import shutil
import time
from pathlib import Path

import torch
import torch.multiprocessing as mp
from datasets import Dataset, DatasetDict, load_from_disk
from easy_logging import EasyFormatter
from transformers import AutoModelForCausalLM, PreTrainedModel

from src.classes.config import Config

handler = logging.StreamHandler()
handler.setFormatter(EasyFormatter())
logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

_N_BUCKETS: list[int] = [350, 400, 450, 600, 800, 1000, 2000, 4000, 6000, 8000, 10000]


def _closest_n(length: int) -> int:
    """Return the N bucket whose value is closest to length."""
    return min(_N_BUCKETS, key=lambda n: abs(n - length))


class CipherEvaluator:
    """Orchestrates the evaluation of a Causal LM on cipher decoding tasks."""

    def __init__(
        self,
        model_path: str,
        use_spaces: bool,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        """Initialize state, sets up configuration, and loads required assets.

        Args:
            model_path (str): The path to the model checkpoint.
            use_spaces (bool): Whether to use spaces or not.
            rank (int): The GPU rank for this worker process.
            world_size (int): Total number of GPU worker processes.

        """
        self.model_path = model_path
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(
            f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
        )

        self.config = Config()
        self.config.use_spaces = use_spaces
        self.config.load_homophones()

        self.output_log_path = Path(self.model_path) / "evaluation_results.jsonl"

        self.model: PreTrainedModel = self._load_model()
        self.dataset = self._load_dataset()
        self.allowed_token_ids = self._build_allowed_token_ids()

        # Build once and pin to device — never rebuild per sample.
        self.logits_processor = self._make_logits_processor()

        # Preflight is expensive; only run on rank 0.
        if rank == 0:
            self._run_preflight_checks()

    def _load_model(self) -> PreTrainedModel:
        """Instantiate the model onto this worker's GPU.

        Returns:
            PreTrainedModel: The loaded model.

        """
        logger.info(f"[Rank {self.rank}] Loading model from {self.model_path}...")
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            dtype=dtype,
            device_map=self.device,
        )
        model.config.use_cache = True
        model.eval()
        logger.info(
            f"[Rank {self.rank}] Loaded {type(model).__name__} on {self.device}"
        )
        return model

    def _load_dataset(self) -> Dataset | DatasetDict:
        """Retrieve the pre-tokenized dataset from the configured directory.

        Returns:
            Dataset | DatasetDict: The loaded dataset.

        """
        test_arrow_path = self.config.tokenized_dir / "Test"
        return load_from_disk(test_arrow_path)

    def _build_allowed_token_ids(self) -> list[int]:
        """Build the list of token IDs corresponding to latin alphabet characters a-z,
        plus the space token if use_spaces is enabled.

        Returns:
            list[int]: The 26 (or 27) valid plaintext token IDs.

        """
        ids = [
            self.config.char_offset + (ord(char) - ord("a"))
            for char in "abcdefghijklmnopqrstuvwxyz"
        ]
        if self.config.use_spaces:
            ids.append(self.config.space_token_id)
        return ids

    def _make_logits_processor(self) -> list:
        """Build a logits-masking processor pinned to this worker's device.

        The mask is constructed once and stays on the GPU for the lifetime of
        the evaluator — no per-batch or per-sample rebuilding.

        Returns:
            list: A single-element list containing the processor callable.

        """
        vocab_size = self.model.config.vocab_size
        base_mask = torch.full((vocab_size,), float("-inf"), device=self.device)
        for token_id in self.allowed_token_ids:
            if token_id < vocab_size:
                base_mask[token_id] = 0.0

        def _restrict_to_alphabet(
            input_ids: torch.Tensor, scores: torch.Tensor
        ) -> torch.Tensor:
            return scores + base_mask

        return [_restrict_to_alphabet]

    def decode_prediction(self, ids: list[int]) -> str:
        """Convert model token IDs back into a plaintext string based on config.

        Token IDs that do not correspond to a valid latin alphabet character or
        (when use_spaces is enabled) the space token are replaced with '?' so
        they register as errors in SER calculation.

        Args:
            ids (list[int]): The list of token IDs to decode.

        Returns:
            str: The decoded plaintext string.

        """
        chars = []
        for idx in ids:
            if self.config.use_spaces and idx == self.config.space_token_id:
                chars.append("_")
            elif idx >= self.config.char_offset:
                chars.append(chr(idx - self.config.char_offset + ord("a")))
            else:
                chars.append("?")
        return "".join(chars)

    def decode_ciphertext(self, ids: list[int]) -> str:
        """Convert integer cipher IDs back to a space-separated string.

        Args:
            ids (list[int]): The list of token IDs to decode.

        Returns:
            str: The decoded string.

        """
        excluded = {self.config.bos_token_id, self.config.sep_token_id}
        return " ".join(str(idx) for idx in ids if idx not in excluded)

    def _parse_sample(self, item: dict, index: int) -> tuple | None:
        """Extract inference-ready components from a raw dataset sample.

        Args:
            item (dict): Raw dataset item.
            index (int): Global dataset index (used for logging and result tagging).

        Returns:
            tuple | None: (index, input_ids, raw_cipher_ids, true_plain,
                           redundancy, target_length) or None if invalid.

        """
        all_ids = item["input_ids"]
        true_plain = item["raw_plaintext"]
        redundancy = int(item["redundancy"])

        try:
            sep_idx = all_ids.index(self.config.sep_token_id)
            input_ids = all_ids[: sep_idx + 1]
            raw_cipher_ids = all_ids[1:sep_idx]
        except ValueError:
            logger.warning(
                f"[Rank {self.rank}] Sample {index} missing SEP token. Skipping."
            )
            return None

        target_length = len(raw_cipher_ids)
        if target_length == 0:
            logger.warning(
                f"[Rank {self.rank}] Sample {index} has empty ciphertext. Skipping."
            )
            return None

        return (index, input_ids, raw_cipher_ids, true_plain, redundancy, target_length)

    def _run_batch(self, batch: list[tuple]) -> list[dict | None]:
        """Run batched generation over a list of parsed samples.

        All inputs are left-padded to the same length so that the generated
        tokens always start at a fixed offset in the output tensor.
        The max target_length in the batch drives max_new_tokens so that
        shorter-target samples just have their output sliced to size.

        Args:
            batch: List of tuples from _parse_sample —
                   (index, input_ids, raw_cipher_ids, true_plain,
                    redundancy, target_length).

        Returns:
            list[dict | None]: One result dict per sample, or None on failure.

        """
        max_input_len = max(len(p[1]) for p in batch)
        max_target_len = max(p[5] for p in batch)

        padded_inputs: list[list[int]] = []
        attention_masks: list[list[int]] = []
        for _, input_ids, *_ in batch:
            pad_len = max_input_len - len(input_ids)
            padded_inputs.append([0] * pad_len + input_ids)
            attention_masks.append([0] * pad_len + [1] * len(input_ids))

        input_tensor = torch.tensor(padded_inputs, device=self.device)
        mask_tensor = torch.tensor(attention_masks, device=self.device)

        start_time = time.perf_counter()
        with torch.no_grad():
            output = self.model.generate(
                input_tensor,
                attention_mask=mask_tensor,
                max_new_tokens=max_target_len,
                min_new_tokens=max_target_len,
                do_sample=False,
                use_cache=True,
                pad_token_id=0,
                logits_processor=self.logits_processor,
            )  # type: ignore
        generation_time = time.perf_counter() - start_time
        per_sample_time = round(generation_time / len(batch), 4)

        results: list[dict | None] = []
        for i, (
            index,
            input_ids,
            raw_cipher_ids,
            true_plain,
            redundancy,
            target_length,
        ) in enumerate(batch):
            # Generated tokens always start at max_input_len (left-pad makes
            # this a fixed offset regardless of the original input length).
            pred_ids = output[i][max_input_len : max_input_len + target_length].tolist()

            if len(pred_ids) != target_length:
                logger.warning(
                    f"[Rank {self.rank}] Sample {index}: expected {target_length} "
                    f"tokens but got {len(pred_ids)}. Skipping."
                )
                results.append(None)
                continue

            pred_plain = self.decode_prediction(pred_ids)
            results.append(
                {
                    "index": index,
                    "redundancy": redundancy,
                    "ciphertext": self.decode_ciphertext(raw_cipher_ids),
                    "plaintext": true_plain,
                    "predicted_plaintext": pred_plain,
                    "ser": self._ser(true_plain, pred_plain),
                    "inference_time_seconds": per_sample_time,
                }
            )

        return results

    def _ser(self, true_plain: str, pred_plain: str) -> float:
        """Calculate the Symbol Error Rate between the true and predicted plaintexts.

        Args:
            true_plain (str): The ground-truth plaintext.
            pred_plain (str): The model-predicted plaintext.

        Returns:
            float: The SER score in [0.0, 1.0].

        Raises:
            ValueError: If true_plain and pred_plain differ in length, since both
                must be exactly target_length characters after decoding.

        """
        if not true_plain:
            return 0.0
        mismatches = sum(t != p for t, p in zip(true_plain, pred_plain, strict=True))
        return mismatches / len(true_plain)

    def run(self, indices: list[int], batch_size: int = 32) -> list[dict]:
        """Evaluate all assigned samples in sorted, batched order.

        Samples are sorted by target_length before batching so that samples in
        the same batch have similar generation lengths, minimising wasted
        padding-driven compute.

        On CUDA OOM the failing batch is retried sample-by-sample so that a
        single very-long sample never kills the whole run.

        Args:
            indices (list[int]): Dataset indices assigned to this rank.
            batch_size (int): Number of samples per GPU call.

        Returns:
            list[dict]: Result dicts for every successfully processed sample.

        """
        logger.info(f"[Rank {self.rank}] Parsing {len(indices)} samples...")
        parsed: list[tuple] = []
        for i in indices:
            p = self._parse_sample(self.dataset[i], i)  # type: ignore[index]
            if p is not None:
                parsed.append(p)

        # Sort ascending by target_length so adjacent batches have similar
        # generation lengths → less padding waste per batch.
        parsed.sort(key=lambda p: p[5])

        all_results: list[dict] = []
        total_ser = 0.0
        processed_count = 0
        num_batches = (len(parsed) + batch_size - 1) // batch_size

        for batch_num, batch_start in enumerate(range(0, len(parsed), batch_size)):
            batch = parsed[batch_start : batch_start + batch_size]

            try:
                batch_results = self._run_batch(batch)
            except torch.cuda.OutOfMemoryError:
                logger.warning(
                    f"[Rank {self.rank}] OOM on batch {batch_num} "
                    f"(size={len(batch)}, max_target={max(p[5] for p in batch)}). "
                    "Retrying sample-by-sample."
                )
                torch.cuda.empty_cache()
                batch_results = []
                for sample in batch:
                    try:
                        batch_results.extend(self._run_batch([sample]))
                    except torch.cuda.OutOfMemoryError:
                        logger.error(
                            f"[Rank {self.rank}] OOM on single sample {sample[0]}. "
                            "Skipping."
                        )
                        batch_results.append(None)
                        torch.cuda.empty_cache()

            for result in batch_results:
                if result is None:
                    continue
                all_results.append(result)
                total_ser += result["ser"]
                processed_count += 1

            if (batch_num + 1) % 20 == 0 or batch_num == num_batches - 1:
                avg_ser = total_ser / processed_count if processed_count else 0.0
                logger.info(
                    f"[Rank {self.rank}] Batch [{batch_num + 1}/{num_batches}] | "
                    f"Processed: {processed_count} | Avg SER: {avg_ser:.4f}"
                )

        logger.info(
            f"[Rank {self.rank}] Done. Processed {processed_count}/{len(indices)} samples."
        )
        return all_results

    def _run_preflight_checks(self) -> None:
        """Run sanity checks before evaluation begins.

        Verifies that the model vocabulary, config token IDs, dataset format,
        and output path are all consistent. Raises RuntimeError on any hard
        failure so a bad evaluation cannot silently proceed.

        """
        logger.info("Running preflight checks...")
        errors: list[str] = []
        warnings: list[str] = []

        # --- 1. Output file ---
        if self.output_log_path.exists():
            warnings.append(
                f"Output file already exists at {self.output_log_path}. "
                "Results will be OVERWRITTEN. Delete it first "
                "if you want to keep the old data."
            )

        # --- 2. Vocab size covers all allowed token IDs ---
        vocab_size: int = self.model.config.vocab_size
        out_of_vocab = [tid for tid in self.allowed_token_ids if tid >= vocab_size]
        if out_of_vocab:
            errors.append(
                f"The following allowed token IDs exceed the model vocab size "
                f"({vocab_size}): {out_of_vocab}. The model cannot generate "
                "these tokens — check that unique_homophones matches what the "
                "model was trained with."
            )

        # --- 3. Special token IDs don't collide with allowed token IDs ---
        special_ids = {
            "bos": self.config.bos_token_id,
            "sep": self.config.sep_token_id,
            "eos": self.config.eos_token_id,
        }
        allowed_set = set(self.allowed_token_ids)
        for name, tid in special_ids.items():
            if tid in allowed_set:
                errors.append(
                    f"Special token {name}_token_id={tid} collides with an "
                    "allowed plaintext token ID. Token ID layout is corrupt."
                )

        # --- 4. Spot-check dataset samples against config ---
        dataset_errors = 0
        n_probe = min(10, len(self.dataset))
        for i in range(n_probe):
            sample = self.dataset[i]["input_ids"]

            if sample[0] != self.config.bos_token_id:
                dataset_errors += 1
                if dataset_errors == 1:
                    errors.append(
                        f"Sample 0: first token is {sample[0]}, expected "
                        f"bos_token_id={self.config.bos_token_id}. The config "
                        "token IDs do not match the dataset — did you run "
                        "load_homophones() with the correct metadata file?"
                    )

            if sample[-1] != self.config.eos_token_id:
                dataset_errors += 1
                if dataset_errors <= 2:
                    errors.append(
                        f"Sample {i}: last token is {sample[-1]}, expected "
                        f"eos_token_id={self.config.eos_token_id}."
                    )

            sep_count = sample.count(self.config.sep_token_id)
            if sep_count != 1:
                dataset_errors += 1
                if dataset_errors <= 3:
                    errors.append(
                        f"Sample {i}: found {sep_count} SEP tokens, expected exactly 1."
                    )

        if dataset_errors > 3:
            errors.append(
                f"... and {dataset_errors - 3} more dataset format errors in "
                f"the first {n_probe} samples. The dataset is likely tokenized "
                "with different token IDs than the current config."
            )

        # --- 5. spaces flag vs dataset consistency ---
        n_space_probe = min(50, len(self.dataset))
        space_token_found = any(
            self.config.space_token_id in self.dataset[i]["input_ids"]
            for i in range(n_space_probe)
        )
        if self.config.use_spaces and not space_token_found:
            warnings.append(
                f"--spaces is set but space_token_id={self.config.space_token_id} "
                f"was not found in the first {n_space_probe} samples. "
                "The dataset may have been tokenized without spaces."
            )
        if not self.config.use_spaces and space_token_found:
            warnings.append(
                f"--spaces is NOT set but space_token_id={self.config.space_token_id} "
                f"was found in the dataset. "
                "The dataset may have been tokenized with spaces — "
                "consider passing --spaces."
            )

        for w in warnings:
            logger.warning(f"PREFLIGHT WARNING: {w}")

        if errors:
            for e in errors:
                logger.error(f"PREFLIGHT ERROR: {e}")
            raise RuntimeError(
                f"Preflight checks failed with {len(errors)} error(s). "
                "Aborting evaluation — fix the issues above before proceeding."
            )

        logger.info("Preflight checks passed.")


# ---------------------------------------------------------------------------
# Module-level worker — must be at module scope for mp.spawn to pickle it.
# ---------------------------------------------------------------------------


def _worker_fn(
    rank: int,
    world_size: int,
    model_path: str,
    use_spaces: bool,
    batch_size: int,
    tmp_dir: str,
) -> None:
    """Entrypoint for each GPU worker spawned by mp.spawn.

    Loads the model on GPU `rank`, processes every (rank)-th sample from the
    dataset, and writes results to a per-rank JSONL file in tmp_dir.

    Args:
        rank (int): GPU index for this worker.
        world_size (int): Total number of workers.
        model_path (str): Path to the model checkpoint.
        use_spaces (bool): Whether the dataset uses space tokens.
        batch_size (int): Samples per generate() call.
        tmp_dir (str): Directory for intermediate per-rank result files.

    """
    # Each spawned process needs its own log handler.
    _h = logging.StreamHandler()
    _h.setFormatter(EasyFormatter())
    logging.getLogger().addHandler(_h)
    logging.getLogger().setLevel(logging.INFO)

    evaluator = CipherEvaluator(
        model_path=model_path,
        use_spaces=use_spaces,
        rank=rank,
        world_size=world_size,
    )

    # Interleaved split: rank 0 → indices 0,4,8,…; rank 1 → 1,5,9,… etc.
    # This ensures each GPU sees a proportional mix of short/long samples.
    all_indices = list(range(len(evaluator.dataset)))
    my_indices = all_indices[rank::world_size]

    results = evaluator.run(my_indices, batch_size=batch_size)

    tmp_path = os.path.join(tmp_dir, f"results_rank{rank}.jsonl")
    with open(tmp_path, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    logger.info(f"[Rank {rank}] Saved {len(results)} results → {tmp_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Handle CLI arguments and acts as the entrypoint for execution."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--spaces", action="store_true")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help=(
            "Samples per generate() call per GPU. "
            "Increase for shorter sequences, decrease if you hit OOM. "
            "(default: 32)"
        ),
    )
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    if world_size == 0:
        logger.warning("No CUDA GPUs detected — falling back to single-process CPU.")
        world_size = 1

    logger.info(
        f"Launching evaluation: {world_size} GPU(s), batch_size={args.batch_size} per GPU."
    )

    tmp_dir = Path(args.model_path) / "_eval_tmp"
    tmp_dir.mkdir(exist_ok=True)
    output_log_path = Path(args.model_path) / "evaluation_results.jsonl"

    # ---- Spawn one worker process per GPU ----
    spawn_kwargs = dict(
        fn=_worker_fn,
        args=(
            world_size,
            args.model_path,
            args.spaces,
            args.batch_size,
            str(tmp_dir),
        ),
        nprocs=world_size,
        join=True,
    )
    if world_size > 1:
        mp.spawn(**spawn_kwargs)  # type: ignore
    else:
        # Single-GPU / CPU: run inline to avoid spawn overhead.
        _worker_fn(0, 1, args.model_path, args.spaces, args.batch_size, str(tmp_dir))

    # ---- Merge per-rank results ----
    logger.info("Merging results from all ranks...")
    all_results: list[dict] = []
    for rank in range(world_size):
        rank_file = tmp_dir / f"results_rank{rank}.jsonl"
        if not rank_file.exists():
            logger.warning(f"Missing result file for rank {rank}: {rank_file}")
            continue
        with open(rank_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    all_results.append(json.loads(line))

    # Restore original dataset order.
    all_results.sort(key=lambda r: r["index"])

    # ---- Compute statistics ----
    total_ser = 0.0
    group_stats: dict[tuple[int, int], dict[str, float | int]] = {}

    with open(output_log_path, "w") as f:
        for result in all_results:
            f.write(json.dumps(result) + "\n")
            total_ser += result["ser"]
            cipher_length = len(result["ciphertext"].split())
            bucket = _closest_n(cipher_length)
            redundancy = result["redundancy"]
            key = (bucket, redundancy)
            if key not in group_stats:
                group_stats[key] = {"total_ser": 0.0, "count": 0}
            group_stats[key]["total_ser"] += result["ser"]
            group_stats[key]["count"] += 1

    processed_count = len(all_results)
    if processed_count == 0:
        logger.warning("No samples were successfully processed.")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return

    global_avg_ser = total_ser / processed_count
    logger.info(f"DONE. Avg SER: {global_avg_ser:.4f} over {processed_count} samples.")

    with open(output_log_path, "a") as f:
        f.write(
            json.dumps(
                {
                    "type": "summary_global",
                    "processed_count": processed_count,
                    "avg_ser": round(global_avg_ser, 6),
                }
            )
            + "\n"
        )

        logger.info("Per-group SER results:")
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

    shutil.rmtree(tmp_dir, ignore_errors=True)
    logger.info(f"Results written to {output_log_path}")


if __name__ == "__main__":
    main()
