import logging
from pathlib import Path

from src.classes.config import EvalConfig

_N_BUCKETS: list[int] = [350, 400, 450, 600, 800, 1000, 2000, 4000, 6000, 8000, 10000]


def closest_n(length: int) -> int:
    """Return the N bucket whose value is closest to length."""
    return min(_N_BUCKETS, key=lambda n: abs(n - length))


def build_allowed_token_ids(config: EvalConfig) -> list[int]:
    """Build the list of token IDs corresponding to latin alphabet characters a-z.

    This collapses the symbol space for dynamic key-recovery.
    """
    ids = [
        config.char_offset + (ord(char) - ord("a"))
        for char in "abcdefghijklmnopqrstuvwxyz"
    ]
    if config.use_spaces:
        ids.append(config.space_token_id)
    return ids


def decode_prediction(ids: list[int], config: EvalConfig) -> str:
    """Convert model token IDs back into a plaintext string based on config."""
    chars = []
    for idx in ids:
        if config.use_spaces and idx == config.space_token_id:
            chars.append("_")
        elif idx >= config.char_offset:
            chars.append(chr(idx - config.char_offset + ord("a")))
        else:
            chars.append("?")
    return "".join(chars)  # type: ignore


def decode_ciphertext(ids: list[int], config: EvalConfig) -> str:
    """Convert integer cipher IDs back to a space-separated string."""
    excluded = {config.bos_token_id, config.sep_token_id}
    return " ".join(str(idx) for idx in ids if idx not in excluded)


def calculate_ser(true_plain: str, pred_plain: str) -> tuple[float, int]:
    """Calculate the Symbol Error Rate (SER), explicitly ignoring spaces."""
    if not true_plain:
        raise ValueError("True plaintext is empty, cannot calculate SER.")

    mismatches = 0
    scored_symbols = 0
    wrong_spaces = 0

    for t, p in zip(true_plain, pred_plain, strict=True):
        if t in ("_", " "):
            if p not in ("_", " "):
                wrong_spaces += 1
            continue

        scored_symbols += 1
        if t != p:
            mismatches += 1

    if scored_symbols == 0:
        return (0.0, 0)

    return (mismatches / scored_symbols, wrong_spaces)


def _check_output_file(output_log_path: Path) -> list[str]:
    """Check if the output log file exists and return a warning if it will be overwritten."""
    if output_log_path.exists():
        return [f"Output file {output_log_path} exists and will be OVERWRITTEN."]
    return []


def _check_vocab_bounds(allowed_token_ids: list[int], vocab_size: int) -> list[str]:
    """Ensure allowed token IDs do not exceed the model's vocabulary size."""
    out_of_vocab = [tid for tid in allowed_token_ids if tid >= vocab_size]
    if out_of_vocab:
        return [f"Allowed token IDs exceed vocab size ({vocab_size}): {out_of_vocab}."]
    return []


def _check_dataset_format(
    dataset: list[dict],
    config: EvalConfig,
    num_samples: int = 10,
) -> list[str]:
    """Verify the mandatory dataset format (BOS, SEP, EOS tokens) on a subset of data."""
    errors = []
    dataset_errors = 0
    n_probe = min(num_samples, len(dataset))

    for i in range(n_probe):
        sample = dataset[i]["input_ids"]
        if sample[0] != config.bos_token_id:
            errors.append(
                f"Sample 0: expected bos_token_id={config.bos_token_id}, got {sample[0]}.",
            )
            dataset_errors += 1
        if sample[-1] != config.eos_token_id:
            errors.append(
                f"Sample {i}: expected eos_token_id={config.eos_token_id}, got {sample[-1]}.",
            )
            dataset_errors += 1
        if sample.count(config.sep_token_id) != 1:
            errors.append(
                f"Sample {i}: expected exactly 1 sep_token_id={config.sep_token_id}, got {sample.count(config.sep_token_id)}.",
            )
            dataset_errors += 1

    if dataset_errors > 0:
        errors.append(
            f"Found {dataset_errors} dataset format errors. Token layout corrupt.",
        )

    return errors


def run_preflight_checks(
    config: EvalConfig,
    dataset: list[dict],
    allowed_token_ids: list[int],
    output_log_path: Path,
    vocab_size: int,
    logger: logging.Logger,
) -> None:
    """Sanity checks to guarantee the Mandatory Training Objective format is intact."""
    logger.info("Running preflight checks...")

    warnings = _check_output_file(output_log_path)

    errors = []
    errors.extend(_check_vocab_bounds(allowed_token_ids, vocab_size))
    errors.extend(_check_dataset_format(dataset, config))

    for w in warnings:
        logger.warning(f"PREFLIGHT WARNING: {w}")

    if errors:
        for e in errors:
            logger.error(f"PREFLIGHT ERROR: {e}")
        raise RuntimeError("Preflight checks failed. Aborting evaluation.")

    logger.info("Preflight checks passed.")
