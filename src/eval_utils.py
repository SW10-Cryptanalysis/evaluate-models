import logging
from pathlib import Path

from src.classes.config import EvalConfig

_N_BUCKETS: list[int] = [350, 400, 450, 600, 800, 1000, 2000, 4000, 6000, 8000, 10000]


def closest_n(length: int) -> int:
    """Return the N bucket whose value is closest to length."""
    return min(_N_BUCKETS, key=lambda n: abs(n - length))


def build_allowed_token_ids(config: EvalConfig) -> list[int]:
    """
    Build the list of token IDs corresponding to latin alphabet characters a-z.
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
    return "".join(chars)


def decode_ciphertext(ids: list[int], config: EvalConfig) -> str:
    """Convert integer cipher IDs back to a space-separated string."""
    excluded = {config.bos_token_id, config.sep_token_id}
    return " ".join(str(idx) for idx in ids if idx not in excluded)


def calculate_ser(true_plain: str, pred_plain: str) -> float:
    """Calculate the Symbol Error Rate (SER)."""
    if not true_plain:
        raise ValueError("True plaintext is empty, cannot calculate SER.")
    mismatches = sum(t != p for t, p in zip(true_plain, pred_plain, strict=True))
    return mismatches / len(true_plain)


def run_preflight_checks(
    config: EvalConfig,
    dataset,
    allowed_token_ids: list[int],
    output_log_path: Path,
    vocab_size: int,
    logger: logging.Logger,
) -> None:
    """Sanity checks to guarantee the Mandatory Training Objective format is intact."""
    logger.info("Running preflight checks...")
    errors: list[str] = []
    warnings: list[str] = []

    if output_log_path.exists():
        warnings.append(
            f"Output file {output_log_path} exists and will be OVERWRITTEN."
        )

    out_of_vocab = [tid for tid in allowed_token_ids if tid >= vocab_size]
    if out_of_vocab:
        errors.append(
            f"Allowed token IDs exceed vocab size ({vocab_size}): {out_of_vocab}."
        )

    dataset_errors = 0
    n_probe = min(10, len(dataset))
    for i in range(n_probe):
        sample = dataset[i]["input_ids"]
        if sample[0] != config.bos_token_id:
            dataset_errors += 1
            if dataset_errors == 1:
                errors.append(
                    f"Sample 0: expected bos_token_id={config.bos_token_id}, got {sample[0]}."
                )
        if sample[-1] != config.eos_token_id:
            dataset_errors += 1
        if sample.count(config.sep_token_id) != 1:
            dataset_errors += 1

    if dataset_errors > 0:
        errors.append(
            f"Found {dataset_errors} dataset format errors. Token layout corrupt."
        )

    for w in warnings:
        logger.warning(f"PREFLIGHT WARNING: {w}")
    if errors:
        for e in errors:
            logger.error(f"PREFLIGHT ERROR: {e}")
        raise RuntimeError("Preflight checks failed. Aborting evaluation.")

    logger.info("Preflight checks passed.")
