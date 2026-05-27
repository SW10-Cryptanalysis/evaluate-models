import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

from easy_logging import EasyFormatter

# Set up logging to match your project's style
handler = logging.StreamHandler()
handler.setFormatter(EasyFormatter())
logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def get_bucket_range(length: int) -> str:
    """
    Returns the string label for the bucket a given length falls into.
    Clamps lengths >= 9000 into the final '9000-10000' bucket so exactly 10 buckets are formed.
    """
    if length >= 9000:
        return "9000-10000"

    lower_bound = (length // 1000) * 1000
    upper_bound = lower_bound + 1000
    return f"{lower_bound}-{upper_bound}"


def generate_bucket_stats(input_log_path: Path, output_stats_path: Path) -> None:
    if not input_log_path.exists():
        logger.error(f"Input file not found: {input_log_path}")
        return

    # Dictionary to hold the running sum of SER and the count for each bucket
    # Keys will be the bucket string, e.g., "0-1000"
    bucket_data = defaultdict(lambda: {"total_ser": 0.0, "count": 0})

    processed_count = 0

    with open(input_log_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)

            # Skip summary log lines at the end of the JSONL file
            if "type" in data and data["type"].startswith("summary_"):
                continue

            # In your eval script, the length of `plaintext` maps 1:1 with target cipher length
            # You could also use len(data["ciphertext"].split())
            cipher_length = len(data["plaintext"])
            ser = data["ser"]

            bucket = get_bucket_range(cipher_length)

            bucket_data[bucket]["total_ser"] += ser
            bucket_data[bucket]["count"] += 1
            processed_count += 1

    # Format the final stats payload
    final_stats = {"processed_count": processed_count, "buckets": {}}

    # Ensure buckets are sorted chronologically in the output JSON
    # We sort by the lower bound integer of the bucket string
    sorted_buckets = sorted(bucket_data.items(), key=lambda x: int(x[0].split("-")[0]))

    for bucket, stats in sorted_buckets:
        count = stats["count"]
        avg_ser = stats["total_ser"] / count if count > 0 else 0.0

        final_stats["buckets"][bucket] = {"count": count, "avg_ser": round(avg_ser, 4)}

        logger.info(f"Bucket {bucket:>11} | count={count:>5} | avg_ser={avg_ser:.4f}")

    # Write out the results
    output_stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_stats_path, "w") as out_f:
        json.dump(final_stats, out_f, indent=4)

    logger.info(f"Successfully processed {processed_count} evaluation results.")
    logger.info(f"1000-character bucket stats written to {output_stats_path}")


def main() -> None:
    input_file = Path("jsonl_files/evaluation_results.jsonl")
    output_file = Path("jsonl_files/evaluation_stats_1000_buckets.json")

    generate_bucket_stats(input_file, output_file)


if __name__ == "__main__":
    main()
