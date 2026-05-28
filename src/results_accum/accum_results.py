import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
from easy_logging import EasyFormatter

# Set up logger
handler = logging.StreamHandler()
handler.setFormatter(EasyFormatter())
logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class ResultsAggregator:
    """Recursively processes model evaluation metrics and generates clustered summaries."""

    def __init__(
        self, 
        base_dir: str | Path = "outputs", 
        target_lengths: List[int] = None
    ) -> None:
        """
        Initializes the aggregator with configuration rules.
        
        :param base_dir: The root folder path to scan for data logs.
        :param target_lengths: List of expected baseline cipher lengths to map targets to.
        """
        self.base_dir = Path(base_dir)
        
        # Fallback to default experimental baselines if none are supplied
        self.target_lengths = target_lengths or [
            350, 400, 450, 600, 800, 1000, 2000, 4000, 6000, 8000, 10000
        ]
        self.target_filename = "evaluation_results.jsonl"
        self.output_filename = "accum_results.jsonl"

    def find_nearest_baseline(self, actual_length: int) -> int:
        """Finds the closest target baseline length from the experimental design."""
        return min(self.target_lengths, key=lambda x: abs(x - actual_length))

    def process_evaluation_file(self, file_path: Path) -> None:
        """Reads a specific evaluation log file, bins values, and writes a local summary."""
        # Group data array buckets dynamically: {(binned_length, redundancy): [sers...]}
        data_groups: Dict[Tuple[int, int], List[float]] = defaultdict(list)

        logger.info(f"Processing: {file_path}")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                        
                        redundancy = record.get("redundancy")
                        ser = record.get("ser")
                        plaintext = record.get("plaintext", "")
                        
                        if redundancy is None or ser is None or not plaintext:
                            continue
                            
                        # Calculate string text scale via character metrics
                        actual_length = len(plaintext)
                        binned_length = self.find_nearest_baseline(actual_length)
                        
                        data_groups[(binned_length, redundancy)].append(float(ser))

                    except (json.JSONDecodeError, ValueError):
                        continue

            if not data_groups:
                logger.warning(f"No valid records found in {file_path}")
                return

            output_path = file_path.parent / self.output_filename
            
            with open(output_path, "w", encoding="utf-8") as out_f:
                for (length, redundancy) in sorted(data_groups.keys()):
                    ser_list = data_groups[(length, redundancy)]
                    
                    mean_ser = float(np.mean(ser_list))
                    median_ser = float(np.median(ser_list))
                    best_ser = float(np.min(ser_list))
                    
                    summary_record = {
                        "cipher_length": length,
                        "redundancy": redundancy,
                        "mean_ser": mean_ser,
                        "median_ser": median_ser,
                        "best_case_ser": best_ser,
                        "sample_size": len(ser_list)
                    }
                    
                    out_f.write(json.dumps(summary_record) + "\n")
                    
            logger.info(f"Successfully created clustered summary -> {output_path}")

        except Exception as e:
            logger.error(f"Failed to process file {file_path}: {e}")

    def run(self) -> None:
        """Executes the complete recursive discovery and aggregation engine pipeline."""
        if not self.base_dir.exists() or not self.base_dir.is_dir():
            logger.error(f"Target base outputs directory not found at: {self.base_dir.resolve()}")
            return

        eval_files = list(self.base_dir.rglob(self.target_filename))

        if not eval_files:
            logger.info(f"No files matching '{self.target_filename}' found.")
            return

        logger.info(f"Discovered {len(eval_files)} files. Commencing pipeline processing...")
        for eval_file in eval_files:
            self.process_evaluation_file(eval_file)
            
        logger.info("All aggregation profiles successfully compiled.")


if __name__ == "__main__":
    # Instantiate and trigger the system using standard parameters
    aggregator = ResultsAggregator(base_dir="outputs")
    aggregator.run()