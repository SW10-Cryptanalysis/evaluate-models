import argparse
import json
import logging
from pathlib import Path
from collections import Counter
from easy_logging import EasyFormatter

handler = logging.StreamHandler()
handler.setFormatter(EasyFormatter())
logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class StrictMapper:
    """Calculate the Strict Mapping Error Rate (SMER) for cryptanalysis.

    Attributes:
        model_path (Path): Directory containing the evaluation files.
        threshold (float): Minimum empirical frequency (0.0 to 1.0) required to
            consider a mapping stable. Defaults to 0.95.
        results_path (Path): Path to the input JSONL file containing model predictions.
        output_path (Path): Path where the calculated SMER metrics will be saved.

    """

    def __init__(self, model_path: str, threshold: float = 0.95) -> None:
        """Initialize the StrictMapper with paths and consistency constraints.

        Args:
            model_path (str): The root directory where 'evaluation_results.jsonl'
                is located and where output will be stored.
            threshold (float, optional): The noise-tolerance consistency threshold.
                Mappings falling below this frequency are treated as unresolved.
                Defaults to 0.95.

        """
        self.model_path = Path(model_path)
        self.threshold = threshold
        self.results_path = self.model_path / "evaluation_results.jsonl"
        self.output_path = self.model_path / "smer_results.jsonl"

    def calculate_smer(self) -> None:
        """Process evaluation results to compute the SMER per entry."""
        if not self.results_path.exists():
            logger.error(f"Input file not found: {self.results_path}")
            return

        output_data = []

        with open(self.results_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(f"Line {i} contains invalid JSON and was skipped.")
                    continue

                if "index" not in data:
                    continue

                result = self._process_entry(data, i)
                if result:
                    output_data.append(result)

        with open(self.output_path, "w", encoding="utf-8") as f:
            for entry in output_data:
                f.write(json.dumps(entry) + "\n")

        logger.info(f"Successfully processed {len(output_data)} entries.")
        logger.info(f"Results saved to: {self.output_path}")

    def _process_entry(self, data: dict, line_idx: int) -> dict | None:
        """Parse data and calculates SMER for a individual line entry."""
        c_raw = data.get("ciphertext")
        p_actual = data.get("plaintext")
        p_pred = data.get("predicted_plaintext")

        if c_raw is None or p_actual is None or p_pred is None:
            missing = [k for k in ["ciphertext", "plaintext", "predicted_plaintext"]
                       if data.get(k) is None]
            logger.warning(f"Line {line_idx} skipped. Missing keys: {missing}")
            return None

        cipher = c_raw.split() if isinstance(c_raw, str) and " " in c_raw else list(c_raw)
        actual = list(p_actual)
        pred = list(p_pred)

        local_counts = {}
        ground_truth = {}

        try:
            for c, truth, guess in zip(cipher, actual, pred, strict=True):
                if c not in local_counts:
                    local_counts[c] = Counter()
                local_counts[c][guess] += 1
                ground_truth[c] = truth
        except ValueError:
            logger.warning(
                f"Line {line_idx}: Length mismatch between ciphertext ({len(cipher)}), "
                f"actual ({len(actual)}), and predicted ({len(pred)}). Skipping.",
            )
            return None

        total_symbols = len(local_counts)
        if total_symbols == 0:
            return None

        invalid = 0
        for c, counts in local_counts.items():
            total_obs = sum(counts.values())
            most_common, freq = counts.most_common(1)[0]

            consistency = freq / total_obs
            stable = consistency >= self.threshold
            correct = (most_common == ground_truth[c])

            if not (stable and correct):
                invalid += 1

        return {
            "index": data.get("index"),
            "smer": invalid / total_symbols,
            "cipher_len": len(cipher),
            "redundancy": data.get("redundancy"),
            "threshold": self.threshold,
        }

def main() -> None:
    """Entry point for calculating SMER."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.95)
    args = parser.parse_args()

    mapper = StrictMapper(args.model_path, args.threshold)
    mapper.calculate_smer()

if __name__ == "__main__":
    main()
