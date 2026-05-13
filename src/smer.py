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
    def __init__(self, model_path: str, threshold: float = 0.95) -> None:
        self.model_path = Path(model_path)
        self.threshold = threshold
        self.results_path = self.model_path / "evaluation_results.jsonl"
        self.output_path = self.model_path / "smer_results.jsonl"

    def calculate_smer(self):
        if not self.results_path.exists():
            logger.error(f"Input file not found: {self.results_path}")
            return

        processed_count = 0
        output_data = []

        with open(self.results_path, "r") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line: continue
                
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(f"Line {i} contains invalid JSON and was skipped.")
                    continue

                if "index" not in data:
                    continue

                c_raw = data.get("ciphertext")
                p_actual = data.get("plaintext")
                p_pred = data.get("predicted_plaintext")

                if c_raw is None or p_actual is None or p_pred is None:
                    missing = [k for k in ["ciphertext", "plaintext", "predicted_plaintext"] if data.get(k) is None]
                    logger.warning(f"Line {i} skipped. Missing keys: {missing}")
                    continue

                cipher = c_raw.split() if isinstance(c_raw, str) and " " in c_raw else list(c_raw)
                actual = list(p_actual)
                pred = list(p_pred)

                local_counts = {}
                ground_truth = {}

                for c, truth, guess in zip(cipher, actual, pred):
                    if c not in local_counts:
                        local_counts[c] = Counter()
                    local_counts[c][guess] += 1
                    ground_truth[c] = truth

                total_symbols = len(local_counts)
                invalid = 0

                for c, counts in local_counts.items():
                    total_obs = sum(counts.values())
                    most_common, freq = counts.most_common(1)[0]
                    
                    consistency = freq / total_obs
                    stable = consistency >= self.threshold
                    correct = (most_common == ground_truth[c])

                    if not (stable and correct):
                        invalid += 1

                entry_smer = invalid / total_symbols if total_symbols > 0 else 0
                
                result = {
                    "index": data.get("index"),
                    "smer": entry_smer,
                    "cipher_len": len(cipher),
                    "redundancy": data.get("redundancy")
                }

                output_data.append(result)
                processed_count += 1

        with open(self.output_path, "w") as f:
            for entry in output_data:
                f.write(json.dumps(entry) + "\n")
            f.write(json.dumps({"threshold": self.threshold}))

        logger.info(f"Successfully processed {processed_count} entries.")
        logger.info(f"Results saved to: {self.output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--threshold", type=float, required=False)
    args = parser.parse_args()

    mapper = StrictMapper(args.model_path, args.threshold)
    mapper.calculate_smer()

if __name__ == "__main__":
    main()