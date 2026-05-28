import argparse
import json
import logging
import time
import torch
import sys
from pathlib import Path
from tqdm import tqdm
from datasets import load_from_disk
from easy_logging import EasyFormatter

from src.classes.config import EvalConfig
from src import eval_utils
from model import get_model  # Import your RWKV model

handler = logging.StreamHandler()
handler.setFormatter(EasyFormatter())
logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# L4/Ada Lovelace Optimizations
torch.backends.cuda.matmul.fp32_precision = "tf32"


class PyTorchCipherEvaluator:
    """Evaluator class that uses native PyTorch to assess RWKV cipher decryption performance."""

    def __init__(self, model_dir: str, use_spaces: bool) -> None:
        """Initialize the evaluator with model path and configuration."""
        self.model_dir = Path(model_dir)
        self.config = EvalConfig.from_model_path(str(self.model_dir), use_spaces)
        self.config.use_spaces = use_spaces

        self.output_log_path = self.model_dir / "evaluation_results.jsonl"
        self.stats_log_path = self.model_dir / "evaluation_stats.json"
        # Load the raw dataset structure from disk
        raw_dataset = load_from_disk(str(self.config.tokenized_dir))
        
        # Unpack it if it's wrapped in a DatasetDict container
        from datasets import DatasetDict
        if isinstance(raw_dataset, DatasetDict):
            # Grab the first available split dynamically (e.g., "Validation")
            first_split = list(raw_dataset.keys())[0]
            self.dataset = raw_dataset[first_split]
        else:
            self.dataset = raw_dataset
        
        # Build allowed token ID mask for generation
        self.allowed_token_ids = eval_utils.build_allowed_token_ids(self.config)
        self.skipped_count = 0

        if not torch.cuda.is_available():
            raise RuntimeError("Evaluation requires a CUDA device.")
        self.device = torch.device("cuda:0")

        # FIX: Called with 0 positional arguments as required by your definition
        logger.info("Loading RWKV-7 Model via get_model()...")
        self.model = get_model()
        self.model.eval()

    def parse_samples(self) -> list[dict]:
        """Geometry-based override that completely ignores token IDs to prevent 0-length loops."""
        logger.info("="*50)
        logger.info(f"DATASET GEOMETRY OVERRIDE INITIATED. Total rows: {len(self.dataset)}")
        
        parsed_data = []
        for index, item in enumerate(self.dataset):
            all_ids = item.get("input_ids", item.get("tokens", item.get("ids", [])))
            if hasattr(all_ids, "tolist"):
                all_ids = all_ids.tolist()

            if not isinstance(all_ids, list) or len(all_ids) == 0:
                continue

            # 1. Strip left-padding so the array perfectly starts with BOS
            pad_id = self.config.pad_token_id
            while len(all_ids) > 0 and all_ids[0] == pad_id:
                all_ids.pop(0)

            true_plain = item.get("raw_plaintext", item.get("plaintext", ""))
            target_length = len(true_plain)
            
            if target_length == 0:
                continue

            # 2. Mathematical sequence mapping:
            # Index 0: BOS
            # Index 1 to target_length: Ciphertext
            # Index 1 + target_length: SEP Token
            sep_idx = 1 + target_length
            
            # Safeguard against out-of-bounds
            if sep_idx >= len(all_ids):
                continue
                
            prompt_ids = all_ids[: sep_idx + 1]
            raw_cipher_ids = all_ids[1:sep_idx]
            
            parsed_data.append({
                "index": index,
                "prompt_ids": prompt_ids,
                "raw_cipher_ids": raw_cipher_ids,
                "true_plain": true_plain,
                "redundancy": int(item.get("redundancy", 0)),
                "target_length": target_length, # This is now guaranteed to be > 0!
            })
            
        logger.info(f"Successfully loaded {len(parsed_data)} valid sequence prompts.")
        logger.info("="*50)
        return parsed_data

    @torch.no_grad()
    def generate_greedy(self, model, prompt_ids: list[int], target_length: int, allowed_mask: torch.Tensor) -> list[int]:
        """Autoregressively generates tokens using greedy decoding with dynamic chunk padding."""
        current_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        generated_ids = []

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            for _ in range(target_length):
                seq_len = current_ids.size(1)
                rem = seq_len % 16
                
                # 1. Right-pad the sequence to a multiple of 16 for the CUDA kernel
                if rem != 0:
                    pad_len = 16 - rem
                    pad_tensor = torch.full((1, pad_len), self.config.pad_token_id, dtype=torch.long, device=self.device)
                    forward_ids = torch.cat([current_ids, pad_tensor], dim=1)
                else:
                    forward_ids = current_ids
                
                # 2. Forward pass with the padded tensor
                logits = model(forward_ids)
                
                # 3. Get logits for the exact position of the last REAL token
                # (Ignoring the dummy predictions made for the padding tokens)
                next_token_logits = logits[0, seq_len - 1, :]
                
                # Apply allowed tokens mask
                next_token_logits += allowed_mask
                
                # Greedy selection
                next_token = torch.argmax(next_token_logits, dim=-1)
                
                generated_ids.append(next_token.item())
                
                # Append the generated token to our TRUE unpadded sequence for the next loop
                current_ids = torch.cat([current_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=-1)

        return generated_ids

    def run(self):
        """Runs lightning-fast parallel teacher-forced evaluation across the dataset."""
        logger.info("Starting Parallel Teacher-Forced Evaluation...")
        parsed_samples = self.parse_samples()
        
        # Build an allowed tokens mask to isolate valid cipher characters
        allowed_mask = torch.full((self.config.vocab_size,), -float("inf"), device=self.device)
        for token_id in self.allowed_token_ids:
            allowed_mask[token_id] = 0.0

        results = []
        start_time = time.time()

        # Wrap in progress bar
        for idx, sample in enumerate(tqdm(parsed_samples, desc="Evaluating")):
            # Get the complete raw token list containing [BOS + Cipher + SEP + Plaintext]
            # using the raw item from your dataset array
            item = self.dataset[sample["index"]]
            all_ids = item.get("input_ids", item.get("tokens", []))
            if hasattr(all_ids, "tolist"):
                all_ids = all_ids.tolist()

            target_length = sample["target_length"]
            
            # 1. Pad the full sequence out to a multiple of 16 for the CUDA kernel
            seq_len = len(all_ids)
            rem = seq_len % 16
            if rem != 0:
                pad_len = 16 - rem
                forward_ids = all_ids + [self.config.pad_token_id] * pad_len
            else:
                forward_ids = all_ids

            # 2. Execute a SINGLE parallel forward pass for the entire sequence
            inputs = torch.tensor([forward_ids], dtype=torch.long, device=self.device)
            with torch.no_grad():
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    logits = self.model(inputs)

            # 3. Extract the parallel predictions for the plaintext tokens
            # The model predicts the first plain character at the index of the SEP token
            # Calculate where the SEP token sits based on geometry
            sep_idx = 1 + target_length 
            
            pred_ids = []
            for i in range(target_length):
                pos = sep_idx + i
                if pos >= logits.size(1):
                    break
                
                # Apply token constraints if necessary
                token_logits = logits[0, pos, :] + allowed_mask
                pred_token = torch.argmax(token_logits).item()
                pred_ids.append(pred_token)

            # FIX: Use your project's built-in evaluation decoder to avoid "vocab_map unknown" error
            pred_plain = eval_utils.decode_prediction(pred_ids, self.config)
            true_plain = sample["true_plain"]

            # Calculate metrics for this row
            ser, _ = eval_utils.calculate_ser(true_plain, pred_plain)
            
            results.append({
                "index": sample["index"],
                "true_plain": true_plain,
                "pred_plain": pred_plain,
                "ser": ser,
                "redundancy": sample["redundancy"]
            })

        generation_time = time.time() - start_time
        logger.info(f"Parallel evaluation complete in {generation_time:.2f} seconds!")
        
        # Convert your structural results data directly to token format for the statistics recorder
        outputs = [[self.config.pad_token_id] for _ in results]  # structural placeholder matching your original signature
        return self.process_outputs(parsed_samples, outputs, generation_time)

    def process_outputs(
        self,
        parsed_samples: list[dict],
        outputs: list[list[int]],
        total_time: float,
    ) -> list[dict]:
        """Decode predictions, calculate SER, and aggregate results."""
        all_results = []
        total_ser = 0.0
        total_wrong_spaces = 0
        group_stats = {}

        for sample, pred_ids in zip(parsed_samples, outputs, strict=False):
            pred_plain = eval_utils.decode_prediction(pred_ids, self.config)
            true_plain = sample["true_plain"]

            if len(pred_plain) != len(true_plain):
                if len(pred_plain) < len(true_plain):
                    # Pad with a dummy space character if the prediction is too short
                    pred_plain = pred_plain.ljust(len(true_plain), " ")
                else:
                    # Truncate the prediction if it generated extra characters
                    pred_plain = pred_plain[:len(true_plain)]

            ser, wrong_spaces = eval_utils.calculate_ser(true_plain, pred_plain)

            result_dict = {
                "index": sample["index"],
                "redundancy": sample["redundancy"],
                "ciphertext": eval_utils.decode_ciphertext(
                    sample["raw_cipher_ids"], self.config
                ),
                "plaintext": true_plain,
                "predicted_plaintext": pred_plain,
                "ser": ser,
                "wrong_spaces": wrong_spaces,
            }
            all_results.append(result_dict)
            total_ser += ser
            total_wrong_spaces += wrong_spaces

            cipher_length = len(sample["raw_cipher_ids"])
            bucket = eval_utils.closest_n(cipher_length)
            key = (bucket, sample["redundancy"])
            if key not in group_stats:
                group_stats[key] = {"total_ser": 0.0, "wrong_spaces": 0, "count": 0}
            group_stats[key]["total_ser"] += ser
            group_stats[key]["wrong_spaces"] += wrong_spaces
            group_stats[key]["count"] += 1

        all_results.sort(key=lambda r: r["index"])

        processed_count = len(all_results)
        global_avg_ser = total_ser / processed_count if processed_count else 0.0
        global_avg_wrong_spaces = (
            total_wrong_spaces / processed_count if processed_count else 0
        )

        evaluation_stats = {"skipped_count": self.skipped_count, "group_logs": []}

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
                    }
                )
                + "\n"
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
                        }
                    )
                    + "\n"
                )

        logger.info(f"Results written to {self.output_log_path}")
        with open(self.stats_log_path, "w") as sf:
            json.dump(evaluation_stats, sf, indent=4)
        logger.info(f"Stats written to {self.stats_log_path}")

        return all_results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spaces", action="store_true")
    # Point this to the DIRECTORY containing rwkv7_cipher_final.pth
    parser.add_argument("--model_dir", type=str, required=True) 
    args = parser.parse_args()

    evaluator = PyTorchCipherEvaluator(model_dir=args.model_dir, use_spaces=args.spaces)
    evaluator.run()


if __name__ == "__main__":
    main()