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

    def run(self) -> list[dict]:
        """Main method to execute the evaluation process."""
        if not EvalConfig.tokenizer_dir.exists():
            logger.error(f"Global tokenizer not found at {EvalConfig.tokenizer_dir}.")
            sys.exit(1)

        logger.info("Initializing Native PyTorch RWKV Model for Evaluation...")

        # 1. Load Model
        model = get_model().to(self.device)
        checkpoint_path = self.model_dir / "rwkv7_cipher_final.pth"
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Could not find model weights at {checkpoint_path}")
            
        state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Capture a raw weight value out of the file before loading
        sample_key = list(state_dict.keys())[0]
        file_weight_sample = state_dict[sample_key].detach().cpu().float().mean().item()
        
        # Load the weights into the model architecture
        # We switch strict=True to FORCE PyTorch to crash and show us the exact naming conflict
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        # Check if the weight actually stuck inside the active layer
        model_weight_sample = dict(model.named_parameters())[sample_key].detach().cpu().float().mean().item()
        logger.info(f"[WEIGHT CHECK] File mean: {file_weight_sample:.6f} | Model mean: {model_weight_sample:.6f}")

        # 2. Setup allowed tokens mask
        vocab_size = self.config.vocab_size
        allowed_mask = torch.full((vocab_size,), float('-inf'), device=self.device)
        valid_ids = [t for t in self.allowed_token_ids if t < vocab_size]
        allowed_mask[valid_ids] = 0.0

        parsed_samples = self.parse_samples()
        parsed_samples.sort(key=lambda x: x["target_length"], reverse=True)

        logger.info(f"Launching sequential inference for {len(parsed_samples)} sequences...")
        start_time = time.perf_counter()

        outputs = []
        # 3. Generate predictions
        for sample in tqdm(parsed_samples, desc="Evaluating"):
            pred_ids = self.generate_greedy(
                model=model,
                prompt_ids=sample["prompt_ids"],
                target_length=sample["target_length"],
                allowed_mask=allowed_mask
            )
            outputs.append(pred_ids)

        generation_time = time.perf_counter() - start_time
        logger.info(f"Inference complete in {generation_time:.2f} seconds.")

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