import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize evaluation results from jsonl."
    )
    parser.add_argument(
        "--eval_file_path",
        type=str,
        required=True,
        help="Path to the evaluation_results.jsonl file",
    )
    args = parser.parse_args()

    eval_file = Path(args.eval_file_path)
    if not eval_file.exists():
        print(f"Error: File not found at {eval_file}")
        return

    lengths = []
    redundancies = []
    sers = []

    # Parse the jsonl file
    with open(eval_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)

            # Skip the summary lines appended at the end of eval.py
            if "type" in data and data["type"].startswith("summary"):
                continue

            length = len(data["plaintext"])
            redundancy = data["redundancy"]
            ser = data["ser"]

            lengths.append(length)
            redundancies.append(redundancy)
            sers.append(ser)

    if not sers:
        print("No valid sample data found in the evaluation file.")
        return

    # Initialize plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Graph 1: SER vs Length (Hexbin)
    # gridsize controls how many hexagons there are; cmap sets the color theme
    hb1 = ax1.hexbin(lengths, sers, gridsize=50, cmap="Blues", mincnt=1)
    ax1.set_title("Symbol Error Rate (SER) vs Sequence Length")
    ax1.set_xlabel("Length (Characters)")
    ax1.set_ylabel("SER")
    ax1.grid(True, linestyle="--", alpha=0.3)
    fig.colorbar(hb1, ax=ax1, label="Number of Sequences")

    # Graph 2: SER vs Redundancy (Hexbin)
    hb2 = ax2.hexbin(redundancies, sers, gridsize=50, cmap="Reds", mincnt=1)
    ax2.set_title("Symbol Error Rate (SER) vs Redundancy")
    ax2.set_xlabel("Redundancy")
    ax2.set_ylabel("SER")
    ax2.grid(True, linestyle="--", alpha=0.3)
    fig.colorbar(hb2, ax=ax2, label="Number of Sequences")

    plt.tight_layout()

    # Save the output
    output_image_path = eval_file.with_name("evaluation_graphs.png")
    plt.savefig(output_image_path, dpi=300, bbox_inches="tight")
    print(f"Graphs successfully generated and saved to:\n{output_image_path}")


if __name__ == "__main__":
    main()
