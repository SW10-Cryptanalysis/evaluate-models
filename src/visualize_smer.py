import argparse
import json
import logging
import math
from pathlib import Path
import matplotlib.pyplot as plt
from easy_logging import EasyFormatter

handler = logging.StreamHandler()
handler.setFormatter(EasyFormatter())
logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def main() -> None:
    """Reads the smer_results.jsonl file and creates two scatter plots for SMER."""
    parser = argparse.ArgumentParser(
        description="Visualize SMER results from jsonl.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the smer_results.jsonl file",
    )
    parser.add_argument(
        "--title",
        type=str,
        required=True,
        help="Custom title for the generated graphs",
    )
    args = parser.parse_args()

    smer_file = Path(args.model_path) / "smer_results.jsonl"
    if not smer_file.exists():
        logger.info(f"Error: File not found at {smer_file}")
        return

    lengths = []
    redundancies = []
    smers = []

    z408_len = None
    z408_redundancy = None
    z408_smer = None

    with open(smer_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)

            if "index" not in data:
                continue

            if data["index"] == "Z408":
                z408_len = data["cipher_len"]
                z408_redundancy = data["redundancy"]
                z408_smer = data["smer"]
            else:
                lengths.append(data["cipher_len"])
                redundancies.append(data["redundancy"])
                smers.append(data["smer"])

    if not smers:
        logger.info("No valid sample data found in the SMER file.")
        return

    min_smer = min(smers)
    y_min = math.floor(min_smer * 10) / 10.0
    y_limit_min = max(0.0, y_min - 0.02)
    y_limit_max = 1.01

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(args.title, fontsize=16, fontweight="bold", y=1.02)

    # Graph 1: SMER vs Length
    ax1.scatter(lengths, smers, alpha=0.4, color="#1f77b4", edgecolors="none")
    if z408_len is not None and z408_smer is not None:
        ax1.scatter(
            z408_len, z408_smer,
            color="gold", marker="*", s=250,
            edgecolors="black", zorder=5,
            label="Z408",
        )
        ax1.annotate(
            "Z408",
            (z408_len, z408_smer),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontweight="bold",
        )
    ax1.set_title("SMER vs Length")
    ax1.set_xlabel("Cipher Length (Symbols)")
    ax1.set_ylabel("SMER")
    ax1.set_ylim(y_limit_min, y_limit_max)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Graph 2: SMER vs Redundancy
    ax2.scatter(redundancies, smers, alpha=0.4, color="#d62728", edgecolors="none")
    if z408_redundancy is not None and z408_smer is not None:
        ax2.scatter(
            z408_redundancy, z408_smer,
            color="gold", marker="*", s=250,
            edgecolors="black", zorder=5,
            label="Z408",
        )
        ax2.annotate(
            "Z408",
            (z408_redundancy, z408_smer),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontweight="bold",
        )
    ax2.set_title("SMER vs Redundancy")
    ax2.set_xlabel("Redundancy")
    ax2.set_ylabel("SMER")
    ax2.set_ylim(y_limit_min, y_limit_max)
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()

    output_image_path = smer_file.with_name("smer_visualization.png")
    plt.savefig(output_image_path, dpi=300, bbox_inches="tight")
    logger.info(f"Graphs successfully generated and saved to:\n{output_image_path}")


if __name__ == "__main__":
    main()
