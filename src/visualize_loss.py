import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import logging
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Plot training and validation loss from trainer_state.json")
    parser.add_argument(
        "--state_path",
        type=str,
        required=True,
        help="Path to the trainer_state.json file",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Model Training Performance",
        help="Main title for the generated graphs",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="The minimum change required in loss values to keep a sequential data point (default: 0.3)",
    )
    return parser.parse_args()


def extract_raw_metrics(log_history: list[dict[str, Any]]) -> tuple[list[float], list[float], list[float], list[float]]:
    """Extract training and validation loss tracking lists from log history."""
    raw_train_epochs, raw_train_losses = [], []
    raw_val_epochs, raw_val_losses = [], []

    for entry in log_history:
        if "loss" in entry and "epoch" in entry:
            raw_train_epochs.append(entry["epoch"])
            raw_train_losses.append(entry["loss"])

        if "eval_loss" in entry and "epoch" in entry:
            raw_val_epochs.append(entry["epoch"])
            raw_val_losses.append(entry["eval_loss"])

    return raw_train_epochs, raw_train_losses, raw_val_epochs, raw_val_losses


def filter_metrics(raw_epochs: list[float], raw_losses: list[float], threshold: float) -> tuple[list[float], list[float]]:
    """Filter down metrics so points are omitted if y-value changes are below the threshold."""
    if not raw_losses:
        return [], []

    filtered_epochs = [raw_epochs[0]]
    filtered_losses = [raw_losses[0]]

    for i in range(1, len(raw_losses)):
        if abs(raw_losses[i] - filtered_losses[-1]) >= threshold:
            filtered_epochs.append(raw_epochs[i])
            filtered_losses.append(raw_losses[i])

    if filtered_epochs[-1] != raw_epochs[-1]:
        filtered_epochs.append(raw_epochs[-1])
        filtered_losses.append(raw_losses[-1])

    return filtered_epochs, filtered_losses


def build_loss_plots(
    title: str,
    train_epochs: list[float],
    train_losses: list[float],
    val_epochs: list[float],
    val_losses: list[float],
) -> plt.Figure:
    """Generate side-by-side training and validation loss charts."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title, fontsize=16, fontweight="bold")

    ax1.plot(train_epochs, train_losses, color="#1f77b4", linewidth=2)
    ax1.set_title("Training Loss vs. Epochs", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Training Loss")
    ax1.grid(True, linestyle="--", alpha=0.6)

    if val_losses:
        ax2.plot(val_epochs, val_losses, color="#d62728", linewidth=2)
        ax2.set_title("Validation Loss vs. Epochs", fontsize=12, fontweight="bold")
    else:
        ax2.text(0.5, 0.5, "No Validation Data Available", ha="center", va="center", fontsize=14)
        ax2.set_title("Validation Loss vs. Epochs (Missing)", fontsize=12, fontweight="bold")

    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Validation Loss")
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def main() -> None:
    """Entry point for creating loss plots."""
    args = parse_arguments()
    state_file = Path(args.state_path)

    if not state_file.exists():
        logger.error(f"File not found at {state_file}")
        return

    with open(state_file, encoding="utf-8") as f:
        state_data = json.load(f)

    log_history = state_data.get("log_history", [])
    raw_train_eps, raw_train_loss, raw_val_eps, raw_val_loss = extract_raw_metrics(log_history)

    if not raw_train_loss and not raw_val_loss:
        logger.warning("No loss metrics discovered in log_history.")
        return

    train_epochs, train_losses = filter_metrics(raw_train_eps, raw_train_loss, threshold=args.threshold)
    val_epochs, val_losses = filter_metrics(raw_val_eps, raw_val_loss, threshold=args.threshold)

    build_loss_plots(args.title, train_epochs, train_losses, val_epochs, val_losses)

    output_image_path = state_file.with_name("loss_curves.pdf")
    plt.savefig(output_image_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Loss curves successfully generated and saved to:\n{output_image_path}")


if __name__ == "__main__":
    main()
