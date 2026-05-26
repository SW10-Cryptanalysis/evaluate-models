import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pytest
from pytest_mock import MockerFixture

from src.visualize_loss import (
    build_loss_plots,
    extract_raw_metrics,
    filter_metrics,
    main,
)


@dataclass
class FilterTestCase:
    """Test case configuration for filter_metrics."""
    raw_epochs: list[float]
    raw_losses: list[float]
    threshold: float
    expected_epochs: list[float]
    expected_losses: list[float]


@pytest.mark.parametrize(
    "test_case",
    [
        FilterTestCase(
            raw_epochs=[],
            raw_losses=[],
            threshold=0.3,
            expected_epochs=[],
            expected_losses=[],
        ),
        FilterTestCase(
            raw_epochs=[1.0, 2.0, 3.0, 4.0],
            raw_losses=[1.0, 0.9, 0.5, 0.45],
            threshold=0.3,
            expected_epochs=[1.0, 3.0, 4.0],
            expected_losses=[1.0, 0.5, 0.45],
        ),
    ],
    ids=["empty_lists", "filters_below_threshold_and_keeps_last"],
)
def test_filter_metrics(test_case: FilterTestCase) -> None:
    """Ensure metrics are correctly decimated based on the threshold delta."""
    epochs, losses = filter_metrics(
        test_case.raw_epochs, test_case.raw_losses, test_case.threshold
    )
    assert epochs == test_case.expected_epochs
    assert losses == test_case.expected_losses


@dataclass
class ExtractTestCase:
    """Test case configuration for extract_raw_metrics."""
    log_history: list[dict[str, Any]]
    expected: tuple[list[float], list[float], list[float], list[float]]


@pytest.mark.parametrize(
    "test_case",
    [
        ExtractTestCase(
            log_history=[
                {"epoch": 1.0, "loss": 0.8},
                {"epoch": 1.0, "eval_loss": 0.9},
                {"epoch": 2.0, "loss": 0.4},
            ],
            expected=([1.0, 2.0], [0.8, 0.4], [1.0], [0.9]),
        )
    ],
    ids=["valid_history_parsing"],
)
def test_extract_raw_metrics(test_case: ExtractTestCase) -> None:
    """Ensure training and validation metrics are correctly extracted from log dicts."""
    result = extract_raw_metrics(test_case.log_history)
    assert result == test_case.expected


def test_build_loss_plots() -> None:
    """Ensure the matplotlib Figure is generated with the correct axes."""
    fig = build_loss_plots(
        title="Test Plot",
        train_epochs=[1.0, 2.0],
        train_losses=[1.0, 0.5],
        val_epochs=[1.0, 2.0],
        val_losses=[1.2, 0.6],
    )

    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 2


def test_main(mocker: MockerFixture, tmp_path: Path) -> None:
    """Verify the main execution flow, file loading, and plot saving."""
    mock_args = mocker.MagicMock()
    mock_args.state_path = str(tmp_path / "trainer_state.json")
    mock_args.title = "Integration Test"
    mock_args.threshold = 0.1

    mocker.patch("src.visualize_loss.parse_arguments", return_value=mock_args)

    state_data = {
        "log_history": [
            {"epoch": 1.0, "loss": 0.8},
            {"epoch": 1.0, "eval_loss": 0.9}
        ]
    }

    state_file = tmp_path / "trainer_state.json"
    with open(state_file, "w", encoding="utf-8") as f:
        json.dump(state_data, f)

    mock_savefig = mocker.patch("matplotlib.pyplot.savefig")
    mocker.patch("matplotlib.pyplot.close")

    main()

    mock_savefig.assert_called_once()
