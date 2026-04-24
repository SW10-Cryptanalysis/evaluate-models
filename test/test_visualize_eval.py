import json
import sys
import pytest
from unittest.mock import patch

# Note: Adjust the import based on where visualize_eval.py is located.
# Assuming it is in the src directory here:
from src.visualize_eval import main


@pytest.fixture
def dummy_jsonl(tmp_path):
    """Creates a temporary jsonl file with dummy data and a summary line."""
    data_file = tmp_path / "evaluation_results.jsonl"
    lines = [
        {"index": 0, "plaintext": "hello", "redundancy": 0, "ser": 0.0},
        {"index": 1, "plaintext": "world", "redundancy": 2, "ser": 0.5},
        {"type": "summary_global", "avg_ser": 0.25},
    ]
    with open(data_file, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")
    return data_file


@pytest.fixture
def empty_jsonl(tmp_path):
    """Creates a temporary jsonl file with only a summary line (no valid data)."""
    data_file = tmp_path / "empty_results.jsonl"
    lines = [{"type": "summary_global", "avg_ser": 0.0}]
    with open(data_file, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")
    return data_file


class TestVisualizeEval:
    @patch("matplotlib.pyplot.savefig")
    def test_main_generates_plot(self, mock_savefig, dummy_jsonl):
        """Test that providing valid data results in a saved plot."""
        test_args = [
            "visualize_eval.py",
            "--eval_file_path",
            str(dummy_jsonl),
            "--title",
            "Test Graph",
        ]

        with patch.object(sys, "argv", test_args):
            main()

        # Verify that matplotlib's savefig was called
        mock_savefig.assert_called_once()

    @patch("matplotlib.pyplot.savefig")
    def test_main_handles_missing_file(self, mock_savefig, tmp_path):
        """Test that the script exits gracefully without plotting if file is missing."""
        missing_file = tmp_path / "does_not_exist.jsonl"
        test_args = [
            "visualize_eval.py",
            "--eval_file_path",
            str(missing_file),
            "--title",
            "Test Graph",
        ]

        with patch.object(sys, "argv", test_args):
            main()

        # Verify plotting is skipped
        mock_savefig.assert_not_called()

    @patch("matplotlib.pyplot.savefig")
    def test_main_handles_no_valid_data(self, mock_savefig, empty_jsonl):
        """Test that the script aborts plotting if no valid samples are found."""
        test_args = [
            "visualize_eval.py",
            "--eval_file_path",
            str(empty_jsonl),
            "--title",
            "Test Graph",
        ]

        with patch.object(sys, "argv", test_args):
            main()

        # Verify plotting is skipped
        mock_savefig.assert_not_called()
