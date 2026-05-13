import json
import sys
import pytest
from unittest.mock import patch
from pathlib import Path

# Ensure the import matches your filename
from src.visualize_smer import main

@pytest.fixture
def model_dir(tmp_path):
    """Creates a temporary directory with a valid smer_results.jsonl file."""
    # The script expects a directory, then finds 'smer_results.jsonl' inside
    data_file = tmp_path / "smer_results.jsonl"
    lines = [
        {"index": 0, "cipher_len": 100, "redundancy": 1.5, "smer": 0.95},
        {"index": 1, "cipher_len": 200, "redundancy": 2.5, "smer": 0.85},
        {"threshold": 0.95}, # Metadata line to be skipped
    ]
    with open(data_file, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")
    return tmp_path


@pytest.fixture
def empty_model_dir(tmp_path):
    """Creates a directory with a jsonl file that has no valid 'index' entries."""
    data_file = tmp_path / "smer_results.jsonl"
    lines = [{"threshold": 0.95}]
    with open(data_file, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")
    return tmp_path


class TestVisualizeSmer:
    
    @patch("matplotlib.pyplot.savefig")
    def test_main_generates_plot(self, mock_savefig, model_dir):
        """Test that valid data in the model path results in a saved plot."""
        test_args = [
            "visualize_smer.py",
            "--model_path", str(model_dir),
            "--title", "Test SMER Plot",
        ]

        with patch.object(sys, "argv", test_args):
            main()

        # Check if savefig was called
        assert mock_savefig.called
        # Verify the filename used in savefig
        args, _ = mock_savefig.call_args
        assert "smer_visualization.png" in str(args[0])

    @patch("matplotlib.pyplot.savefig")
    @patch("src.visualize_smer.logger.info")
    def test_main_handles_missing_file(self, mock_logger, mock_savefig, tmp_path):
        """Test that missing smer_results.jsonl logs an error and exits."""
        empty_dir = tmp_path / "empty_dir"
        empty_dir.mkdir()
        
        test_args = [
            "visualize_smer.py",
            "--model_path", str(empty_dir),
            "--title", "Test",
        ]

        with patch.object(sys, "argv", test_args):
            main()

        mock_savefig.assert_not_called()
        # Verify "File not found" was logged
        assert any("Error: File not found" in call.args[0] for call in mock_logger.call_args_list)

    @patch("matplotlib.pyplot.savefig")
    @patch("src.visualize_smer.logger.info")
    def test_main_handles_no_valid_data(self, mock_logger, mock_savefig, empty_model_dir):
        """Test that script aborts if no lines contain an 'index' key."""
        test_args = [
            "visualize_smer.py",
            "--model_path", str(empty_model_dir),
            "--title", "Test",
        ]

        with patch.object(sys, "argv", test_args):
            main()

        mock_savefig.assert_not_called()
        assert any("No valid sample data found" in call.args[0] for call in mock_logger.call_args_list)

    @patch("matplotlib.pyplot.show")   # This is the second argument (mock_show)
    @patch("matplotlib.pyplot.savefig") # This is the first argument (mock_savefig)
    def test_y_axis_limits(self, mock_savefig, mock_show, model_dir):
        """Verify the dynamic Y-axis logic doesn't crash."""
        test_args = [
            "visualize_smer.py",
            "--model_path", str(model_dir),
            "--title", "Test Dynamic Y",
        ]

        with patch.object(sys, "argv", test_args):
            main()
        
        assert mock_savefig.called