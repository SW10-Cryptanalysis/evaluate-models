import json
import sys
import pytest
from src.visualize_smer import main

@pytest.fixture
def model_dir(tmp_path):
    """Creates a temporary directory with a valid smer_results.jsonl file."""
    data_file = tmp_path / "smer_results.jsonl"
    lines = [
        {"index": 0, "cipher_len": 100, "redundancy": 1, "smer": 0.95},
        {"index": 1, "cipher_len": 200, "redundancy": 2, "smer": 0.85},
        {"threshold": 0.95},
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

    def test_main_generates_plot(self, mocker, model_dir):
        """Test that valid data in the model path results in a saved plot."""
        mocker.patch.object(sys, "argv", [
            "visualize_smer.py",
            "--model_path", str(model_dir),
            "--title", "Test SMER Plot",
        ])
        mock_savefig = mocker.patch("matplotlib.pyplot.savefig")
        mocker.patch("matplotlib.pyplot.show")

        main()

        assert mock_savefig.called
        args, _ = mock_savefig.call_args
        assert "smer_visualization.png" in str(args[0])

    def test_main_handles_missing_file(self, mocker, tmp_path):
        """Test that missing smer_results.jsonl logs an error and exits."""
        mock_logger = mocker.patch("src.visualize_smer.logger.info")
        mock_savefig = mocker.patch("matplotlib.pyplot.savefig")

        empty_dir = tmp_path / "empty_dir"
        empty_dir.mkdir()

        mocker.patch.object(sys, "argv", [
            "visualize_smer.py",
            "--model_path", str(empty_dir),
            "--title", "Test",
        ])

        main()

        mock_savefig.assert_not_called()
        assert any("Error: File not found" in call.args[0] for call in mock_logger.call_args_list)

    def test_main_handles_no_valid_data(self, mocker, empty_model_dir):
        """Test that script aborts if no lines contain an 'index' key."""
        mock_logger = mocker.patch("src.visualize_smer.logger.info")
        mock_savefig = mocker.patch("matplotlib.pyplot.savefig")

        mocker.patch.object(sys, "argv", [
            "visualize_smer.py",
            "--model_path", str(empty_model_dir),
            "--title", "Test",
        ])

        main()

        mock_savefig.assert_not_called()
        assert any("No valid sample data found" in call.args[0] for call in mock_logger.call_args_list)

    def test_y_axis_limits(self, mocker, model_dir):
        """Verify the dynamic Y-axis logic doesn't crash."""
        mock_savefig = mocker.patch("matplotlib.pyplot.savefig")
        mocker.patch("matplotlib.pyplot.show")

        mocker.patch.object(sys, "argv", [
            "visualize_smer.py",
            "--model_path", str(model_dir),
            "--title", "Test Dynamic Y",
        ])

        main()

        assert mock_savefig.called
