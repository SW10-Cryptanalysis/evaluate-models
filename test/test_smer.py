import unittest
import json
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.smer import StrictMapper, main

class TestSMER(unittest.TestCase):
    def setUp(self):
        """Create a temporary directory and mock evaluation data."""
        self.test_dir = Path("test_output")
        self.test_dir.mkdir(exist_ok=True)
        self.results_path = self.test_dir / "evaluation_results.jsonl"

        self.mock_data = [
            # Perfect mapping
            # '1' -> 'a', '2' -> 'b'. consistent, correct.
            {
                "index": 0,
                "ciphertext": "1 1 2 2",
                "plaintext": "aabb",
                "predicted_plaintext": "aabb",
                "redundancy": 2.0
            },
            # Unstable mapping
            # '1' appears twice: mapped to 'a' once and 'x' once (50% freq).
            {
                "index": 1,
                "ciphertext": "1 1 2 2",
                "plaintext": "aabb",
                "predicted_plaintext": "axbb",
                "redundancy": 2.0
            },
            # Stable but incorrect mapping
            {
                "index": 2,
                "ciphertext": "1 1 2 2",
                "plaintext": "aabb",
                "predicted_plaintext": "zzbb",
                "redundancy": 2.0
            }
        ]

        with open(self.results_path, "w") as f:
            for entry in self.mock_data:
                f.write(json.dumps(entry) + "\n")
            f.write(json.dumps({"type": "summary", "total": "ignore me"}))

    def tearDown(self):
        """Clean up the test directory."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    @patch("src.smer.logger.error")
    def test_missing_input_file(self, mocked_logger):
        """Test that the script logs an error and returns if the file is missing."""
        fake_path = self.test_dir / "non_existent_subdir"
        fake_path.mkdir()

        mapper = StrictMapper(model_path=str(fake_path), threshold=0.95)

        mapper.calculate_smer()
        mocked_logger.assert_called_once()
        args, _ = mocked_logger.call_args
        self.assertIn("Input file not found", args[0])

        output_file = fake_path / "smer_results.jsonl"
        self.assertFalse(output_file.exists())

    @patch("src.smer.logger.warning")
    def test_missing_crypto_keys(self, mocked_warning):
        """Test that entries with missing crypto fields are logged and skipped."""
        missing_keys_path = self.test_dir / "missing_keys.jsonl"

        with open(missing_keys_path, "w") as f:
            f.write(json.dumps({
                "index": 1, "ciphertext": "1 2", "plaintext": "ab",
                "predicted_plaintext": "ab", "redundancy": 1.0
            }) + "\n")

            f.write(json.dumps({
                "index": 2, "plaintext": "ab",
                "predicted_plaintext": "ab", "redundancy": 1.0
            }) + "\n")

        mapper = StrictMapper(model_path=str(self.test_dir), threshold=0.95)
        mapper.results_path = missing_keys_path
        mapper.calculate_smer()

        mocked_warning.assert_called_once()
        args, _ = mocked_warning.call_args
        self.assertIn("Missing keys: ['ciphertext']", args[0])

        output_file = self.test_dir / "smer_results.jsonl"
        results = []
        with open(output_file) as f:
            for line in f:
                d = json.loads(line)
                if "index" in d:
                    results.append(d)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["index"], 1)

    def test_json_decode_error_coverage(self):
        """Specifically targets the 'except json.JSONDecodeError' block for coverage."""
        error_path = self.test_dir / "decode_error.jsonl"

        with open(error_path, "w") as f:
            f.write(json.dumps({"index": 1, "ciphertext": "1", "plaintext": "a", "predicted_plaintext": "a", "redundancy": 1.0}) + "\n")

            # Malformed json
            f.write('{"index": 2, "ciphertext": "2", \n')

            f.write(json.dumps({"index": 3, "ciphertext": "3", "plaintext": "c", "predicted_plaintext": "c", "redundancy": 1.0}) + "\n")

        mapper = StrictMapper(model_path=str(self.test_dir), threshold=0.95)
        mapper.results_path = error_path

        mapper.calculate_smer()

        output_file = self.test_dir / "smer_results.jsonl"
        results = []
        with open(output_file) as f:
            for line in f:
                d = json.loads(line)
                if "index" in d:
                    results.append(d)

        assert len(results) == 2
        assert results[0]["index"] == 1
        assert results[1]["index"] == 3

    @patch("src.smer.logger.warning")
    def test_zip_strict_length_mismatch(self, mocked_warning):
        """Tests that mismatched sequence lengths trigger ValueError and a warning."""
        mismatched_data = {
            "index": 42,
            "ciphertext": "1 2 3",
            "plaintext": "abc",
            "predicted_plaintext": "ab",
            "redundancy": 1.0
        }

        mapper = StrictMapper(model_path=str(self.test_dir))
        result = mapper._process_entry(mismatched_data, line_idx=1)
        assert result is None

        mocked_warning.assert_called_once()
        args, _ = mocked_warning.call_args
        assert "Length mismatch" in args[0]
        assert "ciphertext (3)" in args[0]
        assert "predicted (2)" in args[0]

    def test_calculate_smer(self):
        mapper = StrictMapper(model_path=str(self.test_dir), threshold=0.95)
        mapper.calculate_smer()

        output_file = self.test_dir / "smer_results.jsonl"
        self.assertTrue(output_file.exists())

        results = []
        threshold_entry = None

        with open(output_file) as f:
            for line in f:
                data = json.loads(line)
                if "index" in data:
                    results.append(data)
                elif "threshold" in data:
                    threshold_entry = data

        self.assertEqual(results[0]["smer"], 0.0)
        self.assertEqual(results[1]["smer"], 0.5)
        self.assertEqual(results[2]["smer"], 0.5)
        self.assertEqual(threshold_entry["threshold"], 0.95)

@patch("src.smer.argparse.ArgumentParser.parse_args")
@patch("src.smer.StrictMapper")
def test_smer_main_flow(mock_mapper_cls, mock_parse_args):
    mock_args = MagicMock()
    mock_args.model_path = "fake/model/path"
    mock_args.threshold = 0.90
    mock_parse_args.return_value = mock_args

    main()
    mock_mapper_cls.assert_called_once_with("fake/model/path", 0.90)
    mock_mapper_cls.return_value.calculate_smer.assert_called_once()

@patch("src.smer.argparse.ArgumentParser.parse_args")
@patch("src.smer.StrictMapper")
def test_smer_main_default_threshold(mock_mapper_cls, mock_parse_args):
    """Tests main with threshold as None (simulating optional CLI arg)."""
    mock_args = MagicMock()
    mock_args.model_path = "another/path"
    mock_args.threshold = None
    mock_parse_args.return_value = mock_args

    main()

    mock_mapper_cls.assert_called_once_with("another/path", None)

if __name__ == "__main__":
    unittest.main()
