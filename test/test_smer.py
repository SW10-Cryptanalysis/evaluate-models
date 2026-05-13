import json
import pytest
from src.smer import StrictMapper, main

@pytest.fixture
def test_data():
    """Returns mock evaluation data for SMER testing."""
    return [
        {
            "index": 0,
            "ciphertext": "1 1 2 2",
            "plaintext": "aabb",
            "predicted_plaintext": "aabb",
            "redundancy": 2
        },
        {
            "index": 1,
            "ciphertext": "1 1 2 2",
            "plaintext": "aabb",
            "predicted_plaintext": "axbb",
            "redundancy": 2
        },
        {
            "index": 2,
            "ciphertext": "1 1 2 2",
            "plaintext": "aabb",
            "predicted_plaintext": "zzbb",
            "redundancy": 2
        }
    ]

@pytest.fixture
def setup_test_dir(tmp_path, test_data):
    """Creates a temporary directory and an evaluation_results.jsonl file."""
    results_path = tmp_path / "evaluation_results.jsonl"

    with open(results_path, "w", encoding="utf-8") as f:
        for entry in test_data:
            f.write(json.dumps(entry) + "\n")
        f.write(json.dumps({"type": "summary", "total": "ignore me"}))

    return tmp_path

class TestStrictMapper:

    def test_missing_input_file(self, mocker, tmp_path):
        """Test that the script logs an error if the file is missing."""
        mocked_logger = mocker.patch("src.smer.logger.error")
        fake_path = tmp_path / "non_existent_subdir"
        fake_path.mkdir()

        mapper = StrictMapper(model_path=str(fake_path), threshold=0.95)
        mapper.calculate_smer()

        mocked_logger.assert_called_once()
        assert "Input file not found" in mocked_logger.call_args[0][0]
        assert not (fake_path / "smer_results.jsonl").exists()

    def test_missing_crypto_keys(self, mocker, setup_test_dir):
        """Test that entries with missing crypto fields are logged and skipped."""
        mocked_warning = mocker.patch("src.smer.logger.warning")
        test_dir = setup_test_dir
        missing_keys_path = test_dir / "missing_keys.jsonl"

        with open(missing_keys_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "index": 1, "ciphertext": "1 2", "plaintext": "ab",
                "predicted_plaintext": "ab", "redundancy": 1
            }) + "\n")
            f.write(json.dumps({
                "index": 2, "plaintext": "ab",
                "predicted_plaintext": "ab", "redundancy": 1
            }) + "\n")

        mapper = StrictMapper(model_path=str(test_dir), threshold=0.95)
        mapper.results_path = missing_keys_path
        mapper.calculate_smer()

        mocked_warning.assert_called_once()
        assert "Missing keys: ['ciphertext']" in mocked_warning.call_args[0][0]

        output_file = test_dir / "smer_results.jsonl"
        with open(output_file, encoding="utf-8") as f:
            results = [json.loads(line) for line in f if "index" in json.loads(line)]

        assert len(results) == 1
        assert results[0]["index"] == 1

    def test_json_decode_error_coverage(self, setup_test_dir):
        """Specifically targets the 'except json.JSONDecodeError' block."""
        test_dir = setup_test_dir
        error_path = test_dir / "decode_error.jsonl"

        with open(error_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"index": 1, "ciphertext": "1", "plaintext": "a", "predicted_plaintext": "a", "redundancy": 1}) + "\n")
            f.write('{"index": 2, "ciphertext": "2", \n')
            f.write(json.dumps({"index": 3, "ciphertext": "3", "plaintext": "c", "predicted_plaintext": "c", "redundancy": 1}) + "\n")

        mapper = StrictMapper(model_path=str(test_dir), threshold=0.95)
        mapper.results_path = error_path
        mapper.calculate_smer()

        output_file = test_dir / "smer_results.jsonl"
        with open(output_file, encoding="utf-8") as f:
            results = [json.loads(line) for line in f if "index" in json.loads(line)]

        assert len(results) == 2
        assert results[0]["index"] == 1
        assert results[1]["index"] == 3

    def test_zip_strict_length_mismatch(self, mocker, tmp_path):
        """Tests that mismatched sequence lengths trigger ValueError."""
        mocked_warning = mocker.patch("src.smer.logger.warning")
        mismatched_data = {
            "index": 42,
            "ciphertext": "1 2 3",
            "plaintext": "abc",
            "predicted_plaintext": "ab",
            "redundancy": 1
        }

        mapper = StrictMapper(model_path=str(tmp_path))
        result = mapper._process_entry(mismatched_data, line_idx=1)

        assert result is None
        mocked_warning.assert_called_once()
        log_msg = mocked_warning.call_args[0][0]
        assert "Length mismatch" in log_msg

    def test_calculate_smer_integration(self, setup_test_dir):
        """Integration test for the full SMER calculation flow."""
        test_dir = setup_test_dir
        mapper = StrictMapper(model_path=str(test_dir), threshold=0.95)
        mapper.calculate_smer()

        output_file = test_dir / "smer_results.jsonl"
        assert output_file.exists()

        with open(output_file, encoding="utf-8") as f:
            results = [json.loads(line) for line in f if "index" in json.loads(line)]

        assert results[0]["smer"] == 0.0
        assert results[1]["smer"] == 0.5
        assert results[2]["smer"] == 0.5

def test_smer_main_flow(mocker):
    """Verify main() parses arguments and calls calculate_smer using mocker."""
    mock_args = mocker.Mock()
    mock_args.model_path = "fake/model/path"
    mock_args.threshold = 0.90
    mocker.patch("src.smer.argparse.ArgumentParser.parse_args", return_value=mock_args)
    mock_mapper_cls = mocker.patch("src.smer.StrictMapper")

    main()

    mock_mapper_cls.assert_called_once_with("fake/model/path", 0.90)
    mock_mapper_cls.return_value.calculate_smer.assert_called_once()
