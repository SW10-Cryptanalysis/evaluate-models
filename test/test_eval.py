import pytest
from src.classes.config import EvalConfig
from src.eval import VLLMCipherEvaluator


# Native Python classes to simulate vLLM's RequestOutput structure
class DummyTokenOutput:
    def __init__(self, token_ids: list[int]):
        self.token_ids = token_ids


class MockRequestOutput:
    def __init__(self, token_ids: list[int]):
        self.outputs = [DummyTokenOutput(token_ids=token_ids)]


@pytest.fixture
def mock_dataset():
    return [
        {
            "input_ids": [
                5,
                100,
                200,
                3,
                7,
                8,
                6,
            ],  # bos=5, sep=3, eos=6. cipher=[100, 200]. target=[7, 8]
            "raw_plaintext": "ab",
            "redundancy": "0",
        },
        {
            "input_ids": [5, 100, 999, 6],  # Missing SEP token, should be skipped
            "raw_plaintext": "fail",
            "redundancy": "0",
        },
    ]


@pytest.fixture
def setup_evaluator(mocker, tmp_path, mock_dataset):
    mock_config = EvalConfig(
        vocab_size=32000,
        pad_token_id=0,
        bos_token_id=5,
        eos_token_id=6,
        max_context=4096,
        use_spaces=False,
    )
    mocker.patch(
        "src.classes.config.EvalConfig.from_model_path", return_value=mock_config
    )

    mocker.patch("torch.cuda.device_count", return_value=1)

    mocker.patch("src.eval.load_from_disk", return_value=mock_dataset)

    evaluator = VLLMCipherEvaluator(model_path=str(tmp_path), use_spaces=False)
    evaluator.output_log_path = tmp_path / "evaluation_results.jsonl"
    return evaluator


class TestVLLMCipherEvaluator:
    def test_init_raises_error_no_cuda(self, mocker):
        mocker.patch("torch.cuda.device_count", return_value=0)
        mocker.patch("src.classes.config.EvalConfig.from_model_path")
        mocker.patch("src.eval.load_from_disk")

        with pytest.raises(RuntimeError, match="requires CUDA devices"):
            VLLMCipherEvaluator(model_path="dummy", use_spaces=False)

    def test_parse_samples(self, setup_evaluator):
        parsed = setup_evaluator.parse_samples()

        assert len(parsed) == 1

        sample = parsed[0]
        assert sample["index"] == 0
        assert sample["prompt_ids"] == [5, 100, 200, 3]
        assert sample["raw_cipher_ids"] == [100, 200]
        assert sample["target_length"] == 2
        assert sample["true_plain"] == "ab"

    def test_process_outputs(self, setup_evaluator):
        parsed_samples = setup_evaluator.parse_samples()

        outputs = [MockRequestOutput(token_ids=[7, 8])]

        results = setup_evaluator.process_outputs(
            parsed_samples=parsed_samples, outputs=outputs, total_time=1.5
        )

        assert len(results) == 1
        assert results[0]["ser"] == 0.0
        assert results[0]["wrong_spaces"] == 0

        assert setup_evaluator.summary_scores_path.exists()
        log_content = setup_evaluator.summary_scores_path.read_text()

        assert "group_results" in log_content
        assert '"avg_ser": 0.0' in log_content

    def test_run_executes_successfully(self, setup_evaluator, mocker, tmp_path):
        real_tokenizer_dir = tmp_path / "valid_tokenizer"
        real_tokenizer_dir.mkdir()
        mocker.patch("src.classes.config.EvalConfig.tokenizer_dir", real_tokenizer_dir)

        mocker.patch("src.eval_utils.run_preflight_checks")

        mock_llm_class = mocker.patch("src.eval.LLM")
        mock_llm_instance = mock_llm_class.return_value
        mock_llm_instance.generate.return_value = [MockRequestOutput(token_ids=[7, 8])]

        mock_llm_instance.get_tokenizer.return_value = None

        results = setup_evaluator.run()

        assert len(results) == 1
        assert results[0]["ser"] == 0.0
        mock_llm_instance.generate.assert_called_once()

    def test_run_aborts_without_global_tokenizer(
        self, setup_evaluator, mocker, tmp_path
    ):
        missing_tokenizer_dir = tmp_path / "missing_tokenizer"
        mocker.patch(
            "src.classes.config.EvalConfig.tokenizer_dir", missing_tokenizer_dir
        )

        with pytest.raises(SystemExit) as exc_info:
            setup_evaluator.run()

        assert exc_info.value.code == 1
