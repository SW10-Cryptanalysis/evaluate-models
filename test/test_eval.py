from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from src.eval import VLLMCipherEvaluator, main


@pytest.fixture
def mock_eval_config(mocker):
    """Fixture to mock EvalConfig and its properties."""
    config_mock = mocker.MagicMock()
    config_mock.sep_token_id = 99
    config_mock.max_context = 100
    config_mock.vocab_size = 1000
    mocker.patch("src.eval.EvalConfig.from_model_path", return_value=config_mock)
    return config_mock


@pytest.fixture
def base_evaluator(mocker, mock_eval_config):
    """Fixture providing a base initialized VLLMCipherEvaluator."""
    mocker.patch("src.eval.torch.cuda.device_count", return_value=1)
    mocker.patch("src.eval.load_from_disk", return_value=[])
    mocker.patch("src.eval.eval_utils.build_allowed_token_ids", return_value=[1, 2, 3])
    return VLLMCipherEvaluator(
        model_path="/fake/model", use_spaces=False, mapping=False
    )


def test_init_no_gpu(mocker, mock_eval_config):
    """Verify that initialization fails when no CUDA devices are available."""
    mocker.patch("src.eval.torch.cuda.device_count", return_value=0)
    mocker.patch("src.eval.load_from_disk")
    with pytest.raises(RuntimeError, match="vLLM requires CUDA devices"):
        VLLMCipherEvaluator(model_path="/fake/model", use_spaces=False, mapping=False)


@dataclass
class ParseSamplesCase:
    """Defines test cases for the parse_samples method testing mapping branches."""

    name: str
    mapping: bool
    dataset: list[dict]
    max_context: int
    expected_skipped: int
    expected_target_lengths: list[int]


parse_cases = [
    ParseSamplesCase(
        name="mapping_true_unique_length",
        mapping=True,
        dataset=[
            {"input_ids": [1, 10, 11, 10, 99], "raw_plaintext": "ABA", "redundancy": 1}
        ],
        max_context=100,
        expected_skipped=0,
        expected_target_lengths=[2],
    ),
    ParseSamplesCase(
        name="mapping_false_full_length",
        mapping=False,
        dataset=[
            {"input_ids": [1, 10, 11, 10, 99], "raw_plaintext": "ABA", "redundancy": 1}
        ],
        max_context=100,
        expected_skipped=0,
        expected_target_lengths=[3],
    ),
    ParseSamplesCase(
        name="missing_sep_token",
        mapping=False,
        dataset=[
            {"input_ids": [1, 10, 11, 10], "raw_plaintext": "ABA", "redundancy": 1}
        ],
        max_context=100,
        expected_skipped=0,
        expected_target_lengths=[],
    ),
    ParseSamplesCase(
        name="exceeds_max_context",
        mapping=True,
        dataset=[
            {"input_ids": [1, 10, 11, 10, 99], "raw_plaintext": "ABA", "redundancy": 1}
        ],
        max_context=2,
        expected_skipped=1,
        expected_target_lengths=[],
    ),
    ParseSamplesCase(
        name="zero_target_length",
        mapping=False,
        dataset=[{"input_ids": [1, 99], "raw_plaintext": "", "redundancy": 1}],
        max_context=100,
        expected_skipped=0,
        expected_target_lengths=[],
    ),
]


@pytest.mark.parametrize("case", parse_cases, ids=lambda c: c.name)
def test_parse_samples(base_evaluator, case):
    """Test parse_samples with various context limits, lengths, and mapping configurations."""
    base_evaluator.dataset = case.dataset
    base_evaluator.config.max_context = case.max_context
    base_evaluator.config.mapping = case.mapping

    parsed = base_evaluator.parse_samples()

    assert base_evaluator.skipped_count == case.expected_skipped
    assert len(parsed) == len(case.expected_target_lengths)
    for p, expected_len in zip(parsed, case.expected_target_lengths, strict=False):
        assert p["target_length"] == expected_len


def test_derive_plaintext(mocker, base_evaluator):
    """Verify that mapping key dictionaries are constructed and applied properly."""
    mock_decode = mocker.patch(
        "src.eval.eval_utils.decode_prediction", return_value="decoded_string"
    )

    raw_cipher_ids = [30, 40, 30]
    pred_ids = [65, 66]

    result = base_evaluator.derive_plaintext(raw_cipher_ids, pred_ids)

    assert result == "decoded_string"
    mock_decode.assert_called_once_with([65, 66, 65], base_evaluator.config)


@dataclass
class ProcessEntryCase:
    """Defines test cases for branching inside _process_entry."""

    name: str
    mapping: bool
    mock_plain: str


entry_cases = [
    ProcessEntryCase("mapping_true", True, "DERIVED_PLAIN"),
    ProcessEntryCase("mapping_false", False, "DECODED_PLAIN"),
]


@pytest.mark.parametrize("case", entry_cases, ids=lambda c: c.name)
def test_process_entry(mocker, base_evaluator, case):
    """Verify the correct decoding function is called based on the mapping flag."""
    base_evaluator.config.mapping = case.mapping

    mock_derive = mocker.patch.object(
        base_evaluator, "derive_plaintext", return_value=case.mock_plain
    )
    mock_decode = mocker.patch(
        "src.eval.eval_utils.decode_prediction", return_value=case.mock_plain
    )
    mocker.patch("src.eval.eval_utils.calculate_ser", return_value=(0.1, 0))
    mocker.patch("src.eval.eval_utils.decode_ciphertext", return_value="CIPHER")

    sample = {
        "index": 0,
        "target_length": 2,
        "raw_cipher_ids": [10, 11],
        "true_plain": "AB",
        "redundancy": 1,
    }

    output = MagicMock()
    output.outputs = [MagicMock(token_ids=[65, 66, 67])]

    result = base_evaluator._process_entry(sample, output)

    assert result["predicted_plaintext"] == case.mock_plain
    assert result["ser"] == 0.1
    if case.mapping:
        mock_derive.assert_called_once_with([10, 11], [65, 66])
        mock_decode.assert_not_called()
    else:
        mock_decode.assert_called_once_with([65, 66], base_evaluator.config)
        mock_derive.assert_not_called()


def test_write_log(mocker, base_evaluator):
    """Ensure aggregation logging handles empty and populated result batches securely."""
    mock_open = mocker.patch("builtins.open", mocker.mock_open())
    mocker.patch("src.eval.json.dump")

    all_results = [{"index": 0, "ser": 0.1}]
    evaluation_stats = {"group_logs": []}
    group_stats = {(100, 1): {"total_ser": 0.5, "wrong_spaces": 1, "count": 2}}
    total_stats = {"total_ser": 1.0, "wrong_spaces": 0, "total_time": 5.0}

    base_evaluator._write_log(all_results, evaluation_stats, group_stats, total_stats)

    mock_open.assert_called_once_with(base_evaluator.output_log_path, "w")
    handle = mock_open()
    assert handle.write.call_count == 3

    mock_open.reset_mock()
    base_evaluator._write_log([], evaluation_stats, {}, total_stats)
    assert mock_open().write.call_count == 1


def test_process_outputs(mocker, base_evaluator):
    """Assess statistical aggregation logic across the entire result payload."""
    mocker.patch("src.eval.eval_utils.closest_n", return_value=100)
    mock_write = mocker.patch.object(base_evaluator, "_write_log")
    mock_open = mocker.patch("builtins.open", mocker.mock_open())
    mocker.patch("src.eval.json.dump")

    mock_entry_1 = {"index": 1, "ser": 0.1, "wrong_spaces": 0}
    mock_entry_2 = {"index": 0, "ser": 0.2, "wrong_spaces": 1}
    mocker.patch.object(
        base_evaluator, "_process_entry", side_effect=[mock_entry_1, mock_entry_2]
    )

    samples = [
        {"raw_cipher_ids": [10], "redundancy": 1},
        {"raw_cipher_ids": [10], "redundancy": 1},
    ]

    res = base_evaluator.process_outputs(samples, [MagicMock(), MagicMock()], 10.0)

    assert res[0]["index"] == 0
    assert res[1]["index"] == 1

    mock_write.assert_called_once()
    passed_total_stats = mock_write.call_args[0][3]
    assert passed_total_stats["total_ser"] == pytest.approx(0.3)

    mock_open.assert_called_once_with(base_evaluator.stats_log_path, "w")


def test_run_missing_tokenizer(mocker, base_evaluator):
    """Ensure the evaluation terminates gracefully if the global tokenizer directory does not exist."""
    mock_path = mocker.MagicMock()
    mock_path.exists.return_value = False
    mocker.patch(
        "src.eval.EvalConfig.tokenizer_dir",
        new_callable=mocker.PropertyMock,
        return_value=mock_path,
    )

    mock_exit = mocker.patch("src.eval.sys.exit", side_effect=SystemExit)

    with pytest.raises(SystemExit):
        base_evaluator.run()

    mock_exit.assert_called_once_with(1)


def test_run_success(mocker, base_evaluator):
    """Test the standard inference loop spanning tokenizer initialization, generation, and formatting."""
    mock_path = mocker.MagicMock()
    mock_path.exists.return_value = True
    mocker.patch(
        "src.eval.EvalConfig.tokenizer_dir",
        new_callable=mocker.PropertyMock,
        return_value=mock_path,
    )

    mock_llm_instance = mocker.MagicMock()
    mock_llm_instance.get_tokenizer().vocab_size = 1000
    mock_llm_instance.generate.return_value = ["mock_output_1", "mock_output_2"]
    mocker.patch("src.eval.LLM", return_value=mock_llm_instance)

    mocker.patch("src.eval.eval_utils.run_preflight_checks")

    mock_parsed = [
        {"target_length": 2, "prompt_ids": [1]},
        {"target_length": 4, "prompt_ids": [1, 2]},
    ]
    mocker.patch.object(base_evaluator, "parse_samples", return_value=mock_parsed)

    mocker.patch.object(base_evaluator, "process_outputs", return_value=["final_res"])
    mocker.patch("src.eval.SamplingParams")

    res = base_evaluator.run()

    assert res == ["final_res"]
    called_prompts = mock_llm_instance.generate.call_args[0][0]
    assert called_prompts[0]["prompt_token_ids"] == [1, 2]
    assert called_prompts[1]["prompt_token_ids"] == [1]


def test_run_no_tokenizer_fallback(mocker, base_evaluator):
    """Trigger the conditional logic applying config vocab_size if llm.get_tokenizer returns None."""
    mock_path = mocker.MagicMock()
    mock_path.exists.return_value = True
    mocker.patch(
        "src.eval.EvalConfig.tokenizer_dir",
        new_callable=mocker.PropertyMock,
        return_value=mock_path,
    )

    mock_llm_instance = mocker.MagicMock()
    mock_llm_instance.get_tokenizer.return_value = None
    mocker.patch("src.eval.LLM", return_value=mock_llm_instance)

    mocker.patch("src.eval.eval_utils.run_preflight_checks")
    mocker.patch.object(base_evaluator, "parse_samples", return_value=[])
    mocker.patch.object(base_evaluator, "process_outputs", return_value=[])

    res = base_evaluator.run()
    assert res == []


def test_main(mocker):
    """Validate argument passing directly to the VLLMCipherEvaluator module via cli interface."""
    mocker.patch(
        "sys.argv", ["eval.py", "--model_path", "/fake/model", "--spaces", "--mapping"]
    )
    mock_evaluator_cls = mocker.patch("src.eval.VLLMCipherEvaluator")
    mock_instance = mock_evaluator_cls.return_value

    main()

    mock_evaluator_cls.assert_called_once_with(
        model_path="/fake/model", use_spaces=True, mapping=True
    )
    mock_instance.run.assert_called_once()
