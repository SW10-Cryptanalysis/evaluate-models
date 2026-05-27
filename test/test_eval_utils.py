import pytest
import logging
from pathlib import Path
from src import eval_utils
from src.classes.config import EvalConfig


@pytest.fixture
def mock_config():
    return EvalConfig(
        vocab_size=32000,
        pad_token_id=0,
        bos_token_id=5,
        eos_token_id=6,
        max_context=4096,
        use_spaces=True,
        mapping=False,
    )


class TestEvalUtils:
    def test_closest_n(self):
        assert eval_utils.closest_n(350) == 350
        assert eval_utils.closest_n(349) == 350
        assert eval_utils.closest_n(100000) == 10000
        assert eval_utils.closest_n(0) == 350

    def test_build_allowed_token_ids_with_spaces(self, mock_config):
        ids = eval_utils.build_allowed_token_ids(mock_config)
        assert len(ids) == 27
        assert mock_config.space_token_id in ids

    def test_build_allowed_token_ids_without_spaces(self, mock_config):
        mock_config.use_spaces = False
        ids = eval_utils.build_allowed_token_ids(mock_config)
        assert len(ids) == 26
        assert mock_config.space_token_id not in ids

    def test_decode_prediction(self, mock_config):
        ids = [7, 8, 4, 9]
        decoded = eval_utils.decode_prediction(ids, mock_config)
        assert decoded == "ab_c"

    def test_decode_prediction_out_of_bounds(self, mock_config):
        ids = [1]
        decoded = eval_utils.decode_prediction(ids, mock_config)
        assert decoded == "?"

    def test_decode_ciphertext(self, mock_config):
        ids = [5, 100, 200, 3, 300]
        decoded = eval_utils.decode_ciphertext(ids, mock_config)
        assert decoded == "100 200 300"

    def test_calculate_ser_basic(self):
        assert eval_utils.calculate_ser("hello", "hello") == (0.0, 0)
        assert eval_utils.calculate_ser("hello", "hallo") == (0.2, 0)
        assert eval_utils.calculate_ser("hello", "world") == (0.8, 0)

    def test_calculate_ser_ignores_spaces_and_underscores(self):
        # 10 scored symbols, 0 errors. Both space and underscore should be skipped.
        assert eval_utils.calculate_ser("hello world", "hello_world") == (0.0, 0)

        # 10 scored symbols, 1 error ('e' vs 'a'). The space/underscore is skipped.
        assert eval_utils.calculate_ser("hello world", "hallo_world") == (0.1, 0)

        # 3 scored symbols ('a', 'b', 'c'/'z'). 2 skipped. 1 error out of 3.
        assert eval_utils.calculate_ser("a_b_c", "a_b_z") == (1 / 3, 0)

        # Space in prediction, letter in truth. Tuple is skipped entirely.
        # Scored symbols: 2 ('a', 'b'). Errors: 0.
        assert eval_utils.calculate_ser("axb", "a_b") == (1 / 3, 0)

    def test_calculate_ser_wrong_spaces(self):
        # 10 scored symbols, 0 errors, but 1 wrong space. SER should still be 0.0.
        assert eval_utils.calculate_ser("hello world", "hellotworld") == (0.0, 1)

        # 15 scored symbols, 1 error, and 1 wrong space. SER should be 1/15.
        assert eval_utils.calculate_ser("hello_world_today", "halloxworld_today") == (
            1 / 15,
            1,
        )

        # 20 scored symbols, 0 errors, and 2 wrong spaces. SER should be 0.0.
        assert eval_utils.calculate_ser(
            "hello_world_hello_world", "helloxworld_helloyworld"
        ) == (0.0, 2)

    def test_calculate_ser_all_skipped(self):
        # Edge case where all characters are spaces or underscores
        assert eval_utils.calculate_ser("   ", "___") == (0.0, 0)
        assert eval_utils.calculate_ser("_ _", "   ") == (0.0, 0)

    def test_calculate_ser_empty_string(self):
        with pytest.raises(ValueError, match="True plaintext is empty"):
            eval_utils.calculate_ser("", "pred")

    def test_check_output_file_exists(self, tmp_path):
        dummy_file = tmp_path / "exists.txt"
        dummy_file.touch()
        warnings = eval_utils._check_output_file(dummy_file)
        assert len(warnings) == 1
        assert "OVERWRITTEN" in warnings[0]

    def test_check_vocab_bounds(self):
        errors = eval_utils._check_vocab_bounds([100, 200, 3000], vocab_size=500)
        assert len(errors) == 1
        assert "3000" in errors[0]

    def test_check_dataset_format_valid(self, mock_config):
        dataset = [
            {
                "input_ids": [
                    mock_config.bos_token_id,
                    100,
                    mock_config.sep_token_id,
                    200,
                    mock_config.eos_token_id,
                ]
            }
        ]
        errors = eval_utils._check_dataset_format(dataset, mock_config)
        assert len(errors) == 0

    def test_check_dataset_format_invalid(self, mock_config):
        dataset = [
            {"input_ids": [999, 100, 200, mock_config.eos_token_id]}
        ]  # Missing BOS and SEP
        errors = eval_utils._check_dataset_format(dataset, mock_config)
        assert len(errors) > 0

    def test_run_preflight_checks_raises_runtime_error(self, mock_config):
        logger = logging.getLogger("test_logger")
        dataset = [{"input_ids": [1, 2, 3]}]

        with pytest.raises(RuntimeError, match="Preflight checks failed"):
            eval_utils.run_preflight_checks(
                config=mock_config,
                dataset=dataset,
                allowed_token_ids=[100],
                output_log_path=Path("dummy"),
                vocab_size=32000,
                logger=logger,
            )
