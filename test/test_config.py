import json
import pytest
from pathlib import Path
from src.classes.config import EvalConfig


@pytest.fixture
def dummy_config():
    return EvalConfig(
        vocab_size=32000,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        max_context=4096,
        use_spaces=True,
    )


# ---------------------------------------------------------------------------
# Test types (No exact value assertions as requested)
# ---------------------------------------------------------------------------


class TestConfigTypes:
    def test_vocab_size_is_int(self, dummy_config):
        assert isinstance(dummy_config.vocab_size, int)

    def test_pad_token_id_is_int(self, dummy_config):
        assert isinstance(dummy_config.pad_token_id, int)

    def test_bos_token_id_is_int(self, dummy_config):
        assert isinstance(dummy_config.bos_token_id, int)

    def test_eos_token_id_is_int(self, dummy_config):
        assert isinstance(dummy_config.eos_token_id, int)

    def test_max_context_is_int(self, dummy_config):
        assert isinstance(dummy_config.max_context, int)

    def test_use_spaces_is_bool(self, dummy_config):
        assert isinstance(dummy_config.use_spaces, bool)

    def test_sep_token_id_is_int(self, dummy_config):
        assert isinstance(dummy_config.sep_token_id, int)

    def test_space_token_id_is_int(self, dummy_config):
        assert isinstance(dummy_config.space_token_id, int)

    def test_char_offset_is_int(self, dummy_config):
        assert isinstance(dummy_config.char_offset, int)

    def test_tokenizer_dir_is_path(self, dummy_config):
        assert isinstance(dummy_config.tokenizer_dir, Path)

    def test_tokenized_dir_is_path(self, dummy_config):
        assert isinstance(dummy_config.tokenized_dir, Path)


# ---------------------------------------------------------------------------
# Test file loading and error handling
# ---------------------------------------------------------------------------


class TestConfigFromModelPath:
    def test_from_model_path_loads_correctly(self, tmp_path):
        # Setup valid config.json
        model_path = tmp_path / "model"
        model_path.mkdir()
        config_file = model_path / "config.json"

        valid_json = {
            "vocab_size": 10000,
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "max_position_embeddings": 2048,
        }
        config_file.write_text(json.dumps(valid_json))

        config = EvalConfig.from_model_path(model_path, use_spaces=False)
        assert isinstance(config, EvalConfig)
        assert isinstance(config.vocab_size, int)

    def test_from_model_path_missing_file_raises_error(self, tmp_path):
        # Empty directory with no config.json
        model_path = tmp_path / "empty_model"
        model_path.mkdir()

        with pytest.raises(FileNotFoundError):
            EvalConfig.from_model_path(model_path, use_spaces=True)

    def test_from_model_path_invalid_json_fails(self, tmp_path):
        # Corrupt JSON
        model_path = tmp_path / "corrupt_model"
        model_path.mkdir()
        config_file = model_path / "config.json"
        config_file.write_text("{ this is not valid json }")

        with pytest.raises(json.JSONDecodeError):
            EvalConfig.from_model_path(model_path, use_spaces=True)

    def test_from_model_path_missing_key_fails(self, tmp_path):
        # JSON missing required keys
        model_path = tmp_path / "bad_keys_model"
        model_path.mkdir()
        config_file = model_path / "config.json"
        config_file.write_text(json.dumps({"wrong_key": 999}))

        with pytest.raises(KeyError):
            EvalConfig.from_model_path(model_path, use_spaces=True)
