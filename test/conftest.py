import sys
import types
import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest before test collection starts.

    We use this hook to inject a dummy vLLM module on systems where it cannot
    be installed (like Windows). Because pytest executes imports during test
    collection, src/eval.py will throw a ModuleNotFoundError before the tests
    even run. This mock prevents that by pre-caching a dummy module.
    """
    if "vllm" not in sys.modules:
        # Create a blank dummy module natively
        dummy_vllm = types.ModuleType("vllm")

        # Create basic empty classes that can accept any initialization arguments
        class DummyLLM:
            def __init__(self, *args, **kwargs):
                pass

        class DummySamplingParams:
            def __init__(self, *args, **kwargs):
                pass

        class DummyRequestOutput:
            def __init__(self, *args, **kwargs):
                pass

        # Attach them to our dummy module
        dummy_vllm.LLM = DummyLLM  # type: ignore
        dummy_vllm.SamplingParams = DummySamplingParams  # type: ignore
        dummy_vllm.RequestOutput = DummyRequestOutput  # type: ignore

        # Inject the dummy module into the system
        sys.modules["vllm"] = dummy_vllm
