"""Shared fixtures for CLI unit tests."""

import contextlib

import pytest


@pytest.fixture(autouse=True, scope="session")
def _warm_model_caches() -> None:
    """Pre-populate model-config caches once per xdist worker.

    Tests like the model-selector UI tests call `get_available_models()` and
    `get_model_profiles()` during widget init.  Without a warm cache the first
    invocation in each worker process pays ~800-1200 ms of disk I/O to discover
    provider profiles via `importlib.util`.  Paying that cost once per session
    instead of once per test shaves significant time off the overall run.

    Tests that explicitly need a clean cache (e.g. `test_model_config.py`) use
    their own function-scoped `clear_caches()` fixture which overrides this.
    """
    with contextlib.suppress(Exception):
        from deepagents_cli.model_config import get_available_models, get_model_profiles

        get_available_models()
        get_model_profiles()


@pytest.fixture(autouse=True)
def _clear_langsmith_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent LangSmith env vars loaded from .env from leaking into tests.

    ``dotenv.load_dotenv()`` runs at ``deepagents_cli.config`` import time and
    may inject ``LANGSMITH_*`` variables from a local ``.env`` file.  These
    cause spurious failures in unit tests that run with ``--disable-socket``
    because the LangSmith client attempts real HTTP requests.

    Each test that *needs* LangSmith variables should set them explicitly via
    ``monkeypatch.setenv`` or ``patch.dict("os.environ", ...)``.
    """
    for key in (
        "LANGSMITH_API_KEY",
        "LANGCHAIN_API_KEY",
        "LANGSMITH_TRACING",
        "LANGCHAIN_TRACING_V2",
        "LANGSMITH_PROJECT",
        "DEEPAGENTS_CLI_LANGSMITH_PROJECT",
        "DEEPAGENTS_CLI_LANGSMITH_API_KEY",
        "DEEPAGENTS_CLI_LANGCHAIN_API_KEY",
        "DEEPAGENTS_CLI_LANGSMITH_TRACING",
        "DEEPAGENTS_CLI_LANGCHAIN_TRACING_V2",
    ):
        monkeypatch.delenv(key, raising=False)
