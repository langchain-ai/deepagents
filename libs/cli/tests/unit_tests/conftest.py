"""Shared fixtures for CLI unit tests."""

import pytest


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
        "DEEPAGENTS_LANGSMITH_PROJECT",
    ):
        monkeypatch.delenv(key, raising=False)
