from __future__ import annotations

from deepagents.backends.langsmith import LangSmithBackend


def test_import_langsmith_backend() -> None:
    assert LangSmithBackend is not None
