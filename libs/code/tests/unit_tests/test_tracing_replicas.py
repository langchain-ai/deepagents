"""Tests for LangSmith dual-write replica configuration in `config`."""

from __future__ import annotations

from typing import Any

import pytest

from deepagents_code import config
from deepagents_code._env_vars import LANGSMITH_REPLICA_PROJECTS


def test_get_replica_projects_unset_returns_empty(monkeypatch) -> None:
    """No env var means no extra replica destinations."""
    monkeypatch.delenv(LANGSMITH_REPLICA_PROJECTS, raising=False)
    assert config.get_langsmith_replica_projects() == []


def test_get_replica_projects_parses_dedupes_and_strips(monkeypatch) -> None:
    """Comma-separated names are trimmed, de-duplicated, and order-preserved."""
    monkeypatch.setenv(LANGSMITH_REPLICA_PROJECTS, " prod , staging ,prod, ")
    assert config.get_langsmith_replica_projects() == ["prod", "staging"]


def test_replica_context_noop_when_unset(monkeypatch) -> None:
    """Context manager must not touch tracing_context when no extras are set."""
    monkeypatch.delenv(LANGSMITH_REPLICA_PROJECTS, raising=False)

    def _boom(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("tracing_context should not be entered when feature is off")

    monkeypatch.setattr("langsmith.tracing_context", _boom)
    with config.langsmith_replica_context():
        pass


def test_replica_context_noop_when_tracing_inactive(monkeypatch) -> None:
    """Extras with no active primary project (tracing off) is a no-op."""
    monkeypatch.setenv(LANGSMITH_REPLICA_PROJECTS, "prod")
    monkeypatch.setattr(config, "get_langsmith_project_name", lambda: None)

    def _boom(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("tracing_context should not be entered")

    monkeypatch.setattr("langsmith.tracing_context", _boom)
    with config.langsmith_replica_context():
        pass


def test_replica_context_includes_primary_and_extras(monkeypatch) -> None:
    """Replicas list the primary project first, then each de-duplicated extra."""
    monkeypatch.setenv(LANGSMITH_REPLICA_PROJECTS, "deepagents-code, user-proj")
    monkeypatch.setattr(config, "get_langsmith_project_name", lambda: "deepagents-code")

    captured: dict[str, Any] = {}

    class _DummyCtx:
        def __enter__(self) -> None:
            return None

        def __exit__(self, *_exc: object) -> bool:
            return False

    def _fake_tracing_context(**kwargs: Any) -> _DummyCtx:
        captured.update(kwargs)
        return _DummyCtx()

    monkeypatch.setattr("langsmith.tracing_context", _fake_tracing_context)

    with config.langsmith_replica_context():
        pass

    projects = [r["project_name"] for r in captured["replicas"]]
    assert projects == ["deepagents-code", "user-proj"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
