"""Tests for LangSmith dual-write replica configuration in `config`."""

from __future__ import annotations

import logging

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


def test_replica_project_none_when_unset(monkeypatch) -> None:
    """No env var means no project to mirror to."""
    monkeypatch.delenv(LANGSMITH_REPLICA_PROJECTS, raising=False)
    assert config.get_langsmith_replica_project() is None


def test_replica_project_returns_single(monkeypatch) -> None:
    """A single configured project is returned as-is."""
    monkeypatch.setenv(LANGSMITH_REPLICA_PROJECTS, "mason-dual-trace")
    assert config.get_langsmith_replica_project() == "mason-dual-trace"


def test_replica_project_uses_first_and_warns_on_extras(monkeypatch, caplog) -> None:
    """The server mirrors to one project, so only the first is used (with a warning)."""
    monkeypatch.setenv(LANGSMITH_REPLICA_PROJECTS, "first-proj, second-proj")

    with caplog.at_level(logging.WARNING):
        result = config.get_langsmith_replica_project()

    assert result == "first-proj"
    assert "second-proj" in caplog.text
