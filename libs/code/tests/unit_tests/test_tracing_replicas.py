"""Tests for LangSmith dual-write replica configuration in `config`."""

from __future__ import annotations

import logging

from deepagents_code import config
from deepagents_code._env_vars import LANGSMITH_REPLICA_PROJECTS


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
    # The warning must name both the kept project and the dropped one, so a
    # swapped-format-arg regression (claiming the wrong project is used) trips.
    assert "first-proj" in caplog.text
    assert "second-proj" in caplog.text
