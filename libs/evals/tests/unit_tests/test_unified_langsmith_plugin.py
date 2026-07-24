"""Tests for the comparison-aware LangSmith plugin's session naming."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from uuid import NAMESPACE_URL, uuid5

import pytest

from deepagents_harbor.unified_langsmith_plugin import UnifiedComparisonLangSmithPlugin

if TYPE_CHECKING:
    from harbor.job import Job


class _Resp:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


def _fake_job() -> Job:
    # A structural stand-in for harbor's Job (only .id / .config.job_name are read).
    return cast(
        "Job",
        SimpleNamespace(
            id="abcdef1234567890",
            config=SimpleNamespace(job_name="jobname"),
            job_dir="/tmp/job",
        ),
    )


def _plugin(
    *,
    experiment_name: str | None = "exp-name",
    source_branch: str | None = None,
    model: str | None = None,
    category: str | None = None,
) -> UnifiedComparisonLangSmithPlugin:
    return UnifiedComparisonLangSmithPlugin(
        experiment_name=experiment_name,
        api_key="test-key",
        sync_dataset=False,
        source_branch=source_branch,
        model=model,
        category=category,
    )


def test_setup_creates_session_with_exact_unsuffixed_name(monkeypatch) -> None:
    plugin = _plugin(source_branch="main", model="openai:gpt-5.6-terra", category="context")
    calls: list[tuple[str, str, object]] = []

    def fake_request(method: str, path: str, *, json: object = None, **_kwargs: object) -> _Resp:
        calls.append((method, path, json))
        return _Resp(201)

    monkeypatch.setattr(plugin, "_request", fake_request)
    plugin._setup(_fake_job())

    method, path, raw_payload = calls[0]
    payload = cast("dict[str, object]", raw_payload)
    assert (method, path) == ("POST", "/sessions")
    # The whole point: the session name is the exact experiment name, NOT
    # suffixed with the harbor job id.
    assert payload["name"] == "exp-name"
    expected_id = str(uuid5(NAMESPACE_URL, "deepagents-unified:exp-name"))
    assert payload["id"] == expected_id
    assert plugin._experiment_id == expected_id
    extra = cast("dict[str, object]", payload["extra"])
    md = cast("dict[str, object]", extra["metadata"])
    assert md["source_branch"] == "main"
    assert md["model"] == "openai:gpt-5.6-terra"
    assert md["category"] == "context"
    assert md["deepagents_unified"] is True


def test_setup_is_deterministic_across_shards(monkeypatch) -> None:
    ids = []
    for _ in range(2):
        plugin = _plugin()
        monkeypatch.setattr(plugin, "_request", lambda *_a, **_k: _Resp(201))
        plugin._setup(_fake_job())
        ids.append(plugin._experiment_id)
    # Two independent shards resolve to the same session id.
    assert ids[0] == ids[1]


def test_setup_reuses_existing_session_on_conflict(monkeypatch) -> None:
    plugin = _plugin()
    monkeypatch.setattr(plugin, "_request", lambda *_a, **_k: _Resp(409))
    monkeypatch.setattr(plugin, "_find_session", lambda _name: "existing-session-id")
    plugin._setup(_fake_job())
    assert plugin._experiment_id == "existing-session-id"


def test_setup_requires_experiment_name(monkeypatch) -> None:
    monkeypatch.delenv("HARBOR_LANGSMITH_EXPERIMENT", raising=False)
    plugin = UnifiedComparisonLangSmithPlugin(
        experiment_name=None, api_key="test-key", sync_dataset=False
    )
    with pytest.raises(ValueError, match="experiment_name"):
        plugin._setup(_fake_job())


def test_comparison_metadata_drops_empty_fields() -> None:
    plugin = _plugin(source_branch="main", model=None, category="")
    assert plugin._comparison_metadata == {"source_branch": "main"}
