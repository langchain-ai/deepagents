from __future__ import annotations

import urllib.parse
from types import SimpleNamespace
from typing import Any

from deepagents_harbor.unified_langsmith_plugin import (
    UnifiedComparisonLangSmithPlugin,
)


class _Response:
    status_code = 201


def _plugin() -> UnifiedComparisonLangSmithPlugin:
    return UnifiedComparisonLangSmithPlugin(
        dataset_name="dataset",
        experiment_name="deepagents-compare-v1-branch-model-bare-context-1-1",
        api_key="placeholder",
        sync_dataset=False,
        version_id="v1",
        source_branch="feature/todos",
        source_sha="a" * 40,
        agent_config="bare",
        model="openai:gpt",
        category="context",
        workflow_run_id="123",
        workflow_run_attempt="2",
    )


def test_setup_reuses_deterministic_experiment_and_records_source(monkeypatch) -> None:
    plugin = _plugin()
    requests: list[dict[str, Any]] = []
    trial = SimpleNamespace(
        job_id="job-id",
        trial_name="task__attempt-1",
        agent=SimpleNamespace(env={}),
    )
    job = SimpleNamespace(_trial_configs=[trial])

    def request(method: str, path: str, **kwargs: Any) -> _Response:
        requests.append({"method": method, "path": path, **kwargs})
        return _Response()

    monkeypatch.setattr(plugin, "_request", request)
    plugin._setup(job)
    first_id = plugin._experiment_id
    plugin._setup(job)

    assert plugin._experiment_id == first_id
    payload = requests[0]["json"]
    assert payload["name"] == plugin.experiment_name
    assert payload["extra"]["metadata"]["source_sha"] == "a" * 40
    assert payload["extra"]["metadata"]["source_branch"] == "feature/todos"
    assert trial.agent.env["HARBOR_LANGSMITH_PARENT"].endswith(
        plugin._stable_uuid("job-id", "trial", "task__attempt-1")
    )
    baggage = urllib.parse.unquote(trial.agent.env["HARBOR_LANGSMITH_BAGGAGE"])
    assert '"source_sha":"' + "a" * 40 + '"' in baggage
    assert f"langsmith-project={plugin.experiment_name}" in baggage


def test_trial_metadata_includes_comparison_identity(monkeypatch) -> None:
    plugin = _plugin()
    monkeypatch.setattr(
        "harbor_langsmith.LangSmithPlugin._trial_metadata",
        lambda _self, _event: {"source": "harbor"},
    )

    metadata = plugin._trial_metadata(SimpleNamespace())

    assert metadata["source"] == "harbor"
    assert metadata["comparison_version"] == "v1"
    assert metadata["source_sha"] == "a" * 40
