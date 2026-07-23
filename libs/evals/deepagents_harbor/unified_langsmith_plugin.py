"""Comparison-aware Harbor LangSmith plugin that pins one stable experiment.

The stock ``harbor_langsmith.LangSmithPlugin`` names each session
``f"{experiment_name}-{job.id[:8]}"``. Unified evals run one Harbor job per
shard, so that suffix scatters a leaf's rollouts across a different LangSmith
project per shard, none matching the experiment name recorded in
``summary.json``. The usage collector queries by that recorded name, so it
finds nothing. This subclass overrides only ``_setup`` to create (or reuse) a
single session named exactly ``experiment_name`` with a deterministic id, so
every shard of a leaf lands in the one project the collector queries.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import NAMESPACE_URL, uuid5

from harbor_langsmith import LangSmithPlugin

if TYPE_CHECKING:
    from harbor.job import Job
    from harbor.trial.hooks import TrialHookEvent

_CONFLICT = 409


class UnifiedComparisonLangSmithPlugin(LangSmithPlugin):
    """Reuse one LangSmith experiment across every shard of a comparison leaf."""

    def __init__(
        self,
        *,
        source_branch: str | None = None,
        source_sha: str | None = None,
        model: str | None = None,
        category: str | None = None,
        agent_config: str | None = None,
        version_id: str | None = None,
        workflow_run_id: str | None = None,
        workflow_run_attempt: str | None = None,
        dataset_name: str | None = None,
        experiment_name: str | None = None,
        endpoint: str | None = None,
        api_key: str | None = None,
        workspace_id: str | None = None,
        sync_dataset: bool | None = None,
        fail_fast: bool | None = None,
    ) -> None:
        """Initialize comparison provenance plus the normal plugin options."""
        super().__init__(
            dataset_name=dataset_name,
            experiment_name=experiment_name,
            endpoint=endpoint,
            api_key=api_key,
            workspace_id=workspace_id,
            sync_dataset=sync_dataset,
            fail_fast=fail_fast,
        )
        # Only non-empty fields become metadata, so a partial invocation from the
        # workflow does not stamp empty strings onto every run.
        pairs = {
            "comparison_version": version_id,
            "source_branch": source_branch,
            "source_sha": source_sha,
            "agent_config": agent_config,
            "model": model,
            "category": category,
            "github_run_id": workflow_run_id,
            "github_run_attempt": workflow_run_attempt,
        }
        self._comparison_metadata = {k: v for k, v in pairs.items() if v}

    def _setup(self, job: Job) -> None:
        """Create or reuse the exact experiment name across concurrent shards."""
        if not self.api_key:
            msg = "LANGSMITH_API_KEY is required for LangSmithPlugin"
            raise RuntimeError(msg)
        if not self.experiment_name:
            msg = "UnifiedComparisonLangSmithPlugin requires experiment_name"
            raise ValueError(msg)

        endpoint = (self.endpoint or "https://api.smith.langchain.com").rstrip("/")
        self._base_url = endpoint if endpoint.endswith("/api/v1") else f"{endpoint}/api/v1"
        self._session.headers.update({"x-api-key": self.api_key})
        if self.workspace_id:
            self._session.headers.update({"LANGSMITH-WORKSPACE-ID": self.workspace_id})

        if self.sync_dataset:
            self._dataset_id = self._get_or_create_dataset(job)
            self._example_ids = self._get_or_create_examples(job)

        # Deterministic id derived from the experiment name: concurrent shards
        # converge on the same session instead of racing to create distinct ones.
        experiment_id = str(uuid5(NAMESPACE_URL, f"deepagents-unified:{self.experiment_name}"))
        payload: dict[str, Any] = {
            "id": experiment_id,
            "name": self.experiment_name,
            "start_time": self._format_time(datetime.now(UTC)),
            "extra": {
                "metadata": {
                    "ls_runner": "harbor",
                    "deepagents_unified": True,
                    "harbor_job_id": str(job.id),
                    "harbor_job_name": job.config.job_name,
                    **self._comparison_metadata,
                }
            },
        }
        if self._dataset_id is not None:
            payload["reference_dataset_id"] = self._dataset_id

        response = self._request(
            "POST", "/sessions", json=payload, ok_statuses={200, 201, _CONFLICT}
        )
        if response.status_code == _CONFLICT:
            experiment_id = self._find_session(self.experiment_name) or experiment_id
        self._experiment_id = experiment_id

    def _trial_metadata(self, event: TrialHookEvent) -> dict[str, Any]:
        """Stamp the immutable comparison identity onto every Harbor run node."""
        return {**super()._trial_metadata(event), **self._comparison_metadata}
