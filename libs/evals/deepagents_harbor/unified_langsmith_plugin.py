"""Comparison-aware Harbor LangSmith experiment grouping."""

from __future__ import annotations

import json
import urllib.parse
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import NAMESPACE_URL, uuid5

from harbor_langsmith import LangSmithPlugin

if TYPE_CHECKING:
    from harbor.job import Job
    from harbor.trial.hooks import TrialHookEvent

_CONFLICT = 409


class UnifiedComparisonLangSmithPlugin(LangSmithPlugin):
    """Reuse one LangSmith experiment across every shard of a comparison leaf.

    The upstream plugin suffixes each experiment with a random Harbor job ID.
    Unified comparison runs intentionally execute one task per Harbor job, so
    that behavior would split a category into one experiment per task. This
    wrapper uses the already-unique comparison experiment name as a stable
    session identity and records the immutable source on every experiment/run.
    """

    def __init__(
        self,
        *,
        version_id: str,
        source_branch: str,
        source_sha: str,
        agent_config: str,
        model: str,
        category: str,
        workflow_run_id: str,
        workflow_run_attempt: str,
        dataset_name: str | None = None,
        experiment_name: str | None = None,
        endpoint: str | None = None,
        api_key: str | None = None,
        workspace_id: str | None = None,
        sync_dataset: bool | None = None,
        fail_fast: bool | None = None,
    ) -> None:
        """Initialize comparison metadata plus normal plugin options."""
        super().__init__(
            dataset_name=dataset_name,
            experiment_name=experiment_name,
            endpoint=endpoint,
            api_key=api_key,
            workspace_id=workspace_id,
            sync_dataset=sync_dataset,
            fail_fast=fail_fast,
        )
        self._comparison_metadata = {
            "comparison_version": version_id,
            "source_branch": source_branch,
            "source_sha": source_sha,
            "agent_config": agent_config,
            "model": model,
            "category": category,
            "github_run_id": workflow_run_id,
            "github_run_attempt": workflow_run_attempt,
        }

    def _setup(self, job: Job) -> None:
        """Create or reuse the exact experiment name across concurrent shards."""
        if not self.api_key:
            msg = "LANGSMITH_API_KEY is required for LangSmithPlugin"
            raise RuntimeError(msg)
        if not self.experiment_name:
            msg = "comparison plugin requires experiment_name"
            raise ValueError(msg)

        endpoint = (self.endpoint or "https://api.smith.langchain.com").rstrip("/")
        self._base_url = endpoint if endpoint.endswith("/api/v1") else f"{endpoint}/api/v1"
        self._session.headers.update({"x-api-key": self.api_key})
        if self.workspace_id:
            self._session.headers.update({"LANGSMITH-WORKSPACE-ID": self.workspace_id})

        if self.sync_dataset:
            self._dataset_id = self._get_or_create_dataset(job)
            self._example_ids = self._get_or_create_examples(job)

        experiment_id = str(uuid5(NAMESPACE_URL, f"deepagents-unified:{self.experiment_name}"))
        payload: dict[str, object] = {
            "id": experiment_id,
            "name": self.experiment_name,
            "start_time": self._format_time(datetime.now(UTC)),
            "extra": {"metadata": self._comparison_metadata},
        }
        if self._dataset_id is not None:
            payload["reference_dataset_id"] = self._dataset_id
        response = self._request(
            "POST", "/sessions", json=payload, ok_statuses={200, 201, _CONFLICT}
        )
        if response.status_code == _CONFLICT:
            experiment_id = self._find_session(self.experiment_name) or experiment_id
        self._experiment_id = experiment_id
        self._attach_parent_context(job)

    def _attach_parent_context(self, job: Job) -> None:
        """Nest each sandbox graph trace under its matching Harbor rollout."""
        if not self.experiment_name:
            return
        metadata = urllib.parse.quote(
            json.dumps(self._comparison_metadata, separators=(",", ":")), safe=""
        )
        project = urllib.parse.quote(self.experiment_name, safe="")
        baggage = f"langsmith-metadata={metadata},langsmith-project={project}"
        for trial in job._trial_configs:  # noqa: SLF001  # plugin configures each planned rollout before Trial creation
            run_id = self._stable_uuid(trial.job_id, "trial", trial.trial_name)
            timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
            trial.agent.env["HARBOR_LANGSMITH_PARENT"] = f"{timestamp}{run_id}"
            trial.agent.env["HARBOR_LANGSMITH_BAGGAGE"] = baggage

    def _trial_metadata(self, event: TrialHookEvent) -> dict[str, object]:
        """Attach immutable comparison identity to every Harbor run node."""
        return {**super()._trial_metadata(event), **self._comparison_metadata}
