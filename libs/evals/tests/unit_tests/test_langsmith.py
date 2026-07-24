"""Tests for LangSmith feedback helpers."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

import pytest

from deepagents_harbor.langsmith import (
    _dataset_ref,
    _download_dataset,
    _extract_reward,
    _headers,
    _process_trial,
    add_feedback,
    resolve_langsmith_api_key,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable
    from pathlib import Path


@pytest.fixture
def trial_dir(tmp_path: Path) -> Path:
    """Return a temporary trial directory."""
    return tmp_path


def _write_result(trial_dir: Path, data: dict[str, Any]) -> None:
    (trial_dir / "result.json").write_text(json.dumps(data))


class _FakeRegistryClient:
    """Offline fake for Harbor registry client behavior."""

    def __init__(
        self,
        result: list[Any] | Awaitable[list[Any]],
    ) -> None:
        self.result = result
        self.calls: list[tuple[str, bool, Path | None]] = []

    def download_dataset(
        self,
        name: str,
        *,
        overwrite: bool = False,
        output_dir: Path | None = None,
    ) -> list[Any] | Awaitable[list[Any]]:
        """Record the call and return the configured result."""
        self.calls.append((name, overwrite, output_dir))
        return self.result


class TestResolveLangsmithApiKey:
    """Tests for resolve_langsmith_api_key."""

    def test_returns_none_when_no_vars_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LANGSMITH_SANDBOX_API_KEY", raising=False)
        monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
        monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
        assert resolve_langsmith_api_key() is None

    def test_ignores_sandbox_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # LANGSMITH_SANDBOX_API_KEY is intentionally NOT used; LANGSMITH_API_KEY wins.
        monkeypatch.setenv("LANGSMITH_SANDBOX_API_KEY", "sandbox-key")
        monkeypatch.setenv("LANGSMITH_API_KEY", "ls-key")
        monkeypatch.setenv("LANGCHAIN_API_KEY", "lc-key")
        value, name = resolve_langsmith_api_key()  # ty: ignore[not-iterable]
        assert value == "ls-key"
        assert name == "LANGSMITH_API_KEY"

    def test_falls_back_to_langsmith_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LANGSMITH_SANDBOX_API_KEY", raising=False)
        monkeypatch.setenv("LANGSMITH_API_KEY", "ls-key")
        monkeypatch.setenv("LANGCHAIN_API_KEY", "lc-key")
        value, name = resolve_langsmith_api_key()  # ty: ignore[not-iterable]
        assert value == "ls-key"
        assert name == "LANGSMITH_API_KEY"

    def test_falls_back_to_langchain_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LANGSMITH_SANDBOX_API_KEY", raising=False)
        monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
        monkeypatch.setenv("LANGCHAIN_API_KEY", "lc-key")
        value, name = resolve_langsmith_api_key()  # ty: ignore[not-iterable]
        assert value == "lc-key"
        assert name == "LANGCHAIN_API_KEY"

    def test_skips_empty_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LANGSMITH_SANDBOX_API_KEY", "")
        monkeypatch.setenv("LANGSMITH_API_KEY", "ls-key")
        monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
        value, name = resolve_langsmith_api_key()  # ty: ignore[not-iterable]
        assert value == "ls-key"
        assert name == "LANGSMITH_API_KEY"


class TestHeaders:
    """Tests for _headers."""

    def test_returns_api_key_header(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")
        monkeypatch.delenv("LANGSMITH_SANDBOX_API_KEY", raising=False)
        monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
        assert _headers() == {"x-api-key": "test-key"}

    def test_raises_when_no_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LANGSMITH_SANDBOX_API_KEY", raising=False)
        monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
        monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
        with pytest.raises(ValueError, match="No LangSmith API key found"):
            _headers()


class TestDownloadDataset:
    """Tests for Harbor registry client compatibility helpers."""

    def test_dataset_ref_preserves_harbor_dataset_ref(self) -> None:
        assert (
            _dataset_ref("terminal-bench/terminal-bench-2", "2.0")
            == "terminal-bench/terminal-bench-2"
        )

    def test_dataset_ref_ignores_deprecated_version_argument(self) -> None:
        assert _dataset_ref("terminal-bench", "2.0") == "terminal-bench"

    def test_download_dataset_uses_factory_client(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _FakeRegistryClient(result=[])

        monkeypatch.setattr(
            "deepagents_harbor.langsmith.RegistryClientFactory.create",
            lambda: fake,
        )

        result = _download_dataset(
            "terminal-bench/terminal-bench-2",
            overwrite=True,
            output_dir=tmp_path,
        )

        assert result == []
        assert fake.calls == [("terminal-bench/terminal-bench-2", True, tmp_path)]

    def test_download_dataset_unwraps_async_result(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def _result() -> list[Any]:
            return ["downloaded"]

        fake = _FakeRegistryClient(result=_result())

        monkeypatch.setattr(
            "deepagents_harbor.langsmith.RegistryClientFactory.create",
            lambda: fake,
        )

        result = _download_dataset(
            "terminal-bench/terminal-bench-2",
            overwrite=False,
            output_dir=tmp_path,
        )

        assert result == ["downloaded"]


class TestExtractReward:
    """Tests for _extract_reward."""

    def test_normal_reward(self, trial_dir: Path) -> None:
        _write_result(
            trial_dir,
            {"verifier_result": {"rewards": {"reward": 0.75}}},
        )
        reward, comment = _extract_reward(trial_dir)
        assert reward == 0.75
        assert comment is None

    def test_zero_reward(self, trial_dir: Path) -> None:
        _write_result(
            trial_dir,
            {"verifier_result": {"rewards": {"reward": 0.0}}},
        )
        reward, comment = _extract_reward(trial_dir)
        assert reward == 0.0
        assert comment is None

    def test_negative_reward(self, trial_dir: Path) -> None:
        _write_result(
            trial_dir,
            {"verifier_result": {"rewards": {"reward": -0.5}}},
        )
        reward, comment = _extract_reward(trial_dir)
        assert reward == -0.5
        assert comment is None

    def test_integer_reward_returned_as_float(self, trial_dir: Path) -> None:
        _write_result(
            trial_dir,
            {"verifier_result": {"rewards": {"reward": 1}}},
        )
        reward, comment = _extract_reward(trial_dir)
        assert reward == 1.0
        assert isinstance(reward, float)
        assert comment is None

    def test_missing_verifier_result_falls_back(self, trial_dir: Path) -> None:
        _write_result(trial_dir, {"some_other_key": True})
        reward, comment = _extract_reward(trial_dir)
        assert reward == 0.0
        assert comment is not None
        assert "verifier_result" in comment

    def test_empty_verifier_result_falls_back(self, trial_dir: Path) -> None:
        _write_result(trial_dir, {"verifier_result": {}})
        reward, comment = _extract_reward(trial_dir)
        assert reward == 0.0
        assert comment is not None

    def test_none_verifier_result_falls_back(self, trial_dir: Path) -> None:
        _write_result(trial_dir, {"verifier_result": None})
        reward, comment = _extract_reward(trial_dir)
        assert reward == 0.0
        assert comment is not None

    def test_missing_rewards_key_falls_back(self, trial_dir: Path) -> None:
        _write_result(
            trial_dir,
            {"verifier_result": {"something_else": 1}},
        )
        reward, comment = _extract_reward(trial_dir)
        assert reward == 0.0
        assert comment is not None
        assert "reward" in comment

    def test_empty_rewards_falls_back(self, trial_dir: Path) -> None:
        _write_result(trial_dir, {"verifier_result": {"rewards": {}}})
        reward, comment = _extract_reward(trial_dir)
        assert reward == 0.0
        assert comment is not None

    def test_string_reward_falls_back(self, trial_dir: Path) -> None:
        _write_result(
            trial_dir,
            {"verifier_result": {"rewards": {"reward": "high"}}},
        )
        reward, comment = _extract_reward(trial_dir)
        assert reward == 0.0
        assert comment is not None
        assert "str" in comment

    def test_null_reward_falls_back(self, trial_dir: Path) -> None:
        _write_result(
            trial_dir,
            {"verifier_result": {"rewards": {"reward": None}}},
        )
        reward, comment = _extract_reward(trial_dir)
        assert reward == 0.0
        assert comment is not None

    def test_list_reward_falls_back(self, trial_dir: Path) -> None:
        _write_result(
            trial_dir,
            {"verifier_result": {"rewards": {"reward": [1, 2]}}},
        )
        reward, comment = _extract_reward(trial_dir)
        assert reward == 0.0
        assert comment is not None

    def test_missing_file_raises(self, trial_dir: Path) -> None:
        with pytest.raises(FileNotFoundError, match="does not exist"):
            _extract_reward(trial_dir)

    def test_malformed_json_raises(self, trial_dir: Path) -> None:
        (trial_dir / "result.json").write_text("{bad json")
        with pytest.raises(ValueError, match="malformed JSON"):
            _extract_reward(trial_dir)

    def test_malformed_json_preserves_cause(self, trial_dir: Path) -> None:
        (trial_dir / "result.json").write_text("{bad json")
        with pytest.raises(ValueError, match="malformed JSON") as exc_info:
            _extract_reward(trial_dir)
        assert isinstance(exc_info.value.__cause__, json.JSONDecodeError)


class _FakeRun:
    """Minimal stand-in for a LangSmith run object."""

    def __init__(self, run_id: str) -> None:
        self.id = run_id


class _FakeFeedback:
    """Minimal stand-in for a LangSmith feedback object."""

    def __init__(self, key: str) -> None:
        self.key = key


class _FakeClient:
    """Offline fake for the LangSmith `Client` used by `_process_trial`.

    `list_runs` returns `runs` (or raises it, if it is an exception) and
    `list_feedback` returns `feedback`. Every `create_feedback` call is recorded.
    """

    def __init__(
        self,
        *,
        runs: list[Any] | Exception,
        feedback: list[Any] | None = None,
    ) -> None:
        self._runs = runs
        self._feedback = feedback or []
        self.created: list[dict[str, Any]] = []

    def list_runs(self, *, project_name, filter, is_root):  # noqa: A002  # mirrors Client API
        if isinstance(self._runs, Exception):
            raise self._runs
        return iter(self._runs)

    def list_feedback(self, *, run_ids):
        return iter(self._feedback)

    def create_feedback(self, *, run_id, key, score, comment=None):
        self.created.append({"run_id": run_id, "key": key, "score": score, "comment": comment})


class TestProcessTrial:
    """Tests for `_process_trial` status mapping (offline)."""

    def test_success(self, trial_dir: Path) -> None:
        _write_result(trial_dir, {"verifier_result": {"rewards": {"reward": 1.0}}})
        client: Any = _FakeClient(runs=[_FakeRun("run-1")])

        result = _process_trial(client=client, trial_dir=trial_dir, project_name="proj")

        assert result["status"] == "success"
        assert len(client.created) == 1
        assert client.created[0]["run_id"] == "run-1"
        assert client.created[0]["score"] == 1.0

    def test_fallback_when_no_verifier_result(self, trial_dir: Path) -> None:
        _write_result(trial_dir, {"some_other_key": True})
        client: Any = _FakeClient(runs=[_FakeRun("run-1")])

        result = _process_trial(client=client, trial_dir=trial_dir, project_name="proj")

        assert result["status"] == "fallback"
        assert client.created[0]["score"] == 0.0

    def test_skipped_when_feedback_exists(self, trial_dir: Path) -> None:
        _write_result(trial_dir, {"verifier_result": {"rewards": {"reward": 1.0}}})
        client: Any = _FakeClient(
            runs=[_FakeRun("run-1")], feedback=[_FakeFeedback("harbor_reward")]
        )

        result = _process_trial(client=client, trial_dir=trial_dir, project_name="proj")

        assert result["status"] == "skipped"
        assert client.created == []

    def test_error_when_no_trace(self, trial_dir: Path) -> None:
        client: Any = _FakeClient(runs=[])

        result = _process_trial(client=client, trial_dir=trial_dir, project_name="proj")

        assert result["status"] == "error"
        assert "No trace found" in result["message"]

    def test_error_when_multiple_traces(self, trial_dir: Path) -> None:
        client: Any = _FakeClient(runs=[_FakeRun("run-1"), _FakeRun("run-2")])

        result = _process_trial(client=client, trial_dir=trial_dir, project_name="proj")

        assert result["status"] == "error"
        assert "Multiple traces" in result["message"]

    def test_error_when_fetch_fails(self, trial_dir: Path) -> None:
        client: Any = _FakeClient(runs=ValueError("bad filter"))

        result = _process_trial(client=client, trial_dir=trial_dir, project_name="proj")

        assert result["status"] == "error"
        assert "Failed to fetch trace" in result["message"]

    def test_dry_run_does_not_create_feedback(self, trial_dir: Path) -> None:
        _write_result(trial_dir, {"verifier_result": {"rewards": {"reward": 1.0}}})
        client: Any = _FakeClient(runs=[_FakeRun("run-1")])

        result = _process_trial(
            client=client, trial_dir=trial_dir, project_name="proj", dry_run=True
        )

        assert result["status"] == "success"
        assert "Would add" in result["message"]
        assert client.created == []


class _RoutingClient:
    """Fake `Client` that routes per trial by reading the trial name from the filter.

    Lets `add_feedback` run many trials concurrently while each resolves to a
    distinct, deterministic outcome, so we can assert results map back to the
    correct trial regardless of completion order.
    """

    def __init__(self, statuses: dict[str, str]) -> None:
        self._statuses = statuses
        self.created: list[dict[str, Any]] = []

    @staticmethod
    def _trial_name(filter_query: str) -> str:
        match = re.search(r'eq\(metadata_value, "([^"]+)"\)', filter_query)
        return match.group(1) if match else ""

    def list_runs(self, *, project_name, filter, is_root):  # noqa: A002  # mirrors Client API
        name = self._trial_name(filter)
        if self._statuses.get(name) == "none":
            return iter([])
        return iter([_FakeRun(f"run-{name}")])

    def list_feedback(self, *, run_ids):
        name = run_ids[0].removeprefix("run-")
        if self._statuses.get(name) == "skipped":
            return iter([_FakeFeedback("harbor_reward")])
        return iter([])

    def create_feedback(self, *, run_id, key, score, comment=None):
        self.created.append({"run_id": run_id, "key": key, "score": score, "comment": comment})


class TestAddFeedbackOrdering:
    """End-to-end ordering/reassembly check for `add_feedback`."""

    def test_results_map_to_correct_trial(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Three trials with distinct intended outcomes.
        statuses = {"good": "success", "dupe": "skipped", "missing": "none"}
        for name in statuses:
            d = tmp_path / name
            d.mkdir()
            _write_result(d, {"verifier_result": {"rewards": {"reward": 1.0}}})

        routing = _RoutingClient(statuses)
        monkeypatch.setattr("deepagents_harbor.langsmith.Client", lambda: routing)

        add_feedback(tmp_path, project_name="proj")

        # Only the "good" trial should have feedback created (skipped/missing must not).
        assert [c["run_id"] for c in routing.created] == ["run-good"]

        out = capsys.readouterr().out
        assert "Successfully updated: 1" in out
        assert "Skipped (already has feedback): 1" in out
        assert "Errors: 1" in out
