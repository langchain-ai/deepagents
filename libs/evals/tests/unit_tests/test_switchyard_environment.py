"""Tests for the LangSmith Switchyard sidecar environment."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from deepagents_harbor import switchyard_environment

if TYPE_CHECKING:
    from pathlib import Path


def test_forwarded_compose_env_is_allowlisted() -> None:
    env = switchyard_environment._forwarded_compose_env(
        {
            "ANTHROPIC_API_KEY": "anthropic-secret",
            "BASETEN_API_KEY": "",
            "LANGSMITH_API_KEY": "must-not-reach-compose",
            "SWITCHYARD_IMAGE": "registry.example/switchyard@sha256:digest",
            "UNRELATED": "value",
        }
    )

    assert env == {
        "ANTHROPIC_API_KEY": "anthropic-secret",
        "SWITCHYARD_IMAGE": "registry.example/switchyard@sha256:digest",
    }


def test_python_http_command_uses_only_internal_switchyard_origin() -> None:
    health = switchyard_environment._python_http_command("/health", parse_json=False)
    stats = switchyard_environment._python_http_command("/v1/stats", parse_json=True)

    assert "http://switchyard:4000/health" in health
    assert "json.load(response)" not in health
    assert "http://switchyard:4000/v1/stats" in stats
    assert "json.load(response)" in stats


async def test_docker_daemon_start_detaches_from_langsmith_command() -> None:
    environment = object.__new__(switchyard_environment.SwitchyardLangSmithEnvironment)
    environment.logger = SimpleNamespace(debug=lambda *_args: None)
    calls: list[tuple[str, str | None, int | None]] = []
    responses = iter(
        (
            SimpleNamespace(return_code=1, stdout="", stderr="not ready"),
            SimpleNamespace(return_code=0, stdout="DOCKERD_STARTED\n", stderr=""),
        )
    )

    async def fake_exec(
        command: str,
        *,
        cwd: str | None = None,
        timeout_sec: int | None = None,
    ) -> SimpleNamespace:
        calls.append((command, cwd, timeout_sec))
        return next(responses)

    environment._exec_sandbox = fake_exec

    await environment._ensure_docker_daemon()

    start_command, cwd, timeout = calls[1]
    assert "setsid -f dockerd" in start_command
    assert "</dev/null" in start_command
    assert cwd == "/"
    assert timeout == 15


async def test_docker_daemon_start_rejects_missing_detach_marker() -> None:
    environment = object.__new__(switchyard_environment.SwitchyardLangSmithEnvironment)
    environment.logger = SimpleNamespace(debug=lambda *_args: None)
    responses = iter(
        (
            SimpleNamespace(return_code=1, stdout="", stderr="not ready"),
            SimpleNamespace(return_code=0, stdout="", stderr=""),
        )
    )

    async def fake_exec(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return next(responses)

    environment._exec_sandbox = fake_exec

    with pytest.raises(RuntimeError, match="Failed to detach Docker daemon"):
        await environment._ensure_docker_daemon()


async def test_snapshot_switchyard_stats_writes_trial_artifact(tmp_path: Path) -> None:
    environment = object.__new__(switchyard_environment.SwitchyardLangSmithEnvironment)
    environment._compose_mode = True
    environment._switchyard_config = tmp_path / "routes-nano.toml"
    environment.trial_paths = SimpleNamespace(artifacts_dir=tmp_path / "artifacts")
    environment.logger = SimpleNamespace(warning=lambda *_args: None)

    async def fake_exec(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            return_code=0,
            stdout=json.dumps({"total_requests": 3, "models": {"nano": {"calls": 3}}}),
            stderr="",
        )

    environment.exec = fake_exec

    await environment._snapshot_switchyard_stats()

    written = json.loads((tmp_path / "artifacts" / "switchyard-stats.json").read_text())
    assert written["total_requests"] == 3
    assert written["switchyard_config"] == "routes-nano.toml"


@pytest.mark.parametrize("stdout", ["not-json", "[]"])
async def test_snapshot_switchyard_stats_rejects_invalid_payloads(
    tmp_path: Path,
    stdout: str,
) -> None:
    environment = object.__new__(switchyard_environment.SwitchyardLangSmithEnvironment)
    environment._compose_mode = True
    environment._switchyard_config = tmp_path / "routes-nano.toml"
    environment.trial_paths = SimpleNamespace(artifacts_dir=tmp_path / "artifacts")
    warnings: list[str] = []
    environment.logger = SimpleNamespace(warning=lambda message, *_args: warnings.append(message))

    async def fake_exec(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(return_code=0, stdout=stdout, stderr="")

    environment.exec = fake_exec

    await environment._snapshot_switchyard_stats()

    assert warnings
    assert not (tmp_path / "artifacts" / "switchyard-stats.json").exists()
