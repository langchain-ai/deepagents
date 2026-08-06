"""Tests for the LangSmith Switchyard sidecar environment."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from harbor.environments.langsmith import LangSmithEnvironment

from deepagents_harbor import switchyard_environment


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


def test_sandbox_payload_uses_compose_for_agent_network_isolation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = object.__new__(switchyard_environment.SwitchyardLangSmithEnvironment)
    monkeypatch.setattr(
        LangSmithEnvironment,
        "_create_sandbox_payload",
        lambda _self, _snapshot: {
            "name": "sandbox",
            "proxy_config": {"access_control": {"deny_list": ["*"]}},
        },
    )

    payload = environment._create_sandbox_payload(None)

    assert payload == {"name": "sandbox"}


def test_compose_flags_replace_blanket_no_network_overlay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = object.__new__(switchyard_environment.SwitchyardLangSmithEnvironment)
    no_network = "/harbor/compose/docker-compose-no-network.yaml"
    monkeypatch.setattr(
        LangSmithEnvironment,
        "_compose_file_flags",
        lambda _self: [
            "-f",
            "/harbor/compose/resources.yaml",
            "-f",
            "/harbor/environment/switchyard.yaml",
            "-f",
            no_network,
        ],
    )

    flags = environment._compose_file_flags()

    assert flags == [
        "-f",
        "/harbor/compose/resources.yaml",
        "-f",
        "/harbor/environment/switchyard.yaml",
    ]


def test_compose_network_isolates_main_and_gives_switchyard_egress() -> None:
    compose_path = Path(__file__).parents[2] / "switchyard/compose/switchyard.yaml"
    compose = yaml.safe_load(compose_path.read_text())

    assert compose["services"]["main"]["networks"] == ["switchyard-internal"]
    assert compose["services"]["switchyard"]["networks"] == [
        "switchyard-internal",
        "switchyard-egress",
    ]
    assert compose["networks"]["switchyard-internal"]["internal"] is True


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
