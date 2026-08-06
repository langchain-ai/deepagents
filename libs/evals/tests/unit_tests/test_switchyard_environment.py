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

    assert health.startswith("/opt/harbor-langgraph-venv/bin/python ")
    assert "http://switchyard:4000/health" in health
    assert "json.load(response)" not in health
    assert "http://switchyard:4000/v1/stats" in stats
    assert "json.load(response)" in stats


def test_bash_health_command_does_not_require_http_client() -> None:
    command = switchyard_environment._bash_health_command()

    assert command.startswith("bash -lc ")
    assert "/dev/tcp/switchyard/4000" in command
    assert "GET /health HTTP/1.0" in command
    assert " 200 " in command
    assert "python" not in command
    assert "curl" not in command


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


@pytest.mark.parametrize(
    ("memory", "expected"),
    [(1024**3, 2 * 1024**3), (8 * 1024**3, 8 * 1024**3)],
)
def test_sandbox_payload_clamps_only_memory_below_langsmith_minimum(
    monkeypatch: pytest.MonkeyPatch,
    memory: int,
    expected: int,
) -> None:
    environment = object.__new__(switchyard_environment.SwitchyardLangSmithEnvironment)
    monkeypatch.setattr(
        LangSmithEnvironment,
        "_create_sandbox_payload",
        lambda _self, _snapshot: {
            "vcpus": 1,
            "mem_bytes": memory,
            "proxy_config": {"access_control": {"deny_list": ["*"]}},
        },
    )

    payload = environment._create_sandbox_payload(None)

    assert payload == {"vcpus": 1, "mem_bytes": expected}


async def test_compose_up_uses_extended_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    environment = object.__new__(switchyard_environment.SwitchyardLangSmithEnvironment)
    calls: list[tuple[list[str], int | None]] = []
    expected = SimpleNamespace(return_code=0, stdout="", stderr="")

    async def fake_compose(
        _self: object,
        subcommand: list[str],
        timeout_sec: int | None = None,
    ) -> SimpleNamespace:
        calls.append((subcommand, timeout_sec))
        return expected

    monkeypatch.setattr(LangSmithEnvironment, "_compose_exec", fake_compose)

    result = await environment._compose_exec(["up", "-d"], timeout_sec=120)

    assert result is expected
    assert calls == [(["up", "-d"], 1500)]


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


def test_compose_network_gives_main_temporary_setup_egress() -> None:
    compose_path = Path(__file__).parents[2] / "switchyard/compose/switchyard.yaml"
    compose = yaml.safe_load(compose_path.read_text())

    assert compose["services"]["main"]["networks"] == [
        "switchyard-internal",
        "switchyard-egress",
    ]
    assert compose["services"]["switchyard"]["networks"] == [
        "switchyard-internal",
        "switchyard-egress",
    ]
    assert compose["networks"]["switchyard-internal"]["internal"] is True


async def test_isolate_main_after_setup_disconnects_and_verifies_topology() -> None:
    environment = object.__new__(switchyard_environment.SwitchyardLangSmithEnvironment)
    compose_calls: list[list[str]] = []
    sandbox_calls: list[str] = []

    async def fake_compose(
        command: list[str], *, timeout_sec: int | None = None
    ) -> SimpleNamespace:
        compose_calls.append(command)
        assert timeout_sec == 15
        return SimpleNamespace(return_code=0, stdout="main-container\n", stderr="")

    responses = iter(
        (
            SimpleNamespace(return_code=0, stdout="", stderr=""),
            SimpleNamespace(
                return_code=0,
                stdout=json.dumps(
                    {"harbor-project_switchyard-internal": {"NetworkID": "internal"}}
                ),
                stderr="",
            ),
        )
    )

    async def fake_sandbox(command: str, **_kwargs: object) -> SimpleNamespace:
        sandbox_calls.append(command)
        return next(responses)

    async def fake_exec(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(return_code=0, stdout="", stderr="")

    environment._compose_exec = fake_compose
    environment._exec_sandbox = fake_sandbox
    environment._compose_project_name = lambda: "harbor-project"
    environment.exec = fake_exec

    await environment.isolate_main_after_setup()

    assert compose_calls == [["ps", "-q", "main"]]
    assert (
        "docker network disconnect harbor-project_switchyard-egress main-container"
        in sandbox_calls[0]
    )
    assert "docker inspect --format" in sandbox_calls[1]


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
