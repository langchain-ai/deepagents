"""Tests for the LangSmith Switchyard sidecar environment."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from harbor.environments.docker.docker import DockerEnvironment
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


def test_docker_health_and_stats_use_loopback() -> None:
    health = switchyard_environment._bash_health_command("127.0.0.1")
    stats = switchyard_environment._python_http_command(
        "/v1/stats",
        parse_json=True,
        base_url="http://127.0.0.1:4000",
    )

    assert "/dev/tcp/127.0.0.1/4000" in health
    assert "http://127.0.0.1:4000/v1/stats" in stats


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
    ("vcpus", "memory", "expected_vcpus", "expected_memory"),
    [
        (1, 1024**3, 1, 2 * 1024**3),
        (1, 4 * 1024**3, 1, 4 * 1024**3),
        (1, 8 * 1024**3, 2, 8 * 1024**3),
    ],
)
def test_sandbox_payload_preserves_resources_with_valid_langsmith_ratio(
    monkeypatch: pytest.MonkeyPatch,
    vcpus: int,
    memory: int,
    expected_vcpus: int,
    expected_memory: int,
) -> None:
    environment = object.__new__(switchyard_environment.DeepAgentsLangSmithEnvironment)
    monkeypatch.setattr(
        LangSmithEnvironment,
        "_create_sandbox_payload",
        lambda _self, _snapshot: {
            "vcpus": vcpus,
            "mem_bytes": memory,
            "proxy_config": {"access_control": {"deny_list": ["*"]}},
        },
    )

    payload = environment._create_sandbox_payload(None)

    assert payload == {
        "vcpus": expected_vcpus,
        "mem_bytes": expected_memory,
        "proxy_config": {"access_control": {"deny_list": ["*"]}},
    }


async def test_directory_upload_archives_each_nested_path_once(tmp_path: Path) -> None:
    source = tmp_path / "source"
    nested = source / "nested"
    nested.mkdir(parents=True)
    (source / "root.txt").write_text("root")
    (nested / "child.txt").write_text("child")
    environment = object.__new__(switchyard_environment.DeepAgentsLangSmithEnvironment)
    archived: list[str] = []
    commands: list[str] = []

    async def fake_upload(source_path: Path, _target_path: str) -> None:
        with tarfile.open(source_path, "r:gz") as archive:
            archived.extend(member.name for member in archive.getmembers())

    async def fake_exec(command: str, **_kwargs: object) -> SimpleNamespace:
        commands.append(command)
        return SimpleNamespace(return_code=0, stdout="", stderr="")

    environment._upload_file_to_sandbox = fake_upload
    environment._exec_sandbox = fake_exec

    await environment._upload_dir_to_sandbox(source, "/installed-agent")

    assert sorted(archived) == ["nested", "nested/child.txt", "root.txt"]
    assert len(archived) == len(set(archived))
    assert "tar -xzf" in commands[0]


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


def test_docker_compose_sidecar_shares_main_network_namespace() -> None:
    compose_path = Path(__file__).parents[2] / "switchyard/compose/switchyard-docker.yaml"
    compose = yaml.safe_load(compose_path.read_text())
    sidecar = compose["services"]["switchyard"]

    assert sidecar["network_mode"] == "service:main"
    assert "127.0.0.1" in sidecar["command"]
    assert sidecar["volumes"][0]["source"].startswith("${SWITCHYARD_CONFIG_PATH:")
    assert sidecar["environment"] == {
        "ANTHROPIC_API_KEY": "${SWITCHYARD_ANTHROPIC_API_KEY:-}",
        "BASETEN_API_KEY": "${SWITCHYARD_BASETEN_API_KEY:-}",
        "GOOGLE_API_KEY": "${SWITCHYARD_GOOGLE_API_KEY:-}",
        "NVIDIA_API_KEY": "${SWITCHYARD_NVIDIA_API_KEY:-}",
    }


def test_docker_compose_env_includes_validated_route_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = tmp_path / "routes-opus.toml"
    config.write_text("schema_version = 1")
    environment = object.__new__(switchyard_environment.SwitchyardDockerEnvironment)
    environment._switchyard_config = config
    monkeypatch.setattr(
        DockerEnvironment,
        "_compose_env_vars",
        lambda _self, include_os_env=True: {"BASE": str(include_os_env)},
    )

    env = environment._compose_env_vars(include_os_env=False)

    assert env == {"BASE": "False", "SWITCHYARD_CONFIG_PATH": str(config)}


@pytest.mark.parametrize("has_overlay", [False, True])
async def test_docker_start_waits_only_when_switchyard_overlay_is_present(
    monkeypatch: pytest.MonkeyPatch,
    has_overlay: bool,
) -> None:
    environment = object.__new__(switchyard_environment.SwitchyardDockerEnvironment)
    environment.extra_docker_compose_paths = (
        [Path("switchyard-docker.yaml")] if has_overlay else []
    )
    started: list[bool] = []
    waited: list[bool] = []

    async def fake_start(_self: object, force_build: bool) -> None:
        started.append(force_build)

    async def fake_wait() -> None:
        waited.append(True)

    monkeypatch.setattr(DockerEnvironment, "start", fake_start)
    environment._wait_for_switchyard = fake_wait

    await environment.start(force_build=True)

    assert started == [True]
    assert waited == ([True] if has_overlay else [])


async def test_docker_capture_writes_stats_and_stops_sidecar(tmp_path: Path) -> None:
    environment = object.__new__(switchyard_environment.SwitchyardDockerEnvironment)
    environment._switchyard_config = tmp_path / "routes-opus.toml"
    environment._switchyard_stopped = False
    environment.extra_docker_compose_paths = [Path("switchyard-docker.yaml")]
    environment.trial_paths = SimpleNamespace(artifacts_dir=tmp_path / "artifacts")
    environment.logger = SimpleNamespace(warning=lambda *_args: None)
    commands: list[str] = []
    stopped: list[str] = []

    async def fake_exec(command: str, **_kwargs: object) -> SimpleNamespace:
        commands.append(command)
        return SimpleNamespace(
            return_code=0,
            stdout=json.dumps({"total_requests": 2, "models": {}}),
            stderr="",
        )

    async def fake_stop(service: str) -> None:
        stopped.append(service)

    environment.exec = fake_exec
    environment.stop_service = fake_stop

    await environment.capture_and_stop_switchyard()
    await environment.capture_and_stop_switchyard()

    written = json.loads((tmp_path / "artifacts/switchyard-stats.json").read_text())
    assert written["total_requests"] == 2
    assert "http://127.0.0.1:4000/v1/stats" in commands[0]
    assert stopped == ["switchyard"]


async def test_docker_capture_is_noop_without_switchyard_overlay() -> None:
    environment = object.__new__(switchyard_environment.SwitchyardDockerEnvironment)
    environment._switchyard_stopped = False
    environment.extra_docker_compose_paths = []

    async def unexpected_snapshot(**_kwargs: object) -> bool:
        pytest.fail("verifier environment must not capture Switchyard stats")

    async def unexpected_stop(_service: str) -> None:
        pytest.fail("verifier environment must not stop a Switchyard service")

    environment._snapshot_switchyard_stats = unexpected_snapshot
    environment.stop_service = unexpected_stop

    await environment.capture_and_stop_switchyard()


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
