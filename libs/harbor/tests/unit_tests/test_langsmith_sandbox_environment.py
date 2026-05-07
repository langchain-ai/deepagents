"""Unit tests for the LangSmith sandbox Harbor environment."""

from pathlib import Path
from typing import Any

import pytest
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths

from deepagents_harbor import LangSmithSandboxEnvironment
from deepagents_harbor.langsmith_sandbox_environment import (
    _k8s_name,
    _snapshot_name,
    _validate_ttl_seconds,
)


class CapturingLangSmithSandboxEnvironment(LangSmithSandboxEnvironment):
    """Test double that captures JSON requests instead of performing HTTP."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.seen_payloads: list[dict[str, Any]] = []
        super().__init__(*args, **kwargs)

    async def _request_json(
        self,
        method: str,
        url: str,
        *,
        body: dict[str, Any] | None = None,
        expected_statuses: set[int],
    ) -> dict[str, Any]:
        self.seen_payloads.append(
            {
                "method": method,
                "url": url,
                "body": body,
                "expected_statuses": expected_statuses,
            }
        )
        return {"stdout": "/app\n", "stderr": "", "exit_code": 0}


def _make_environment(
    tmp_path: Path,
    *,
    task_env_config: EnvironmentConfig | None = None,
    environment_class: type[LangSmithSandboxEnvironment] = LangSmithSandboxEnvironment,
    **kwargs: Any,
) -> LangSmithSandboxEnvironment:
    environment_dir = tmp_path / "environment"
    environment_dir.mkdir()
    trial_paths = TrialPaths(tmp_path / "trial")
    trial_paths.mkdir()
    return environment_class(
        environment_dir=environment_dir,
        environment_name="Smoke Task",
        session_id="trial_ABC/123",
        trial_paths=trial_paths,
        task_env_config=task_env_config or EnvironmentConfig(docker_image="python:3.12-slim"),
        api_key="test-api-key",
        **kwargs,
    )


def test_k8s_name_is_safe_and_stable() -> None:
    name = _k8s_name("harbor", "Trial ABC/123_with-symbols")

    assert name == _k8s_name("harbor", "Trial ABC/123_with-symbols")
    assert name.startswith("harbor-trial-abc-123-with-symbols-")
    assert len(name) <= 63
    assert name.strip("-") == name


def test_snapshot_name_changes_on_force_build() -> None:
    cached = _snapshot_name("Smoke Task", "python:3.12-slim", False, "trial-a")
    forced = _snapshot_name("Smoke Task", "python:3.12-slim", True, "trial-a")

    assert cached != forced
    assert cached == _snapshot_name("Smoke Task", "python:3.12-slim", False, "trial-b")


def test_ttl_validation_requires_minute_alignment() -> None:
    assert _validate_ttl_seconds("idle_ttl_seconds", 0) == 0
    assert _validate_ttl_seconds("idle_ttl_seconds", 120) == 120

    with pytest.raises(ValueError, match="multiple of 60"):
        _validate_ttl_seconds("idle_ttl_seconds", 45)

    with pytest.raises(ValueError, match=">= 0"):
        _validate_ttl_seconds("idle_ttl_seconds", -60)


def test_claim_payload_maps_harbor_config(tmp_path: Path) -> None:
    environment = _make_environment(
        tmp_path,
        task_env_config=EnvironmentConfig(
            docker_image="python:3.12-slim",
            cpus=2,
            memory_mb=4096,
            storage_mb=20480,
            allow_internet=False,
        ),
        idle_ttl_seconds=0,
        delete_after_stop_seconds=3600,
    )

    payload = environment._create_claim_payload("smoke-snapshot")

    assert payload["name"].startswith("harbor-trial-abc-123-")
    assert payload["snapshot_name"] == "smoke-snapshot"
    assert payload["vcpus"] == 2
    assert payload["mem_bytes"] == 4096 * 1024 * 1024
    assert payload["fs_capacity_bytes"] == 20480 * 1024 * 1024
    assert payload["idle_ttl_seconds"] == 0
    assert payload["delete_after_stop_seconds"] == 3600
    assert payload["proxy_config"] == {
        "rules": [],
        "no_proxy": [],
        "access_control": {"deny_list": ["*"]},
    }


def test_ttl_seconds_is_delete_after_stop_alias(tmp_path: Path) -> None:
    environment = _make_environment(tmp_path, ttl_seconds=1800)

    payload = environment._create_claim_payload("smoke-snapshot")

    assert payload["delete_after_stop_seconds"] == 1800


def test_validate_definition_rejects_dockerfile_without_image(tmp_path: Path) -> None:
    environment_dir = tmp_path / "environment"
    environment_dir.mkdir()
    (environment_dir / "Dockerfile").write_text("FROM python:3.12-slim\n")
    trial_paths = TrialPaths(tmp_path / "trial")
    trial_paths.mkdir()

    with pytest.raises(ValueError, match="Dockerfile build/push support"):
        LangSmithSandboxEnvironment(
            environment_dir=environment_dir,
            environment_name="dockerfile-task",
            session_id="trial",
            trial_paths=trial_paths,
            task_env_config=EnvironmentConfig(),
            api_key="test-api-key",
        )


async def test_exec_defaults_to_app_workdir(tmp_path: Path) -> None:
    environment = _make_environment(
        tmp_path, environment_class=CapturingLangSmithSandboxEnvironment
    )
    assert isinstance(environment, CapturingLangSmithSandboxEnvironment)
    environment._dataplane_url = "https://sandbox.example"

    result = await environment.exec("pwd")

    assert result.return_code == 0
    assert environment.seen_payloads == [
        {
            "method": "POST",
            "url": "https://sandbox.example/execute",
            "body": {"command": "pwd", "cwd": "/app"},
            "expected_statuses": {200},
        }
    ]
