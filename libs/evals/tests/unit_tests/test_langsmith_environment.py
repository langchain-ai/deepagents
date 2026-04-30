"""Tests for the LangSmith harbor environment adapter."""

from __future__ import annotations

import logging
import textwrap
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths

from deepagents_harbor.langsmith_environment import (
    LangSmithEnvironment,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def _fake_langsmith_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide a fake API key so unit tests never require real credentials."""
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2-test-fake-key")


@dataclass
class _FakeExecResult:
    """Minimal stand-in for langsmith ExecutionResult."""

    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0


@dataclass
class _FakeSandbox:
    """Minimal stand-in for langsmith AsyncSandbox."""

    name: str = "test-sandbox"
    _run_calls: list[tuple] = field(default_factory=list)
    _written_files: dict[str, bytes] = field(default_factory=dict)
    _read_files: dict[str, bytes] = field(default_factory=dict)

    async def run(
        self,
        command: str,
        *,
        timeout: int = 60,  # noqa: ASYNC109 -- mirrors AsyncSandbox.run() signature
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> _FakeExecResult:
        self._run_calls.append((command, timeout, cwd, env))
        return _FakeExecResult()

    async def write(self, path: str, content: str | bytes) -> None:
        if isinstance(content, str):
            content = content.encode()
        self._written_files[path] = content

    async def read(self, path: str) -> bytes:
        return self._read_files.get(path, b"")


def _make_env(
    tmp_path: Path,
    *,
    docker_image: str | None = None,
    dockerfile_content: str | None = None,
    cpus: int = 1,
    memory_mb: int = 2048,
    storage_mb: int = 10240,
) -> LangSmithEnvironment:
    """Create a LangSmithEnvironment with a temp directory.

    Args:
        tmp_path: Temporary directory for environment files.
        docker_image: Prebuilt image name (skips Dockerfile).
        dockerfile_content: Dockerfile content to write.
        cpus: CPU count for the task.
        memory_mb: Memory in MB for the task.
        storage_mb: Storage in MB for the task.
    """
    env_dir = tmp_path / "environment"
    env_dir.mkdir()
    if dockerfile_content:
        (env_dir / "Dockerfile").write_text(dockerfile_content)
    elif not docker_image:
        (env_dir / "Dockerfile").write_text("FROM ubuntu:24.04\n")

    trial_dir = tmp_path / "trial"
    trial_dir.mkdir()

    config = EnvironmentConfig(
        docker_image=docker_image,
        cpus=cpus,
        memory_mb=memory_mb,
        storage_mb=storage_mb,
    )
    trial_paths = TrialPaths(trial_dir=trial_dir)
    trial_paths.mkdir()

    return LangSmithEnvironment(
        environment_dir=env_dir,
        environment_name="test-task",
        session_id="test-session-001",
        trial_paths=trial_paths,
        task_env_config=config,
    )


@dataclass
class _FakeSnapshot:
    """Stand-in for langsmith Snapshot with a safe ``name`` attribute.

    MagicMock reserves ``.name`` as a descriptor for the mock itself, so we
    use a plain dataclass here to make assertions on ``snap.name`` behave
    normally.
    """

    name: str
    status: str = "ready"
    id: str = "snap-id-test"


def _mock_async_client(*, existing_snapshots: list[Any] | None = None) -> AsyncMock:
    """Build a mock AsyncSandboxClient wired for start() tests.

    Defaults mirror the "happy path": no existing snapshot → `create_snapshot`
    will be invoked → `create_sandbox` returns a fake sandbox. Tests can
    override by passing `existing_snapshots` or reassigning the individual
    method mocks.
    """
    mock = AsyncMock()
    fake_sb = _FakeSandbox()
    mock.create_sandbox.return_value = fake_sb
    mock.list_snapshots = AsyncMock(return_value=existing_snapshots or [])
    mock.create_snapshot = AsyncMock(return_value=_FakeSnapshot(name="snap-test", status="ready"))
    return mock


class TestValidation:
    """Tests for __init__-time validation."""

    def test_valid_dockerfile(self, tmp_path: Path) -> None:
        env = _make_env(tmp_path)
        assert env is not None

    def test_valid_docker_image(self, tmp_path: Path) -> None:
        env = _make_env(tmp_path, docker_image="python:3.12-slim")
        assert env is not None

    def test_missing_dockerfile_and_image_raises(self, tmp_path: Path) -> None:
        env_dir = tmp_path / "environment"
        env_dir.mkdir()

        trial_dir = tmp_path / "trial"
        trial_dir.mkdir()
        config = EnvironmentConfig()
        trial_paths = TrialPaths(trial_dir=trial_dir)
        trial_paths.mkdir()

        with pytest.raises(FileNotFoundError, match="LangSmith environment requires"):
            LangSmithEnvironment(
                environment_dir=env_dir,
                environment_name="test",
                session_id="s1",
                trial_paths=trial_paths,
                task_env_config=config,
            )

    def test_gpu_requirement_raises(self, tmp_path: Path) -> None:
        env_dir = tmp_path / "environment"
        env_dir.mkdir()
        (env_dir / "Dockerfile").write_text("FROM ubuntu:24.04\n")

        trial_dir = tmp_path / "trial"
        trial_dir.mkdir()
        config = EnvironmentConfig(gpus=1)
        trial_paths = TrialPaths(trial_dir=trial_dir)
        trial_paths.mkdir()

        with pytest.raises(RuntimeError, match="GPU"):
            LangSmithEnvironment(
                environment_dir=env_dir,
                environment_name="test",
                session_id="s1",
                trial_paths=trial_paths,
                task_env_config=config,
            )

    def test_internet_disabled_raises(self, tmp_path: Path) -> None:
        env_dir = tmp_path / "environment"
        env_dir.mkdir()
        (env_dir / "Dockerfile").write_text("FROM ubuntu:24.04\n")

        trial_dir = tmp_path / "trial"
        trial_dir.mkdir()
        config = EnvironmentConfig(allow_internet=False)
        trial_paths = TrialPaths(trial_dir=trial_dir)
        trial_paths.mkdir()

        with pytest.raises(ValueError, match="internet"):
            LangSmithEnvironment(
                environment_dir=env_dir,
                environment_name="test",
                session_id="s1",
                trial_paths=trial_paths,
                task_env_config=config,
            )

    def test_accepts_factory_kwargs(self, tmp_path: Path) -> None:
        """Harbor's EnvironmentFactory passes logger, override_* kwargs."""
        env_dir = tmp_path / "environment"
        env_dir.mkdir()
        (env_dir / "Dockerfile").write_text("FROM ubuntu:24.04\n")

        trial_dir = tmp_path / "trial"
        trial_dir.mkdir()
        config = EnvironmentConfig()
        trial_paths = TrialPaths(trial_dir=trial_dir)
        trial_paths.mkdir()

        test_logger = logging.getLogger("test.harbor")
        env = LangSmithEnvironment(
            environment_dir=env_dir,
            environment_name="test",
            session_id="s1",
            trial_paths=trial_paths,
            task_env_config=config,
            logger=test_logger,
            override_cpus=4,
            override_memory_mb=8192,
            override_storage_mb=20480,
            override_gpus=0,
            suppress_override_warnings=True,
        )
        assert env is not None
        assert env.task_env_config.cpus == 4
        assert env.task_env_config.memory_mb == 8192


class TestResolveImage:
    """Tests for image resolution from Dockerfile or config."""

    def test_prefers_docker_image(self, tmp_path: Path) -> None:
        env = _make_env(tmp_path, docker_image="my-custom:latest")
        assert env._resolve_image() == "my-custom:latest"

    def test_parses_from_dockerfile(self, tmp_path: Path) -> None:
        env = _make_env(
            tmp_path,
            dockerfile_content=textwrap.dedent("""\
                FROM python:3.12-slim
                RUN apt-get update
                WORKDIR /app
            """),
        )
        assert env._resolve_image() == "python:3.12-slim"

    def test_empty_from_raises(self, tmp_path: Path) -> None:
        env = _make_env(tmp_path, dockerfile_content="# no FROM\n")
        with pytest.raises(ValueError, match="Could not extract FROM"):
            env._resolve_image()


class TestProperties:
    """Tests for static properties."""

    def test_is_mounted(self, tmp_path: Path) -> None:
        env = _make_env(tmp_path)
        assert env.is_mounted is False

    def test_supports_gpus(self, tmp_path: Path) -> None:
        env = _make_env(tmp_path)
        assert env.supports_gpus is False

    def test_can_disable_internet(self, tmp_path: Path) -> None:
        env = _make_env(tmp_path)
        assert env.can_disable_internet is False

    def test_type_raises(self, tmp_path: Path) -> None:
        with pytest.raises(NotImplementedError):
            LangSmithEnvironment.type()


class TestSanitizeName:
    """Tests for LangSmith resource name sanitization."""

    def test_lowercase_and_replace_underscores(self) -> None:
        assert (
            LangSmithEnvironment._sanitize_name("gpt2-codegolf__UxLAidb") == "gpt2-codegolf-uxlaidb"
        )

    def test_replaces_special_chars(self) -> None:
        assert LangSmithEnvironment._sanitize_name("my/image:3.12-slim") == "my-image-3-12-slim"

    def test_collapses_consecutive_hyphens(self) -> None:
        assert LangSmithEnvironment._sanitize_name("a___b---c") == "a-b-c"

    def test_strips_leading_trailing_hyphens(self) -> None:
        assert LangSmithEnvironment._sanitize_name("--hello--") == "hello"

    def test_prepends_prefix_if_starts_with_number(self) -> None:
        result = LangSmithEnvironment._sanitize_name("123abc")
        assert result[0].isalpha()
        assert result == "h-123abc"

    def test_truncates_to_63_chars(self) -> None:
        long_name = "a" * 100
        assert len(LangSmithEnvironment._sanitize_name(long_name)) == 63

    def test_empty_string(self) -> None:
        result = LangSmithEnvironment._sanitize_name("")
        assert result[0].isalpha()
        assert not result.endswith("-")

    def test_no_trailing_hyphen_after_truncation(self) -> None:
        """Truncation at 63 chars must not leave a trailing hyphen."""
        raw = "a" * 62 + ":b"
        result = LangSmithEnvironment._sanitize_name(raw)
        assert not result.endswith("-")
        assert len(result) <= 63


class TestResourceConversion:
    """Tests for task resource config → LangSmith create_sandbox/create_snapshot kwargs.

    Memory and CPU live on ``create_sandbox`` (vcpus/mem_bytes). Filesystem
    capacity is passed to both ``create_snapshot`` (when building) and
    ``create_sandbox`` so per-trial disk sizing takes effect.
    """

    async def test_memory_under_1gb(self, tmp_path: Path) -> None:
        env = _make_env(tmp_path, memory_mb=512)

        with patch("deepagents_harbor.langsmith_environment.AsyncSandboxClient") as mock_cls:
            mock_client = _mock_async_client()
            mock_cls.return_value = mock_client

            await env.start(force_build=True)

            assert mock_client.create_sandbox.call_args.kwargs["mem_bytes"] == 512 * 1024 * 1024

    async def test_memory_over_1gb(self, tmp_path: Path) -> None:
        env = _make_env(tmp_path, memory_mb=2048)

        with patch("deepagents_harbor.langsmith_environment.AsyncSandboxClient") as mock_cls:
            mock_client = _mock_async_client()
            mock_cls.return_value = mock_client

            await env.start(force_build=True)

            assert mock_client.create_sandbox.call_args.kwargs["mem_bytes"] == 2048 * 1024 * 1024

    async def test_cpu_conversion(self, tmp_path: Path) -> None:
        env = _make_env(tmp_path, cpus=2)

        with patch("deepagents_harbor.langsmith_environment.AsyncSandboxClient") as mock_cls:
            mock_client = _mock_async_client()
            mock_cls.return_value = mock_client

            await env.start(force_build=True)

            assert mock_client.create_sandbox.call_args.kwargs["vcpus"] == 2

    async def test_storage_bytes_on_snapshot_and_sandbox(self, tmp_path: Path) -> None:
        env = _make_env(tmp_path, storage_mb=10240)

        with patch("deepagents_harbor.langsmith_environment.AsyncSandboxClient") as mock_cls:
            mock_client = _mock_async_client()
            mock_cls.return_value = mock_client

            await env.start(force_build=True)

            expected_bytes = 10240 * 1024 * 1024
            assert (
                mock_client.create_snapshot.call_args.kwargs["fs_capacity_bytes"] == expected_bytes
            )
            assert (
                mock_client.create_sandbox.call_args.kwargs["fs_capacity_bytes"] == expected_bytes
            )

    async def test_storage_passes_exact_bytes(self, tmp_path: Path) -> None:
        """No more Gi rounding -- we forward whatever the task requested, byte-for-byte."""
        env = _make_env(tmp_path, storage_mb=1500)

        with patch("deepagents_harbor.langsmith_environment.AsyncSandboxClient") as mock_cls:
            mock_client = _mock_async_client()
            mock_cls.return_value = mock_client

            await env.start(force_build=True)

            expected_bytes = 1500 * 1024 * 1024
            assert (
                mock_client.create_snapshot.call_args.kwargs["fs_capacity_bytes"] == expected_bytes
            )
            assert (
                mock_client.create_sandbox.call_args.kwargs["fs_capacity_bytes"] == expected_bytes
            )


class TestStartSnapshotProvisioning:
    """Tests for shared-per-image snapshot provisioning in start().

    Snapshots are keyed purely by image and reused across trials; trial-local
    resource sizing moves onto ``create_sandbox``.
    """

    async def test_builds_snapshot_when_missing(self, tmp_path: Path) -> None:
        env = _make_env(tmp_path)

        with patch("deepagents_harbor.langsmith_environment.AsyncSandboxClient") as mock_cls:
            mock_client = _mock_async_client(existing_snapshots=[])
            mock_cls.return_value = mock_client

            await env.start(force_build=False)

            expected_name = LangSmithEnvironment._build_snapshot_name("ubuntu:24.04")
            mock_client.create_snapshot.assert_called_once()
            kwargs = mock_client.create_snapshot.call_args.kwargs
            assert kwargs["name"] == expected_name
            assert kwargs["docker_image"] == "ubuntu:24.04"

    async def test_reuses_existing_ready_snapshot(self, tmp_path: Path) -> None:
        env = _make_env(tmp_path)
        expected_name = LangSmithEnvironment._build_snapshot_name("ubuntu:24.04")

        with patch("deepagents_harbor.langsmith_environment.AsyncSandboxClient") as mock_cls:
            mock_client = _mock_async_client(
                existing_snapshots=[_FakeSnapshot(name=expected_name, status="ready")]
            )
            mock_cls.return_value = mock_client

            await env.start(force_build=False)

            mock_client.create_snapshot.assert_not_called()
            create_kwargs = mock_client.create_sandbox.call_args.kwargs
            assert create_kwargs["snapshot_name"] == expected_name
            assert "snapshot_id" not in create_kwargs
            assert "template_name" not in create_kwargs

    async def test_raises_when_existing_snapshot_not_ready(self, tmp_path: Path) -> None:
        env = _make_env(tmp_path)
        expected_name = LangSmithEnvironment._build_snapshot_name("ubuntu:24.04")

        with patch("deepagents_harbor.langsmith_environment.AsyncSandboxClient") as mock_cls:
            mock_client = _mock_async_client(
                existing_snapshots=[_FakeSnapshot(name=expected_name, status="building")]
            )
            mock_cls.return_value = mock_client

            with pytest.raises(RuntimeError, match="building"):
                await env.start(force_build=False)

            mock_client.create_snapshot.assert_not_called()
            mock_client.create_sandbox.assert_not_called()

    async def test_list_snapshots_called_with_name_contains(self, tmp_path: Path) -> None:
        """Readiness check must use the server-side ``name_contains`` filter."""
        env = _make_env(tmp_path)
        expected_name = LangSmithEnvironment._build_snapshot_name("ubuntu:24.04")

        with patch("deepagents_harbor.langsmith_environment.AsyncSandboxClient") as mock_cls:
            mock_client = _mock_async_client()
            mock_cls.return_value = mock_client

            await env.start(force_build=False)

            mock_client.list_snapshots.assert_awaited_once_with(name_contains=expected_name)

    async def test_creates_sandbox_from_snapshot(self, tmp_path: Path) -> None:
        env = _make_env(tmp_path)

        with patch("deepagents_harbor.langsmith_environment.AsyncSandboxClient") as mock_cls:
            mock_client = _mock_async_client()
            mock_cls.return_value = mock_client

            await env.start(force_build=False)

            expected_name = LangSmithEnvironment._build_snapshot_name("ubuntu:24.04")
            mock_client.create_sandbox.assert_called_once()
            kwargs = mock_client.create_sandbox.call_args.kwargs
            assert kwargs["snapshot_name"] == expected_name
            assert kwargs["vcpus"] == 1
            assert kwargs["mem_bytes"] == 2048 * 1024 * 1024
            assert kwargs["fs_capacity_bytes"] == 10240 * 1024 * 1024
            assert kwargs["timeout"] == 120
            assert "template_name" not in kwargs
            assert "snapshot_id" not in kwargs

    async def test_force_build_is_noop(self, tmp_path: Path) -> None:
        """force_build accepted for interface compat but does not change calls."""
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        env_a = _make_env(tmp_path / "a")
        env_b = _make_env(tmp_path / "b")

        with patch("deepagents_harbor.langsmith_environment.AsyncSandboxClient") as mock_cls:
            mock_a = _mock_async_client()
            mock_cls.return_value = mock_a
            await env_a.start(force_build=False)

        with patch("deepagents_harbor.langsmith_environment.AsyncSandboxClient") as mock_cls:
            mock_b = _mock_async_client()
            mock_cls.return_value = mock_b
            await env_b.start(force_build=True)

        assert mock_a.create_sandbox.call_count == mock_b.create_sandbox.call_count == 1


class TestBuildSnapshotName:
    """Tests for _build_snapshot_name shape and sharing contract."""

    def test_starts_with_harbor_prefix(self) -> None:
        name = LangSmithEnvironment._build_snapshot_name("ubuntu:24.04")
        assert name.startswith("harbor-")
        assert len(name) <= 63

    def test_long_image_stays_within_limit(self) -> None:
        long_image = "alexgshaw/log-summary-date-ranges:20251031"
        name = LangSmithEnvironment._build_snapshot_name(long_image)
        assert len(name) <= 63

    def test_deterministic(self) -> None:
        a = LangSmithEnvironment._build_snapshot_name("img:v1")
        b = LangSmithEnvironment._build_snapshot_name("img:v1")
        assert a == b

    def test_same_image_different_sessions_share_name(self) -> None:
        """Snapshots are shared across trials: name depends on image alone."""
        image = "alexgshaw/log-summary-date-ranges:20251031"
        # _build_snapshot_name takes no session_id; calling it from multiple
        # "trials" must produce the exact same name.
        names = {LangSmithEnvironment._build_snapshot_name(image) for _ in range(5)}
        assert len(names) == 1

    def test_name_starts_with_letter(self) -> None:
        name = LangSmithEnvironment._build_snapshot_name("123image:latest")
        assert name[0].isalpha()


class TestExec:
    """Tests for command execution."""

    async def test_exec_delegates_to_sandbox(self, tmp_path: Path) -> None:
        env = _make_env(tmp_path)
        sandbox = _FakeSandbox()
        env._sandbox = sandbox  # type: ignore[assignment]

        result = await env.exec("echo hello")

        assert result.return_code == 0
        assert sandbox._run_calls[0][0] == "echo hello"

    async def test_exec_passes_cwd_and_env(self, tmp_path: Path) -> None:
        env = _make_env(tmp_path)
        sandbox = _FakeSandbox()
        env._sandbox = sandbox  # type: ignore[assignment]

        await env.exec("ls", cwd="/app", env={"FOO": "bar"})

        _, _, cwd, cmd_env = sandbox._run_calls[0]
        assert cwd == "/app"
        assert cmd_env == {"FOO": "bar"}

    async def test_exec_uses_default_cwd_when_none_provided(self, tmp_path: Path) -> None:
        """Regression: without this, LangSmith's dataplane defaults to "/",
        which causes terminal-bench verifier scripts to abort early without
        writing /logs/verifier/reward.txt.
        """
        env = _make_env(tmp_path)
        sandbox = _FakeSandbox()
        env._sandbox = sandbox  # type: ignore[assignment]
        env._default_cwd = "/app"

        await env.exec("ls")

        _, _, cwd, _ = sandbox._run_calls[0]
        assert cwd == "/app"

    async def test_exec_explicit_cwd_overrides_default(self, tmp_path: Path) -> None:
        env = _make_env(tmp_path)
        sandbox = _FakeSandbox()
        env._sandbox = sandbox  # type: ignore[assignment]
        env._default_cwd = "/app"

        await env.exec("ls", cwd="/tmp")

        _, _, cwd, _ = sandbox._run_calls[0]
        assert cwd == "/tmp"

    async def test_exec_uses_default_timeout(self, tmp_path: Path) -> None:
        env = _make_env(tmp_path)
        sandbox = _FakeSandbox()
        env._sandbox = sandbox  # type: ignore[assignment]

        await env.exec("echo hello")

        assert sandbox._run_calls[0][1] == 30 * 60

    async def test_exec_forwards_custom_timeout(self, tmp_path: Path) -> None:
        env = _make_env(tmp_path)
        sandbox = _FakeSandbox()
        env._sandbox = sandbox  # type: ignore[assignment]

        await env.exec("echo hello", timeout_sec=10)

        assert sandbox._run_calls[0][1] == 10

    async def test_exec_without_start_raises(self, tmp_path: Path) -> None:
        env = _make_env(tmp_path)
        with pytest.raises(RuntimeError, match="start"):
            await env.exec("echo fail")


class TestWorkdirDetection:
    """Tests for container WORKDIR detection at start()."""

    @staticmethod
    def _install_probe_sandbox(
        mock_client: MagicMock,
        *,
        readlink_stdout: str = "",
        readlink_exit_code: int = 0,
        dir_probe_stdout: str = "",
    ) -> _FakeSandbox:
        """Install a sandbox whose `run()` answers the workdir probes.

        Both probes (`readlink /proc/1/cwd` and the `/app`-existence fallback)
        are dispatched from the same method so assertions stay simple.
        """
        sandbox = _FakeSandbox()

        async def _run(
            command: str,
            *,
            timeout: int = 60,  # noqa: ASYNC109
            cwd: str | None = None,
            env: dict[str, str] | None = None,
        ) -> _FakeExecResult:
            sandbox._run_calls.append((command, timeout, cwd, env))
            if "readlink /proc/1/cwd" in command:
                return _FakeExecResult(stdout=readlink_stdout, exit_code=readlink_exit_code)
            if "-d /app" in command:
                return _FakeExecResult(stdout=dir_probe_stdout)
            return _FakeExecResult()

        sandbox.run = _run  # type: ignore[assignment]
        mock_client.create_sandbox.return_value = sandbox
        return sandbox

    async def test_detects_workdir_from_pid1(self, tmp_path: Path) -> None:
        env = _make_env(tmp_path)

        with patch("deepagents_harbor.langsmith_environment.AsyncSandboxClient") as mock_cls:
            mock_client = AsyncMock()
            self._install_probe_sandbox(mock_client, readlink_stdout="/app\n")
            mock_cls.return_value = mock_client

            await env.start(force_build=False)

        assert env._default_cwd == "/app"

    async def test_detects_non_app_workdir(self, tmp_path: Path) -> None:
        """Images with non-standard WORKDIRs (e.g. /workspace) must be honored."""
        env = _make_env(tmp_path)

        with patch("deepagents_harbor.langsmith_environment.AsyncSandboxClient") as mock_cls:
            mock_client = AsyncMock()
            self._install_probe_sandbox(mock_client, readlink_stdout="/workspace\n")
            mock_cls.return_value = mock_client

            await env.start(force_build=False)

        assert env._default_cwd == "/workspace"

    async def test_falls_back_to_app_when_root(self, tmp_path: Path) -> None:
        """When the container has no WORKDIR, prefer /app if it exists."""
        env = _make_env(tmp_path)

        with patch("deepagents_harbor.langsmith_environment.AsyncSandboxClient") as mock_cls:
            mock_client = AsyncMock()
            self._install_probe_sandbox(
                mock_client,
                readlink_stdout="/\n",
                dir_probe_stdout="/app\n",
            )
            mock_cls.return_value = mock_client

            await env.start(force_build=False)

        assert env._default_cwd == "/app"

    async def test_falls_back_to_app_on_probe_failure(self, tmp_path: Path) -> None:
        """A broken readlink probe must not prevent start()."""
        env = _make_env(tmp_path)

        with patch("deepagents_harbor.langsmith_environment.AsyncSandboxClient") as mock_cls:
            mock_client = AsyncMock()
            sandbox = _FakeSandbox()

            async def _run(
                command: str,
                *,
                timeout: int = 60,  # noqa: ASYNC109
                cwd: str | None = None,
                env: dict[str, str] | None = None,
            ) -> _FakeExecResult:
                sandbox._run_calls.append((command, timeout, cwd, env))
                if "readlink /proc/1/cwd" in command:
                    msg = "connection reset"
                    raise RuntimeError(msg)
                return _FakeExecResult()

            sandbox.run = _run  # type: ignore[assignment]
            mock_client.create_sandbox.return_value = sandbox
            mock_cls.return_value = mock_client

            await env.start(force_build=False)

        assert env._default_cwd == "/app"

    async def test_falls_back_to_app_when_nothing_resolves(self, tmp_path: Path) -> None:
        """When /app doesn't exist either, still default to /app rather
        than "/" to preserve terminal-bench verifier PWD semantics."""
        env = _make_env(tmp_path)

        with patch("deepagents_harbor.langsmith_environment.AsyncSandboxClient") as mock_cls:
            mock_client = AsyncMock()
            self._install_probe_sandbox(
                mock_client,
                readlink_stdout="/\n",
                dir_probe_stdout="/\n",
            )
            mock_cls.return_value = mock_client

            await env.start(force_build=False)

        assert env._default_cwd == "/app"

    async def test_stop_clears_default_cwd(self, tmp_path: Path) -> None:
        env = _make_env(tmp_path)
        env._sandbox = _FakeSandbox()  # type: ignore[assignment]
        env._client = AsyncMock()
        env._snapshot_name = "snap-tmpl"
        env._default_cwd = "/app"

        await env.stop(delete=False)

        assert env._default_cwd is None


class TestFileOps:
    """Tests for file upload/download operations."""

    async def test_upload_file(self, tmp_path: Path) -> None:
        env = _make_env(tmp_path)
        sandbox = _FakeSandbox()
        env._sandbox = sandbox  # type: ignore[assignment]

        src = tmp_path / "local.txt"
        src.write_text("hello world")

        await env.upload_file(src, "/app/remote.txt")

        assert sandbox._written_files["/app/remote.txt"] == b"hello world"

    async def test_download_file(self, tmp_path: Path) -> None:
        env = _make_env(tmp_path)
        sandbox = _FakeSandbox()
        sandbox._read_files["/app/data.txt"] = b"file content"
        env._sandbox = sandbox  # type: ignore[assignment]

        dest = tmp_path / "downloaded.txt"
        await env.download_file("/app/data.txt", dest)

        assert dest.read_bytes() == b"file content"

    async def test_upload_dir(self, tmp_path: Path) -> None:
        env = _make_env(tmp_path)
        sandbox = _FakeSandbox()
        env._sandbox = sandbox  # type: ignore[assignment]

        src_dir = tmp_path / "mydir"
        src_dir.mkdir()
        (src_dir / "a.txt").write_text("aaa")
        sub = src_dir / "sub"
        sub.mkdir()
        (sub / "b.txt").write_text("bbb")

        await env.upload_dir(src_dir, "/app/dest")

        assert sandbox._written_files["/app/dest/a.txt"] == b"aaa"
        assert sandbox._written_files["/app/dest/sub/b.txt"] == b"bbb"

    async def test_download_dir(self, tmp_path: Path) -> None:
        env = _make_env(tmp_path)
        sandbox = _FakeSandbox()
        sandbox._read_files["/remote/a.txt"] = b"aaa"
        sandbox._read_files["/remote/sub/b.txt"] = b"bbb"
        env._sandbox = sandbox  # type: ignore[assignment]

        async def _fake_run(_cmd: str, **_kw: Any) -> _FakeExecResult:
            return _FakeExecResult(stdout="/remote/a.txt\n/remote/sub/b.txt\n")

        sandbox.run = _fake_run  # type: ignore[assignment]

        dest = tmp_path / "downloaded"
        await env.download_dir("/remote", dest)

        assert (dest / "a.txt").read_bytes() == b"aaa"
        assert (dest / "sub" / "b.txt").read_bytes() == b"bbb"

    async def test_download_dir_partial_failure(self, tmp_path: Path) -> None:
        """Files that fail to download are skipped; successful ones are kept."""
        env = _make_env(tmp_path)
        sandbox = _FakeSandbox()
        sandbox._read_files["/remote/good.txt"] = b"ok"
        env._sandbox = sandbox  # type: ignore[assignment]

        async def _fake_run(_cmd: str, **_kw: Any) -> _FakeExecResult:
            return _FakeExecResult(stdout="/remote/good.txt\n/remote/bad.txt\n")

        sandbox.run = _fake_run  # type: ignore[assignment]

        # bad.txt is not in _read_files, so download_file → sandbox.read
        # returns b"" by default, but we override read to raise for bad.txt.
        original_read = sandbox.read

        async def _failing_read(path: str) -> bytes:
            if path == "/remote/bad.txt":
                msg = "not found"
                raise FileNotFoundError(msg)
            return await original_read(path)

        sandbox.read = _failing_read  # type: ignore[assignment]

        dest = tmp_path / "downloaded"
        await env.download_dir("/remote", dest)

        assert (dest / "good.txt").read_bytes() == b"ok"
        assert not (dest / "bad.txt").exists()

    async def test_upload_dir_without_start_raises(self, tmp_path: Path) -> None:
        env = _make_env(tmp_path)
        src_dir = tmp_path / "mydir"
        src_dir.mkdir()
        with pytest.raises(RuntimeError, match="start"):
            await env.upload_dir(src_dir, "/app/dest")

    async def test_download_dir_empty(self, tmp_path: Path) -> None:
        env = _make_env(tmp_path)
        sandbox = _FakeSandbox()
        env._sandbox = sandbox  # type: ignore[assignment]

        async def _fake_run(_cmd: str, **_kw: Any) -> _FakeExecResult:
            return _FakeExecResult(exit_code=1, stderr="No such file or directory")

        sandbox.run = _fake_run  # type: ignore[assignment]

        dest = tmp_path / "downloaded"
        await env.download_dir("/nonexistent", dest)

        assert dest.exists()
        assert list(dest.iterdir()) == []


class TestStop:
    """Tests for teardown.

    Snapshots are shared resources — ``stop()`` must never delete them,
    regardless of the ``delete`` flag.
    """

    async def test_stop_deletes_sandbox_but_not_snapshot(self, tmp_path: Path) -> None:
        env = _make_env(tmp_path)
        mock_client = AsyncMock()
        mock_sandbox = _FakeSandbox(name="my-sandbox")
        env._sandbox = mock_sandbox  # type: ignore[assignment]
        env._client = mock_client
        env._snapshot_name = "harbor-ubuntu-24-04"

        await env.stop(delete=True)

        mock_client.delete_sandbox.assert_called_once_with("my-sandbox")
        mock_client.delete_snapshot.assert_not_called()
        mock_client.aclose.assert_called_once()
        assert env._sandbox is None
        assert env._client is None
        assert env._snapshot_name is None

    async def test_stop_no_delete_skips_cleanup(self, tmp_path: Path) -> None:
        env = _make_env(tmp_path)
        mock_client = AsyncMock()
        env._sandbox = _FakeSandbox()  # type: ignore[assignment]
        env._client = mock_client
        env._snapshot_name = "snap-tmpl"

        await env.stop(delete=False)

        mock_client.delete_sandbox.assert_not_called()
        mock_client.delete_snapshot.assert_not_called()
        mock_client.aclose.assert_called_once()

    async def test_stop_continues_after_sandbox_delete_fails(self, tmp_path: Path) -> None:
        """If sandbox deletion fails, aclose still runs and snapshot is untouched."""
        env = _make_env(tmp_path)
        mock_client = AsyncMock()
        mock_client.delete_sandbox.side_effect = RuntimeError("API timeout")
        env._sandbox = _FakeSandbox(name="my-sandbox")  # type: ignore[assignment]
        env._client = mock_client
        env._snapshot_name = "harbor-my-image"

        await env.stop(delete=True)

        mock_client.delete_sandbox.assert_called_once_with("my-sandbox")
        mock_client.delete_snapshot.assert_not_called()
        mock_client.aclose.assert_called_once()
        assert env._sandbox is None
        assert env._client is None
        assert env._snapshot_name is None

    async def test_stop_clean_after_failed_start(self, tmp_path: Path) -> None:
        """If create_snapshot fails, _snapshot_name stays None and stop is safe."""
        env = _make_env(tmp_path)

        with patch("deepagents_harbor.langsmith_environment.AsyncSandboxClient") as mock_cls:
            mock_client = _mock_async_client()
            mock_client.create_snapshot = AsyncMock(side_effect=RuntimeError("API 422"))
            mock_cls.return_value = mock_client

            with pytest.raises(RuntimeError, match="API 422"):
                await env.start(force_build=False)

        assert env._snapshot_name is None
        assert env._sandbox is None

        await env.stop(delete=True)
        mock_client.delete_snapshot.assert_not_called()
