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


@dataclass
class _FakeExecResult:
    """Minimal stand-in for langsmith ExecutionResult."""

    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0


@dataclass
class _FakeTemplate:
    """Minimal stand-in for langsmith SandboxTemplate."""

    name: str = "harbor-ubuntu-24-04"

    @dataclass
    class _Resources:
        cpu: str = "1000m"
        memory: str = "2Gi"
        storage: str = "10Gi"

    resources: _Resources = field(default_factory=_Resources)


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


def _mock_async_client(
    *,
    template: _FakeTemplate | None = None,
    get_template_side_effect: Exception | None = None,
) -> MagicMock:
    """Build a mock AsyncSandboxClient wired for start() tests."""
    mock = AsyncMock()
    fake_sb = _FakeSandbox()
    mock.create_sandbox.return_value = fake_sb
    if get_template_side_effect:
        mock.get_template.side_effect = get_template_side_effect
    else:
        mock.get_template.return_value = template or _FakeTemplate()
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
    """Tests for task resource config → LangSmith format.

    Uses force_build=True so create_template is always called regardless
    of _ensure_template reuse logic.
    """

    async def test_memory_under_1gb(self, tmp_path: Path) -> None:
        env = _make_env(tmp_path, memory_mb=512)

        with patch("langsmith.sandbox.AsyncSandboxClient") as mock_cls:
            mock_client = _mock_async_client()
            mock_cls.return_value = mock_client

            await env.start(force_build=True)

            assert mock_client.create_template.call_args.kwargs["memory"] == "512Mi"

    async def test_memory_over_1gb(self, tmp_path: Path) -> None:
        env = _make_env(tmp_path, memory_mb=2048)

        with patch("langsmith.sandbox.AsyncSandboxClient") as mock_cls:
            mock_client = _mock_async_client()
            mock_cls.return_value = mock_client

            await env.start(force_build=True)

            assert mock_client.create_template.call_args.kwargs["memory"] == "2Gi"

    async def test_cpu_conversion(self, tmp_path: Path) -> None:
        env = _make_env(tmp_path, cpus=2)

        with patch("langsmith.sandbox.AsyncSandboxClient") as mock_cls:
            mock_client = _mock_async_client()
            mock_cls.return_value = mock_client

            await env.start(force_build=True)

            assert mock_client.create_template.call_args.kwargs["cpu"] == "2000m"

    async def test_storage_always_gi(self, tmp_path: Path) -> None:
        env = _make_env(tmp_path, storage_mb=10240)

        with patch("langsmith.sandbox.AsyncSandboxClient") as mock_cls:
            mock_client = _mock_async_client()
            mock_cls.return_value = mock_client

            await env.start(force_build=True)

            assert mock_client.create_template.call_args.kwargs["storage"] == "10Gi"

    async def test_storage_rounds_up(self, tmp_path: Path) -> None:
        env = _make_env(tmp_path, storage_mb=1500)

        with patch("langsmith.sandbox.AsyncSandboxClient") as mock_cls:
            mock_client = _mock_async_client()
            mock_cls.return_value = mock_client

            await env.start(force_build=True)

            assert mock_client.create_template.call_args.kwargs["storage"] == "2Gi"


class TestEnsureTemplate:
    """Tests for template reuse via _ensure_template."""

    async def test_reuses_template_when_resources_match(self, tmp_path: Path) -> None:
        env = _make_env(tmp_path)

        with patch("langsmith.sandbox.AsyncSandboxClient") as mock_cls:
            mock_client = _mock_async_client(
                template=_FakeTemplate(
                    resources=_FakeTemplate._Resources(cpu="1000m", memory="2Gi", storage="10Gi")
                ),
            )
            mock_cls.return_value = mock_client

            await env.start(force_build=False)

            mock_client.get_template.assert_called_once()
            mock_client.create_template.assert_not_called()

    async def test_recreates_template_when_resources_differ(self, tmp_path: Path) -> None:
        env = _make_env(tmp_path, cpus=4)

        with patch("langsmith.sandbox.AsyncSandboxClient") as mock_cls:
            mock_client = _mock_async_client(
                template=_FakeTemplate(
                    resources=_FakeTemplate._Resources(cpu="1000m", memory="2Gi", storage="10Gi")
                ),
            )
            mock_cls.return_value = mock_client

            await env.start(force_build=False)

            mock_client.delete_template.assert_called_once()
            mock_client.create_template.assert_called_once()
            assert mock_client.create_template.call_args.kwargs["cpu"] == "4000m"

    async def test_creates_template_when_not_found(self, tmp_path: Path) -> None:
        from langsmith.sandbox import ResourceNotFoundError

        env = _make_env(tmp_path)

        with patch("langsmith.sandbox.AsyncSandboxClient") as mock_cls:
            mock_client = _mock_async_client(
                get_template_side_effect=ResourceNotFoundError(
                    "not found", resource_type="template"
                ),
            )
            mock_cls.return_value = mock_client

            await env.start(force_build=False)

            mock_client.create_template.assert_called_once()

    async def test_force_build_always_recreates(self, tmp_path: Path) -> None:
        env = _make_env(tmp_path)

        with patch("langsmith.sandbox.AsyncSandboxClient") as mock_cls:
            mock_client = _mock_async_client()
            mock_cls.return_value = mock_client

            await env.start(force_build=True)

            mock_client.get_template.assert_not_called()
            mock_client.delete_template.assert_called_once()
            mock_client.create_template.assert_called_once()

    async def test_recreates_template_when_storage_differs(self, tmp_path: Path) -> None:
        env = _make_env(tmp_path, storage_mb=20480)

        with patch("langsmith.sandbox.AsyncSandboxClient") as mock_cls:
            mock_client = _mock_async_client(
                template=_FakeTemplate(
                    resources=_FakeTemplate._Resources(cpu="1000m", memory="2Gi", storage="10Gi")
                ),
            )
            mock_cls.return_value = mock_client

            await env.start(force_build=False)

            mock_client.delete_template.assert_called_once()
            mock_client.create_template.assert_called_once()
            assert mock_client.create_template.call_args.kwargs["storage"] == "20Gi"

    async def test_template_name_derived_from_image(self, tmp_path: Path) -> None:
        env = _make_env(tmp_path, docker_image="ghcr.io/my-org/my-image:v1.2")

        with patch("langsmith.sandbox.AsyncSandboxClient") as mock_cls:
            mock_client = _mock_async_client()
            mock_cls.return_value = mock_client

            await env.start(force_build=False)

            expected = "harbor-ghcr-io-my-org-my-image-v1-2"
            mock_client.get_template.assert_called_once_with(expected)


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
    """Tests for teardown."""

    async def test_stop_deletes_sandbox_and_template(self, tmp_path: Path) -> None:
        env = _make_env(tmp_path)
        mock_client = AsyncMock()
        mock_sandbox = _FakeSandbox(name="my-sandbox")
        env._sandbox = mock_sandbox  # type: ignore[assignment]
        env._client = mock_client
        env._template_name = "my-template"

        await env.stop(delete=True)

        mock_client.delete_sandbox.assert_called_once_with("my-sandbox")
        mock_client.delete_template.assert_called_once_with("my-template")
        mock_client.aclose.assert_called_once()
        assert env._sandbox is None
        assert env._client is None

    async def test_stop_no_delete_skips_cleanup(self, tmp_path: Path) -> None:
        env = _make_env(tmp_path)
        mock_client = AsyncMock()
        env._sandbox = _FakeSandbox()  # type: ignore[assignment]
        env._client = mock_client
        env._template_name = "tmpl"

        await env.stop(delete=False)

        mock_client.delete_sandbox.assert_not_called()
        mock_client.delete_template.assert_not_called()
        mock_client.aclose.assert_called_once()

    async def test_stop_continues_after_sandbox_delete_fails(self, tmp_path: Path) -> None:
        """If sandbox deletion fails, template deletion and aclose still run."""
        env = _make_env(tmp_path)
        mock_client = AsyncMock()
        mock_client.delete_sandbox.side_effect = RuntimeError("API timeout")
        env._sandbox = _FakeSandbox(name="my-sandbox")  # type: ignore[assignment]
        env._client = mock_client
        env._template_name = "my-template"

        await env.stop(delete=True)

        mock_client.delete_sandbox.assert_called_once_with("my-sandbox")
        mock_client.delete_template.assert_called_once_with("my-template")
        mock_client.aclose.assert_called_once()
        assert env._sandbox is None
        assert env._client is None
        assert env._template_name is None

    async def test_stop_clean_after_failed_start(self, tmp_path: Path) -> None:
        """If create_template fails, _template_name stays None and stop is safe."""
        from langsmith.sandbox import ResourceNotFoundError

        env = _make_env(tmp_path)

        with patch("langsmith.sandbox.AsyncSandboxClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.get_template.side_effect = ResourceNotFoundError(
                "not found", resource_type="template"
            )
            mock_client.create_template.side_effect = RuntimeError("API 422")
            mock_cls.return_value = mock_client

            with pytest.raises(RuntimeError, match="API 422"):
                await env.start(force_build=False)

        assert env._template_name is None
        assert env._sandbox is None

        await env.stop(delete=True)
