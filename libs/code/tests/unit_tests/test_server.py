"""Tests for server lifecycle helpers."""

from __future__ import annotations

import os
import socket
import threading
from types import SimpleNamespace
from typing import TYPE_CHECKING, Self
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deepagents_code.server import (
    ServerProcess,
    _find_free_port,
    _port_in_use,
    wait_for_server_healthy,
)

if TYPE_CHECKING:
    from pathlib import Path


class _FakeSocket:
    """Small socket stand-in for unit tests running with `--disable-socket`."""

    def __init__(
        self,
        *,
        bind_error: OSError | None = None,
        sockname: tuple[str, int] = ("127.0.0.1", 0),
    ) -> None:
        self._bind_error = bind_error
        self._sockname = sockname
        self.bound_addr: tuple[str, int] | None = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def bind(self, addr: tuple[str, int]) -> None:
        """Record the bind call or raise the configured error."""
        if self._bind_error is not None:
            raise self._bind_error
        self.bound_addr = addr

    def getsockname(self) -> tuple[str, int]:
        """Return the configured socket name tuple."""
        return self._sockname


class _FakeAsyncClient:
    """Minimal async `httpx.AsyncClient` stand-in for readiness tests."""

    def __init__(self, response: object) -> None:
        self.response = response
        self.urls: list[str] = []

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    async def get(
        self,
        url: str,
        *,
        timeout: float,  # noqa: ARG002, ASYNC109  # mirrors httpx.AsyncClient.get
    ) -> object:
        self.urls.append(url)
        if isinstance(self.response, BaseException):
            raise self.response
        return self.response


class TestPortInUse:
    def test_free_port(self) -> None:
        fake_socket = _FakeSocket()

        with patch("socket.socket", return_value=fake_socket) as socket_cls:
            assert not _port_in_use("127.0.0.1", 2024)

        socket_cls.assert_called_once_with(socket.AF_INET, socket.SOCK_STREAM)
        assert fake_socket.bound_addr == ("127.0.0.1", 2024)

    def test_occupied_port(self) -> None:
        fake_socket = _FakeSocket(bind_error=OSError("port already in use"))

        with patch("socket.socket", return_value=fake_socket) as socket_cls:
            assert _port_in_use("127.0.0.1", 2024)

        socket_cls.assert_called_once_with(socket.AF_INET, socket.SOCK_STREAM)
        assert fake_socket.bound_addr is None


class TestFindFreePort:
    def test_returns_valid_port(self) -> None:
        fake_socket = _FakeSocket(sockname=("127.0.0.1", 43210))

        with patch("socket.socket", return_value=fake_socket) as socket_cls:
            port = _find_free_port("127.0.0.1")

        socket_cls.assert_called_once_with(socket.AF_INET, socket.SOCK_STREAM)
        assert fake_socket.bound_addr == ("127.0.0.1", 0)
        assert 1 <= port <= 65535

    def test_returns_port_reported_by_socket(self) -> None:
        fake_socket = _FakeSocket(sockname=("127.0.0.1", 53123))

        with patch("socket.socket", return_value=fake_socket):
            port = _find_free_port("127.0.0.1")

        assert port == 53123


class TestServerPortSelection:
    """Port resolution in `ServerProcess.start()`."""

    @staticmethod
    def _make_server(tmp_path: Path, port: int = 0) -> ServerProcess:
        config_dir = tmp_path / "runtime"
        config_dir.mkdir()
        (config_dir / "langgraph.json").write_text("{}")
        return ServerProcess(config_dir=config_dir, port=port)

    @staticmethod
    def _mock_log_file(tmp_path: Path) -> MagicMock:
        log_file = MagicMock()
        log_file.name = str(tmp_path / "server.log")
        return log_file

    async def test_default_uses_ephemeral_port(self, tmp_path: Path) -> None:
        """Default port (0) resolves via `_find_free_port`, never squats 2024."""
        server = self._make_server(tmp_path)
        assert server.port == 0

        process = MagicMock(pid=1234)
        process.poll.return_value = None
        with (
            patch(
                "deepagents_code.server.tempfile.NamedTemporaryFile",
                return_value=self._mock_log_file(tmp_path),
            ),
            patch("deepagents_code.server.subprocess.Popen", return_value=process),
            patch("deepagents_code.server.wait_for_server_healthy", new=AsyncMock()),
            patch(
                "deepagents_code.server._find_free_port", return_value=43210
            ) as find_free,
            patch("deepagents_code.server._port_in_use") as in_use,
        ):
            await server.start()

        find_free.assert_called_once_with("127.0.0.1")
        in_use.assert_not_called()
        assert server.port == 43210

    async def test_explicit_free_port_is_kept(self, tmp_path: Path) -> None:
        """An explicit, free port is honored without searching for another."""
        server = self._make_server(tmp_path, port=2024)

        process = MagicMock(pid=1234)
        process.poll.return_value = None
        with (
            patch(
                "deepagents_code.server.tempfile.NamedTemporaryFile",
                return_value=self._mock_log_file(tmp_path),
            ),
            patch("deepagents_code.server.subprocess.Popen", return_value=process),
            patch("deepagents_code.server.wait_for_server_healthy", new=AsyncMock()),
            patch("deepagents_code.server._port_in_use", return_value=False) as in_use,
            patch("deepagents_code.server._find_free_port") as find_free,
        ):
            await server.start()

        in_use.assert_called_once_with("127.0.0.1", 2024)
        find_free.assert_not_called()
        assert server.port == 2024

    async def test_explicit_busy_port_falls_back(self, tmp_path: Path) -> None:
        """An explicit but busy port falls back to a free port."""
        server = self._make_server(tmp_path, port=2024)

        process = MagicMock(pid=1234)
        process.poll.return_value = None
        with (
            patch(
                "deepagents_code.server.tempfile.NamedTemporaryFile",
                return_value=self._mock_log_file(tmp_path),
            ),
            patch("deepagents_code.server.subprocess.Popen", return_value=process),
            patch("deepagents_code.server.wait_for_server_healthy", new=AsyncMock()),
            patch("deepagents_code.server._port_in_use", return_value=True) as in_use,
            patch(
                "deepagents_code.server._find_free_port", return_value=43210
            ) as find_free,
        ):
            await server.start()

        in_use.assert_called_once_with("127.0.0.1", 2024)
        find_free.assert_called_once_with("127.0.0.1")
        assert server.port == 43210


class TestWaitForServerHealthy:
    """Tests for the health-check polling loop."""

    async def test_returns_on_200(self) -> None:
        """Happy path: server responds 200 immediately."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_client):
            await wait_for_server_healthy("http://localhost:2024", timeout=5)

        mock_client.get.assert_awaited_once()

    async def test_raises_on_early_process_exit(self) -> None:
        """Process dies before health check succeeds -> fail fast."""
        process = MagicMock()
        process.poll.return_value = 1
        process.returncode = 1

        with pytest.raises(RuntimeError, match="exited with code 1"):
            await wait_for_server_healthy(
                "http://localhost:2024",
                timeout=5,
                process=process,
            )

    async def test_early_exit_includes_log_output(self) -> None:
        """read_log output is included in the error message."""
        process = MagicMock()
        process.poll.return_value = 1
        process.returncode = 1

        with pytest.raises(RuntimeError, match="some log output"):
            await wait_for_server_healthy(
                "http://localhost:2024",
                timeout=5,
                process=process,
                read_log=lambda: "some log output",
            )

    async def test_early_exit_promotes_marked_startup_error(self) -> None:
        """Marked server startup errors should survive app error trimming."""
        process = MagicMock()
        process.poll.return_value = 1
        process.returncode = 3

        log = (
            "Traceback (most recent call last):\n"
            "ValueError: No Runloop API key found\n"
            "Sandbox creation failed for 'runloop': No Runloop API key found. "
            "Set RUNLOOP_API_KEY or DEEPAGENTS_CODE_RUNLOOP_API_KEY.\n"
            "DEEPAGENTS_STARTUP_ERROR:Sandbox creation failed for 'runloop': "
            "No Runloop API key found. Set RUNLOOP_API_KEY or "
            "DEEPAGENTS_CODE_RUNLOOP_API_KEY.\n"
            "2026-05-11T03:37:44.911664Z [error    ] "
            "Application startup failed. Exiting. [uvicorn.error]"
        )

        with pytest.raises(RuntimeError) as exc_info:
            await wait_for_server_healthy(
                "http://localhost:2024",
                timeout=5,
                process=process,
                read_log=lambda: log,
            )

        first_line = str(exc_info.value).splitlines()[0]
        assert first_line == (
            "Server process exited with code 3: Sandbox creation failed for "
            "'runloop': No Runloop API key found. Set RUNLOOP_API_KEY or "
            "DEEPAGENTS_CODE_RUNLOOP_API_KEY."
        )

    async def test_raises_on_timeout(self) -> None:
        """Timeout exhaustion raises RuntimeError."""
        import httpx

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            patch("deepagents_code.server._HEALTH_POLL_INTERVAL_LOCAL", 0),
            patch("deepagents_code.server._HEALTH_POLL_INTERVAL_REMOTE", 0),
            pytest.raises(RuntimeError, match="did not become healthy"),
        ):
            await wait_for_server_healthy("http://localhost:2024", timeout=0.01)

    async def test_timeout_reports_last_status(self) -> None:
        """Timeout error includes the last HTTP status code."""
        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            patch("deepagents_code.server._HEALTH_POLL_INTERVAL_LOCAL", 0),
            patch("deepagents_code.server._HEALTH_POLL_INTERVAL_REMOTE", 0),
            pytest.raises(RuntimeError, match="last status: 503"),
        ):
            await wait_for_server_healthy("http://localhost:2024", timeout=0.01)


class TestServerProcess:
    async def test_wait_for_graph_ready_resolves_graph_endpoint(self) -> None:
        """Graph readiness should force LangGraph to resolve graph factories."""
        client = _FakeAsyncClient(SimpleNamespace(status_code=200))
        process = MagicMock()
        process.poll.return_value = None
        server = ServerProcess(host="127.0.0.1", port=2024)
        server._process = process

        with patch("httpx.AsyncClient", return_value=client):
            await server.wait_for_graph_ready("agent")

        assert client.urls == ["http://127.0.0.1:2024/assistants/agent/graph"]

    async def test_wait_for_graph_ready_surfaces_startup_marker(
        self, tmp_path: Path
    ) -> None:
        """Readiness failures should preserve marked subprocess startup errors."""
        log_path = tmp_path / "server.log"
        log_path.write_text(
            "booting\n"
            "DEEPAGENTS_STARTUP_ERROR:Sandbox creation failed for 'modal': boom\n"
        )

        log_file = MagicMock()
        log_file.name = str(log_path)

        client = _FakeAsyncClient(SimpleNamespace(status_code=500))
        process = MagicMock()
        process.poll.return_value = None
        server = ServerProcess(host="127.0.0.1", port=2024)
        server._process = process
        server._log_file = log_file

        with (
            patch("httpx.AsyncClient", return_value=client),
            pytest.raises(RuntimeError, match="Sandbox creation failed"),
        ):
            await server.wait_for_graph_ready("agent")

    async def test_wait_for_graph_ready_checks_logs_after_transport_error(
        self, tmp_path: Path
    ) -> None:
        """Dropped graph requests should still surface startup markers."""
        log_path = tmp_path / "server.log"
        log_path.write_text(
            "booting\nDEEPAGENTS_STARTUP_ERROR:ModelConfigError: missing API key\n"
        )

        log_file = MagicMock()
        log_file.name = str(log_path)

        client = _FakeAsyncClient(OSError("connection closed"))
        process = MagicMock()
        process.poll.return_value = 1
        process.returncode = 3
        server = ServerProcess(host="127.0.0.1", port=2024)
        server._process = process
        server._log_file = log_file

        with (
            patch("httpx.AsyncClient", return_value=client),
            pytest.raises(RuntimeError, match="ModelConfigError: missing API key"),
        ):
            await server.wait_for_graph_ready("agent")

    async def test_start_cleans_up_partial_state_on_health_failure(
        self, tmp_path: Path
    ) -> None:
        """Failed startup should stop the process and remove owned resources."""
        config_dir = tmp_path / "runtime"
        config_dir.mkdir()
        (config_dir / "langgraph.json").write_text("{}")

        log_path = tmp_path / "server.log"
        log_path.write_text("booting")

        process = MagicMock()
        process.pid = 1234
        process.poll.return_value = None

        log_file = MagicMock()
        log_file.name = str(log_path)

        server = ServerProcess(config_dir=config_dir, owns_config_dir=True)

        with (
            patch("deepagents_code.server._find_free_port", return_value=12345),
            patch(
                "deepagents_code.server.tempfile.NamedTemporaryFile",
                return_value=log_file,
            ),
            patch("deepagents_code.server.subprocess.Popen", return_value=process),
            patch(
                "deepagents_code.server.wait_for_server_healthy",
                new=AsyncMock(side_effect=RuntimeError("boom")),
            ),
            pytest.raises(RuntimeError, match="boom"),
        ):
            await server.start()

        process.send_signal.assert_called_once()
        process.wait.assert_called_once()
        log_file.close.assert_called_once()
        assert server._process is None
        assert server._log_file is None
        assert not config_dir.exists()
        assert not log_path.exists()

    async def test_start_rescaffolds_when_config_missing(self, tmp_path: Path) -> None:
        """A missing langgraph.json should be rebuilt via the scaffold hook."""
        config_dir = tmp_path / "runtime"
        config_dir.mkdir()

        def scaffold(work_dir: Path) -> None:
            (work_dir / "langgraph.json").write_text("{}")

        scaffold_mock = MagicMock(side_effect=scaffold)

        process = MagicMock()
        process.pid = 1234
        process.poll.return_value = None

        log_file = MagicMock()
        log_file.name = str(tmp_path / "server.log")

        server = ServerProcess(config_dir=config_dir, scaffold=scaffold_mock)

        with (
            patch("deepagents_code.server._find_free_port", return_value=12345),
            patch(
                "deepagents_code.server.tempfile.NamedTemporaryFile",
                return_value=log_file,
            ),
            patch("deepagents_code.server.subprocess.Popen", return_value=process),
            patch(
                "deepagents_code.server.wait_for_server_healthy",
                new=AsyncMock(),
            ),
        ):
            await server.start()

        scaffold_mock.assert_called_once_with(config_dir)
        assert (config_dir / "langgraph.json").exists()

    async def test_start_raises_when_scaffold_does_not_restore_config(
        self, tmp_path: Path
    ) -> None:
        """A scaffold hook that runs but produces no config still raises.

        The error must report the failed rescaffold (not the misleading
        "call generate_langgraph_json() first") and the hook must have run.
        """
        config_dir = tmp_path / "runtime"
        config_dir.mkdir()

        scaffold_mock = MagicMock()
        server = ServerProcess(config_dir=config_dir, scaffold=scaffold_mock)

        with pytest.raises(RuntimeError, match=r"did not produce langgraph\.json"):
            await server.start()

        scaffold_mock.assert_called_once_with(config_dir)

    async def test_start_raises_without_scaffold_when_config_missing(
        self, tmp_path: Path
    ) -> None:
        """With no scaffold hook, a missing config raises the original error."""
        config_dir = tmp_path / "runtime"
        config_dir.mkdir()

        server = ServerProcess(config_dir=config_dir)

        with pytest.raises(RuntimeError, match=r"langgraph\.json not found"):
            await server.start()

    async def test_start_wraps_scaffold_oserror(self, tmp_path: Path) -> None:
        """An OSError raised mid-scaffold surfaces as RuntimeError, cause kept."""
        config_dir = tmp_path / "runtime"
        config_dir.mkdir()

        boom = OSError("No space left on device")
        server = ServerProcess(
            config_dir=config_dir, scaffold=MagicMock(side_effect=boom)
        )

        with pytest.raises(RuntimeError, match=r"Failed to rescaffold") as exc_info:
            await server.start()

        assert exc_info.value.__cause__ is boom

    async def test_start_creates_work_dir_when_purged(self, tmp_path: Path) -> None:
        """A fully purged work dir is recreated before the scaffold runs.

        Exercises the `mkdir(parents=True)` recovery: the directory itself —
        not just `langgraph.json` — is gone (the OS tmp reaper removing the
        whole temp dir), so the scaffold would fail without the mkdir.
        """
        # Deliberately not created: the directory is missing entirely.
        config_dir = tmp_path / "runtime"

        def scaffold(work_dir: Path) -> None:
            (work_dir / "langgraph.json").write_text("{}")

        scaffold_mock = MagicMock(side_effect=scaffold)

        process = MagicMock()
        process.pid = 1234
        process.poll.return_value = None

        log_file = MagicMock()
        log_file.name = str(tmp_path / "server.log")

        server = ServerProcess(config_dir=config_dir, scaffold=scaffold_mock)

        with (
            patch("deepagents_code.server._find_free_port", return_value=12345),
            patch(
                "deepagents_code.server.tempfile.NamedTemporaryFile",
                return_value=log_file,
            ),
            patch("deepagents_code.server.subprocess.Popen", return_value=process),
            patch(
                "deepagents_code.server.wait_for_server_healthy",
                new=AsyncMock(),
            ),
        ):
            await server.start()

        scaffold_mock.assert_called_once_with(config_dir)
        assert config_dir.is_dir()
        assert (config_dir / "langgraph.json").exists()

    async def test_start_does_not_rescaffold_when_config_present(
        self, tmp_path: Path
    ) -> None:
        """The scaffold hook is a recovery path only; skip it when config exists."""
        config_dir = tmp_path / "runtime"
        config_dir.mkdir()
        (config_dir / "langgraph.json").write_text("{}")

        scaffold_mock = MagicMock()

        process = MagicMock()
        process.pid = 1234
        process.poll.return_value = None

        log_file = MagicMock()
        log_file.name = str(tmp_path / "server.log")

        server = ServerProcess(config_dir=config_dir, scaffold=scaffold_mock)

        with (
            patch("deepagents_code.server._find_free_port", return_value=12345),
            patch(
                "deepagents_code.server.tempfile.NamedTemporaryFile",
                return_value=log_file,
            ),
            patch("deepagents_code.server.subprocess.Popen", return_value=process),
            patch(
                "deepagents_code.server.wait_for_server_healthy",
                new=AsyncMock(),
            ),
        ):
            await server.start()

        scaffold_mock.assert_not_called()

    async def test_restart_rescaffolds_after_config_purged(
        self, tmp_path: Path
    ) -> None:
        """restart() rebuilds a config purged between boot and the restart.

        This is the motivating scenario: the server boots with config present,
        the work dir is purged externally, and a later `/restart` recovers via
        the scaffold hook rather than failing.
        """
        config_dir = tmp_path / "runtime"
        config_dir.mkdir()
        config_path = config_dir / "langgraph.json"
        config_path.write_text("{}")

        def scaffold(work_dir: Path) -> None:
            (work_dir / "langgraph.json").write_text("{}")

        scaffold_mock = MagicMock(side_effect=scaffold)

        process = MagicMock()
        process.pid = 1234
        process.poll.return_value = None

        log_file = MagicMock()
        log_file.name = str(tmp_path / "server.log")

        server = ServerProcess(config_dir=config_dir, scaffold=scaffold_mock)

        with (
            patch("deepagents_code.server._find_free_port", return_value=12345),
            patch("deepagents_code.server._port_in_use", return_value=False),
            patch(
                "deepagents_code.server.tempfile.NamedTemporaryFile",
                return_value=log_file,
            ),
            patch("deepagents_code.server.subprocess.Popen", return_value=process),
            patch(
                "deepagents_code.server.wait_for_server_healthy",
                new=AsyncMock(),
            ),
        ):
            await server.start()
            # Config present on boot: the scaffold hook must not have fired yet.
            scaffold_mock.assert_not_called()

            # Simulate the OS tmp reaper purging the work dir.
            config_path.unlink()

            await server.restart()

        scaffold_mock.assert_called_once_with(config_dir)
        assert config_path.exists()

    async def test_update_env_and_restart(self, tmp_path: Path) -> None:
        """update_env stages overrides that restart() applies."""
        config_dir = tmp_path / "runtime"
        config_dir.mkdir()
        (config_dir / "langgraph.json").write_text("{}")

        log_path = tmp_path / "server.log"
        log_path.write_text("")

        process = MagicMock()
        process.pid = 1234
        process.poll.return_value = None

        log_file = MagicMock()
        log_file.name = str(log_path)

        server = ServerProcess(config_dir=config_dir, owns_config_dir=False)

        with (
            patch("deepagents_code.server._find_free_port", return_value=12345),
            patch("deepagents_code.server._port_in_use", return_value=False),
            patch(
                "deepagents_code.server.tempfile.NamedTemporaryFile",
                return_value=log_file,
            ),
            patch("deepagents_code.server.subprocess.Popen", return_value=process),
            patch(
                "deepagents_code.server.wait_for_server_healthy",
                new=AsyncMock(),
            ),
        ):
            await server.start()
            assert server.running

            server.update_env(DEEPAGENTS_CODE_SERVER_MODEL="anthropic:claude-opus-4-6")

            # Restart: should stop the old process and start a new one
            await server.restart()

        # Env override was applied
        env_key = "DEEPAGENTS_CODE_SERVER_MODEL"
        assert os.environ.get(env_key) == "anthropic:claude-opus-4-6"
        # Overrides cleared after successful restart
        assert server._env_overrides == {}

    async def test_restart_runs_blocking_stop_off_event_loop(
        self, tmp_path: Path
    ) -> None:
        """restart() must run the blocking subprocess stop off the event loop.

        `_stop_process` blocks up to `_SHUTDOWN_TIMEOUT` (plus a SIGKILL grace
        wait) on `process.wait`; running it directly on the loop freezes the
        Textual reactor so `/restart` wedges the TUI input. `restart()` must
        offload it to a worker thread, so the stop executes on a thread other
        than the one running the event loop.
        """
        config_dir = tmp_path / "runtime"
        config_dir.mkdir()
        (config_dir / "langgraph.json").write_text("{}")

        server = ServerProcess(config_dir=config_dir, owns_config_dir=False)
        server._process = MagicMock()

        loop_thread_id = threading.get_ident()
        stop_thread_id: int | None = None

        def recording_stop() -> None:
            nonlocal stop_thread_id
            stop_thread_id = threading.get_ident()
            server._process = None

        # Patch only `start` (avoid spawning a real server) and `_stop_process`
        # (record its executing thread). The real `restart()` and real
        # `asyncio.to_thread` run, so a regression to a direct call would run
        # `_stop_process` on the loop thread and fail the off-loop assertion.
        with (
            patch.object(server, "start", new=AsyncMock()),
            patch.object(server, "_stop_process", new=recording_stop),
        ):
            await server.restart()

        assert stop_thread_id is not None
        assert stop_thread_id != loop_thread_id

    async def test_persistent_env_applies_to_later_restarts(
        self, tmp_path: Path
    ) -> None:
        """Persistent env overrides should apply without restaging."""
        config_dir = tmp_path / "runtime"
        config_dir.mkdir()
        (config_dir / "langgraph.json").write_text("{}")

        first_process = MagicMock()
        first_process.pid = 1234
        first_process.poll.return_value = None
        second_process = MagicMock()
        second_process.pid = 5678
        second_process.poll.return_value = None

        first_log_file = MagicMock()
        first_log_file.name = str(tmp_path / "server-1.log")
        second_log_file = MagicMock()
        second_log_file.name = str(tmp_path / "server-2.log")

        env_key = "DEEPAGENTS_CODE_SERVER_RUBRIC_MAX_ITERATIONS"
        server = ServerProcess(config_dir=config_dir, owns_config_dir=False)
        server.persist_env(**{env_key: "12"})

        with (
            patch("deepagents_code.server._find_free_port", return_value=12345),
            patch("deepagents_code.server._port_in_use", return_value=False),
            patch(
                "deepagents_code.server.tempfile.NamedTemporaryFile",
                side_effect=[first_log_file, second_log_file],
            ),
            patch(
                "deepagents_code.server.subprocess.Popen",
                side_effect=[first_process, second_process],
            ) as popen,
            patch(
                "deepagents_code.server.wait_for_server_healthy",
                new=AsyncMock(),
            ),
        ):
            await server.start()
            assert server._env_overrides == {}

            await server.restart()

        first_env = popen.call_args_list[0].kwargs["env"]
        second_env = popen.call_args_list[1].kwargs["env"]
        assert first_env[env_key] == "12"
        assert second_env[env_key] == "12"
        assert server._env_overrides == {}

    async def test_one_shot_override_wins_over_persisted(self, tmp_path: Path) -> None:
        """A one-shot `update_env` must override a persisted default on restart.

        Regression: persisting a value (via `persist_env`) and then staging a
        different value (via `update_env`) before a restart must launch the
        subprocess with the freshly staged value, not the stale persisted one.
        """
        config_dir = tmp_path / "runtime"
        config_dir.mkdir()
        (config_dir / "langgraph.json").write_text("{}")

        first_process = MagicMock()
        first_process.pid = 1234
        first_process.poll.return_value = None
        second_process = MagicMock()
        second_process.pid = 5678
        second_process.poll.return_value = None
        first_log_file = MagicMock()
        first_log_file.name = str(tmp_path / "server-1.log")
        second_log_file = MagicMock()
        second_log_file.name = str(tmp_path / "server-2.log")

        env_key = "DEEPAGENTS_CODE_SERVER_RUBRIC_MAX_ITERATIONS"
        server = ServerProcess(config_dir=config_dir, owns_config_dir=False)
        server.persist_env(**{env_key: "10"})

        with (
            patch("deepagents_code.server._find_free_port", return_value=12345),
            patch("deepagents_code.server._port_in_use", return_value=False),
            patch(
                "deepagents_code.server.tempfile.NamedTemporaryFile",
                side_effect=[first_log_file, second_log_file],
            ),
            patch(
                "deepagents_code.server.subprocess.Popen",
                side_effect=[first_process, second_process],
            ) as popen,
            patch(
                "deepagents_code.server.wait_for_server_healthy",
                new=AsyncMock(),
            ),
        ):
            await server.start()
            # Stage a different value for the next restart only.
            server.update_env(**{env_key: "12"})
            await server.restart()

        # The restart that applies the staged override must use it, not the
        # persisted default it temporarily supersedes.
        second_env = popen.call_args_list[1].kwargs["env"]
        assert second_env[env_key] == "12"

    async def test_restart_rollback_on_failure(self, tmp_path: Path) -> None:
        """Env overrides are rolled back when restart fails."""
        config_dir = tmp_path / "runtime"
        config_dir.mkdir()
        (config_dir / "langgraph.json").write_text("{}")

        process = MagicMock()
        process.pid = 1234
        process.poll.return_value = None

        server = ServerProcess(config_dir=config_dir, owns_config_dir=False)
        server._process = process  # simulate already started

        old_value = os.environ.get("DEEPAGENTS_CODE_SERVER_MODEL")

        async def failing_start(*, timeout: float = 60) -> None:  # noqa: ARG001, ASYNC109, RUF029
            msg = "restart failed"
            raise RuntimeError(msg)

        server.start = failing_start  # ty: ignore
        server.update_env(DEEPAGENTS_CODE_SERVER_MODEL="should-be-rolled-back")

        with pytest.raises(RuntimeError, match="restart failed"):
            await server.restart()

        # Env should be rolled back
        assert os.environ.get("DEEPAGENTS_CODE_SERVER_MODEL") == old_value
        # Overrides NOT cleared (available for retry)
        assert "DEEPAGENTS_CODE_SERVER_MODEL" in server._env_overrides
