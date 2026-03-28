"""Tests for server lifecycle helpers."""

from __future__ import annotations

import os
import socket
from typing import TYPE_CHECKING, Self
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deepagents_cli.server import (
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

    async def test_raises_on_timeout(self) -> None:
        """Timeout exhaustion raises RuntimeError."""
        import httpx

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            patch("deepagents_cli.server._HEALTH_POLL_INTERVAL_LOCAL", 0),
            patch("deepagents_cli.server._HEALTH_POLL_INTERVAL_REMOTE", 0),
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
            patch("deepagents_cli.server._HEALTH_POLL_INTERVAL_LOCAL", 0),
            patch("deepagents_cli.server._HEALTH_POLL_INTERVAL_REMOTE", 0),
            pytest.raises(RuntimeError, match="last status: 503"),
        ):
            await wait_for_server_healthy("http://localhost:2024", timeout=0.01)


class TestServerProcess:
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
            patch("deepagents_cli.server._port_in_use", return_value=False),
            patch(
                "deepagents_cli.server.tempfile.NamedTemporaryFile",
                return_value=log_file,
            ),
            patch("deepagents_cli.server.subprocess.Popen", return_value=process),
            patch(
                "deepagents_cli.server.wait_for_server_healthy",
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
            patch("deepagents_cli.server._port_in_use", return_value=False),
            patch(
                "deepagents_cli.server.tempfile.NamedTemporaryFile",
                return_value=log_file,
            ),
            patch("deepagents_cli.server.subprocess.Popen", return_value=process),
            patch(
                "deepagents_cli.server.wait_for_server_healthy",
                new=AsyncMock(),
            ),
        ):
            await server.start()
            assert server.running

            server.update_env(DA_SERVER_MODEL="anthropic:claude-opus-4-6")

            # Restart: should stop the old process and start a new one
            await server.restart()

        # Env override was applied
        assert os.environ.get("DA_SERVER_MODEL") == "anthropic:claude-opus-4-6"
        # Overrides cleared after successful restart
        assert server._env_overrides == {}

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

        old_value = os.environ.get("DA_SERVER_MODEL")

        async def failing_start(*, timeout: float = 60) -> None:  # noqa: ARG001, ASYNC109, RUF029
            msg = "restart failed"
            raise RuntimeError(msg)

        server.start = failing_start  # type: ignore[assignment]
        server.update_env(DA_SERVER_MODEL="should-be-rolled-back")

        with pytest.raises(RuntimeError, match="restart failed"):
            await server.restart()

        # Env should be rolled back
        assert os.environ.get("DA_SERVER_MODEL") == old_value
        # Overrides NOT cleared (available for retry)
        assert "DA_SERVER_MODEL" in server._env_overrides
