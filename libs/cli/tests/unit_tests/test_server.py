"""Tests for server lifecycle helpers."""

from __future__ import annotations

import socket
from typing import TYPE_CHECKING, Self
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deepagents_cli.server import ServerProcess, _find_free_port, _port_in_use

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
