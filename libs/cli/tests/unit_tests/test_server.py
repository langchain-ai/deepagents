"""Tests for server lifecycle helpers."""

from __future__ import annotations

import socket
from typing import Self
from unittest.mock import patch

from deepagents_cli.server import _find_free_port, _port_in_use


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
