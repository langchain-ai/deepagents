"""Tests for server lifecycle helpers."""

from __future__ import annotations

import io
import socket

from deepagents_cli.server import _find_free_port, _port_in_use, _read_process_output


class TestPortInUse:
    def test_free_port(self) -> None:
        port = _find_free_port("127.0.0.1")
        assert not _port_in_use("127.0.0.1", port)

    def test_occupied_port(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
            assert _port_in_use("127.0.0.1", port)


class TestFindFreePort:
    def test_returns_valid_port(self) -> None:
        port = _find_free_port("127.0.0.1")
        assert 1 <= port <= 65535

    def test_port_is_actually_free(self) -> None:
        port = _find_free_port("127.0.0.1")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", port))


class _FakeProc:
    """Minimal stand-in for a finished subprocess.Popen."""

    def __init__(
        self,
        stdout: bytes | None = None,
        stderr: bytes | None = None,
    ) -> None:
        self.stdout = io.BytesIO(stdout) if stdout is not None else None
        self.stderr = io.BytesIO(stderr) if stderr is not None else None


class TestReadProcessOutput:
    def test_reads_stdout_and_stderr(self) -> None:
        proc = _FakeProc(stdout=b"stdout line", stderr=b"stderr line")
        result = _read_process_output(proc)  # type: ignore[arg-type]
        assert "stdout line" in result
        assert "stderr line" in result

    def test_empty_output(self) -> None:
        proc = _FakeProc(stdout=b"", stderr=b"")
        assert _read_process_output(proc) == ""  # type: ignore[arg-type]

    def test_no_pipes(self) -> None:
        proc = _FakeProc()
        assert _read_process_output(proc) == ""  # type: ignore[arg-type]
