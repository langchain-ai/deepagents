"""Pipe ownership and resource cleanup for server process generations."""

from __future__ import annotations

import errno
import gc
import os
import sys
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest

from deepagents_code.client.launch import server as server_module
from deepagents_code.client.launch.server import ServerProcess

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import TextIO

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX pipe ownership")


@dataclass
class _Launch:
    reader: int
    process: MagicMock
    log: TextIO

    def close_reader(self) -> None:
        """Release the fake child's copy without closing a reused descriptor."""
        os.close(self.reader)
        self.reader = -1


def _make_server(tmp_path: Path) -> ServerProcess:
    config_dir = tmp_path / "runtime"
    config_dir.mkdir()
    (config_dir / "langgraph.json").write_text("{}")
    return ServerProcess(config_dir=config_dir)


def _assert_writer_open(reader: int) -> None:
    with pytest.raises(BlockingIOError):
        os.read(reader, 1)


def _assert_eof(reader: int) -> None:
    assert os.read(reader, 1) == b""


def _assert_log_removed(launch: _Launch) -> None:
    assert launch.log.closed
    assert not Path(launch.log.name).exists()


def _descriptor_identity(descriptor: int) -> tuple[int, int] | None:
    try:
        stat = os.fstat(descriptor)
    except OSError as error:
        if error.errno != errno.EBADF:
            raise
        return None
    return stat.st_dev, stat.st_ino


def _descriptor_count() -> int:
    return len(tuple(Path("/proc/self/fd").iterdir()))


@pytest.fixture(autouse=True)
def _server_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resource tests do not require the application's credential bootstrap."""
    monkeypatch.setattr(
        server_module, "_server_env_with_overrides", MagicMock(return_value={})
    )


@pytest.fixture
def launches(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[_Launch]]:
    records: list[_Launch] = []

    def popen(
        _command: list[str],
        *,
        pass_fds: tuple[int, ...],
        stdout: TextIO,
        **_kwargs: object,
    ) -> MagicMock:
        reader = os.dup(pass_fds[0])
        os.set_blocking(reader, False)
        process = MagicMock(pid=4321)
        process.poll.return_value = None
        records.append(_Launch(reader, process, stdout))
        return process

    def terminate(process: MagicMock) -> None:
        process.poll.return_value = 0

    monkeypatch.delenv("DEEPAGENTS_CODE_DEBUG", raising=False)
    monkeypatch.setattr(server_module.subprocess, "Popen", popen)
    monkeypatch.setattr(server_module, "_terminate_server_process", terminate)
    monkeypatch.setattr(server_module, "wait_for_server_healthy", AsyncMock())
    monkeypatch.setattr(server_module, "_find_free_port", MagicMock(return_value=43210))
    monkeypatch.setattr(server_module, "_port_in_use", MagicMock(return_value=False))
    yield records
    for launch in records:
        if launch.reader >= 0:
            launch.close_reader()
        launch.log.close()
        Path(launch.log.name).unlink(missing_ok=True)


@pytest.fixture
def server(tmp_path: Path, launches: list[_Launch]) -> Iterator[ServerProcess]:  # noqa: ARG001  # ensures process mocks outlive manager cleanup
    instance = _make_server(tmp_path)
    yield instance
    instance.stop()


async def test_explicit_stop_retains_writer_through_grace_period(
    server: ServerProcess,
    launches: list[_Launch],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit termination owns the entire grace period before causing EOF."""
    await server.start()
    launch = launches[-1]
    _assert_writer_open(launch.reader)

    def terminate(process: MagicMock) -> None:
        _assert_writer_open(launch.reader)
        process.poll.return_value = 0

    monkeypatch.setattr(server_module, "_terminate_server_process", terminate)
    server.stop()

    _assert_eof(launch.reader)
    _assert_log_removed(launch)


def test_failed_spawn_closes_pipe_and_log(
    server: ServerProcess,
    launches: list[_Launch],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Popen exception releases both local pipe ends and the generation log."""
    descriptors: dict[int, tuple[int, int]] = {}

    def fail(
        _command: list[str],
        *,
        pass_fds: tuple[int, ...],
        stdout: TextIO,
        **_kwargs: object,
    ) -> None:
        writer = server._parent_pipe_writer
        assert writer is not None
        for descriptor in (pass_fds[0], writer):
            stat = os.fstat(descriptor)
            descriptors[descriptor] = (stat.st_dev, stat.st_ino)
        reader = os.dup(pass_fds[0])
        os.set_blocking(reader, False)
        launches.append(_Launch(reader, MagicMock(), stdout))
        msg = "cannot spawn server"
        raise OSError(msg)

    monkeypatch.setattr(server_module.subprocess, "Popen", fail)
    with pytest.raises(OSError, match="cannot spawn server"):
        server._spawn_process(expected_stop_generation=None)

    # Compare identity as well as the number, so an unrelated descriptor reused
    # by the test runner cannot make this report a false resource leak.
    for descriptor, identity in descriptors.items():
        assert _descriptor_identity(descriptor) != identity
    launch = launches[-1]
    _assert_eof(launch.reader)
    _assert_log_removed(launch)
    assert not server.running


async def test_exited_generation_releases_writer_before_replacement(
    server: ServerProcess, launches: list[_Launch]
) -> None:
    """Restarting an already-exited root cannot retain its old writer or log."""
    await server.start()
    old = launches[-1]
    old.process.poll.return_value = 0

    await server.start()

    _assert_eof(old.reader)
    _assert_log_removed(old)
    _assert_writer_open(launches[-1].reader)
    assert server.running


@pytest.mark.skipif(sys.platform != "linux", reason="counts /proc/self/fd")
async def test_start_stop_and_restart_do_not_accumulate_descriptors(
    server: ServerProcess, launches: list[_Launch]
) -> None:
    """Repeated generations return to the original descriptor count."""
    baseline = _descriptor_count()
    for _ in range(5):
        await server.start()
        old = launches[-1]
        await server.restart()
        _assert_eof(old.reader)
        old.close_reader()
        active = launches[-1]
        _assert_writer_open(active.reader)
        server.stop()
        _assert_eof(active.reader)
        active.close_reader()
        assert _descriptor_count() == baseline


def test_abandoned_manager_releases_writer(
    tmp_path: Path, launches: list[_Launch]
) -> None:
    """Garbage collection makes EOF visible even without an explicit stop."""
    instance = _make_server(tmp_path)
    instance._spawn_process(expected_stop_generation=None)
    launch = launches[-1]
    _assert_writer_open(launch.reader)
    reference = weakref.ref(instance)

    del instance
    gc.collect()

    assert reference() is None
    _assert_eof(launch.reader)


async def test_windows_launch_keeps_direct_cli_without_lifetime_pipe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Windows continues to use its console group and direct LangGraph CLI."""
    instance = _make_server(tmp_path)
    process = MagicMock(pid=4321)
    process.poll.return_value = None
    popen = MagicMock(return_value=process)
    monkeypatch.setattr(server_module, "wait_for_server_healthy", AsyncMock())
    monkeypatch.setattr(server_module, "_find_free_port", MagicMock(return_value=43210))
    monkeypatch.setattr(server_module, "_terminate_server_process", MagicMock())
    monkeypatch.setattr(server_module.subprocess, "Popen", popen)
    monkeypatch.setattr(
        server_module.os,
        "pipe",
        MagicMock(side_effect=AssertionError("unexpected pipe")),
    )
    monkeypatch.setattr(server_module.sys, "platform", "win32")
    try:
        await instance.start()
        assert popen.call_args.args[0][2] == "langgraph_cli"
        assert "pass_fds" not in popen.call_args.kwargs
        assert popen.call_args.kwargs["start_new_session"] is False
        assert (
            popen.call_args.kwargs["creationflags"]
            == server_module._WINDOWS_CREATE_NEW_PROCESS_GROUP
        )
    finally:
        process.poll.return_value = 0
        instance.stop()
