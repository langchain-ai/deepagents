"""Focused owner-pipe lifecycle checks for the POSIX server launcher."""

from __future__ import annotations

import errno
import os
from pathlib import Path
from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock, MagicMock

import anyio
import pytest

from deepagents_code.client.launch import server as server_module
from deepagents_code.client.launch.server import ServerProcess

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from typing import TextIO

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX pipe ownership")


@pytest.fixture
def launched_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[ServerProcess, list[tuple[int, MagicMock, TextIO]]]]:
    """Retain fake child readers so EOF exposes leaked parent writers."""
    directory = tmp_path / "runtime"
    directory.mkdir()
    (directory / "langgraph.json").write_text("{}")
    server = ServerProcess(config_dir=directory)
    children: list[tuple[int, MagicMock, TextIO]] = []

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
        children.append((reader, process, stdout))
        return process

    def terminate(process: MagicMock, *, grace: float | None = None) -> None:  # noqa: ARG001
        process.poll.return_value = 0

    monkeypatch.delenv("DEEPAGENTS_CODE_DEBUG", raising=False)
    monkeypatch.setattr(
        server_module, "_server_env_with_overrides", MagicMock(return_value={})
    )
    monkeypatch.setattr(server_module.subprocess, "Popen", popen)
    monkeypatch.setattr(server_module, "_terminate_server_process", terminate)
    monkeypatch.setattr(server_module, "wait_for_server_healthy", AsyncMock())
    monkeypatch.setattr(server_module, "_find_free_port", MagicMock(return_value=43210))
    monkeypatch.setattr(server_module, "_port_in_use", MagicMock(return_value=False))
    try:
        yield server, children
    finally:
        server.stop()
        for reader, _, log in children:
            os.close(reader)
            log.close()
            Path(log.name).unlink(missing_ok=True)


async def test_normal_stop_keeps_writer_until_termination(
    launched_server: tuple[ServerProcess, list[tuple[int, MagicMock, TextIO]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server, children = launched_server
    await server.start()
    reader, _, log = children[-1]

    def terminate(process: MagicMock, *, grace: float | None = None) -> None:  # noqa: ARG001
        with pytest.raises(BlockingIOError):
            os.read(reader, 1)
        process.poll.return_value = 0

    monkeypatch.setattr(server_module, "_terminate_server_process", terminate)
    server.stop()
    assert os.read(reader, 1) == b""
    assert log.closed
    assert not (await anyio.Path(log.name).exists())


def test_failed_launch_closes_reader_writer_and_log(
    launched_server: tuple[ServerProcess, list[tuple[int, MagicMock, TextIO]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server, children = launched_server
    inherited: list[int] = []

    def fail(
        _command: list[str],
        *,
        pass_fds: tuple[int, ...],
        stdout: TextIO,
        **_kwargs: object,
    ) -> None:
        writer = server._parent_pipe_writer
        assert writer is not None
        inherited.extend((pass_fds[0], writer))
        reader = os.dup(pass_fds[0])
        os.set_blocking(reader, False)
        children.append((reader, MagicMock(), stdout))
        message = "cannot spawn server"
        raise OSError(message)

    monkeypatch.setattr(server_module.subprocess, "Popen", fail)
    with pytest.raises(OSError, match="cannot spawn server"):
        server._spawn_process(expected_stop_generation=None)

    for descriptor in inherited:
        with pytest.raises(OSError, match=r"\[Errno 9\]") as error:
            os.fstat(descriptor)
        assert error.value.errno == errno.EBADF
    reader, _, log = children[-1]
    assert os.read(reader, 1) == b""
    assert log.closed
    assert not Path(log.name).exists()
    assert not server.running


@pytest.mark.parametrize("already_exited", [False, True], ids=["restart", "exited"])
async def test_replacement_closes_old_pipe_and_log(
    launched_server: tuple[ServerProcess, list[tuple[int, MagicMock, TextIO]]],
    already_exited: bool,
) -> None:
    server, children = launched_server
    await server.start()
    old_reader, old_process, old_log = children[-1]
    if already_exited:
        old_process.poll.return_value = 0
        await server.start()
    else:
        await server.restart()
    assert os.read(old_reader, 1) == b""
    assert old_log.closed
    assert not (await anyio.Path(old_log.name).exists())
    with pytest.raises(BlockingIOError):
        os.read(children[-1][0], 1)
    server.stop()
    assert os.read(children[-1][0], 1) == b""


async def test_server_inherits_the_session_lock(
    launched_server: tuple[ServerProcess, list[tuple[int, MagicMock, TextIO]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A server that outlives its client must keep the install marked in use."""
    from deepagents_code._session_lock import hold_session_lock, session_lock_fds

    hold_session_lock()
    (lock_fd,) = session_lock_fds()
    server, _ = launched_server
    # The fixture already swapped in its fake.
    fake_popen = cast("Callable[..., MagicMock]", server_module.subprocess.Popen)
    launches: list[tuple[list[str], tuple[int, ...]]] = []

    def popen(
        command: list[str], *, pass_fds: tuple[int, ...], **kwargs: object
    ) -> MagicMock:
        launches.append((command, pass_fds))
        return fake_popen(command, pass_fds=pass_fds, **kwargs)

    monkeypatch.setattr(server_module.subprocess, "Popen", popen)
    await server.start()

    ((command, pass_fds),) = launches
    assert lock_fd in pass_fds
    # The bootstrap must learn the descriptor too, to keep it from grandchildren.
    assert command[3].split(",")[1:] == [str(lock_fd)]


def test_server_interpreter_caches_bytecode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cache server imports inside the install without writing into projects."""
    import sys

    from deepagents_code.client.launch import _server_bootstrap as bootstrap

    project = tmp_path / "project"
    project.mkdir()
    (project / "project_extension.py").write_text("value = 43\n")
    monkeypatch.syspath_prepend(str(project))
    run_module = bootstrap.runpy.run_module
    monkeypatch.setattr(bootstrap, "_ParentGuard", MagicMock())
    monkeypatch.setattr(
        bootstrap.runpy,
        "run_module",
        lambda *_args, **_kwargs: run_module("project_extension"),
    )
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "install"))
    monkeypatch.setattr(sys, "pycache_prefix", None)
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    monkeypatch.setattr(sys, "argv", ["bootstrap", "0", "5", "dev"])

    bootstrap._main()

    assert list((tmp_path / "install" / "__pycache__").rglob("*.pyc"))
    assert not (project / "__pycache__").exists()
    assert os.environ["PYTHONDONTWRITEBYTECODE"] == "1"
