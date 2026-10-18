"""Exercise the real server spawn and bootstrap across owner process death.

The synthetic LangGraph CLI uses only the standard library and explicit file
barriers. Linux pidfds pin every cleanup signal to a process created here; a
scoped subreaper lets the tests reap servers orphaned by the killed owner.
"""

from __future__ import annotations

import contextlib
import ctypes
import json
import os
import select
import signal
import subprocess
import sys
import textwrap
import time
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

pytestmark = pytest.mark.skipif(
    sys.platform != "linux" or not hasattr(os, "pidfd_open"),
    reason="process cleanup and orphan reaping use Linux pidfds and subreapers",
)

_OWNER = """
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from deepagents_code.client.launch import server as module

root = Path(sys.argv[1])
module._SHUTDOWN_TIMEOUT = float(sys.argv[2])
# Keep the production spawn/pipe/bootstrap path, replacing only the test CLI.
module._server_env_with_overrides = lambda *_: dict(os.environ)
module._port_in_use = lambda *_: False
server = module.ServerProcess(config_dir=root, port=54321)
process = server._spawn_process(expected_stop_generation=None)
state = {"server": process.pid}
if os.environ["LIFETIME_MODE"] == "descendant":
    descendant = subprocess.Popen(
        [sys.executable, "-c", "import signal; signal.pause()"],
        close_fds=False,
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    state["descendant"] = descendant.pid
temporary = root / "state.tmp"
temporary.write_text(json.dumps(state))
temporary.replace(root / "state.json")
while not (root / "stop").exists():
    time.sleep(0.01)
server.stop()
(root / "result.json").write_text(json.dumps({"returncode": process.returncode}))
"""

_CLI_INIT = """
import os
import signal
from pathlib import Path

if os.environ["LIFETIME_MODE"] == "import":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    Path(os.environ["LIFETIME_ROOT"], "ready").touch()
    while True:
        signal.pause()
"""

_CLI_MAIN = """
import atexit
import os
import signal
import time
from pathlib import Path

root = Path(os.environ["LIFETIME_ROOT"])
mode = os.environ["LIFETIME_MODE"]

def graceful(*_):
    with (root / "terms").open("a") as output:
        output.write("term\\n")
    (root / "stopping").touch()
    if mode == "grace":
        while not (root / "release").exists():
            time.sleep(0.01)
    (root / "finished").touch()
    raise SystemExit(0)

def stall():
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    (root / "ready").touch()
    while True:
        signal.pause()

if mode == "atexit":
    atexit.register(stall)
else:
    signal.signal(signal.SIGTERM, graceful)
    (root / "ready").touch()
    while True:
        signal.pause()
"""


def _wait_for_file(path: Path, process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 10
    while not path.exists() and process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert path.exists(), f"Missing barrier {path.name}; owner status={process.poll()}"


def _exited(pidfd: int, *, timeout: float) -> bool:
    return bool(select.select([pidfd], [], [], timeout)[0])


class _Owner:
    def __init__(self, root: Path, mode: str, timeout: float) -> None:
        self.root = root
        package = root / "langgraph_cli"
        package.mkdir()
        (package / "__init__.py").write_text(textwrap.dedent(_CLI_INIT))
        (package / "__main__.py").write_text(textwrap.dedent(_CLI_MAIN))
        (root / "langgraph.json").write_text("{}")
        env = dict(os.environ)
        env.update(
            PYTHONPATH=str(root),
            LIFETIME_ROOT=str(root),
            LIFETIME_MODE=mode,
            LANGSMITH_TRACING="false",
            LANGCHAIN_TRACING_V2="false",
            DEEPAGENTS_HOME=str(root / "home"),
            TMPDIR=str(root),
        )
        self.process = subprocess.Popen(
            [sys.executable, "-c", textwrap.dedent(_OWNER), str(root), str(timeout)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self.handles: dict[int, int] = {}
        try:
            _wait_for_file(root / "state.json", self.process)
            self.state = json.loads((root / "state.json").read_text())
            for pid in self.state.values():
                self.handles[pid] = os.pidfd_open(pid)
            self.wait("ready")
        except BaseException:
            self.close()
            raise

    @property
    def server_fd(self) -> int:
        return self.handles[self.state["server"]]

    def wait(self, name: str) -> None:
        _wait_for_file(self.root / name, self.process)

    def kill(self) -> None:
        self.process.kill()
        self.process.wait(timeout=5)

    def close(self) -> None:
        if self.process.poll() is None:
            self.kill()
        for pid, pidfd in self.handles.items():
            if not _exited(pidfd, timeout=0):
                with contextlib.suppress(ProcessLookupError):
                    signal.pidfd_send_signal(pidfd, signal.SIGKILL)
            assert _exited(pidfd, timeout=5), f"Fixture process {pid} survived cleanup"
            with contextlib.suppress(ChildProcessError):
                os.waitpid(pid, 0)
            os.close(pidfd)
        self.handles.clear()
        output, _ = self.process.communicate(timeout=5)
        assert self.process.returncode in {0, -signal.SIGKILL}, output


@pytest.fixture
def owner_factory(tmp_path: Path) -> Iterator[Callable[..., _Owner]]:
    # Preserve the runner's state and reap only PIDs recorded by this fixture.
    libc = ctypes.CDLL(None, use_errno=True)
    previous = ctypes.c_int()
    assert libc.prctl(37, ctypes.byref(previous), 0, 0, 0) == 0
    assert libc.prctl(36, 1, 0, 0, 0) == 0
    owners: list[_Owner] = []

    def create(mode: str, timeout: float = 0.5) -> _Owner:
        root = tmp_path / str(len(owners))
        root.mkdir()
        owner = _Owner(root, mode, timeout)
        owners.append(owner)
        return owner

    try:
        yield create
    finally:
        try:
            for owner in owners:
                owner.close()
        finally:
            assert libc.prctl(36, previous.value, 0, 0, 0) == 0


def test_owner_sigkill_stops_running_server(owner_factory) -> None:
    owner = owner_factory("ready")
    owner.kill()
    assert _exited(owner.server_fd, timeout=5), "Server survived its owner"
    assert (owner.root / "finished").exists(), "Graceful SIGTERM was not delivered"


@pytest.mark.parametrize("mode", ["import", "atexit"])
def test_owner_sigkill_bounds_stalled_startup_and_exit(
    owner_factory, mode: str
) -> None:
    owner = owner_factory(mode)
    started = time.monotonic()
    owner.kill()
    assert _exited(owner.server_fd, timeout=5), f"Server stalled during {mode}"
    assert time.monotonic() - started >= 0.45, "Grace period ended early"
    status = os.waitid(os.P_PIDFD, owner.server_fd, os.WEXITED | os.WNOWAIT)
    assert status is not None
    assert status.si_status == signal.SIGKILL, "Watchdog did not escalate"


def test_normal_stop_preserves_graceful_shutdown(owner_factory) -> None:
    owner = owner_factory("grace", timeout=2)
    (owner.root / "stop").touch()
    owner.wait("stopping")
    # Keep real teardown running long enough to expose a second owner-loss signal.
    assert not _exited(owner.server_fd, timeout=0.3)
    (owner.root / "release").touch()
    owner.wait("result.json")
    assert json.loads((owner.root / "result.json").read_text()) == {"returncode": 0}
    assert (owner.root / "terms").read_text().splitlines() == ["term"]
    assert (owner.root / "finished").exists()
    assert owner.process.wait(timeout=5) == 0


def test_exec_descendant_does_not_keep_server_owner_alive(owner_factory) -> None:
    owner = owner_factory("descendant")
    descendant_fd = owner.handles[owner.state["descendant"]]
    owner.kill()
    assert _exited(owner.server_fd, timeout=5), "Exec descendant inherited owner pipe"
    assert not _exited(descendant_fd, timeout=0), "Fixture descendant exited early"


def test_already_dead_owner_does_not_dispatch_cli(tmp_path: Path) -> None:
    package = tmp_path / "langgraph_cli"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "__main__.py").write_text(
        "from pathlib import Path\nPath('dispatched').touch()\n"
    )
    env = dict(os.environ, PYTHONPATH=str(tmp_path))
    reader, writer = os.pipe()
    os.close(writer)
    try:
        with subprocess.Popen(
            [
                sys.executable,
                "-m",
                "deepagents_code.client.launch._server_bootstrap",
                str(reader),
                "0.5",
                "dev",
            ],
            cwd=tmp_path,
            env=env,
            pass_fds=(reader,),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        ) as process:
            try:
                output, _ = process.communicate(timeout=5)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=5)
        assert process.returncode == 0, output
        assert not (tmp_path / "dispatched").exists()
    finally:
        os.close(reader)
