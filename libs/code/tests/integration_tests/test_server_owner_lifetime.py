"""Exercise owner shutdown through the real LangGraph CLI and HTTP server.

Linux-only: subreaper adoption lets the test reap the server after deliberately
SIGKILLing its owner. pidfds keep cleanup scoped to the process we launched.
No model, provider credentials, or tracing service is involved.
"""

from __future__ import annotations

import contextlib
import ctypes
import json
import os
import select
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import IO

pytestmark = [
    pytest.mark.skipif(
        sys.platform != "linux" or not hasattr(os, "pidfd_open"),
        reason="orphan cleanup requires Linux subreaper adoption and pidfds",
    ),
    pytest.mark.timeout(90),
]

_OWNER = """
import asyncio
import json
import sys
from pathlib import Path

from deepagents_code.client.launch.server import (
    ServerProcess,
    wait_for_server_healthy,
)
from deepagents_code.config import _ensure_bootstrap

async def main():
    directory = Path(sys.argv[1])
    _ensure_bootstrap()
    server = ServerProcess(config_dir=directory)
    try:
        process = server._spawn_process(expected_stop_generation=None)
        metadata = directory / "spawned.json.tmp"
        metadata.write_text(json.dumps({
            "pid": process.pid,
            "url": server.url,
            "log": server._log_file.name,
        }))
        metadata.replace(directory / "spawned.json")
        await wait_for_server_healthy(
            server.url, timeout=45, process=process,
            read_log=server._read_log_file, local=True,
        )
        await server.wait_for_graph_ready(timeout=10)
        (directory / "ready").touch()
        await asyncio.to_thread(sys.stdin.readline)
    finally:
        server.stop()

asyncio.run(main())
"""

_GRAPH = """
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from starlette.applications import Starlette

class State(TypedDict):
    value: str

def echo(state: State) -> State:
    return state

graph = (
    StateGraph(State)
    .add_node("echo", echo)
    .add_edge(START, "echo")
    .add_edge("echo", END)
    .compile()
)

@asynccontextmanager
async def lifespan(app):
    yield
    directory = Path(__file__).parent
    (directory / "shutdown_started").touch()
    await asyncio.sleep(2.3)
    (directory / "shutdown_completed").touch()

app = Starlette(lifespan=lifespan)
"""


@pytest.fixture(scope="session")
def langsmith_client() -> None:
    """Override the integration fixture: this test never uses LangSmith."""


@pytest.fixture
def _subreaper() -> Iterator[None]:
    libc = ctypes.CDLL(None, use_errno=True)
    original = ctypes.c_int()
    assert libc.prctl(37, ctypes.byref(original), 0, 0, 0) == 0
    assert libc.prctl(36, 1, 0, 0, 0) == 0
    try:
        yield
    finally:
        assert libc.prctl(36, original.value, 0, 0, 0) == 0


def _isolated_env(directory: Path) -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("LANGSMITH_", "LANGCHAIN_", "LANGGRAPH_", "DEEPAGENTS_"))
        and not key.endswith("_API_KEY")
        and not key.lower().endswith("_proxy")
        and key not in {"PYTHONPATH", "PYTHONHOME"}
    }
    env.update(
        HOME=str(directory),
        LANGSMITH_TRACING="false",
        LANGCHAIN_TRACING_V2="false",
        LANGGRAPH_CLI_NO_ANALYTICS="1",
        DEEPAGENTS_CODE_NO_UPDATE_CHECK="1",
        DEEPAGENTS_HOME=str(directory / "profile"),
    )
    return env


def _wait_for_file(path: Path, owner: subprocess.Popen[str], output: IO[str]) -> None:
    deadline = time.monotonic() + 50
    while not path.exists():
        if owner.poll() is not None or time.monotonic() >= deadline:
            output.flush()
            output.seek(0)
            pytest.fail(f"Owner did not reach {path.name}:\n{output.read()}")
        time.sleep(0.05)


@contextlib.contextmanager
def _running_owner(directory: Path) -> Iterator[tuple[subprocess.Popen[str], int, str]]:
    # `pytestmark` already skips other platforms at runtime; this guard is what
    # lets `ty` resolve the Linux-only pidfd APIs when run on macOS.
    if sys.platform != "linux":
        pytest.skip("pidfd APIs are Linux-only")
    (directory / "graph.py").write_text(_GRAPH)
    (directory / "langgraph.json").write_text(
        json.dumps(
            {
                "dependencies": ["."],
                "graphs": {"agent": "./graph.py:graph"},
                "http": {"app": "./graph.py:app"},
            }
        )
    )
    with (directory / "owner.log").open("w+") as output:
        owner = subprocess.Popen(
            [sys.executable, "-c", _OWNER, str(directory)],
            cwd=directory,
            env=_isolated_env(directory),
            stdin=subprocess.PIPE,
            stdout=output,
            stderr=subprocess.STDOUT,
            text=True,
        )
        pidfd: int | None = None
        metadata: dict[str, int | str] = {}
        try:
            _wait_for_file(directory / "spawned.json", owner, output)
            metadata = json.loads((directory / "spawned.json").read_text())
            pidfd = os.pidfd_open(int(metadata["pid"]))
            _wait_for_file(directory / "ready", owner, output)
            yield owner, pidfd, str(metadata["url"])
        finally:
            if owner.poll() is None:
                owner.kill()
            owner.wait(timeout=5)
            if owner.stdin is not None:
                owner.stdin.close()
            if pidfd is not None:
                with contextlib.suppress(ProcessLookupError):
                    signal.pidfd_send_signal(pidfd, signal.SIGKILL)
                assert select.select([pidfd], [], [], 5)[0]
                with contextlib.suppress(ChildProcessError):
                    os.waitpid(int(metadata["pid"]), 0)
                os.close(pidfd)
            if "log" in metadata:
                Path(str(metadata["log"])).unlink(missing_ok=True)


@pytest.mark.parametrize("kill_owner", [False, True], ids=["stop", "owner-sigkill"])
@pytest.mark.usefixtures("_subreaper")
def test_real_server_preserves_graceful_shutdown_budget(
    tmp_path: Path, kill_owner: bool
) -> None:
    """Both explicit stop and owner death finish slow real lifespan teardown."""
    with _running_owner(tmp_path) as (owner, pidfd, url):
        with httpx.Client(base_url=url, trust_env=False) as client:
            assert client.get("/ok").status_code == 200
            response = client.get("/assistants/agent/graph")
            response.raise_for_status()
            assert any(node["id"] == "echo" for node in response.json()["nodes"])

        started = time.monotonic()
        if kill_owner:
            owner.kill()
        else:
            assert owner.stdin is not None
            owner.stdin.write("stop\n")
            owner.stdin.flush()
        owner.wait(timeout=10)
        assert owner.returncode == (-signal.SIGKILL if kill_owner else 0)
        assert select.select([pidfd], [], [], 10)[0], "LangGraph outlived its owner"
        assert (tmp_path / "shutdown_started").exists()
        assert (tmp_path / "shutdown_completed").exists(), "lifespan was interrupted"
        assert time.monotonic() - started < 8
        port = int(url.rsplit(":", 1)[1])
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", port))
