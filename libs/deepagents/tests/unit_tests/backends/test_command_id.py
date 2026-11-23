"""Tests for passing a stable `command_id` through sandbox backends."""

from deepagents.backends.composite import CompositeBackend
from deepagents.backends.protocol import (
    ExecuteResponse,
    SandboxBackendProtocol,
    execute_accepts_command_id,
    execute_command_id,
)
from deepagents.backends.sandbox import BaseSandbox
from deepagents.backends.state import StateBackend


class ReattachingBackend(SandboxBackendProtocol):
    def __init__(self) -> None:
        self.command_ids: list[str | None] = []

    @property
    def id(self) -> str:
        return "reattaching"

    def execute(self, command: str, *, timeout: int | None = None, command_id: str | None = None) -> ExecuteResponse:
        self.command_ids.append(command_id)
        return ExecuteResponse(output=command, exit_code=0)

    async def aexecute(self, command: str, *, timeout: int | None = None, command_id: str | None = None) -> ExecuteResponse:  # noqa: ASYNC109
        return self.execute(command, timeout=timeout, command_id=command_id)


class OlderBackend(SandboxBackendProtocol):
    @property
    def id(self) -> str:
        return "older"

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        return ExecuteResponse(output=command, exit_code=0)


def test_execute_command_id_is_stable_and_specific() -> None:
    first = execute_command_id(thread_id="t", message_id="m", tool_call_id="c", command="make")

    assert first == execute_command_id(thread_id="t", message_id="m", tool_call_id="c", command="make")
    assert first.startswith("da-")
    assert len(first) == 43
    assert first != execute_command_id(thread_id="t2", message_id="m", tool_call_id="c", command="make")
    assert first != execute_command_id(thread_id="t", message_id="m2", tool_call_id="c", command="make")
    assert first != execute_command_id(thread_id="t", message_id="m", tool_call_id="c2", command="make")
    assert first != execute_command_id(thread_id="t", message_id="m", tool_call_id="c", command="make test")


def test_execute_accepts_command_id() -> None:
    assert execute_accepts_command_id(ReattachingBackend, "execute") is True
    assert execute_accepts_command_id(OlderBackend, "execute") is False


async def test_composite_forwards_command_id_to_default_backend() -> None:
    backend = ReattachingBackend()
    composite = CompositeBackend(default=backend, routes={"/memories/": StateBackend()})

    composite.execute("make", command_id="da-1")
    await composite.aexecute("make", command_id="da-2")

    assert backend.command_ids == ["da-1", "da-2"]


async def test_composite_leaves_it_out_for_older_default_backend() -> None:
    composite = CompositeBackend(default=OlderBackend(), routes={"/memories/": StateBackend()})

    assert composite.execute("make", command_id="da-1").output == "make"
    assert (await composite.aexecute("make", command_id="da-1")).output == "make"


class CapturingSandbox(BaseSandbox):
    """A sandbox whose `aexecute` records the id each wrapped command runs under."""

    enable_capture_offload = True

    def __init__(self) -> None:
        self.command_ids: list[str | None] = []

    @property
    def id(self) -> str:
        return "capturing"

    def execute(self, command: str, *, timeout: int | None = None, command_id: str | None = None) -> ExecuteResponse:
        raise NotImplementedError

    async def aexecute(self, command: str, *, timeout: int | None = None, command_id: str | None = None) -> ExecuteResponse:  # noqa: ASYNC109
        self.command_ids.append(command_id)
        return ExecuteResponse(output="__DEEPAGENTS_EXEC_META__ 0 0 0 0\nok", exit_code=0)

    def upload_files(self, files):
        raise NotImplementedError

    def download_files(self, paths):
        raise NotImplementedError


async def test_capture_wrapper_runs_under_the_callers_command_id() -> None:
    sandbox = CapturingSandbox()

    await sandbox.aexecute_with_offload("make", "/capture/call_0", max_inline_bytes=100, command_id="da-1")

    assert sandbox.command_ids == ["da-1"]
