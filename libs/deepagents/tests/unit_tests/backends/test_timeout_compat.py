"""Tests for timeout forwarding on sandbox execution helpers."""

from deepagents.backends.composite import CompositeBackend
from deepagents.backends.protocol import ExecuteResponse, SandboxBackendProtocol


class RecordingBackend(SandboxBackendProtocol):
    """Sandbox backend that records the last timeout it received."""

    def __init__(self) -> None:
        self.received_timeout: int | None = None

    @property
    def id(self) -> str:
        return "recording"

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        self.received_timeout = timeout
        return ExecuteResponse(output=command, exit_code=0)


class TestSandboxBackendProtocolAexecute:
    async def test_aexecute_forwards_timeout(self) -> None:
        backend = RecordingBackend()
        result = await backend.aexecute("ls", timeout=42)

        assert result.output == "ls"
        assert backend.received_timeout == 42

    async def test_aexecute_forwards_none_timeout(self) -> None:
        backend = RecordingBackend()
        result = await backend.aexecute("ls")

        assert result.output == "ls"
        assert backend.received_timeout is None


class TestCompositeTimeoutForwarding:
    def test_execute_forwards_timeout(self) -> None:
        backend = RecordingBackend()
        comp = CompositeBackend(default=backend, routes={})

        result = comp.execute("ls", timeout=60)

        assert result.output == "ls"
        assert backend.received_timeout == 60

    def test_execute_forwards_none_timeout(self) -> None:
        backend = RecordingBackend()
        comp = CompositeBackend(default=backend, routes={})

        result = comp.execute("ls")

        assert result.output == "ls"
        assert backend.received_timeout is None

    async def test_aexecute_forwards_timeout(self) -> None:
        backend = RecordingBackend()
        comp = CompositeBackend(default=backend, routes={})

        result = await comp.aexecute("ls", timeout=60)

        assert result.output == "ls"
        assert backend.received_timeout == 60

    async def test_aexecute_forwards_none_timeout(self) -> None:
        backend = RecordingBackend()
        comp = CompositeBackend(default=backend, routes={})

        result = await comp.aexecute("ls")

        assert result.output == "ls"
        assert backend.received_timeout is None
