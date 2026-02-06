"""Tests for SandboxClient protocol and related types."""

from typing import Any, Literal

from typing_extensions import TypedDict

from deepagents.backends.protocol import ExecuteResponse, FileDownloadResponse, FileUploadResponse, SandboxBackendProtocol
from deepagents.backends.sandbox import (
    BaseSandbox,
    SandboxClient,
)


class MockMetadata(TypedDict, total=False):
    """Example typed metadata for sandboxes."""

    status: Literal["running", "stopped"]
    template: str


class MockSandboxBackend(BaseSandbox):
    """Mock implementation of SandboxBackendProtocol for testing."""

    def __init__(self, sandbox_id: str) -> None:
        self._id = sandbox_id

    def execute(
        self,
        command: str,
    ) -> ExecuteResponse:
        return ExecuteResponse(
            output="Got: " + command,
        )

    @property
    def id(self) -> str:
        return self._id

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Mock upload implementation."""
        return [FileUploadResponse(path=path) for path, _ in files]

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Mock download implementation."""
        return [FileDownloadResponse(path=path, content=b"mock content") for path in paths]


class MockSandboxClient(SandboxClient):
    """Mock provider implementation for testing.

    This demonstrates how to implement the SandboxClient ABC
    with custom kwargs types and typed metadata.
    """

    def __init__(self) -> None:
        self.sandboxes: dict[str, MockMetadata] = {
            "sb_001": {"status": "running", "template": "python-3.11"},
            "sb_002": {"status": "stopped", "template": "node-20"},
        }

    def get(
        self,
        *,
        sandbox_id: str,
        **kwargs: Any,
    ) -> SandboxBackendProtocol:
        """Get existing sandbox."""
        _ = kwargs

        if sandbox_id not in self.sandboxes:
            msg = f"Sandbox {sandbox_id} not found"
            raise ValueError(msg)

        return MockSandboxBackend(sandbox_id)

    def create(
        self,
        *,
        template_id: str = "default",
        timeout_minutes: int | None = None,
        **kwargs: Any,
    ) -> SandboxBackendProtocol:
        """Create a new sandbox.

        Args:
            sandbox_id: ID of existing sandbox to retrieve, or None to create new.
            template_id: Template to use for new sandboxes.
            timeout_minutes: Timeout for sandbox operations (unused).
            **kwargs: Additional provider-specific options (unused).
        """
        _ = timeout_minutes
        _ = kwargs

        new_id = f"sb_{len(self.sandboxes) + 1:03d}"
        self.sandboxes[new_id] = {"status": "running", "template": template_id}
        return MockSandboxBackend(new_id)

    def delete(
        self,
        *,
        sandbox_id: str,
        force: bool = False,
        **kwargs: Any,
    ) -> None:
        """Delete a sandbox.

        Idempotent - does not raise an error if sandbox doesn't exist.

        Args:
            sandbox_id: ID of sandbox to delete.
            force: Force deletion even if sandbox is running (unused).
            **kwargs: Additional provider-specific options (unused).
        """
        _ = force  # Unused in simple implementation
        _ = kwargs  # No additional options supported

        # Idempotent - silently succeed if sandbox doesn't exist
        if sandbox_id in self.sandboxes:
            del self.sandboxes[sandbox_id]


def test_client_get_existing() -> None:
    """Test getting an existing sandbox."""
    client = MockSandboxClient()
    sandbox = client.get(sandbox_id="sb_001")

    assert sandbox.id == "sb_001"


def test_client_create_new() -> None:
    """Test creating a new sandbox."""
    client = MockSandboxClient()
    sandbox = client.create(template_id="python-3.11", timeout_minutes=60)

    assert sandbox.id == "sb_003"
    assert "sb_003" in client.sandboxes


def test_client_delete() -> None:
    """Test deleting a sandbox."""
    client = MockSandboxClient()
    assert "sb_001" in client.sandboxes

    client.delete(sandbox_id="sb_001")

    assert "sb_001" not in client.sandboxes


def test_client_delete_idempotent() -> None:
    """Test that delete is idempotent (doesn't error on non-existent sandbox)."""
    client = MockSandboxClient()

    # Delete non-existent sandbox - should not raise an error
    client.delete(sandbox_id="sb_999")

    # Delete existing sandbox twice - should not raise an error
    client.delete(sandbox_id="sb_001")
    client.delete(sandbox_id="sb_001")  # Second delete should succeed

    assert "sb_001" not in client.sandboxes


def test_client_protocol_compliance() -> None:
    """Test that MockSandboxClient satisfies the protocol."""
    client: SandboxClient = MockSandboxClient()

    backend = client.create()
    assert isinstance(backend.id, str)

    reconnected = client.get(sandbox_id=backend.id)
    assert reconnected.id == backend.id


async def test_client_async_get() -> None:
    """Test async get method."""
    client = MockSandboxClient()
    sandbox = await client.aget(sandbox_id="sb_001")

    assert sandbox.id == "sb_001"


async def test_client_async_delete() -> None:
    """Test async delete method."""
    client = MockSandboxClient()
    assert "sb_001" in client.sandboxes

    await client.adelete(sandbox_id="sb_001")

    assert "sb_001" not in client.sandboxes
