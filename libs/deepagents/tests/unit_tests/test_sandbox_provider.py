"""Tests for SandboxProvider protocol and related types."""

from typing import Any, Literal

from typing_extensions import TypedDict

from deepagents.backends.protocol import ExecuteResponse, FileDownloadResponse, FileUploadResponse, SandboxBackendProtocol
from deepagents.backends.sandbox import (
    BaseSandbox,
    SandboxProvider,
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


class MockSandboxProvider(SandboxProvider):
    """Mock provider implementation for testing.

    This demonstrates how to implement the SandboxProvider ABC
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


def test_provider_get_existing() -> None:
    """Test getting an existing sandbox."""
    provider = MockSandboxProvider()
    sandbox = provider.get(sandbox_id="sb_001")

    assert sandbox.id == "sb_001"


def test_provider_create_new() -> None:
    """Test creating a new sandbox."""
    provider = MockSandboxProvider()
    sandbox = provider.create(template_id="python-3.11", timeout_minutes=60)

    assert sandbox.id == "sb_003"
    assert "sb_003" in provider.sandboxes


def test_provider_delete() -> None:
    """Test deleting a sandbox."""
    provider = MockSandboxProvider()
    assert "sb_001" in provider.sandboxes

    provider.delete(sandbox_id="sb_001")

    assert "sb_001" not in provider.sandboxes


def test_provider_delete_idempotent() -> None:
    """Test that delete is idempotent (doesn't error on non-existent sandbox)."""
    provider = MockSandboxProvider()

    # Delete non-existent sandbox - should not raise an error
    provider.delete(sandbox_id="sb_999")

    # Delete existing sandbox twice - should not raise an error
    provider.delete(sandbox_id="sb_001")
    provider.delete(sandbox_id="sb_001")  # Second delete should succeed

    assert "sb_001" not in provider.sandboxes


def test_provider_protocol_compliance() -> None:
    """Test that MockSandboxProvider satisfies the protocol."""
    provider: SandboxProvider = MockSandboxProvider()

    backend = provider.create()
    assert isinstance(backend.id, str)

    reconnected = provider.get(sandbox_id=backend.id)
    assert reconnected.id == backend.id


async def test_provider_async_get() -> None:
    """Test async get method."""
    provider = MockSandboxProvider()
    sandbox = await provider.aget(sandbox_id="sb_001")

    assert sandbox.id == "sb_001"


async def test_provider_async_delete() -> None:
    """Test async delete method."""
    provider = MockSandboxProvider()
    assert "sb_001" in provider.sandboxes

    await provider.adelete(sandbox_id="sb_001")

    assert "sb_001" not in provider.sandboxes
