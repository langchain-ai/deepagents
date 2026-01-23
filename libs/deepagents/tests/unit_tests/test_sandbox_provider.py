"""Tests for SandboxProvider protocol and related types."""

from typing import Any, Literal

from typing_extensions import TypedDict

from deepagents.backends.protocol import ExecuteResponse, FileDownloadResponse, FileUploadResponse, SandboxBackendProtocol
from deepagents.backends.sandbox import (
    BaseSandbox,
    SandboxInfo,
    SandboxListResponse,
    SandboxProvider,
)


class MockMetadata(TypedDict, total=False):
    """Example typed metadata for sandboxes."""

    status: Literal["running", "stopped"]
    template: str


class MockListKwargs(TypedDict, total=False):
    """Example kwargs for list operation (documentation only).

    While implementations accept **kwargs: Any, defining these TypedDicts
    serves as documentation for what kwargs are supported by this provider.
    """

    status: Literal["running", "stopped"]
    template_id: str


class MockCreateKwargs(TypedDict, total=False):
    """Example kwargs for get_or_create operation (documentation only)."""

    template_id: str
    timeout_minutes: int


class MockDeleteKwargs(TypedDict, total=False):
    """Example kwargs for delete operation (documentation only)."""

    force: bool


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


class MockSandboxProvider(SandboxProvider[MockMetadata]):
    """Mock provider implementation for testing.

    This demonstrates how to implement the SandboxProvider ABC
    with custom kwargs types and typed metadata.
    """

    def __init__(self) -> None:
        self.sandboxes: dict[str, MockMetadata] = {
            "sb_001": {"status": "running", "template": "python-3.11"},
            "sb_002": {"status": "stopped", "template": "node-20"},
        }

    def list(
        self,
        *,
        cursor: str | None = None,
        status: Literal["running", "stopped"] | None = None,
        template_id: str | None = None,
        **kwargs: Any,
    ) -> SandboxListResponse[MockMetadata]:
        """List sandboxes with optional filtering.

        Args:
            cursor: Pagination cursor (unused in this simple implementation).
            status: Filter by sandbox status.
            template_id: Filter by template ID.
            **kwargs: Additional provider-specific filters (unused).
        """
        _ = cursor  # Unused in simple implementation
        _ = kwargs  # No additional filters supported

        items: list[SandboxInfo[MockMetadata]] = []

        for sandbox_id, metadata in self.sandboxes.items():
            # Apply status filter
            if status and metadata.get("status") != status:
                continue
            # Apply template filter
            if template_id and metadata.get("template") != template_id:
                continue

            items.append(
                {
                    "sandbox_id": sandbox_id,
                    "metadata": metadata,
                }
            )

        return {
            "items": items,
            "cursor": None,  # Simple implementation without pagination
        }

    def get_or_create(
        self,
        *,
        sandbox_id: str | None = None,
        template_id: str = "default",
        timeout_minutes: int | None = None,
        **kwargs: Any,
    ) -> SandboxBackendProtocol:
        """Get existing or create new sandbox.

        Args:
            sandbox_id: ID of existing sandbox to retrieve, or None to create new.
            template_id: Template to use for new sandboxes.
            timeout_minutes: Timeout for sandbox operations (unused).
            **kwargs: Additional provider-specific options (unused).
        """
        _ = timeout_minutes  # Unused in simple implementation
        _ = kwargs  # No additional options supported

        if sandbox_id is None:
            # Create new sandbox
            new_id = f"sb_{len(self.sandboxes) + 1:03d}"
            self.sandboxes[new_id] = {"status": "running", "template": template_id}
            return MockSandboxBackend(new_id)

        # Get existing sandbox
        if sandbox_id not in self.sandboxes:
            msg = f"Sandbox {sandbox_id} not found"
            raise ValueError(msg)

        return MockSandboxBackend(sandbox_id)

    def delete(
        self,
        sandbox_id: str,
        *,
        force: bool = False,
        **kwargs: Any,
    ) -> None:
        """Delete a sandbox.

        Args:
            sandbox_id: ID of sandbox to delete.
            force: Force deletion even if sandbox is running (unused).
            **kwargs: Additional provider-specific options (unused).
        """
        _ = force  # Unused in simple implementation
        _ = kwargs  # No additional options supported

        if sandbox_id not in self.sandboxes:
            msg = f"Sandbox {sandbox_id} not found"
            raise ValueError(msg)

        del self.sandboxes[sandbox_id]


def test_sandbox_info_structure() -> None:
    """Test SandboxInfo TypedDict structure."""
    info: SandboxInfo[MockMetadata] = {
        "sandbox_id": "sb_123",
        "metadata": {"status": "running", "template": "python-3.11"},
    }

    assert info["sandbox_id"] == "sb_123"
    metadata = info.get("metadata")
    assert metadata is not None
    assert metadata["status"] == "running"


def test_sandbox_list_response() -> None:
    """Test SandboxListResponse structure."""
    response: SandboxListResponse[MockMetadata] = {
        "items": [
            {"sandbox_id": "sb_001", "metadata": {"status": "running", "template": "python-3.11"}},
            {"sandbox_id": "sb_002"},  # metadata is optional
        ],
        "cursor": "next_page_token",
    }

    assert len(response["items"]) == 2
    assert response["cursor"] == "next_page_token"


def test_provider_list_all() -> None:
    """Test listing all sandboxes."""
    provider = MockSandboxProvider()
    result = provider.list()

    assert len(result["items"]) == 2
    assert result["cursor"] is None


def test_provider_list_with_filter() -> None:
    """Test listing with status filter."""
    provider = MockSandboxProvider()
    result = provider.list(status="running")

    assert len(result["items"]) == 1
    assert result["items"][0]["sandbox_id"] == "sb_001"


def test_provider_get_or_create_existing() -> None:
    """Test getting an existing sandbox."""
    provider = MockSandboxProvider()
    sandbox = provider.get_or_create(sandbox_id="sb_001")

    assert sandbox.id == "sb_001"


def test_provider_get_or_create_new() -> None:
    """Test creating a new sandbox."""
    provider = MockSandboxProvider()
    sandbox = provider.get_or_create(sandbox_id=None, template_id="python-3.11", timeout_minutes=60)

    assert sandbox.id == "sb_003"
    assert "sb_003" in provider.sandboxes


def test_provider_delete() -> None:
    """Test deleting a sandbox."""
    provider = MockSandboxProvider()
    assert "sb_001" in provider.sandboxes

    provider.delete(sandbox_id="sb_001")

    assert "sb_001" not in provider.sandboxes


def test_provider_protocol_compliance() -> None:
    """Test that MockSandboxProvider satisfies the protocol."""
    provider: SandboxProvider = MockSandboxProvider()  # type: ignore[type-arg]

    # Should be able to call protocol methods
    result = provider.list()
    assert isinstance(result, dict)
    assert "items" in result
    assert "cursor" in result


async def test_provider_async_list() -> None:
    """Test async list method (defaults to running sync in thread)."""
    provider = MockSandboxProvider()
    result = await provider.alist()

    assert len(result["items"]) == 2
    assert result["cursor"] is None


async def test_provider_async_get_or_create() -> None:
    """Test async get_or_create method."""
    provider = MockSandboxProvider()
    sandbox = await provider.aget_or_create(sandbox_id="sb_001")

    assert sandbox.id == "sb_001"


async def test_provider_async_delete() -> None:
    """Test async delete method."""
    provider = MockSandboxProvider()
    assert "sb_001" in provider.sandboxes

    await provider.adelete(sandbox_id="sb_001")

    assert "sb_001" not in provider.sandboxes
