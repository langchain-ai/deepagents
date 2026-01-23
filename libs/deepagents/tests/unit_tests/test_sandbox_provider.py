"""Tests for SandboxProvider protocol and related types."""

from typing import Literal, Never

from typing_extensions import TypedDict, Unpack

from deepagents.backends.protocol import SandboxBackendProtocol
from deepagents.backends.sandbox import (
    SandboxInfo,
    SandboxListResponse,
    SandboxProvider,
)


class MockMetadata(TypedDict, total=False):
    """Example typed metadata for sandboxes."""

    status: Literal["running", "stopped"]
    template: str


class MockListKwargs(TypedDict, total=False):
    """Example kwargs for list operation."""

    status: Literal["running", "stopped"]
    template_id: str


class MockCreateKwargs(TypedDict, total=False):
    """Example kwargs for get_or_create operation."""

    template_id: str
    timeout_minutes: int


class MockDeleteKwargs(TypedDict, total=False):
    """Example kwargs for delete operation."""

    force: bool


class MockSandboxBackend(SandboxBackendProtocol):
    """Mock implementation of SandboxBackendProtocol for testing."""

    def __init__(self, sandbox_id: str) -> None:
        self._id = sandbox_id

    @property
    def id(self) -> str:
        return self._id

    # Minimal implementation - other methods would raise NotImplementedError
    def execute(self, command: str) -> Never:  # type: ignore[override]
        raise NotImplementedError

    def ls_info(self, path: str) -> Never:  # type: ignore[override]
        raise NotImplementedError

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> Never:  # type: ignore[override]
        raise NotImplementedError

    def write(self, file_path: str, content: str) -> Never:  # type: ignore[override]
        raise NotImplementedError

    def edit(  # type: ignore[override]
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        *,
        replace_all: bool = False,
    ) -> Never:
        raise NotImplementedError

    def grep_raw(self, pattern: str, path: str | None = None, glob: str | None = None) -> Never:  # type: ignore[override]
        raise NotImplementedError

    def glob_info(self, pattern: str, path: str = "/") -> Never:  # type: ignore[override]
        raise NotImplementedError

    def upload_files(self, files: list[tuple[str, bytes]]) -> Never:  # type: ignore[override]
        raise NotImplementedError

    def download_files(self, paths: list[str]) -> Never:  # type: ignore[override]
        raise NotImplementedError


class MockSandboxProvider:
    """Mock provider implementation for testing.

    This demonstrates how to implement the SandboxProvider protocol
    with custom kwargs types.
    """

    def __init__(self) -> None:
        self.sandboxes = {
            "sb_001": {"status": "running", "template": "python-3.11"},
            "sb_002": {"status": "stopped", "template": "node-20"},
        }

    def list(
        self,
        *,
        cursor: str | None = None,
        **kwargs: Unpack[MockListKwargs],
    ) -> SandboxListResponse[MockMetadata]:
        """List sandboxes with optional filtering."""
        # Note: cursor is part of the protocol but not used in this simple implementation
        _ = cursor  # Mark as intentionally unused

        items: list[SandboxInfo[MockMetadata]] = []

        # Apply filters from kwargs
        status_filter = kwargs.get("status")
        template_filter = kwargs.get("template_id")

        for sandbox_id, metadata in self.sandboxes.items():
            if status_filter and metadata.get("status") != status_filter:
                continue
            if template_filter and metadata.get("template") != template_filter:
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
        **kwargs: Unpack[MockCreateKwargs],
    ) -> SandboxBackendProtocol:
        """Get existing or create new sandbox."""
        if sandbox_id is None:
            # Create new sandbox
            new_id = f"sb_{len(self.sandboxes) + 1:03d}"
            template = kwargs.get("template_id", "default")
            self.sandboxes[new_id] = {"status": "running", "template": template}
            return MockSandboxBackend(new_id)

        # Get existing sandbox
        if sandbox_id not in self.sandboxes:
            msg = f"Sandbox {sandbox_id} not found"
            raise ValueError(msg)

        return MockSandboxBackend(sandbox_id)

    def delete(
        self,
        sandbox_id: str,
        **kwargs: Unpack[MockDeleteKwargs],
    ) -> None:
        """Delete a sandbox."""
        # Note: kwargs is part of the protocol but not used in this simple implementation
        _ = kwargs  # Mark as intentionally unused

        if sandbox_id not in self.sandboxes:
            msg = f"Sandbox {sandbox_id} not found"
            raise ValueError(msg)

        del self.sandboxes[sandbox_id]


def test_sandbox_info_structure() -> None:
    """Test SandboxInfo TypedDict structure."""
    info: SandboxInfo = {
        "sandbox_id": "sb_123",
        "metadata": {"status": "running", "created_at": "2024-01-01T00:00:00Z"},
    }

    assert info["sandbox_id"] == "sb_123"
    assert info["metadata"]["status"] == "running"  # type: ignore[index]


def test_sandbox_list_response() -> None:
    """Test SandboxListResponse structure."""
    response: SandboxListResponse = {
        "items": [
            {"sandbox_id": "sb_001", "metadata": {"status": "running"}},
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
