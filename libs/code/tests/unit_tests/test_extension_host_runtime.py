"""Tests for registrations made after extension initialization."""

from pathlib import Path
from typing import Any, cast

import pytest
from deepagents.backends import (
    FilesystemBackend,
    LocalShellBackend,
    StateBackend,
)
from langchain.agents.middleware.types import (
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
)
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from deepagents_code.extensions.hosting import (
    ExtensionRuntimeMiddleware,
    bind_runtime_host_policy,
    validate_backend_route,
)
from deepagents_code.extensions.registry import (
    ExtensionError,
    ExtensionRegistry,
    RegisteredUnit,
    SourceInfo,
)


@tool("late")
def _late_tool() -> str:
    """Return a runtime-registration marker."""
    return "ready"


def test_backend_route_policy_rejects_sandbox_filesystems(tmp_path: Path) -> None:
    """A sandbox cannot directly mount host filesystem backend subclasses."""
    source = SourceInfo(tmp_path / "extension.py")
    backends = (
        FilesystemBackend(root_dir=tmp_path, virtual_mode=True),
        LocalShellBackend(root_dir=tmp_path),
    )

    for backend in backends:
        route = RegisteredUnit("/workspace/", backend, source)
        with pytest.raises(ExtensionError, match="cannot mount"):
            validate_backend_route(route, set(), sandbox_active=True)


def test_backend_route_policy_rejects_reserved_overlap(tmp_path: Path) -> None:
    """Extensions cannot claim a parent of internal storage."""
    route = RegisteredUnit(
        "/artifacts/", StateBackend(), SourceInfo(tmp_path / "extension.py")
    )

    with pytest.raises(ExtensionError, match="overlaps an internal route"):
        validate_backend_route(
            route,
            {"/artifacts/history/"},
            sandbox_active=False,
        )
