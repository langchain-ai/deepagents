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


def test_runtime_tool_is_advertised_and_executable(tmp_path: Path) -> None:
    """A late tool is visible on the next request and selected for execution."""
    registry = ExtensionRegistry()
    middleware = ExtensionRuntimeMiddleware(registry)
    registry.add_tool(_late_tool, SourceInfo(tmp_path / "extension.py"))
    model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))
    request = ModelRequest(model=model, messages=[], tools=[])
    observed: list[ModelRequest[Any]] = []

    def handle(value: ModelRequest[Any]) -> ModelResponse[Any]:
        observed.append(value)
        return ModelResponse(result=[AIMessage(content="ok")])

    middleware.wrap_model_call(request, handle)

    assert [getattr(item, "name", None) for item in observed[0].tools] == ["late"]
    tool_request = ToolCallRequest(
        tool_call={"name": "late", "args": {}, "id": "call"},
        tool=None,
        state={},
        runtime=cast("Any", object()),
    )
    selected: list[object] = []
    middleware.wrap_tool_call(
        tool_request,
        lambda value: (
            selected.append(value.tool)
            or ToolMessage(content="ready", name="late", tool_call_id="call")
        ),
    )
    assert selected == [_late_tool]


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


def test_invalid_runtime_route_rolls_back_registration(tmp_path: Path) -> None:
    """Runtime policy rolls back invalid routes and flags valid ones."""
    registry = ExtensionRegistry()
    bind_runtime_host_policy(registry, {"/artifacts/"})
    source = SourceInfo(tmp_path / "extension.py")

    with pytest.raises(ExtensionError, match="overlaps an internal route"):
        registry.add_backend_route(
            "/artifacts/private/",
            StateBackend(),
            source,
        )

    assert not registry.backend_routes
    assert not registry.restart_required
    registry.add_backend_route("/memories/", StateBackend(), source)
    assert registry.restart_required
