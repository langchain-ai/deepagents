"""Tests for registrations made after extension initialization."""

from typing import TYPE_CHECKING, Any

import pytest
from deepagents.backends import (
    FilesystemBackend,
    LocalShellBackend,
    StateBackend,
)
from langchain.agents.middleware.types import (
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
)
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool, tool

from deepagents_code.extensions.hosting import (
    ExtensionRuntimeMiddleware,
    ExtensionToolsState,
    validate_backend_route,
)
from deepagents_code.extensions.registry import (
    ExtensionError,
    ExtensionRegistry,
    RegisteredUnit,
    SourceInfo,
)

if TYPE_CHECKING:
    from pathlib import Path

    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import AnyMessage

_ANCHORS = "_extension_tool_anchors"


@tool("late")
def _late_tool() -> str:
    """Return a runtime-registration marker."""
    return "ready"


@tool("initial")
def _initial_tool() -> str:
    """Return a startup-registration marker."""
    return "initial"


@tool("existing")
def _existing() -> str:
    """Represent a tool bound when the graph was built."""
    return "existing"


def _call(
    middleware: ExtensionRuntimeMiddleware,
    model: BaseChatModel,
    messages: list[AnyMessage],
    anchors: dict[str, str | None] | None = None,
) -> tuple[ModelRequest[Any], dict[str, str | None] | None]:
    """Run one model call and return the request sent and any anchors recorded."""
    seen: list[ModelRequest[Any]] = []

    def handler(request: ModelRequest[Any]) -> ModelResponse[Any]:
        seen.append(request)
        return ModelResponse(result=[AIMessage("ok")])

    state = ExtensionToolsState(messages=messages)
    if anchors is not None:
        state["_extension_tool_anchors"] = anchors
    request = ModelRequest(
        model=model,
        messages=messages,
        tools=[_existing],
        state=state,
    )
    result = middleware.wrap_model_call(request, handler)
    command = result.command if isinstance(result, ExtendedModelResponse) else None
    update = command.update if command is not None else None
    return seen[0], update[_ANCHORS] if isinstance(update, dict) else None


def _names(tools: list[BaseTool | dict[str, Any]]) -> list[str]:
    return [t.name if isinstance(t, BaseTool) else t["name"] for t in tools]


def _anthropic() -> ChatAnthropic:
    model = ChatAnthropic(model="claude-sonnet-4-5", api_key="test")
    model.profile = {"mid_conversation_tools": True}
    return model


def test_tool_registered_mid_thread_is_disclosed_inline_at_a_stable_position(
    tmp_path: Path,
) -> None:
    """A late tool stays out of `tools` and keeps its disclosure's position."""
    registry = ExtensionRegistry()
    source = SourceInfo(tmp_path / "extension.py")
    registry.add_tool(_initial_tool, source)
    middleware = ExtensionRuntimeMiddleware(registry)
    model = _anthropic()

    first: list[AnyMessage] = [HumanMessage("hi", id="h1")]
    request, anchors = _call(middleware, model, first)
    assert _names(request.tools) == ["existing", "initial"]
    assert anchors == {"initial": None}

    registry.add_tool(_late_tool, source)
    second = [*first, AIMessage("hello", id="a1"), HumanMessage("go", id="h2")]
    request, anchors = _call(middleware, model, second, anchors)
    assert _names(request.tools) == ["existing", "initial"]
    assert request.messages[:3] == second
    disclosure = request.messages[3]
    assert isinstance(disclosure, SystemMessage)
    block = disclosure.content[0]
    assert isinstance(block, dict)
    assert block["type"] == "tool_addition"
    assert block["tool"]["definition"]["name"] == "late"
    assert anchors == {"initial": None, "late": "h2"}

    call = {"name": "late", "args": {}, "id": "c1", "type": "tool_call"}
    third = [
        *second,
        AIMessage("", tool_calls=[call], id="a2"),
        ToolMessage("ready", tool_call_id="c1", id="t1"),
    ]
    request, update = _call(middleware, model, third, anchors)
    assert _names(request.tools) == ["existing", "initial"]
    assert request.messages == [*second, disclosure, *third[3:]]
    assert update is None


def test_tool_registered_mid_thread_is_bound_without_inline_support(
    tmp_path: Path,
) -> None:
    """A model that can't take tools mid-conversation gets them in `tools`."""
    registry = ExtensionRegistry()
    middleware = ExtensionRuntimeMiddleware(registry)
    model = GenericFakeChatModel(messages=iter([]))
    first: list[AnyMessage] = [HumanMessage("hi", id="h1")]
    _, anchors = _call(middleware, model, first)

    registry.add_tool(_late_tool, SourceInfo(tmp_path / "extension.py"))
    request, anchors = _call(middleware, model, first, anchors)
    assert _names(request.tools) == ["existing", "late"]
    assert request.messages == first
    assert anchors == {"late": "h1"}


def test_lost_anchor_moves_to_the_latest_message_and_removed_tools_drop(
    tmp_path: Path,
) -> None:
    """A compacted anchor moves to the tail; unregistered tools leave the record."""
    registry = ExtensionRegistry()
    registry.add_tool(_late_tool, SourceInfo(tmp_path / "extension.py"))
    middleware = ExtensionRuntimeMiddleware(registry)
    messages: list[AnyMessage] = [HumanMessage("summary", id="s1")]

    _, anchors = _call(
        middleware, _anthropic(), messages, {"late": "gone", "removed": None}
    )
    assert anchors == {"late": "s1"}


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
