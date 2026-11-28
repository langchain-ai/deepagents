"""Host startup and runtime extension registrations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, NotRequired

from deepagents.backends.filesystem import FilesystemBackend

# Private SDK module: `deepagents-code` pins an exact `deepagents` version.
from deepagents.middleware._tool_disclosure import (  # noqa: PLC2701
    ToolDisclosure,
    disclose_tools,
    tool_name,
)
from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ExtendedModelResponse,
    PrivateStateAttr,
    TracePolicy,
    omit_payload,
)
from langgraph.types import Command

from deepagents_code.extensions.registry import ExtensionError

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Collection, Mapping, Sequence

    from langchain.agents.middleware.types import (
        ModelRequest,
        ModelResponse,
        ToolCallRequest,
    )
    from langchain_core.messages import AIMessage, AnyMessage, ToolMessage
    from langchain_core.tools import BaseTool

    from deepagents_code.extensions.registry import ExtensionRegistry, RegisteredUnit

_ANCHORS_KEY = "_extension_tool_anchors"
"""Private state key mapping each runtime extension tool to where it is disclosed."""

_Anchors = dict[str, str | None]
"""Tool name to the ID of the message it is disclosed after, or `None` if in `tools`."""


class ExtensionToolsState(AgentState):
    """Where each extension tool missing from the built graph reaches the model."""

    _extension_tool_anchors: Annotated[NotRequired[_Anchors], PrivateStateAttr]
    """Tools registered by the thread's first model call map to `None` and stay
    in `tools`; later ones map to the message they were first disclosed after,
    so their inline disclosure keeps its position and the prompt cache."""


def _stored_anchors(state: Mapping[str, object]) -> _Anchors | None:
    """Return the recorded anchors, or `None` before the thread's first model call.

    A malformed record reads as missing, so every live tool is bound in `tools`.
    """
    record = state.get(_ANCHORS_KEY)
    if not isinstance(record, dict):
        return None
    anchors: _Anchors = {}
    for name, anchor in record.items():
        if not isinstance(name, str) or not (anchor is None or isinstance(anchor, str)):
            return None
        anchors[name] = anchor
    return anchors


def _next_anchors(
    stored: _Anchors | None, names: Sequence[str], messages: Sequence[AnyMessage]
) -> _Anchors:
    """Anchor each of `names`, keeping a recorded anchor while its message is visible.

    A tool new to the record, or whose anchor message was compacted away, is
    anchored at the latest message. Unregistered tools drop out of the record,
    so one registered again is disclosed where it reappears.

    Returns:
        The anchors for `names`.
    """
    if stored is None:
        return dict.fromkeys(names)
    visible = {message.id for message in messages if message.id}
    tail = next((message.id for message in reversed(messages) if message.id), None)
    anchors: _Anchors = {}
    for name in names:
        kept = name in stored and (stored[name] is None or stored[name] in visible)
        anchors[name] = stored[name] if kept else tail
    return anchors


class ExtensionRuntimeMiddleware(AgentMiddleware):
    """Expose tools registered after the agent graph was built.

    Tools registered by a thread's first model call are bound in `tools`. Later
    ones are disclosed inline after the message they first appeared at, when the
    model supports it, so the request's tool list and cached prefix stay stable.
    """

    name = "__deepagents_extension_runtime__"

    state_schema = ExtensionToolsState

    trace_policy = TracePolicy(process_inputs=omit_payload)
    """Omit hook inputs from traces by default; set a `TracePolicy` to override."""

    def __init__(self, registry: ExtensionRegistry) -> None:
        """Bind dynamic model and tool dispatch to `registry`."""
        self._registry = registry

    def _prepare(
        self, request: ModelRequest[Any]
    ) -> tuple[ModelRequest[Any], _Anchors | None]:
        """Expose the live extension tools to `request`.

        Returns:
            The request to send, and the anchors to record if they changed.
        """
        live = self._registry.tool_units()
        bound = {tool_name(tool) for tool in request.tools}
        stored = _stored_anchors(request.state)
        unbound = [item.name for item in live if item.name not in bound]
        anchors = _next_anchors(stored, unbound, request.messages)
        request = _expose(request, live, anchors)
        return request, None if anchors == stored else anchors

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any] | AIMessage | ExtendedModelResponse[Any]:
        """Expose the latest extension-tool snapshot to a sync model call.

        Returns:
            The wrapped model response, recording any new tool anchors.
        """
        request, anchors = self._prepare(request)
        return _respond(handler(request), anchors)

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any] | AIMessage | ExtendedModelResponse[Any]:
        """Expose the latest extension-tool snapshot to an async model call.

        Returns:
            The wrapped model response, recording any new tool anchors.
        """
        request, anchors = self._prepare(request)
        return _respond(await handler(request), anchors)

    def _tool_request(self, request: ToolCallRequest) -> ToolCallRequest:
        registered = self._registry.find_tool(request.tool_call["name"])
        return request if registered is None else request.override(tool=registered.unit)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Execute a sync runtime tool through LangChain's normal handler.

        Returns:
            The wrapped tool result.
        """
        return handler(self._tool_request(request))

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Execute an async runtime tool through LangChain's normal handler.

        Returns:
            The wrapped tool result.
        """
        return await handler(self._tool_request(request))


def _expose(
    request: ModelRequest[Any],
    live: Sequence[RegisteredUnit[BaseTool]],
    anchors: _Anchors,
) -> ModelRequest[Any]:
    """Bind the graph's and initial extension tools, and disclose the later ones.

    A registered tool replaces a bound tool of the same name in place, so its
    current schema reaches the model.

    Returns:
        The request with the live extension tools exposed.
    """
    tools = list(request.tools)
    indexes = {tool_name(tool): index for index, tool in enumerate(tools)}
    positions = {message.id: i for i, message in enumerate(request.messages)}
    disclosure = ToolDisclosure()
    for item in live:
        anchor = anchors.get(item.name)
        if item.name in indexes:
            tools[indexes[item.name]] = item.unit
        elif anchor is None:
            tools.append(item.unit)
        else:
            disclosure.added[item.name] = item.unit
            disclosure.anchors[item.name] = positions[anchor]
    return disclose_tools(request.override(tools=tools), disclosure)


def _respond(
    response: ModelResponse[Any], anchors: _Anchors | None
) -> ModelResponse[Any] | ExtendedModelResponse[Any]:
    """Return `response`, recording `anchors` in state when they changed."""
    if anchors is None:
        return response
    return ExtendedModelResponse(
        model_response=response, command=Command(update={_ANCHORS_KEY: anchors})
    )


def validate_backend_route(
    item: RegisteredUnit[Any],
    protected_routes: Collection[str],
    *,
    sandbox_active: bool,
) -> None:
    """Reject a backend route that violates host storage boundaries.

    The sandbox check is deliberately shallow. Wrapping a filesystem backend
    in another `BackendProtocol` implementation changes who owns that safety
    contract, so dcode does not inspect arbitrary backend object graphs.

    Args:
        item: Backend route registration to validate.
        protected_routes: Internal route prefixes unavailable to extensions.
        sandbox_active: Whether the default execution backend is sandboxed.

    Raises:
        ExtensionError: If the route overlaps internal storage or directly
            exposes a host filesystem backend to a sandboxed agent.
    """
    if any(
        item.name.startswith(prefix) or prefix.startswith(item.name)
        for prefix in protected_routes
    ):
        msg = (
            f"Extension backend route {item.name!r} from {item.source.label} "
            "overlaps an internal route"
        )
        raise ExtensionError(msg)
    if sandbox_active and isinstance(item.unit, FilesystemBackend):
        msg = (
            f"Extension backend route {item.name!r} from {item.source.label} "
            f"cannot mount {type(item.unit).__name__} in sandbox mode"
        )
        raise ExtensionError(msg)


def bind_runtime_host_policy(
    registry: ExtensionRegistry,
    protected_routes: Collection[str],
    *,
    sandbox_active: bool = False,
) -> None:
    """Validate late routes and flag graph-bound registrations for restart."""

    def apply(kind: str, item: RegisteredUnit[Any]) -> None:
        if kind == "middleware":
            registry.require_restart()
            return
        if kind == "backend_route":
            validate_backend_route(
                item, protected_routes, sandbox_active=sandbox_active
            )
            registry.require_restart()

    registry.subscribe_to_registrations(apply)
