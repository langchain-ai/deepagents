"""Middleware for renaming tools at the model boundary.

Translates tool names at the model boundary so the model sees its
training vocabulary while the rest of the Deep Agents stack (routing,
permissions, HITL, logging, tests) stays on canonical names.

## Outbound (canonical -> alias)

Rewrites, in order:

1. `request.tools` — the current tool catalog (`execute` -> `shell_command`).
2. `AIMessage.tool_calls` and `AIMessage.invalid_tool_calls` in
    `request.messages` — past tool calls in conversation history.
3. `ToolMessage.name` in `request.messages` — tool result messages from
    prior turns.

Steps 2 and 3 are what give the model a consistent vocabulary across
turns. If only the tool catalog were renamed and history left on
canonical names, the model would see an unfamiliar name for the tool it
supposedly called last turn — the very distribution mismatch this
middleware exists to avoid.

## Inbound (alias -> canonical)

Rewrites tool-call names on the response (`AIMessage` directly,
`ModelResponse.result`, or `ExtendedModelResponse.model_response.result`)
so downstream tool routing and state persistence only ever observe
canonical names.

## Positioning

This middleware MUST be the innermost name-aware entry in the stack —
placed after every other name-aware middleware (tool exclusion, HITL,
permissions). Any entry before aliasing sees canonical names, which
matches the names users specify in tool exclusion, HITL `interrupt_on`
rules, and permission configs. Any entry after aliasing must be
name-agnostic (e.g. `AnthropicPromptCachingMiddleware` adds only
positional cache markers and is unaffected by rename order).
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from deepagents.middleware._tool_exclusion import _tool_name

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping

    from langchain.agents.middleware.types import (
        ExtendedModelResponse,
        ModelRequest,
        ModelResponse,
        ResponseT,
    )
    from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)


def _validate_tool_aliases(aliases: Mapping[str, str]) -> None:
    """Validate that an alias map is safe to round-trip.

    Two invariants are enforced:

    1. **Injective values.** Two canonical names mapping to the same alias
        would make the reverse map drop an entry, so a round-tripped tool
        call could resolve to the wrong canonical tool.
    2. **Disjoint keys and values.** A canonical name that also appears as
        an alias for a different tool would make the model's catalog
        contain a name that is simultaneously a canonical identifier
        elsewhere in the system, rendering round-tripping ambiguous.

    Args:
        aliases: Mapping from canonical tool name to model-facing alias.

    Raises:
        TypeError: If `aliases` is not a mapping or contains non-string
            keys or values.
        ValueError: If either invariant is violated. Empty maps pass.
    """
    if not aliases:
        return

    for key, value in aliases.items():
        if not isinstance(key, str) or not isinstance(value, str):
            msg = f"`tool_aliases` keys and values must be strings, got {type(key).__name__} -> {type(value).__name__}"
            raise TypeError(msg)
        if not key or not value:
            msg = f"`tool_aliases` keys and values must be non-empty strings, got {key!r} -> {value!r}"
            raise ValueError(msg)

    value_counts = Counter(aliases.values())
    duplicates = sorted(v for v, count in value_counts.items() if count > 1)
    if duplicates:
        msg = (
            "`tool_aliases` values must be unique so the reverse map is "
            f"well-defined. Duplicate alias targets: {duplicates}. "
            f"Aliases: {dict(aliases)}"
        )
        raise ValueError(msg)

    collisions = sorted(set(aliases.keys()) & set(aliases.values()))
    if collisions:
        msg = (
            "`tool_aliases` keys and values must be disjoint — a name cannot be "
            "both a canonical tool and an alias for a different tool. "
            f"Collisions: {collisions}. Aliases: {dict(aliases)}"
        )
        raise ValueError(msg)


def _rename_tool(
    tool: BaseTool | dict[str, Any] | object,
    new_name: str,
) -> BaseTool | dict[str, Any] | object:
    """Return a copy of `tool` with its name replaced.

    `BaseTool` instances are copied via `model_copy`; dict tools get a
    shallow `dict.copy()`; other shapes (plain callables, unusual custom
    tool classes without `model_copy`) cannot be renamed and are returned
    unchanged with a warning — the model will see the tool under its
    original name, which defeats the point of aliasing for that entry.
    """
    if isinstance(tool, dict):
        copied = cast("dict[str, Any]", tool).copy()
        copied["name"] = new_name
        return copied
    model_copy = getattr(tool, "model_copy", None)
    if model_copy is not None:
        return model_copy(update={"name": new_name})
    logger.warning(
        "Tool %r matched alias target %r but cannot be renamed (no "
        "`model_copy` support and not a dict). Model will see the original "
        "name; this may degrade performance on models trained on the alias.",
        getattr(tool, "name", tool),
        new_name,
    )
    return tool


def _rewrite_ai_message_tool_calls(
    message: AIMessage,
    aliases: Mapping[str, str],
) -> AIMessage:
    """Rewrite tool-call names in `message` per `aliases`, returning a copy.

    Returns `message` unchanged (same object identity) when no tool calls
    match the alias map, so callers can use `is` to detect no-ops.
    """
    if not message.tool_calls and not message.invalid_tool_calls:
        return message

    changed = False
    rewritten: list[dict[str, Any]] = []
    for tc in message.tool_calls:
        if tc["name"] in aliases:
            rewritten.append({**tc, "name": aliases[tc["name"]]})
            changed = True
        else:
            rewritten.append(cast("dict[str, Any]", tc))

    rewritten_invalid: list[dict[str, Any]] = []
    for tc in message.invalid_tool_calls:
        if tc["name"] in aliases:
            rewritten_invalid.append({**tc, "name": aliases[tc["name"]]})
            changed = True
        else:
            rewritten_invalid.append(cast("dict[str, Any]", tc))

    if not changed:
        return message

    return message.model_copy(
        update={"tool_calls": rewritten, "invalid_tool_calls": rewritten_invalid},
    )


def _rewrite_tool_message_name(
    message: ToolMessage,
    aliases: Mapping[str, str],
) -> ToolMessage:
    """Rewrite `ToolMessage.name` per `aliases`, returning a copy.

    Returns `message` unchanged (same object identity) when the name is
    absent or not in the alias map.
    """
    name = message.name
    if name is None or name not in aliases:
        return message
    return message.model_copy(update={"name": aliases[name]})


def _rewrite_message_list(
    messages: list[BaseMessage],
    aliases: Mapping[str, str],
) -> tuple[list[BaseMessage], bool]:
    """Rewrite tool-related names across a list of messages.

    Rewrites both `AIMessage.tool_calls` (and `invalid_tool_calls`) and
    `ToolMessage.name` via `aliases`. Other message types pass through
    unchanged.

    Returns the (possibly new) list and whether any change was made. When
    nothing changed the returned list still contains the original message
    objects, so callers can skip `request.override` entirely on the
    `changed=False` path.
    """
    changed = False
    result: list[BaseMessage] = []
    for msg in messages:
        if isinstance(msg, AIMessage):
            rewritten: BaseMessage = _rewrite_ai_message_tool_calls(msg, aliases)
        elif isinstance(msg, ToolMessage):
            rewritten = _rewrite_tool_message_name(msg, aliases)
        else:
            result.append(msg)
            continue
        if rewritten is not msg:
            changed = True
        result.append(rewritten)
    return result, changed


def _rewrite_response_tool_names(
    response: object,
    aliases: Mapping[str, str],
) -> object:
    """Rewrite tool-call names in a model response using `aliases`.

    Handles:

    - `AIMessage` directly (bare message response).
    - `ModelResponse` (has `.result: list[BaseMessage]`).
    - `ExtendedModelResponse` (wraps a `ModelResponse` in `.model_response`).

    Other response shapes are returned unchanged.
    """
    if isinstance(response, AIMessage):
        return _rewrite_ai_message_tool_calls(response, aliases)

    result_msgs = getattr(response, "result", None)
    if isinstance(result_msgs, list):
        new_msgs, changed = _rewrite_message_list(result_msgs, aliases)
        if changed:
            return replace(cast("Any", response), result=new_msgs)
        return response

    inner = getattr(response, "model_response", None)
    if inner is not None:
        inner_msgs = getattr(inner, "result", None)
        if isinstance(inner_msgs, list):
            new_msgs, changed = _rewrite_message_list(inner_msgs, aliases)
            if changed:
                return replace(
                    cast("Any", response),
                    model_response=replace(cast("Any", inner), result=new_msgs),
                )
        return response

    return response


class _ToolAliasingMiddleware(AgentMiddleware[Any, Any, Any]):
    """Rename tools at the model boundary for better model-distribution fit.

    Must be positioned as the innermost name-aware middleware in the
    stack so that tool exclusion, HITL rules, and permission configs
    keyed on canonical tool names continue to match. The translation is
    symmetric: outbound, both the tool catalog and any past tool calls /
    tool messages in `request.messages` are rewritten canonical -> alias;
    inbound, tool-call names on the response are reverted alias ->
    canonical.

    Args:
        aliases: Mapping of canonical tool name to model-facing alias
            (e.g. `{"execute": "shell_command"}`).

    Raises:
        TypeError: If `aliases` contains non-string keys or values.
        ValueError: If `aliases` has duplicate values or keys/values
            overlap. See `_validate_tool_aliases`.
    """

    def __init__(self, *, aliases: Mapping[str, str]) -> None:
        _validate_tool_aliases(aliases)
        self._forward: dict[str, str] = dict(aliases)
        self._reverse: dict[str, str] = {v: k for k, v in aliases.items()}

    def _apply_outbound(self, request: ModelRequest[Any]) -> ModelRequest[Any]:
        """Translate tool catalog and message history canonical -> alias.

        Returns the original request unchanged (same identity) when the
        alias map is empty or no tool/message actually needed renaming,
        so downstream cache keys keyed on request identity remain stable
        when aliasing is a no-op.
        """
        if not self._forward:
            return request

        renamed_tools: list[Any] = []
        tools_changed = False
        for tool in request.tools:
            name = _tool_name(tool)
            if name is not None and name in self._forward:
                renamed_tools.append(_rename_tool(tool, self._forward[name]))
                tools_changed = True
            else:
                renamed_tools.append(tool)

        new_messages, messages_changed = _rewrite_message_list(list(request.messages), self._forward)

        if not tools_changed and not messages_changed:
            return request

        overrides: dict[str, Any] = {}
        if tools_changed:
            overrides["tools"] = renamed_tools
        if messages_changed:
            overrides["messages"] = new_messages
        return request.override(**overrides)

    def _apply_inbound(
        self,
        response: ModelResponse[Any] | AIMessage | ExtendedModelResponse[Any],
    ) -> ModelResponse[Any] | AIMessage | ExtendedModelResponse[Any]:
        """Translate response tool-call names alias -> canonical."""
        if not self._reverse:
            return response
        return cast(
            "ModelResponse[Any] | AIMessage | ExtendedModelResponse[Any]",
            _rewrite_response_tool_names(response, self._reverse),
        )

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        """Rename outbound request, revert inbound response tool-call names."""
        response = handler(self._apply_outbound(request))
        return cast("ModelResponse[Any]", self._apply_inbound(response))

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT] | AIMessage | ExtendedModelResponse[ResponseT]:
        """Async variant of `wrap_model_call`."""
        response = await handler(self._apply_outbound(request))
        return cast(
            "ModelResponse[ResponseT] | AIMessage | ExtendedModelResponse[ResponseT]",
            self._apply_inbound(response),
        )
