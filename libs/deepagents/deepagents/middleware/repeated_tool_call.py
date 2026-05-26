"""Middleware that blocks repeated identical tool calls."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Protocol, TypeAlias, TypeGuard, cast

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

if TYPE_CHECKING:
    from langchain.tools.tool_node import ToolCallRequest
    from langchain_core.tools import BaseTool
    from langgraph.types import Command


JsonValue: TypeAlias = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]


class _ModelDump(Protocol):
    def model_dump(self, *, mode: str, exclude_none: bool) -> object:
        """Dump a Pydantic v2 model to JSON-compatible data."""


class _DictDump(Protocol):
    def dict(self) -> object:
        """Dump a Pydantic v1 model to Python data."""


class _ModelValidateSchema(Protocol):
    def model_validate(self, obj: object) -> _ModelDump:
        """Validate input with a Pydantic v2 schema."""


class _ParseObjSchema(Protocol):
    def parse_obj(self, obj: object) -> _DictDump:
        """Validate input with a Pydantic v1 schema."""


class RepeatedToolCallMiddleware(AgentMiddleware):
    """Block execution of consecutively repeated tool calls.

    This middleware compares tool calls by tool name plus schema-normalized
    arguments. It blocks only the repeated call; the agent can continue with
    different arguments, a different tool, or a final response.

    Args:
        max_repeats: Number of consecutive identical tool calls to allow before
            blocking later repeats.

    Raises:
        ValueError: If `max_repeats` is less than 1.
    """

    def __init__(self, *, max_repeats: int = 5) -> None:
        """Initialize the middleware with a repeat threshold."""
        super().__init__()
        if max_repeats < 1:
            msg = "max_repeats must be >= 1"
            raise ValueError(msg)
        self.max_repeats = max_repeats

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[object]],
    ) -> ToolMessage | Command[object]:
        """Block the current tool call if it repeats too many times."""
        if self._should_block(request):
            return self._blocked_tool_message(request)
        return handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[object]]],
    ) -> ToolMessage | Command[object]:
        """Async version of `wrap_tool_call`."""
        if self._should_block(request):
            return self._blocked_tool_message(request)
        return await handler(request)

    def _should_block(self, request: ToolCallRequest) -> bool:
        return _consecutive_repeat_count(request) > self.max_repeats

    def _blocked_tool_message(self, request: ToolCallRequest) -> ToolMessage:
        tool_name = request.tool_call["name"]
        content = (
            f"Repeated tool call blocked: `{tool_name}` was called more than "
            f"{self.max_repeats} consecutive times with the same normalized "
            "arguments. The repeated call was not executed. Continue the task "
            "using the existing context or another approach."
        )
        return ToolMessage(
            content=content,
            name=tool_name,
            tool_call_id=request.tool_call["id"],
            status="error",
        )


def _consecutive_repeat_count(request: ToolCallRequest) -> int:
    current = request.tool_call
    current_id = current.get("id")
    current_fingerprint = _tool_call_fingerprint(current, request.tool)
    calls = _tool_calls_since_last_human_message(request.state.get("messages", []))

    for index, call in enumerate(calls):
        if current_id is not None and call.get("id") == current_id:
            calls = calls[: index + 1]
            break
    else:
        calls.append(current)

    count = 0
    for call in reversed(calls):
        tool = request.tool if call.get("name") == current.get("name") else None
        if _tool_call_fingerprint(call, tool) != current_fingerprint:
            break
        count += 1
    return count


def _tool_calls_since_last_human_message(messages: Sequence[object]) -> list[Mapping[str, object]]:
    calls: list[Mapping[str, object]] = []
    for message in messages:
        if isinstance(message, HumanMessage):
            calls = []
            continue
        if isinstance(message, AIMessage):
            calls.extend(cast("Sequence[Mapping[str, object]]", message.tool_calls))
    return calls


def _tool_call_fingerprint(tool_call: Mapping[str, object], tool: BaseTool | None) -> str:
    name = str(tool_call.get("name", ""))
    raw_args = tool_call.get("args", {})
    args = _coerce_args(raw_args)
    normalized = _normalize_args(args, tool if _tool_name(tool) == name else None)
    return _stable_json({"name": name, "args": normalized})


def _coerce_args(raw_args: object) -> dict[str, object]:
    if isinstance(raw_args, Mapping):
        return {str(key): val for key, val in raw_args.items()}
    return {"__arg__": raw_args}


def _normalize_args(args: Mapping[str, object], tool: BaseTool | None) -> JsonValue:
    schema = getattr(tool, "args_schema", None)
    if schema is not None:
        try:
            if _has_model_validate(schema):
                return _to_jsonable(schema.model_validate(args).model_dump(mode="json", exclude_none=False))
            if _has_parse_obj(schema):
                return _to_jsonable(schema.parse_obj(args).dict())
        except (AttributeError, TypeError, ValueError):
            return _to_jsonable(args)
    return _to_jsonable(args)


def _has_model_validate(schema: object) -> TypeGuard[_ModelValidateSchema]:
    return hasattr(schema, "model_validate")


def _has_parse_obj(schema: object) -> TypeGuard[_ParseObjSchema]:
    return hasattr(schema, "parse_obj")


def _tool_name(tool: BaseTool | None) -> str | None:
    if tool is None:
        return None
    return getattr(tool, "name", None)


def _stable_json(value: object) -> str:
    return json.dumps(_to_jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _to_jsonable(value: object) -> JsonValue:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(val) for key, val in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_to_jsonable(item) for item in value]
    return repr(value)


__all__ = ["RepeatedToolCallMiddleware"]
