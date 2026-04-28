"""Formatting and output-coercion helpers for the QuickJS REPL."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from langchain_core.messages import ToolMessage
from langgraph.types import Command
from quickjs_rs import UNDEFINED

if TYPE_CHECKING:
    from langchain_quickjs._repl import EvalOutcome

_TRUNCATE_MARKER = "… [truncated {n} chars]"


def format_handle(handle: Any) -> str:
    """Describe a ``Handle`` value in REPL-style shorthand."""
    kind = handle.type_of
    if kind == "function":
        try:
            arity_h = handle.get("length")
            try:
                arity = arity_h.to_python()
            finally:
                arity_h.dispose()
        except Exception:  # noqa: BLE001 — best-effort
            return "[Function]"
        return f"[Function] arity={arity}"
    return f"[{kind}]"


def stringify(value: Any) -> str:
    """Best-effort string form for a console arg or eval result."""
    return _format_jsvalue(value)


def _format_jsvalue(value: Any) -> str:
    if value is None:
        return "null"
    if value is UNDEFINED:
        return "undefined"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return str(value)
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "[" + ", ".join(_format_nested(v) for v in value) + "]"
    if isinstance(value, dict):
        return (
            "{" + ", ".join(f"{k}: {_format_nested(v)}" for k, v in value.items()) + "}"
        )
    return repr(value)


def _format_nested(value: Any) -> str:
    """Like ``_format_jsvalue`` but quotes nested strings."""
    if isinstance(value, str):
        return f'"{value}"'
    return _format_jsvalue(value)


def coerce_tool_output(value: Any) -> str:
    """Coerce arbitrary tool return values to the JS-visible string output."""
    if isinstance(value, str):
        return value
    if isinstance(value, Command):
        return _coerce_command_output(value)
    if isinstance(value, ToolMessage):
        return _coerce_tool_message_output(value)
    if isinstance(value, list):
        for entry in reversed(value):
            if isinstance(entry, ToolMessage):
                return _coerce_tool_message_output(entry)
            if isinstance(entry, Command):
                return _coerce_command_output(entry)
    return _coerce_message_content(value)


def _coerce_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, default=str)
    except (TypeError, ValueError):
        return str(content)


def _coerce_tool_message_output(message: ToolMessage) -> str:
    return _coerce_message_content(message.content)


def _coerce_command_output(command: Command) -> str:
    update = command.update
    if isinstance(update, dict):
        messages = update.get("messages")
        if isinstance(messages, list):
            for entry in reversed(messages):
                content = getattr(entry, "content", None)
                if content is not None:
                    return _coerce_message_content(content)
    return str(update)


def format_outcome(
    outcome: EvalOutcome,
    *,
    max_result_chars: int,
) -> str:
    """Render an EvalOutcome-like object as the tool's wire format."""
    parts: list[str] = []
    if outcome.stdout:
        parts.append(
            f"<stdout>\n{_truncate(outcome.stdout, max_result_chars)}\n</stdout>"
        )
    if outcome.error_type is not None:
        inner = outcome.error_message
        if outcome.error_stack:
            inner = f"{inner}\n{outcome.error_stack}"
        parts.append(
            f'<error type="{_xml_escape(outcome.error_type)}">'
            f"{_xml_escape(_truncate(inner, max_result_chars))}"
            f"</error>"
        )
    else:
        body = outcome.result if outcome.result is not None else "undefined"
        kind_attr = f' kind="{outcome.result_kind}"' if outcome.result_kind else ""
        body_xml = _xml_escape(_truncate(body, max_result_chars))
        parts.append(f"<result{kind_attr}>{body_xml}</result>")
    return "\n".join(parts)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    keep = max(0, limit - len(_TRUNCATE_MARKER.format(n=0)))
    dropped = len(text) - keep
    return text[:keep] + _TRUNCATE_MARKER.format(n=dropped)


def _xml_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
