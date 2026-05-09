"""Formatting and output-coercion helpers for the QuickJS REPL."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from langchain_core.messages import ToolMessage
from langgraph.types import Command
from pydantic import BaseModel
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


# Scalar types the quickjs_rs binding marshals natively. Compound shapes
# (``dict`` / ``list`` / ``tuple``) are walked recursively in
# ``_coerce_for_marshal``; anything else becomes ``str(value)`` so the JS
# side can still see a usable value.
_NATIVE_JS_SCALARS = (str, bool, int, float, type(None))


def coerce_tool_output_for_ptc(value: Any) -> Any:
    """Coerce a tool result for the PTC bridge, preserving native types.

    The quickjs_rs ``register`` bridge marshals Python primitives, ``list``,
    and ``dict`` directly to native JS values, so the model can use them
    without an explicit ``JSON.parse``. This helper unwraps LangChain's
    ``ToolMessage`` / ``Command`` envelopes (matching ``coerce_tool_output``'s
    selection rules) and returns the underlying value typed.

    Compound returns are walked recursively: nested values that the binding
    cannot marshal natively (``datetime``, Pydantic models, custom classes)
    are stringified in place via ``str(value)`` so the surrounding object
    structure remains navigable from JS. Cyclic structures hit Python's
    recursion limit and surface as a host error in the eval — same outcome
    as ``json.dumps`` on a self-referencing dict.
    """
    if isinstance(value, Command):
        return coerce_tool_output_for_ptc(_extract_command_content(value))
    if isinstance(value, ToolMessage):
        return coerce_tool_output_for_ptc(value.content)
    if isinstance(value, list):
        for entry in reversed(value):
            if isinstance(entry, ToolMessage):
                return coerce_tool_output_for_ptc(entry.content)
            if isinstance(entry, Command):
                return coerce_tool_output_for_ptc(_extract_command_content(entry))
    return _coerce_for_marshal(value)


def _coerce_for_marshal(value: Any) -> Any:
    """Convert *value* into a shape the quickjs_rs bridge can marshal."""
    if isinstance(value, _NATIVE_JS_SCALARS):
        return value
    if isinstance(value, dict):
        return {str(k): _coerce_for_marshal(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_coerce_for_marshal(v) for v in value]
    # Pydantic models are dumped to a plain dict so the JS side sees the
    # field shape its return-type signature advertises (rather than
    # ``str(model)``). Nested models / datetimes are handled by recursion.
    if isinstance(value, BaseModel):
        return _coerce_for_marshal(value.model_dump())
    return str(value)


def _extract_command_content(command: Command) -> Any:
    """Return the trailing message content from a ``Command`` update, if any."""
    update = command.update
    if isinstance(update, dict):
        messages = update.get("messages")
        if isinstance(messages, list):
            for entry in reversed(messages):
                content = getattr(entry, "content", None)
                if content is not None:
                    return content
    return str(update)


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
    if outcome.stdout or outcome.stdout_truncated_chars > 0:
        stdout = outcome.stdout
        if outcome.stdout_truncated_chars > 0:
            stdout = _truncate(
                outcome.stdout,
                max_result_chars,
                dropped=outcome.stdout_truncated_chars,
            )
        parts.append(f"<stdout>\n{stdout}\n</stdout>")
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


def _truncate(text: str, limit: int, *, dropped: int | None = None) -> str:
    # stdout path -- we've already truncated the result, so we just add the marker
    if dropped is not None:
        marker = _TRUNCATE_MARKER.format(n=dropped)
        if len(marker) >= limit:
            return marker[: max(0, limit)]
        keep = max(0, limit - len(marker))
        return text[:keep] + marker
    # result/error path -- these values need to be truncated
    if len(text) <= limit:
        return text
    keep = max(0, limit - len(_TRUNCATE_MARKER.format(n=0)))
    dropped = len(text) - keep
    return text[:keep] + _TRUNCATE_MARKER.format(n=dropped)


def _xml_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
