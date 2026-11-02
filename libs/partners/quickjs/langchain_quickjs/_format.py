"""Formatting and output-coercion helpers for the QuickJS REPL."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Literal

from langchain_core.messages import ToolMessage
from langgraph.types import Command
from pydantic import BaseModel
from pydantic_core import PydanticSerializationError, to_jsonable_python
from quickjs_rs import UNDEFINED

if TYPE_CHECKING:
    from langchain_quickjs._repl import EvalOutcome

_TRUNCATE_MARKER = "… [truncated {n} chars]"


def format_handle(handle: Any) -> str:
    """Describe a `Handle` value in REPL-style shorthand."""
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
    """Like `_format_jsvalue` but quotes nested strings."""
    if isinstance(value, str):
        return f'"{value}"'
    return _format_jsvalue(value)


# Scalar types the quickjs_rs binding marshals natively. `bytes` and
# `bytearray` become `Uint8Array`. Compound shapes are walked recursively in
# `_coerce_for_marshal`.
_NATIVE_JS_SCALARS = (str, bool, int, float, type(None), bytes, bytearray)


def coerce_tool_output_for_ptc(
    value: Any,
    *,
    response_format: Literal["content", "content_and_artifact"] = "content",
) -> Any:
    """Return the data value JavaScript gets for a tool's return value.

    A tool's artifact is its data when set: the second element of a
    `content_and_artifact` pair, or `ToolMessage.artifact` on a message the
    tool returns alone, at the tail of a list, or in a `Command` update.
    Without an artifact, the content is the data. An error message always
    gives its content.

    The result is JSON-shaped so the JS side sees the type its signature
    advertises: Pydantic models and dataclasses become objects, and values
    such as `datetime` take their Pydantic JSON form. Cyclic structures hit
    Python's recursion limit and surface as a host error in the eval.
    """
    if (
        response_format == "content_and_artifact"
        and isinstance(value, tuple)
        and len(value) == 2  # noqa: PLR2004 — a (content, artifact) pair
    ):
        content, artifact = value
        if artifact is not None:
            return _coerce_for_marshal(artifact)
        value = content
    if isinstance(value, list):
        envelopes = (
            e for e in reversed(value) if isinstance(e, (ToolMessage, Command))
        )
        value = next(envelopes, value)
    if isinstance(value, Command):
        value = _command_result(value)
    if isinstance(value, ToolMessage):
        if value.status != "error" and value.artifact is not None:
            return _coerce_for_marshal(value.artifact)
        return _coerce_for_marshal(value.content)
    return _coerce_for_marshal(value)


def _coerce_for_marshal(value: Any) -> Any:
    """Convert *value* into a shape the quickjs_rs bridge can marshal."""
    if isinstance(value, _NATIVE_JS_SCALARS):
        return value
    if isinstance(value, dict):
        return {_coerce_key(k): _coerce_for_marshal(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_coerce_for_marshal(v) for v in value]
    if isinstance(value, BaseModel):
        # By alias, so the keys match the JSON Schema the signature renders.
        return _coerce_for_marshal(value.model_dump(by_alias=True))
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        fields = dataclasses.fields(value)
        return {f.name: _coerce_for_marshal(getattr(value, f.name)) for f in fields}
    if isinstance(value, memoryview):
        return value.tobytes()
    try:
        jsonable = to_jsonable_python(value, fallback=str)
    except PydanticSerializationError:
        return str(value)
    return _coerce_for_marshal(jsonable)


def _coerce_key(key: Any) -> str:
    if isinstance(key, str):
        return key
    return str(_coerce_for_marshal(key))


def _command_result(command: Command) -> Any:
    """Return the trailing message in a `Command` update, or the update's text.

    A `ToolMessage` is returned whole so its artifact stays reachable; any
    other message gives its content.
    """
    update = command.update
    if isinstance(update, dict):
        messages = update.get("messages")
        if isinstance(messages, list):
            for entry in reversed(messages):
                content = getattr(entry, "content", None)
                if content is not None:
                    return entry if isinstance(entry, ToolMessage) else content
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
    notes = getattr(outcome, "notes", None)
    if notes:
        joined = _xml_escape(_truncate("\n\n".join(notes), max_result_chars))
        parts.append(f"<notes>\n{joined}\n</notes>")
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
