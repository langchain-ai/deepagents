"""Gemini-specific tool-schema repair and error translation.

The `langchain-google-genai` integration produces schemas that Gemini rejects
with `400 INVALID_ARGUMENT` when an `array`-typed property lacks an `items`
sub-schema. Because the 400 propagates cleanly through every middleware and
resolves to an empty AIMessage, the user sees a silent empty response.

This module provides two helpers used by `ConfigurableModelMiddleware` when
the resolved provider is `google_genai`:

* `repair_tools_for_gemini` — walks each tool's parameter schema and injects
  a safe `{"type": "string"}` `items` default on any array property missing
  one. BaseTool instances are converted to OpenAI-format dicts so the
  repaired schema survives the round-trip through `bind_tools`.
* `translate_gemini_tool_schema_error` — converts a
  `400 INVALID_ARGUMENT` referencing `GenerateContentRequest.tools[...]
  .function_declarations` into a user-facing message that names the
  offending function declarations, replacing the silent empty AIMessage.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)


_DEFAULT_ARRAY_ITEMS: dict[str, str] = {"type": "string"}
"""Safe default injected for `type: array` properties missing an `items` schema."""

_SCHEMA_COMPOSITION_KEYS: frozenset[str] = frozenset({"anyOf", "oneOf", "allOf"})
_PROPERTIES_MAP_KEYS: frozenset[str] = frozenset({"properties", "$defs"})


def _repair_schema(schema: Any) -> Any:  # noqa: ANN401  # schema is arbitrary JSON
    """Return a copy of `schema` with `items` injected on any array missing it."""
    if isinstance(schema, list):
        return [_repair_schema(item) for item in schema]
    if not isinstance(schema, dict):
        return schema

    repaired: dict[str, Any] = {}
    for raw_key, value in schema.items():
        key = str(raw_key)
        if key in _PROPERTIES_MAP_KEYS and isinstance(value, dict):
            repaired[key] = {
                prop: _repair_schema(prop_schema) for prop, prop_schema in value.items()
            }
        elif key in _SCHEMA_COMPOSITION_KEYS and isinstance(value, list):
            repaired[key] = [_repair_schema(sub) for sub in value]
        elif key == "items" or isinstance(value, dict):
            repaired[key] = _repair_schema(value)
        elif isinstance(value, list):
            repaired[key] = [
                _repair_schema(item) if isinstance(item, (dict, list)) else item
                for item in value
            ]
        else:
            repaired[key] = value

    if repaired.get("type") == "array":
        items = repaired.get("items")
        if not isinstance(items, (dict, list)) or (
            isinstance(items, dict) and not items
        ):
            repaired["items"] = dict(_DEFAULT_ARRAY_ITEMS)

    return repaired


def _tool_to_openai_dict(tool: Any) -> dict[str, Any] | None:  # noqa: ANN401  # BaseTool | Callable | dict
    """Return an OpenAI-format function dict for `tool`, or `None` on failure."""
    try:
        from langchain_core.utils.function_calling import convert_to_openai_tool
    except ImportError:
        logger.debug("convert_to_openai_tool unavailable; skipping Gemini repair")
        return None
    try:
        converted = convert_to_openai_tool(tool)
    except (ValueError, TypeError, AttributeError):
        logger.debug("convert_to_openai_tool failed for %r", tool, exc_info=True)
        return None
    if not isinstance(converted, dict):
        return None
    return converted


def repair_tools_for_gemini(tools: Iterable[Any]) -> list[Any]:
    """Return `tools` with array properties patched to include an `items` schema.

    Gemini rejects tool schemas whose array properties lack `items`. Each
    `BaseTool` is converted to an OpenAI-format function dict so the repaired
    parameters survive the round-trip through `ChatGoogleGenerativeAI
    .bind_tools`; existing dict tools are copied and repaired in place.
    Tool execution is unaffected because `ToolNode` looks tools up by name
    from the agent's original registration, not from `request.tools`.
    """
    repaired: list[object] = []
    for tool in tools:
        if isinstance(tool, dict):
            repaired.append(_repair_schema(tool))
            continue
        as_dict = _tool_to_openai_dict(tool)
        if as_dict is None:
            repaired.append(tool)
            continue
        repaired.append(_repair_schema(as_dict))
    return repaired


_TOOL_SCHEMA_ERROR_HINT = "GenerateContentRequest.tools"
_FUNCTION_DECL_RE = re.compile(
    r"function_declarations\[(\d+)\](?:\.parameters\.properties\[([^\]]+)\])?"
)


def _extract_error_body(exc: BaseException) -> str:
    """Return the best available string representation of a provider error body."""
    parts: list[str] = []
    for attr in ("message", "body", "details", "response"):
        value = getattr(exc, attr, None)
        if value is None:
            continue
        try:
            parts.append(str(value))
        except Exception:
            logger.debug("Failed to stringify %s on %r", attr, exc, exc_info=True)
    parts.append(str(exc))
    return "\n".join(parts)


def translate_gemini_tool_schema_error(exc: BaseException) -> str | None:
    """Return a user-facing message for a Gemini tool-schema `400`, else `None`.

    The wrapper looks for the sentinel `GenerateContentRequest.tools` /
    `function_declarations` strings emitted by the Gemini validator. When
    present, it parses out the offending declaration indexes and property
    names and formats them into a single sentence. Any other exception
    shape returns `None` so the caller re-raises unchanged.
    """
    body = _extract_error_body(exc)
    if _TOOL_SCHEMA_ERROR_HINT not in body or "function_declarations" not in body:
        return None

    matches = _FUNCTION_DECL_RE.findall(body)
    if not matches:
        return (
            "The Gemini model rejected the tool schemas in this request. "
            "This is usually caused by an array-typed parameter missing "
            "an `items` sub-schema."
        )

    descriptions: list[str] = []
    seen: set[str] = set()
    for index, prop in matches:
        label = (
            f"function_declarations[{index}].parameters.properties[{prop}]"
            if prop
            else f"function_declarations[{index}]"
        )
        if label in seen:
            continue
        seen.add(label)
        descriptions.append(label)

    joined = ", ".join(descriptions)
    return f"The Gemini model rejected tool schemas for: {joined}."
