"""Shared streaming tool-call buffering and hook-payload construction.

Both execution surfaces reassemble the same streamed tool-call state and fire
the same `tool.use` / `tool.result` hook payloads:

- the interactive Textual TUI (`deepagents_code.textual_adapter`), and
- the headless runner (`deepagents_code.non_interactive`).

This module holds the single implementation of that logic so the two surfaces
cannot drift. Each surface still calls `dispatch_hook_fire_and_forget` from its
own namespace (the dispatch seam tests patch); only the buffering, argument
parsing, and payload shapes live here.

The payload schema is documented in `deepagents_code.hooks`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from deepagents_code.hooks import HOOK_TOOL_OUTPUT_LIMIT

logger = logging.getLogger(__name__)


def tool_call_buffer_key(
    index: int | str | None, tool_id: str | None, count: int
) -> int | str:
    """Compute a stable key for buffering an in-progress streamed tool call.

    Prefers the streaming `index` (stable across fragments of one call), then
    the tool-call `id`, falling back to a positional placeholder so unrelated
    id-less calls don't collide.

    Args:
        index: The `index` field from the streamed tool-call chunk, if any.
            Typed loosely because it is read from an untyped content-block dict.
        tool_id: The tool-call `id` from the chunk, if any.
        count: The current number of buffered calls, used to make the fallback
            placeholder unique.

    Returns:
        The chunk `index`, else the `tool_id`, else a unique placeholder string.
    """
    if index is not None:
        return index
    if tool_id is not None:
        return tool_id
    return f"unknown-{count}"


@dataclass
class ToolCallBuffer:
    """In-progress state for a single streamed tool call.

    `args` and `args_parts` are two representations of the arguments, intended
    to be used one at a time: `args` holds a fully materialized value when a
    chunk delivers it in one piece, while `args_parts` collects JSON string
    fragments that are reassembled by `parse_args` once the payload is complete.
    `displayed` is a one-shot latch guarding the single "Calling tool" console
    line (used only by the headless surface).
    """

    name: str | None = None
    tool_id: str | None = None
    args: Any = None  # provider-shaped: dict, scalar, or None
    args_parts: list[str] = field(default_factory=list)
    displayed: bool = False

    def ingest(
        self,
        *,
        name: str | None,
        tool_id: str | None,
        args: Any,  # noqa: ANN401  # provider-shaped tool-call args chunk
    ) -> None:
        """Fold one streamed tool-call chunk's fields into the buffer.

        A dict `args` replaces any accumulated fragments (the provider delivered
        the whole value at once); a string `args` is appended as a fragment,
        skipping an immediate duplicate of the previous fragment; any other
        non-`None` value is stored as-is.

        Args:
            name: The tool name from this chunk, if present.
            tool_id: The tool-call id from this chunk, if present.
            args: The `args` field from this chunk (dict, string fragment, or
                other scalar).
        """
        if name:
            self.name = name
        if tool_id:
            self.tool_id = tool_id

        if isinstance(args, dict):
            self.args = args
            self.args_parts = []
        elif isinstance(args, str):
            if args and (not self.args_parts or args != self.args_parts[-1]):
                self.args_parts.append(args)
        elif args is not None:
            self.args = args

    def parse_args(self) -> dict[str, Any] | None:
        """Return the tool-call args once enough data has arrived, else `None`.

        A non-object JSON value (a bare scalar or list — rare for tool calls) is
        wrapped as `{"value": ...}` so a caller's `tool_args` is always an
        object.

        Returns:
            Parsed tool-call arguments, or `None` when the args are not yet
                complete (still streaming) or empty.
        """
        if isinstance(self.args, dict):
            return self.args
        if self.args is not None:
            return {"value": self.args}

        if not self.args_parts:
            return None
        joined = "".join(self.args_parts)
        stripped = joined.strip()
        if not stripped:
            return None
        # Cheap structural pre-check: bail while a JSON object/array is still
        # open so we don't attempt to parse a partial streamed fragment. A
        # well-formed object's closing brace is always its last char. The join
        # above is unavoidable per fragment, but deferring `json.loads` until
        # the value looks complete is what avoids re-parsing the whole prefix
        # on every fragment.
        if stripped[0] in "{[" and not stripped.endswith(("}", "]")):
            return None
        try:
            parsed = json.loads(joined)
        except json.JSONDecodeError:
            # Args that look structurally complete (bracketed and closed) but
            # still fail to parse are malformed, not mid-stream — surface them
            # rather than silently dropping the tool.use hook. `repr` escapes
            # any control characters in the model-generated fragment.
            if stripped[0] in "{[" and stripped.endswith(("}", "]")):
                logger.warning(
                    "Tool-call args look complete but failed to parse: %r",
                    joined[:200],
                )
            return None
        if not isinstance(parsed, dict):
            return {"value": parsed}
        return parsed


def build_tool_use_payload(
    tool_name: str, tool_id: str | None, tool_args: dict[str, Any]
) -> dict[str, Any]:
    """Build the `tool.use` hook payload (schema documented in `hooks`).

    Returns:
        The `tool.use` payload dict.
    """
    return {
        "tool_name": tool_name,
        "tool_id": tool_id,
        "tool_args": tool_args,
    }


def build_tool_error_payload(tool_name: str) -> dict[str, Any]:
    """Build the `tool.error` hook payload (schema documented in `hooks`).

    Returns:
        The `tool.error` payload dict.
    """
    return {"tool_names": [tool_name]}


def build_tool_result_payload(
    tool_name: str,
    tool_id: str | None,
    tool_args: dict[str, Any],
    tool_status: str,
    tool_output: str,
) -> dict[str, Any]:
    """Build the `tool.result` hook payload (schema documented in `hooks`).

    `tool_output` is truncated to `HOOK_TOOL_OUTPUT_LIMIT` here so both surfaces
    apply the identical cap regardless of where the raw output originates.

    Returns:
        The `tool.result` payload dict with `tool_output` truncated.
    """
    return {
        "tool_name": tool_name,
        "tool_id": tool_id,
        "tool_args": tool_args,
        "tool_status": tool_status,
        "tool_output": tool_output[:HOOK_TOOL_OUTPUT_LIMIT],
    }
