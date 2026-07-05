"""Shared streaming tool-call buffering and hook-payload construction.

Both execution surfaces reassemble the same streamed tool-call state and fire
the same `tool.use` / `tool.result` / `tool.error` hook payloads:

- the interactive Textual TUI (`deepagents_code.textual_adapter`), and
- the headless runner (`deepagents_code.non_interactive`).

This module holds the single implementation of the buffering, argument parsing,
and payload-shape logic, so the two surfaces cannot drift *in those layers*. The
dispatch, gating, result-correlation, and buffer-lifecycle layers are still
implemented separately in each surface (each calls
`dispatch_hook_fire_and_forget` from its own namespace — the dispatch seam tests
patch) and must be kept in sync by hand.

Why the split exists
--------------------

The two surfaces have fundamentally different object lifecycles, which prevents
a single shared dispatch function:

- **TUI**: tool calls are backed by `ToolCallMessage` widgets that persist
    parsed args as instance state and render success/error visually. `tool.use`
    fires at widget-mount time because that is when the parsed args are available
    from the widget constructor (or from a validated HITL interrupt). Result
    correlation reads args from `tool_msg.args` (a widget property).

- **Headless**: there are no widgets. Tool-call state lives in a plain dict on
    `StreamState` (`in_flight_tool_calls`, keyed by tool-call id). `tool.use`
    fires in the stream loop once args parse and the tool-call id is known.
    Result correlation pops the record from that dict.

Additionally, only the TUI has interactive HITL approval widgets that can be
rejected, so only it needs `_dispatch_terminal_tool_result_hooks` and the
`completed_tool_result_ids` duplicate-suppression set for synthetic middleware
`ToolMessage` re-arrivals after a resumed turn. The headless runner has no
interrupt/resume flow and therefore no equivalent race.

Unifying these into a single function would require either a common
widget-or-dict abstraction (indirection with no behavioral gain) or pushing
widget-aware logic into this shared module (coupling headless mode to TUI
concepts it does not use). The split is deliberate: share everything that *can*
drift (payload shapes, arg parsing, status normalization, output truncation)
here, and keep everything that *must* differ (when to dispatch, where args come
from, HITL handling) in each surface.

Parity contract for hook consumers
----------------------------------

A hook consumer can rely on these guarantees being identical across surfaces:

- **Payload shape**: every `tool.use`, `tool.result`, and `tool.error` payload
    is built by the shared builders in this module, so field names and
    truncation are the same.
- **Event completeness**: every executed tool emits `tool.result`, including
    tools whose args never parsed or that carried no tool-call id (both surfaces
    emit with `{}` args in that case). A tool whose stream is aborted before its
    result — a cancelled turn or a mid-stream error — is also closed with a
    terminal `tool.error`/`tool.result` (TUI via
    `_dispatch_terminal_tool_result_hooks`, headless via
    `_dispatch_orphaned_tool_result_hooks`), so no `tool.use` is left dangling.
- **Fire-once-per-tool**: `tool.use` fires at most once per in-flight
    tool-call id on both surfaces (TUI via `displayed_tool_ids`, headless via
    `in_flight_tool_calls`).
- **`tool.error` co-firing**: whenever `tool_status` is `"error"`, both
    surfaces emit `tool.error` alongside `tool.result`.

What is allowed to differ (and is not part of the contract):

- **Dispatch timing**: the TUI dispatches `tool.use` at widget mount;
    headless dispatches it in the stream loop once args parse. A hook subscriber
    should not assume `tool.use` fires at an identical point in the surface's
    internal lifecycle — only that it fires before `tool.result` for the same
    `tool_id`.
- **HITL rejection events**: only the TUI emits terminal `tool.error`/
    `tool.result` for rejected tools. Headless mode has no interactive approval
    path.

When changing the dispatch or gating logic in one surface, verify the parity
contract still holds against the other surface.

The payload schema is documented in `deepagents_code.hooks`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict, cast

from deepagents_code.hooks import HOOK_TOOL_OUTPUT_LIMIT

logger = logging.getLogger(__name__)

ToolStatus = Literal["success", "error"]
"""Terminal status of a tool call, mirroring `ToolMessage.status`."""

TOOL_OUTPUT_TRUNCATION_MARKER = "…[output truncated]"
"""Suffix appended to a `tool.result` `tool_output` that hit
`HOOK_TOOL_OUTPUT_LIMIT`, so a consumer can distinguish a capped result from a
genuinely short one. The marker is counted within the cap, so the final
`tool_output` length never exceeds `HOOK_TOOL_OUTPUT_LIMIT`."""

UNRENDERABLE_TOOL_OUTPUT = "<tool output could not be rendered>"
"""Sentinel `tool_output` used when formatting/coercing a tool result raises.

Lets both surfaces keep the terminal `tool.result` dispatch unconditional
without re-touching the offending content (whose `__str__`/`__repr__` may itself
raise), so a hook consumer still sees the result rather than a dropped event."""


def normalize_tool_status(raw_status: object, tool_name: str) -> ToolStatus:
    """Map a raw `ToolMessage.status` to the two-value hook domain, fail-closed.

    `"error"` and `"success"` pass through. Any other *present* value — a future
    provider status, an explicit `None`, or a typo — is unexpected and treated as
    `"error"` (and logged), so an audit or notification hook is never told a
    non-successful tool succeeded. Callers pass
    `getattr(message, "status", "success")`, so a missing status arrives as
    `"success"` and is not warned about.

    Args:
        raw_status: The raw `status` value read off the `ToolMessage`.
        tool_name: The tool name, included in the warning for context.

    Returns:
        `"success"` or `"error"`.
    """
    if raw_status == "error":
        return "error"
    if raw_status == "success":
        return "success"
    logger.warning(
        "Unexpected ToolMessage.status %r for tool %s; treating as error",
        raw_status,
        tool_name,
    )
    return "error"


class ToolUsePayload(TypedDict):
    """`tool.use` hook payload (schema documented in `hooks`)."""

    tool_name: str
    """The tool being invoked."""

    tool_id: str
    """The tool-call id. Always a real id — a call with no id never produces a
    `tool.use` (see `hooks`), so unlike `tool.result` this is never `None`."""

    tool_args: dict[str, Any]
    """The parsed tool-call arguments."""


class ToolErrorPayload(TypedDict):
    """`tool.error` hook payload (schema documented in `hooks`)."""

    tool_names: list[str]
    """Names of the tools whose calls failed or were rejected."""


class ToolResultPayload(TypedDict):
    """`tool.result` hook payload (schema documented in `hooks`)."""

    tool_name: str
    """The tool that produced the result."""

    tool_id: str | None
    """The tool-call id, or `None` when it could not be correlated."""

    tool_args: dict[str, Any]
    """The parsed tool-call arguments, or `{}` when uncorrelated."""

    tool_status: ToolStatus
    """`"success"`, or `"error"` for a failed, rejected, or cancelled call."""

    tool_output: str
    """The tool's returned content, capped to `HOOK_TOOL_OUTPUT_LIMIT`. When the
    full output exceeded the cap, the value ends with
    `TOOL_OUTPUT_TRUNCATION_MARKER` so a consumer (e.g. a secret/policy scanner)
    can tell a truncated result from a genuinely short one."""


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

    `args` and `args_parts` are two representations of the arguments used one at
    a time, depending on whether the provider delivers the value whole or in
    JSON string fragments.
    """

    name: str | None = None
    """The tool name, once a chunk has supplied it."""

    tool_id: str | None = None
    """The tool-call id, once a chunk has supplied it."""

    args: object = None  # provider-shaped: dict, scalar, or None
    """A fully materialized arguments value delivered by a single chunk (dict or,
    rarely, a scalar). Mutually exclusive with `args_parts` (see `__post_init__`
    and `ingest`)."""

    args_parts: list[str] = field(default_factory=list)
    """JSON string fragments accumulated across chunks, reassembled by
    `parse_args` once the payload looks complete. Mutually exclusive with
    `args` (see `__post_init__` and `ingest`)."""

    displayed: bool = False
    """One-shot latch guarding the single "Calling tool" console line (used only
    by the headless surface)."""

    warned: bool = False
    """One-shot latch so a malformed-but-complete payload is logged at most once,
    even though `parse_args` re-runs on every later chunk for the retained
    buffer."""

    def __post_init__(self) -> None:
        """Enforce the `args` XOR `args_parts` invariant at construction.

        `ingest` maintains this on every chunk; asserting it here as well means
        the illegal both-populated state is unrepresentable regardless of how a
        buffer is built, so `parse_args` can read `args` first without masking a
        conflicting `args_parts`.

        Raises:
            ValueError: If both `args` and `args_parts` are set.
        """
        if self.args is not None and self.args_parts:
            msg = "ToolCallBuffer cannot hold both args and args_parts"
            raise ValueError(msg)

    def ingest(
        self,
        *,
        name: str | None,
        tool_id: str | None,
        args: Any,  # noqa: ANN401  # provider-shaped tool-call args chunk
    ) -> None:
        """Fold one streamed tool-call chunk's fields into the buffer.

        A dict `args` replaces any accumulated fragments (the provider delivered
        the whole value at once); a string `args` is appended as a fragment; any
        other non-`None` value is stored as-is.

        Args:
            name: The tool name from this chunk, if present.
            tool_id: The tool-call id from this chunk, if present.
            args: The `args` field from this chunk (dict, string fragment, or
                other scalar).
        """
        # A differing id on the same buffer key means a *new* call has reused
        # this streaming index. Indices restart per message, so a buffer retained
        # from an earlier message or HITL-resume round (e.g. one whose args never
        # parsed) can collide here. Reset the accumulated arg state so the new
        # call's fragments never append onto the old call's leftover — which would
        # otherwise leave both unparseable and silently drop the new call's
        # `tool.use`. Per-call metadata is reset too, so stale name/display state
        # cannot bleed into the new call if its first chunk only carries the id.
        if tool_id and self.tool_id and tool_id != self.tool_id:
            self.name = None
            self.args = None
            self.args_parts = []
            self.displayed = False
            self.warned = False

        if name:
            self.name = name
        if tool_id:
            self.tool_id = tool_id

        # `args` (a whole value) and `args_parts` (JSON fragments) are mutually
        # exclusive; each branch clears the counterpart so the buffer never holds
        # both. Together with the `__post_init__` guard this keeps the invariant
        # true for every buffer, so `parse_args` reads `args` first without
        # masking a conflicting `args_parts` via read order. A provider streams a
        # single call as either whole values or fragments, never a mix, so no
        # real sequence loses data.
        if isinstance(args, dict):
            self.args = args
            self.args_parts = []
        elif isinstance(args, str):
            if args:
                self.args = None
                self.args_parts.append(args)
        elif args is not None:
            self.args = args
            self.args_parts = []

    def parse_args(self) -> dict[str, Any] | None:
        """Return the tool-call args once enough data has arrived, else `None`.

        A non-object JSON value (a bare scalar or list — rare for tool calls) is
        wrapped as `{"value": ...}` so a caller's `tool_args` is always an
        object.

        Returns:
            Parsed tool-call arguments, or `None` when the args are not yet
                complete (still streaming), empty, or structurally complete but
                malformed (the malformed case is logged once via `warned`).
        """
        if isinstance(self.args, dict):
            # A whole-value dict delivered by the provider; its keys are
            # provider-shaped, so narrow the `object` field to the declared arg
            # type at this single return site.
            return cast("dict[str, Any]", self.args)
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
            # rather than silently dropping the tool.use hook. `%r` escapes any
            # control characters in the model-generated fragment. The `warned`
            # latch keeps this to one line per call, since the buffer is retained
            # and parse_args re-runs on every later chunk for the same call.
            if (
                stripped[0] in "{["
                and stripped.endswith(("}", "]"))
                and not self.warned
            ):
                self.warned = True
                logger.warning(
                    "Tool-call args look complete but failed to parse: %r",
                    joined[:200],
                )
            # A non-bracketed complete-but-malformed value is deliberately not
            # warned here: it is indistinguishable from a still-streaming scalar
            # fragment, so warning would be noisy. If such a call still executes,
            # its `tool.result` logs the correlation miss at info; if it never
            # executes there is no result to audit, so nothing is lost.
            return None
        if not isinstance(parsed, dict):
            return {"value": parsed}
        return parsed


def build_tool_use_payload(
    tool_name: str, tool_id: str, tool_args: dict[str, Any]
) -> ToolUsePayload:
    """Build the `tool.use` hook payload (schema documented in `hooks`).

    Args:
        tool_name: The tool being invoked.
        tool_id: The tool-call id. Always a real id for an emitted `tool.use`
            (a call with no id never produces one).
        tool_args: The parsed tool-call arguments.

    Returns:
        The `tool.use` payload dict.
    """
    return {
        "tool_name": tool_name,
        "tool_id": tool_id,
        "tool_args": tool_args,
    }


def build_tool_error_payload(tool_name: str) -> ToolErrorPayload:
    """Build the `tool.error` hook payload (schema documented in `hooks`).

    Args:
        tool_name: The tool whose call failed or was rejected.

    Returns:
        The `tool.error` payload dict.
    """
    return {"tool_names": [tool_name]}


def build_tool_result_payload(
    tool_name: str,
    tool_id: str | None,
    tool_args: dict[str, Any],
    tool_status: ToolStatus,
    tool_output: str,
) -> ToolResultPayload:
    """Build the `tool.result` hook payload (schema documented in `hooks`).

    `tool_output` is capped to `HOOK_TOOL_OUTPUT_LIMIT` here so both surfaces
    apply the identical cap regardless of where the raw output originates. When
    the cap fires the value ends with `TOOL_OUTPUT_TRUNCATION_MARKER` (counted
    within the cap, so the result never exceeds the limit) so a consumer can tell
    a capped result from a short one. `tool_args` is intentionally not truncated
    (see `HOOK_TOOL_OUTPUT_LIMIT`).

    Args:
        tool_name: The tool that produced the result.
        tool_id: The tool-call id, or `None` when it could not be correlated.
        tool_args: The parsed tool-call arguments, or `{}` when uncorrelated.
        tool_status: `"success"` or `"error"`.
        tool_output: The tool's returned content (capped in the payload).

    Returns:
        The `tool.result` payload dict with `tool_output` capped and marked when
            truncation occurred.
    """
    if len(tool_output) > HOOK_TOOL_OUTPUT_LIMIT:
        keep = HOOK_TOOL_OUTPUT_LIMIT - len(TOOL_OUTPUT_TRUNCATION_MARKER)
        capped_output = tool_output[:keep] + TOOL_OUTPUT_TRUNCATION_MARKER
    else:
        capped_output = tool_output
    return {
        "tool_name": tool_name,
        "tool_id": tool_id,
        "tool_args": tool_args,
        "tool_status": tool_status,
        "tool_output": capped_output,
    }
