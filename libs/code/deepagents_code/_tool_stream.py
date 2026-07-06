"""Shared streaming tool-call buffering and hook-payload construction.

Both execution surfaces reassemble the same streamed tool-call state and fire
the same `tool.use` / `tool.result` / `tool.error` hook payloads:

- the interactive Textual TUI (`deepagents_code.tui.textual_adapter`), and
- the headless runner (`deepagents_code.client.non_interactive`).

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
    — they are what the widget is constructed with (or come from a validated HITL
    interrupt). Result correlation reads args from `tool_msg.args` (a widget
    property).

- **Headless**: there are no widgets. Tool-call state lives in a plain dict on
    `StreamState` (`in_flight_tool_calls`, keyed by tool-call id). `tool.use`
    fires in the stream loop once args parse and the tool-call id is known.
    Result correlation pops the record from that dict.

Additionally, only the TUI has interactive HITL approval *widgets* that can be
rejected before any `ToolMessage` streams back, so only it needs
`_dispatch_terminal_tool_result_hooks` and the `completed_tool_result_ids`
duplicate-suppression set for synthetic middleware `ToolMessage` re-arrivals
after a resumed turn. The headless runner *does* have a HITL interrupt/resume
flow, but it never pre-dispatches terminal hooks before a resume — a rejection
arrives as a synthetic `ToolMessage` handled by the normal result path — so
there is no already-emitted terminal hook to suppress and therefore no
equivalent race.

Unifying these into a single function would require either a common
widget-or-dict abstraction (indirection with no behavioral gain) or pushing
widget-aware logic into this shared module (coupling headless mode to TUI
concepts it does not use). The split is deliberate: share everything that *can*
drift (payload shapes, arg parsing, status normalization, output truncation, and
the end-of-stream classification of tool calls that never emitted a `tool.use`)
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
- **Fire-once-per-id**: `tool.use` fires at most once per tool-call id on both
    surfaces, each gated by a monotonic id set that is never discarded within a
    turn (TUI via `displayed_tool_ids`, headless via `emitted_tool_use_ids`). A
    redelivery of the same id's arg chunks *after* its result — non-standard for
    `stream_mode="messages"` — is therefore ignored on both surfaces rather than
    re-firing `tool.use`. Headless additionally keeps `in_flight_tool_calls` for
    result correlation and the orphan drain; that map *is* cleared per result,
    so it is deliberately not the fire-once guard.
- **`tool.error` co-firing**: whenever `tool_status` is `"error"`, both
    surfaces emit `tool.error` alongside `tool.result`.

What is allowed to differ (and is not part of the contract):

- **Dispatch timing**: the TUI dispatches `tool.use` at widget mount;
    headless dispatches it in the stream loop once args parse. A hook subscriber
    should not assume `tool.use` fires at an identical point in the surface's
    internal lifecycle — only that it fires before `tool.result` for the same
    `tool_id`.
- **HITL rejection events**: both surfaces emit `tool.error`/`tool.result` for
    a rejected tool, but by different routes. Headless (and the TUI on a resumed
    turn) emits them from the synthetic `ToolMessage` on the normal result path;
    the TUI *additionally* emits them via the dedicated
    `_dispatch_terminal_tool_result_hooks` drain for tools rejected or cancelled
    in the interactive UI *before* any `ToolMessage` streams back. Only that
    pre-`ToolMessage` terminal drain is TUI-only — headless has no interactive
    approval path that can reject a call before its result arrives.

When changing the dispatch or gating logic in one surface, verify the parity
contract still holds against the other surface.

The payload schema is documented in `deepagents_code.hooks`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, TypedDict

from deepagents_code.hooks import HOOK_TOOL_OUTPUT_LIMIT

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)

ToolStatus = Literal["success", "error"]
"""Terminal status of a tool call, mirroring `ToolMessage.status`."""

ToolCallBufferKey = int | str
"""Key for buffering an in-progress streamed tool call.

The streaming `index` (an `int`) when present, else the tool-call `id` (a
`str`), else a unique placeholder string (see `tool_call_buffer_key`). Exported
so both surfaces annotate their buffer maps identically rather than each
spelling the union — the exact drift this module exists to prevent.
"""

ProviderToolArgs = dict[str, Any] | list[Any] | str | int | float | bool | None
"""A whole tool-call arguments value delivered by a provider in one chunk.

Usually a `dict`; rarely a bare scalar or list. `None` means no whole value has
arrived yet (the args may instead be streaming as fragments in `args_parts`).
An `isinstance` check narrows it to the declared `dict[str, Any]` arg type at the
single `parse_args` return site.
"""

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
    """The tool that produced the result. Usually the real tool name; falls back
    to `""` in the rare uncorrelated case where the `ToolMessage` carries no
    `name` and no `tool.use` was correlated to supply one."""

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
) -> ToolCallBufferKey:
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


def _looks_structurally_complete(s: str) -> bool:
    """Return whether `s` is a balanced JSON container, string-state aware.

    Scans for bracket balance while tracking whether the cursor is inside a
    string literal (honoring backslash escapes), so a value is "complete" only
    when every `{`/`[` is matched and the text does not end mid-string. Used
    solely to decide whether a `json.loads` failure is worth warning about: a
    balanced-but-unparseable value is malformed, whereas a partial stream — e.g.
    a chunk boundary landing right after an inner `}` in `{"edits": [{"a": 1}` —
    leaves an outer container open and returns `False`, so no false warning fires
    on a healthy mid-stream fragment. A stray closer (`depth < 0`) is unbalanced
    and can never be completed by more input, so it is reported complete
    (malformed) too. Iterative, so it never raises `RecursionError` on
    pathologically nested input.

    Args:
        s: The accumulated, stripped tool-call argument text.

    Returns:
        `True` if the brackets are matched (or over-closed) and the text does not
            end inside a string; `False` while a container is still open.
    """
    depth = 0
    in_string = False
    escaped = False
    for ch in s:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
            if depth < 0:
                # More closers than openers: unbalanced, not a partial stream.
                return True
    return depth == 0 and not in_string


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

    args: ProviderToolArgs = None
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
        """Enforce the `args` XOR `args_parts` invariant at construction time.

        `ingest` maintains this on every chunk. This guard catches an illegal
        buffer built directly with both fields set; the fields are public and
        mutable, though, so a caller can still reach the both-populated state by
        assigning them after construction. `parse_args` re-checks the invariant
        at read time (raising rather than silently reading `args` first and
        masking a conflicting `args_parts`), so the illegal state fails loudly
        wherever it originates.

        Raises:
            ValueError: If both `args` and `args_parts` are set.
        """
        if self.args is not None and self.args_parts:
            msg = "ToolCallBuffer cannot hold both args and args_parts"
            raise ValueError(msg)

    def _reset_for_new_call(self) -> None:
        """Discard retained state before this buffer is reused for another call."""
        self.name = None
        self.tool_id = None
        self.args = None
        self.args_parts = []
        self.displayed = False
        self.warned = False

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
        #
        # Some providers deliver the new call's name/args before its replacement
        # id. In that delayed-id shape, a retained `self.tool_id` is already stale
        # even though this chunk has no id to compare against; clear it before
        # folding the new name/args so parsed args cannot dispatch under the old
        # call id.
        #
        # This assumes the standard LangChain streaming contract: a call's name
        # arrives only on its first chunk, so id-less continuation chunks for the
        # same call carry only args and keep accumulating below. A non-standard
        # provider that *repeats* the name on an id-less continuation chunk would
        # trip `delayed_id_reuses_index` mid-call and reset the accumulated args.
        # That degrades gracefully rather than crashing: the call's `tool.use`
        # (and its parsed args) is lost, but the tool still executes and its
        # `tool.result` still fires via the uncorrelated `{}`-args path. No
        # observed provider (Anthropic/OpenAI) streams that shape.
        new_id_reuses_index = (
            tool_id is not None and self.tool_id is not None and tool_id != self.tool_id
        )
        delayed_id_reuses_index = (
            tool_id is None
            and self.tool_id is not None
            and name is not None
            and self.name is not None
        )
        if new_id_reuses_index or delayed_id_reuses_index:
            self._reset_for_new_call()

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
            # Append every non-empty fragment unconditionally. An earlier
            # TUI-only version skipped a fragment equal to the immediately
            # preceding one to dedup accidental redelivery; that guard was
            # dropped because a stream can legitimately emit two identical
            # consecutive deltas (e.g. two `", "` fragments) and skipping one
            # corrupts the reassembled JSON. Standard `stream_mode="messages"`
            # streaming never redelivers a fragment, so unconditional append is
            # both lossless and correct in practice.
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
                unparseable — malformed or too deeply nested (the
                structurally-complete malformed case is logged once via
                `warned`).

        Raises:
            ValueError: If both `args` and `args_parts` are populated, violating
                the invariant `ingest`/`__post_init__` maintain. Guarded here so
                a caller that assigned the public fields directly fails loudly
                instead of silently dropping the fragments (read order would
                otherwise return `args` and discard `args_parts`).
        """
        if self.args is not None and self.args_parts:
            msg = "ToolCallBuffer cannot hold both args and args_parts"
            raise ValueError(msg)
        if isinstance(self.args, dict):
            # A whole-value dict delivered by the provider. The `isinstance`
            # narrows `ProviderToolArgs` to the declared `dict[str, Any]` arg
            # type, so this returns without a cast.
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
        # well-formed object's closing brace is always its last char. The
        # `"".join` still runs per fragment (so accumulation stays O(n^2) across
        # the stream); the win is deferring the costlier `json.loads` until the
        # value looks complete, rather than re-parsing the whole prefix on every
        # fragment.
        #
        # A bracketed value with trailing junk after its close (e.g.
        # `{"a": 1} x`) also returns here and is treated as still-streaming, so
        # it never reaches the malformed warning below — the same deliberate
        # trade as the non-bracketed-scalar exclusion documented at the end of
        # this method. Real provider arg streams are pure JSON, so this shape
        # does not occur in practice; if such a call still executes, its
        # `tool.result` logs the correlation miss.
        if stripped[0] in "{[" and not stripped.endswith(("}", "]")):
            return None
        try:
            parsed = json.loads(joined)
        except (json.JSONDecodeError, RecursionError):
            # Args that are structurally balanced (all brackets matched, not
            # ending mid-string) but still fail to parse are malformed, not
            # mid-stream — surface them rather than silently dropping the
            # tool.use hook. `_looks_structurally_complete` does a string-aware
            # balance check rather than the cheaper "starts with {/[ and ends
            # with }/]" heuristic, which false-positives on a normal chunk
            # boundary inside nested args (e.g. `{"edits": [{"a": 1}`) and would
            # warn on a perfectly healthy stream. `RecursionError` is caught
            # alongside the decode error: pathologically nested model output
            # makes `json.loads` exceed the interpreter recursion limit, and that
            # is one malformed call to skip, not a reason to let the exception
            # escape and abort the whole turn. `%r` escapes any control
            # characters in the model-generated fragment. The `warned` latch
            # keeps this to one line per call, since the buffer is retained and
            # parse_args re-runs on every later chunk for the same call.
            if (
                stripped[0] in "{["
                and _looks_structurally_complete(stripped)
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


class UnemittedToolCalls(NamedTuple):
    """Counts of buffered tool calls that never emitted a `tool.use`.

    A `NamedTuple` so callers can still unpack positionally while the field names
    keep the two same-typed slots from being load-bearing at every call site (a
    bare `(int, int)` invites a silent transposition). Both surfaces log the two
    counts as separate diagnostics.
    """

    unparsed: int
    """Named buffers whose args never parsed."""

    idless_parsed: int
    """Named buffers whose args parsed but whose `tool_id` stayed `None`."""


def count_unemitted_tool_calls(buffers: Iterable[ToolCallBuffer]) -> UnemittedToolCalls:
    """Classify buffered tool calls that never emitted a `tool.use`.

    Both surfaces log the same end-of-stream diagnostic for tool calls still in
    their buffer map when the stream ends: those whose args never parsed, and
    those whose args parsed but whose `tool_id` stayed `None` (so `tool.use` was
    gated out). Sharing the classification here keeps the two diagnostics from
    drifting; each surface still emits its own log lines. `parse_args` is safe to
    re-run (idempotent bar its one-shot `warned` latch).

    Args:
        buffers: The in-progress tool-call buffers remaining at stream end.

    Returns:
        The `unparsed` and `idless_parsed` counts (see `UnemittedToolCalls`).
    """
    unparsed = 0
    idless_parsed = 0
    for buffer in buffers:
        if buffer.name is None:
            continue
        if buffer.parse_args() is None:
            unparsed += 1
        elif buffer.tool_id is None:
            idless_parsed += 1
    return UnemittedToolCalls(unparsed, idless_parsed)


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
        # `max(..., 0)` guards a future `HOOK_TOOL_OUTPUT_LIMIT` set below the
        # marker length: a negative `keep` would slice from the end and keep the
        # tail instead of truncating. Positive with today's constants (2000 vs a
        # ~19-char marker); this only future-proofs a constant change.
        keep = max(HOOK_TOOL_OUTPUT_LIMIT - len(TOOL_OUTPUT_TRUNCATION_MARKER), 0)
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
