"""Programmatic tool calling (PTC) support for `CodeInterpreterMiddleware`.

PTC exposes the agent's LangChain tools inside the JavaScript REPL as
`tools.<camelCaseName>(input)` async functions. Instead of issuing
N serial tool calls, the model writes one `eval` that loops / parallelises
/ chains tools in-code:

    const [a, b] = await Promise.all([
        tools.search({query: "foo"}),
        tools.search({query: "bar"}),
    ]);

Three pieces live here:

- filtering — turn the live agent toolset into the subset exposed to PTC
- prompt rendering — render a short TS-ish API-reference block describing
    each exposed tool, so the model knows the call shape
- bridge helpers — per-eval call budget, tool input coercion, injected
    args, and the JS that binds `globalThis.tools`

The host functions that actually invoke each tool live on the slot in
`_repl.py`, next to the rest of the context wiring.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, get_type_hints

from langchain_core.tools import BaseTool
from langchain_core.tools.base import (
    InjectedToolCallId,
    _is_injected_arg_type,
    get_all_basemodel_annotations,
)
from langgraph.prebuilt.tool_node import _get_all_injected_args
from quickjs_rs import Undefined

from langchain_quickjs import _prompt

if TYPE_CHECKING:
    import asyncio
    from collections.abc import Sequence

    from deepagents.backends.protocol import BackendProtocol
    from langgraph.prebuilt import ToolRuntime


PTCOption = list[str | BaseTool]

_RESERVED_SUBAGENT_TASK_NAME = "task"

_TASK_IN_PTC_MSG = (
    "The subagent `task` tool cannot be exposed via `ptc`. It is always "
    "available as the top-level `task()` global inside the REPL (with "
    "`subagentType`, `label`, and `responseSchema` support); exposing it through "
    "the `tools.*` namespace would create a second, conflicting dispatch path "
    'that drops `responseSchema`. Remove "task" from `ptc`.'
)


def filter_tools_for_ptc(
    tools: Sequence[BaseTool],
    config: PTCOption,
    *,
    self_tool_name: str,
) -> list[BaseTool]:
    """Return the subset of `tools` exposed inside the REPL.

    `self_tool_name` is the REPL's own tool name; it is *always* excluded
    to prevent the model from recursing `tools.eval("tools.eval(...)")`.
    If the model wants a nested eval, it can just write nested code in one
    call — that's the whole point of PTC.

    `config` is allowlist-only:

    - `str` entries: expose matching tool names from `tools`.
    - `BaseTool` entries: expose those tools directly (minus
        `self_tool_name`).

    Mixed lists are supported and merged. Explicit `BaseTool` entries
    are included first, then name-matched agent tools are appended.
    Duplicate tool names are deduplicated.

    The subagent `task` tool is reserved and may not appear in `config`
    (by name or instance) — it is always available as the `task()` global,
    so a `tools.task` PTC variant would be a conflicting, degraded duplicate.
    A `"task"` entry raises `ValueError`.

    Warning:
        PTC tool calls execute through the REPL bridge and currently do
        not respect `interrupt_on` / HITL approval hooks for each
        individual tool invocation.
    """
    if isinstance(config, list):
        explicit_tools: list[BaseTool] = []
        allow_names: set[str] = set()
        for entry in config:
            if isinstance(entry, BaseTool):
                if entry.name == _RESERVED_SUBAGENT_TASK_NAME:
                    raise ValueError(_TASK_IN_PTC_MSG)
                if entry.name != self_tool_name:
                    explicit_tools.append(entry)
                continue
            if isinstance(entry, str):
                if entry == _RESERVED_SUBAGENT_TASK_NAME:
                    raise ValueError(_TASK_IN_PTC_MSG)
                allow_names.add(entry)
                continue
            msg = "ptc list entries must be str or BaseTool"
            raise TypeError(msg)
        selected = [
            *explicit_tools,
            *[t for t in tools if t.name != self_tool_name and t.name in allow_names],
        ]
        deduped: list[BaseTool] = []
        seen_names: set[str] = set()
        for tool in selected:
            if tool.name in seen_names:
                continue
            seen_names.add(tool.name)
            deduped.append(tool)
        selected = deduped
        _raise_on_invalid_ptc_tools(selected)
        return selected
    msg = (
        "Unsupported `ptc` config type. "
        "Use a list of tool names, list of BaseTool instances, or disable PTC."
    )
    raise TypeError(msg)


def to_camel_case(name: str) -> str:
    """Convert `snake_case` / `kebab-case` → `camelCase`."""
    return _prompt.to_camel_case(name)


def is_valid_js_identifier(name: str) -> bool:
    """Return whether `name` is a valid JavaScript identifier."""
    return _prompt.is_valid_js_identifier(name)


def is_valid_ptc_tool_name(name: str) -> bool:
    """Return whether a tool can be exposed as `tools.<camelCaseName>`."""
    return _prompt.is_valid_ptc_tool_name(name)


def _raise_on_invalid_ptc_tools(tools: Sequence[BaseTool]) -> None:
    for tool in tools:
        camel = to_camel_case(tool.name)
        if is_valid_js_identifier(camel):
            continue
        msg = (
            f"PTC tool name {tool.name!r} cannot be exposed as JavaScript "
            f"identifier {camel!r}. Tool names must map to "
            "`/^[A-Za-z_$][A-Za-z0-9_$]*$/`."
        )
        raise ValueError(msg)


def render_ptc_prompt(tools: Sequence[BaseTool], *, tool_name: str = "eval") -> str:
    """Build the `tools` namespace section of the system prompt."""
    if not tools:
        return ""
    _raise_on_invalid_ptc_tools(tools)
    return _prompt.render_ptc_prompt(tools, tool_name=tool_name)


class PTCCallBudgetExceededError(RuntimeError):
    """Raised when one eval exceeds its configured PTC call budget."""

    def __init__(self, *, limit: int, attempted: int, function_name: str) -> None:
        self.limit = limit
        self.attempted = attempted
        self.function_name = function_name
        msg = (
            "PTC call budget exceeded "
            f"(limit={limit}, attempted={attempted}, "
            f"function={function_name})"
        )
        super().__init__(msg)

    def render_message(self) -> str:
        return (
            "PTC call budget exceeded "
            f"(limit={self.limit}, attempted={self.attempted}, "
            f"function={self.function_name})"
        )


@dataclass(frozen=True, slots=True)
class PTCState:
    """Per-eval PTC state (reset on each eval call)."""

    remaining_calls: int | None
    outer_runtime: ToolRuntime | None = None
    outer_loop: asyncio.AbstractEventLoop | None = None
    fs: BackendProtocol | None = None
    task_dispatch_count: int = 0
    """Ordinal for the next `task()` dispatch in this eval.

    Starts at 0 for every eval, so a replayed eval that dispatches in the
    same order reproduces the same ordinals and therefore the same derived
    dispatch ids.
    """

    def consume_call_budget(
        self, *, function_name: str, max_ptc_calls: int | None
    ) -> PTCState:
        """Count one PTC bridge call and enforce the per-eval limit."""
        if self.remaining_calls is None:
            return self
        if self.remaining_calls > 0:
            return replace(self, remaining_calls=self.remaining_calls - 1)

        normalized_limit = max_ptc_calls if max_ptc_calls is not None else 0
        raise PTCCallBudgetExceededError(
            limit=normalized_limit,
            attempted=normalized_limit + 1,
            function_name=function_name,
        )


def _strip_undefined(value: Any) -> Any:
    """Recursively drop ``undefined`` object properties so defaults apply.

    JS ``undefined`` means "absent", so a dict key whose value is
    ``undefined`` is omitted, letting the tool's schema defaults take over.
    Array entries are converted to ``None`` (JS ``null``) instead of being
    dropped, since dropping would shift indices and corrupt the array.
    """
    if isinstance(value, dict):
        return {
            key: _strip_undefined(val)
            for key, val in value.items()
            if not isinstance(val, Undefined)
        }
    if isinstance(value, list):
        return [
            None if isinstance(item, Undefined) else _strip_undefined(item)
            for item in value
        ]
    return value


def normalize_tool_input(raw: Any) -> dict[str, Any]:
    """Coerce whatever JS passed into `tools.X(...)` to a dict.

    LangChain tools accept a dict. QuickJS marshals JS objects to dicts
    already; we just want to guard against the model passing `null`,
    `undefined`, a bare string, or a number (none of which a well-
    formed tool call should produce, but the model is the model).
    """
    if raw is None or isinstance(raw, Undefined):
        return {}
    if isinstance(raw, dict):
        return _strip_undefined(raw)
    # Bare scalar / list — wrap under a conventional key so the tool's
    # schema validation produces an informative error rather than a
    # silent miss.
    return {"input": _strip_undefined(raw)}


def synth_tool_call_id(tool_name: str) -> str:
    """Mint a synthetic tool_call_id for a PTC-driven tool invocation."""
    return f"ptc_{tool_name}_{uuid.uuid4().hex[:8]}"


def inject_tool_args(
    tool: Any,
    payload: dict[str, Any],
    outer_runtime: Any,
    tool_call_id: str,
) -> dict[str, Any]:
    """Mirror LangGraph's `ToolNode._inject_tool_args` for PTC calls.

    LangChain tools that declare `ToolRuntime` / `InjectedState` /
    `InjectedStore` only see those values when a real `ToolNode` wires
    them in. PTC calls bypass it, so we replicate the detection logic here.
    The outer runtime (captured from the active `eval` tool invocation)
    provides state/store/context/config; `tool_call_id` is freshly minted
    per sub-call. `InjectedToolCallId` is handled separately via
    `BaseTool.arun(..., tool_call_id=...)` at the bridge site.
    """
    enriched = dict(payload)

    injected = _get_all_injected_args(tool)
    if not injected or outer_runtime is None:
        return enriched

    # Build a ToolRuntime matching the outer one but with a fresh
    # tool_call_id. `type(outer_runtime)` rather than a literal import
    # so the shape stays in lockstep with whatever langgraph ships.
    derived = type(outer_runtime)(
        state=outer_runtime.state,
        tool_call_id=tool_call_id,
        config=outer_runtime.config,
        context=outer_runtime.context,
        store=outer_runtime.store,
        stream_writer=outer_runtime.stream_writer,
        tools=outer_runtime.tools,
        execution_info=getattr(outer_runtime, "execution_info", None),
        server_info=getattr(outer_runtime, "server_info", None),
    )

    if injected.runtime:
        enriched[injected.runtime] = derived
    # InjectedState: state can be injected under one or more arg names.
    if injected.state:
        for arg_name, state_field in injected.state.items():
            if state_field:
                enriched[arg_name] = (
                    outer_runtime.state.get(state_field)
                    if isinstance(outer_runtime.state, dict)
                    else getattr(outer_runtime.state, state_field, None)
                )
            else:
                enriched[arg_name] = outer_runtime.state
    if injected.store and outer_runtime.store is not None:
        enriched[injected.store] = outer_runtime.store
    return enriched


def uses_injected_tool_call_id(tool: Any) -> bool:
    """Return whether a tool declares an injected tool call ID."""
    schema_annotations = get_all_basemodel_annotations(tool.get_input_schema())
    func = getattr(tool, "func", None) or getattr(tool, "coroutine", None)
    func_annotations = get_type_hints(func, include_extras=True) if func else {}
    return any(
        _is_injected_arg_type(type_, injected_type=InjectedToolCallId)
        for type_ in {**func_annotations, **schema_annotations}.values()
    )


def bridge_symbol_name(tool_name: str) -> str:
    """Build a stable, JS-safe global symbol for one PTC bridge."""
    # Keep only identifier-safe characters and salt with a short hash to
    # avoid collisions when different source names sanitize similarly.
    normalized = "".join(
        c if c.isalnum() or c in {"_", "$"} else "_" for c in tool_name
    )
    if not normalized or normalized[0].isdigit():
        normalized = f"_{normalized}"
    digest = hashlib.sha256(tool_name.encode("utf-8")).hexdigest()[:8]
    return f"__tools_{normalized}_{digest}"


def render_tools_namespace_assignment(bridges: dict[str, str]) -> str:
    """Return JS that atomically rebuilds `globalThis.tools` from bridges."""
    statements = ["globalThis.tools = {};"]
    for tool_name, bridge_symbol in sorted(bridges.items()):
        quoted_tool_name = json.dumps(tool_name)
        quoted_bridge_symbol = json.dumps(bridge_symbol)
        statements.append(
            "globalThis.tools"
            f"[{quoted_tool_name}] = globalThis[{quoted_bridge_symbol}];"
        )
    statements.append("undefined")
    return " ".join(statements)
