"""Built-in NVIDIA Nemotron 3 Ultra harness profile.

Registers a `HarnessProfile` for NVIDIA Nemotron 3 Ultra
(`nvidia/nemotron-3-ultra-550b-a55b`) that pairs a behavior-shaping
`system_prompt_suffix` with three middleware:

- `NemotronToolCallShim`: fixes two Nemotron tool-call payload quirks at the
  `wrap_tool_call` layer — (1) remaps a stray `path` arg to the schema-required
  `file_path` for `read_file`/`write_file`/`edit_file` (Nemotron often uses the
  wrong key and burns turns on the validation error), and (2) coerces empty/None
  tool-result content to a non-empty placeholder (`ChatNVIDIA` rejects null
  `role="tool"` content, crashing the run). No-op on well-formed calls/results.
- `ReadFileContinuationNoticeMiddleware`: the `read_file` line-limit path returns
  exactly `limit` lines with no truncation signal (the truncation message only
  fires on the token-size path), so the model can assume it has seen the whole
  file. This appends an in-result continuation notice.
- `ToolRetryMiddleware` (stock LangChain): retries a failed filesystem tool call
  once.

Per-model keys (not the `"NVIDIA"`/`"nvidia"` provider prefix) keep other
NVIDIA-catalog models unchanged. Both provider casings are registered: a
`ChatNVIDIA` instance resolves to `NVIDIA:<id>` (its LangSmith provider is
capitalized) while an `init_chat_model("nvidia:...")` spec resolves to
`nvidia:<id>`.

Source: https://developer.nvidia.com/blog/nvidia-nemotron-3-ultra-powers-faster-more-efficient-reasoning-for-long-running-agents/
"""

# ruff: noqa: E501
# The prompt suffix is kept verbatim from the tuned/measured profile, and the
# middleware classes are inlined so the profile is self-contained.

from __future__ import annotations

import re
import uuid
from typing import TYPE_CHECKING, Annotated, NotRequired

from langchain.agents.middleware import ToolRetryMiddleware
from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ExtendedModelResponse,
    ModelResponse,
    PrivateStateAttr,
    hook_config,
)
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.config import get_config

from deepagents.profiles.harness._fireworks_glm_5p2_middleware import (
    RambleMiddleware,
)
from deepagents.profiles.harness.harness_profiles import (
    HarnessProfile,
    _register_harness_profile_impl,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from typing import Any

    from langchain.agents.middleware.types import ModelCallResult, ModelRequest, ToolCallRequest
    from langchain_core.messages.tool import ToolCall
    from langgraph.runtime import Runtime
    from langgraph.types import Command

_NEMOTRON_ULTRA_MODEL_SPECS: tuple[str, ...] = (
    # NVIDIA's own API (ChatNVIDIA): instance form is capitalized `NVIDIA`,
    # the init_chat_model spec form is lowercase `nvidia`.
    "NVIDIA:nvidia/nemotron-3-ultra-550b-a55b",
    "nvidia:nvidia/nemotron-3-ultra-550b-a55b",
    # Nemotron 3 Ultra as served by other providers.
    "fireworks:accounts/fireworks/models/nemotron-3-ultra-nvfp4",
    "fireworks:accounts/fireworks/models/nemotron-3-ultra-bf16",
    "baseten:nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B",
    "openrouter:nvidia/nemotron-3-ultra-550b-a55b",
)
"""Model specs that receive the Nemotron 3 Ultra harness profile.

Registered per-model (not provider-wide), so other models on these providers are
unchanged. Each key is `<ls_provider>:<model id>`, and the provider segment must
match what the chat-model client reports as its LangSmith provider, so casing
matters: `ChatNVIDIA` reports `NVIDIA`, while an `init_chat_model("nvidia:...")`
spec is lowercase `nvidia` (both are registered). Covers NVIDIA's own API plus
Nemotron 3 Ultra as served by Fireworks, Baseten, and OpenRouter.
"""

_FILESYSTEM_TOOLS: tuple[str, ...] = ("ls", "read_file", "write_file", "edit_file", "glob", "grep")
"""Tools the scoped tool-retry applies to."""

_EMPTY_TOOL_PLACEHOLDER = "(empty tool result)"


def _tool_content_is_empty(content: str | list[Any] | None) -> bool:
    """True if `content` would serialize to empty/None (mirrors ChatNVIDIA content normalization)."""
    if content is None:
        return True
    if isinstance(content, str):
        return content == ""
    if isinstance(content, list):
        # Non-empty if any block is non-text (multimodal is preserved) or has text.
        for block in content:
            if not (isinstance(block, dict) and block.get("type") == "text"):
                return False
            if block.get("text"):
                return False
        return True
    return False


# Filesystem tools whose schema requires `file_path`. Nemotron frequently calls
# these with the key `path` instead, so the shim below remaps it.
_FILE_PATH_TOOLS: frozenset[str] = frozenset({"read_file", "write_file", "edit_file"})


class NemotronToolCallShim(AgentMiddleware):
    """Fix Nemotron's tool-call payload quirks at the `wrap_tool_call` layer.

    Two Nemotron-specific tool-call compatibility fixes, kept in one interceptor
    (same hook, same concern) rather than separate middleware:

    - Request side (`_fix_args`): Nemotron frequently calls `read_file`/`write_file`/
      `edit_file` with `{"path": ...}` instead of the schema-required
      `{"file_path": ...}`, and does not self-correct — it re-issues the same wrong
      key and burns turns on the `pydantic ValidationError: file_path Field
      required`. This renames the key before the tool runs (scoped to those tools, so
      `ls` etc. keep their legitimate `path`); the tool's own path validation still
      runs on the value. Covers structured and text-parsed calls. No-op when
      `file_path` is already present or `path` is absent.
    - Result side (`_normalize`): `ChatNVIDIA`'s payload builder rejects a
      `role="tool"` message whose content normalizes to null (an empty string
      collapses to null), crashing the run. This coerces empty/None tool content to
      a non-empty placeholder. Idempotent; a no-op for non-empty results.

    Implements BOTH sync and async hooks: Deep Agents executes tools asynchronously,
    so a sync-only `wrap_tool_call` raises `NotImplementedError` and breaks every
    async tool call.
    """

    name = "NemotronToolCallShim"

    @staticmethod
    def _fix_args(request: ToolCallRequest) -> ToolCallRequest:
        tool_call = request.tool_call
        if tool_call.get("name") not in _FILE_PATH_TOOLS:
            return request
        args = tool_call.get("args") or {}
        if "path" not in args or "file_path" in args:
            return request
        new_args = {k: v for k, v in args.items() if k != "path"}
        new_args["file_path"] = args["path"]
        return request.override(tool_call={**tool_call, "args": new_args})

    @staticmethod
    def _normalize(result: ToolMessage | Command[Any]) -> ToolMessage | Command[Any]:
        if isinstance(result, ToolMessage) and _tool_content_is_empty(result.content):
            return result.model_copy(update={"content": _EMPTY_TOOL_PLACEHOLDER})
        return result

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        return self._normalize(handler(self._fix_args(request)))

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        return self._normalize(await handler(self._fix_args(request)))


_READ_NOTICE_DEFAULT_LIMIT = 100  # Deep Agents default read limit (FilesystemMiddleware).


class ReadFileContinuationNoticeMiddleware(AgentMiddleware):
    """Append a continuation notice to exactly-at-limit `read_file` results.

    The `read_file` line-limit path returns a bare slice with no truncation
    signal (the truncation message only fires on the token-size path), and the
    stock tool description implies omitting `limit` reads the whole file while the
    backend still caps at the default read limit. A model that receives exactly
    `limit` lines therefore has no way to know the file continues. This restores
    the missing signal in the tool result itself. It only annotates one
    tool-result string; the model still has to issue the follow-up reads.
    """

    name = "ReadFileContinuationNoticeMiddleware"

    @staticmethod
    def _annotate(request: ToolCallRequest, result: ToolMessage | Command[Any]) -> ToolMessage | Command[Any]:
        if not isinstance(result, ToolMessage):
            return result
        if request.tool_call.get("name") != "read_file":
            return result
        content = result.text
        if not content or content.startswith("Error"):
            return result
        args = request.tool_call.get("args", {}) or {}
        try:
            offset = int(args.get("offset") or 0)
        except (TypeError, ValueError):
            offset = 0
        try:
            limit = int(args.get("limit") or _READ_NOTICE_DEFAULT_LIMIT)
        except (TypeError, ValueError):
            limit = _READ_NOTICE_DEFAULT_LIMIT
        # Count source lines, not rendered rows: read_file uses cat -n format and
        # splits long lines into continuation rows (e.g. "5.1") that do NOT count
        # against the source-line `limit`. Source-line rows have a bare-integer
        # line-number prefix; continuation rows have a "<int>.<int>" prefix.
        n_lines = sum(1 for row in content.split("\n") if "\t" in row and row.split("\t", 1)[0].strip().isdigit())
        if n_lines < limit:
            return result
        notice = (
            f"\n\n[read_file returned {limit} lines starting at offset {offset}, the "
            f"per-read limit. The file likely continues past this window. To read "
            f"further, call read_file again with offset={offset + limit}. Do not assume "
            f"you have seen the end of the file.]"
        )
        return result.model_copy(update={"content": content + notice})

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        return self._annotate(request, handler(request))

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        return self._annotate(request, await handler(request))


_FUNCTION_BLOCK_RE = re.compile(r"<function=([^>\s]+)\s*>(.*?)</function>", re.DOTALL)
"""Matches a `<function=NAME> ... </function>` text tool-call block."""

_PARAMETER_RE = re.compile(r"<parameter\s+name=([^>\s]+)\s*>(.*?)</parameter>", re.DOTALL)
"""Matches a `<parameter name=KEY>VALUE</parameter>` pair inside a function block."""


def _parse_text_tool_calls(content: str) -> tuple[list[ToolCall], str]:
    """Parse Nemotron's text-format tool calls into structured tool calls.

    Some Nemotron deployments (notably via OpenRouter) intermittently emit a tool
    call as message *content* using a `<function=NAME><parameter name=K>V</parameter>
    </function>` template instead of a structured tool call, so the agent never
    executes it. This converts each such block into a LangChain ``ToolCall``. The
    template is tolerated loosely: unquoted `name=`, multi-line values, stray
    nameless `<parameter>` openers, and an orphan `</tool_call>` are all handled.

    Args:
        content: The model message text.

    Returns:
        A ``(tool_calls, leftover_content)`` pair. ``tool_calls`` is empty (and the
        content returned unchanged) when no `<function=...>` block is present.
    """
    calls: list[ToolCall] = []
    for block in _FUNCTION_BLOCK_RE.finditer(content):
        name = block.group(1).strip("\"'")
        args = {param.group(1).strip("\"'"): param.group(2).strip() for param in _PARAMETER_RE.finditer(block.group(2))}
        calls.append({"name": name, "args": args, "id": uuid.uuid4().hex, "type": "tool_call"})
    if not calls:
        return [], content
    leftover = _FUNCTION_BLOCK_RE.sub("", content).replace("</tool_call>", "").strip()
    return calls, leftover


class NemotronTextToolCallParser(AgentMiddleware):
    """Repair tool calls the model emits as text content instead of structured calls.

    When the model returns an ``AIMessage`` with no structured ``tool_calls`` but a
    `<function=...>` block in its content, parse that block into structured
    ``tool_calls`` so the agent executes it rather than stalling on unrun text.
    Messages that already carry structured ``tool_calls`` (the common case) and
    ordinary prose answers are returned untouched.
    """

    name = "NemotronTextToolCallParser"

    @staticmethod
    def _repair_message(message: AIMessage) -> AIMessage:
        if message.tool_calls:
            return message
        content = message.content
        text = content if isinstance(content, str) else "".join(part.get("text", "") for part in content if isinstance(part, dict))
        if "<function=" not in text:
            return message
        calls, leftover = _parse_text_tool_calls(text)
        if not calls:
            return message
        return message.model_copy(update={"tool_calls": calls, "content": leftover})

    @staticmethod
    def _repair_response(response: ModelResponse) -> ModelResponse:
        return ModelResponse(
            result=[NemotronTextToolCallParser._repair_message(m) if isinstance(m, AIMessage) else m for m in response.result],
            structured_response=response.structured_response,
        )

    @staticmethod
    def _repair(result: ModelCallResult) -> ModelCallResult:
        if isinstance(result, ExtendedModelResponse):
            return ExtendedModelResponse(
                model_response=NemotronTextToolCallParser._repair_response(result.model_response),
                command=result.command,
            )
        if isinstance(result, AIMessage):
            return NemotronTextToolCallParser._repair_message(result)
        if isinstance(result, ModelResponse):
            return NemotronTextToolCallParser._repair_response(result)
        return result

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelCallResult],
    ) -> ModelCallResult:
        return self._repair(handler(request))

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelCallResult]],
    ) -> ModelCallResult:
        return self._repair(await handler(request))


_STALL_THRESHOLD = 3
"""Consecutive identical failing tool results that count as a stall."""

_STALL_FAILURE_MARKERS: tuple[str, ...] = (
    "error",
    "traceback",
    "command failed",
    "no such",
    "not found",
    "exit code 1",
    "failed",
)

_STALL_NUDGE_TEXT = (
    "You have hit the same failure several times in a row and keep retrying essentially "
    "the same fix. Stop repeating it — the identical approach will not start working. "
    "Step back and either rewrite the failing component from scratch or take a "
    "fundamentally different approach, and re-examine your assumptions about the cause."
)


class StallBreakerMiddleware(AgentMiddleware):
    """Break no-progress loops where the model re-tries the same failing action.

    When the last `_STALL_THRESHOLD` tool results are all failures with the same
    signature (the model keeps applying the same fix to the same error), inject a
    one-time nudge before the next model turn telling it to change approach. General
    and model-agnostic: keys only on repeated identical failing tool output, not on
    any task. A no-op when results vary, succeed, or progress.
    """

    name = "StallBreakerMiddleware"

    @staticmethod
    def _text(content: str | list[Any] | None) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
        return ""

    @staticmethod
    def _is_failure(text: str) -> bool:
        low = text.lower()
        return any(marker in low for marker in _STALL_FAILURE_MARKERS)

    @staticmethod
    def _signature(text: str) -> str:
        return " ".join(text.split())[:200]

    def _nudge(self, state: AgentState) -> dict[str, Any] | None:
        messages = state.get("messages") or []
        tool_texts = [self._text(m.content) for m in messages if isinstance(m, ToolMessage)]
        recent = tool_texts[-_STALL_THRESHOLD:]
        if len(recent) < _STALL_THRESHOLD:
            return None
        if not all(self._is_failure(t) for t in recent):
            return None
        if len({self._signature(t) for t in recent}) != 1:
            return None
        # Don't re-nudge within the same streak: skip if our nudge is already nearby.
        window = messages[-(2 * _STALL_THRESHOLD + 2) :]
        if any(isinstance(m, HumanMessage) and self._text(m.content) == _STALL_NUDGE_TEXT for m in window):
            return None
        return {"messages": [HumanMessage(_STALL_NUDGE_TEXT)]}

    def before_model(
        self,
        state: AgentState,
        runtime: Runtime[Any],  # noqa: ARG002  (part of the hook signature)
    ) -> dict[str, Any] | None:
        return self._nudge(state)

    async def abefore_model(
        self,
        state: AgentState,
        runtime: Runtime[Any],  # noqa: ARG002  (part of the hook signature)
    ) -> dict[str, Any] | None:
        return self._nudge(state)


_VERIFY_NUDGE_TEXT = (
    "Before you finish: you have not yet proven this works. Do not stop yet. Re-read the "
    "task's exact wording — every required output file, path, name, format, and every "
    "specified case, mode, size, or count — and PROVE each requirement holds by running a "
    "concrete command in the shell and reading its output. Verify the real artifact the "
    "way it will be checked, not a proxy and not your own prior reasoning; a claim like "
    "'all tests pass' or 'the file is correct' is not evidence unless a command you just "
    "ran shows it. If any check fails or a required artifact is missing or misnamed, fix "
    "it and re-verify. Only once a command has demonstrated the deliverable meets the "
    "spec may you finish."
)


class VerifyBeforeFinalizeState(AgentState):
    """State schema for ``VerifyBeforeFinalizeMiddleware``."""

    # `PrivateStateAttr` keeps the gate flag out of the input/output schema while it
    # persists internally across turns (default last-value reducer).
    verify_gate_fired: NotRequired[Annotated[bool, PrivateStateAttr]]


class VerifyBeforeFinalizeMiddleware(AgentMiddleware):
    """Force one verification pass before the agent is allowed to finish.

    Targets the dominant observed failure mode: overconfident false completion, where
    the model ends with a confident "done / verified / all checks pass" while the grader
    disagrees (e.g. claiming "zero overfull hbox warnings" when its own compile output
    showed one). When the latest turn is a finalization — an ``AIMessage`` with no tool
    calls and non-empty content, i.e. the agent is about to end — this injects a
    just-in-time demand to verify each requirement by running a real command, and jumps
    back to the model. Fires at most once per run via a `PrivateStateAttr` flag, so it
    adds a single verify turn and cannot loop (no external backstop is required). This
    is a control-flow gate at the finish line, not standing prompt advice.
    """

    name = "VerifyBeforeFinalizeMiddleware"
    state_schema = VerifyBeforeFinalizeState  # type: ignore[assignment]

    @staticmethod
    def _is_finalizing(message: AIMessage) -> bool:
        if message.tool_calls:
            return False
        content = message.content
        if isinstance(content, str):
            return bool(content.strip())
        if isinstance(content, list):
            return any(isinstance(p, dict) and str(p.get("text", "")).strip() for p in content)
        return False

    def _gate(self, state: VerifyBeforeFinalizeState) -> dict[str, Any] | None:
        if state.get("verify_gate_fired"):
            return None
        messages = state.get("messages") or []
        if not messages:
            return None
        last = messages[-1]
        if not isinstance(last, AIMessage) or not self._is_finalizing(last):
            return None
        return {
            "messages": [HumanMessage(content=_VERIFY_NUDGE_TEXT)],
            "jump_to": "model",
            "verify_gate_fired": True,
        }

    @hook_config(can_jump_to=["model"])
    def after_model(
        self,
        state: VerifyBeforeFinalizeState,
        runtime: Runtime[Any],  # noqa: ARG002  (part of the hook signature)
    ) -> dict[str, Any] | None:
        return self._gate(state)

    @hook_config(can_jump_to=["model"])
    async def aafter_model(
        self,
        state: VerifyBeforeFinalizeState,
        runtime: Runtime[Any],  # noqa: ARG002  (part of the hook signature)
    ) -> dict[str, Any] | None:
        return self._gate(state)


_SYSTEM_PROMPT_SUFFIX: str = """<approach>
Plan briefly before acting. When several reads or lookups are independent, issue them as parallel tool calls rather than one at a time.
</approach>

<grounding>
Verify state with tools instead of recalling it: read a file before describing it, run a check before claiming a result. After each tool result, reflect briefly before choosing the next action.
</grounding>

<reasoning_discipline>
Reason only as much as needed to choose the next action, then act. If a tool call fails, read the error and change the call before retrying — never re-issue the same failing call unchanged.
</reasoning_discipline>

<verification_discipline>
Before treating a task as done:

- Cover every output and constraint. Re-read the request and list every output
  it names — each file path, and each field, section, format, name, ordering,
  value range, or "all vs. one" rule stated about it. Confirm each one against
  your work (`ls`, `cat`); a single missing output or unmet constraint leaves
  the task unfinished.

- Verify the real behavior, not a proxy. Exercise the actual required operation
  end-to-end against adversarial and boundary inputs — the specific scenarios,
  parameter names, and edge cases the task describes — not a happy-path case you
  picked yourself. A check that only runs inputs you chose can pass while the
  behavior is still wrong. Exercise every configuration the task specifies — each
  size, mode, count, or degree of parallelism or concurrency — never just a single
  trivial or degenerate case (a one-element, one-process, or empty run) where the
  logic collapses to a no-op and the real behavior is never tested.

- Make it reproducible from a clean state. Your work has to function for someone
  starting fresh, not only in the shell you built it in. A service must keep
  running on its own — a managed or persistent service, not a process tied to
  the shell that launched it (which dies when that shell exits). A script must
  run using only what is already installed; if you installed something ad hoc to
  make it work, it will fail elsewhere. Confirm it from a brand-new shell —
  restart the service, open a fresh session, re-run the script — not just where
  you built it.

- Stay within the task's scope. Modify only what the task asks you to produce or
  change, and don't reach into state it never named — don't rewrite shared
  history, migrate schemas, or regenerate or delete files you weren't asked to.
  This limits what you touch, not how much you do: doing the full work a task
  needs — installing packages, configuring and starting services, building out a
  complete setup — is expected, not a violation.
</verification_discipline>

<stop_condition>
Verify each result once, against the task's literal requirements — the exact names,
paths, and formats it states, not ones you chose yourself. A check that only
exercises your own invented interface proves nothing; confirm the real contract
first. When a spec-anchored check produces the expected output, record it and move
on — do not re-run the same check on the same inputs, and do not re-derive a value
you have already confirmed. Repeating a check that already passed cannot change the
answer; it only spends the turn budget you need to finish the task. Once the task's
required outputs exist on disk and that check has passed, write your final answer
and stop — do not open another round of edits or re-verification.
</stop_condition>

<work_in_batches>
When iterating — building, testing, debugging, or reverse-engineering — do as
much as possible per command rather than one probe per turn. Script the whole
cycle (build, run, check) so it prints one consolidated result you can act on,
instead of running a command, reading a single value, and stopping. When
inspecting an unknown file, binary, or data structure, extract the specific
values you need in one pass rather than querying them one at a time. If one step
is unavoidably long (a large training, sampling, or build run), start it in the
background with a timeout and poll for completion, rather than blocking on a
single multi-minute command.
</work_in_batches>"""
"""Text appended to the assembled base system prompt.

Coding-task disciplines only. Generic-assistant blocks (clarification, follow-up
defaults, context-compaction, final-answer-completeness) were removed: there is no
human to clarify with and no multi-task conversation in this harness, so they were
dead weight that diluted the coding-relevant guidance.
"""


# Escalating completion nudges keyed on how close the run is to the LangGraph
# ``recursion_limit`` (read live from the runnable config). Pure nudges: this never
# terminates the run — it lets the run end on its own at the recursion limit or the
# harness wall-clock timeout, and simply presses harder to secure a deliverable as
# the step budget runs down. A fixed turn cap fired far too early (~29% of the agent
# wall-clock budget on fast rollouts); measuring against the real step ceiling adapts
# to each run's per-turn cost. Tone escalates (intrinsic discipline early, firm
# consequence near the ceiling); lower tiers fire once when first crossed, the top
# tier repeats every turn as the ceiling nears.
_PRESSURE_TIERS: tuple[tuple[float, str], ...] = (
    (
        0.55,
        "You are past the halfway point of your step budget. Make sure every remaining "
        "action moves toward a concrete deliverable on disk — converge on the task's "
        "required output rather than exploring further.",
    ),
    (
        0.70,
        "You have used about 70% of your step budget. Write a working solution to the "
        "exact path the task requires now, then confirm it exists with `ls`/`cat` and "
        "run the task's own check against it. Do not open new investigations.",
    ),
    (
        0.82,
        "Step budget is running low (~82% used). Ship the best solution you have to the "
        "required path this turn — a finished best-effort artifact is a usable result, an "
        "unfinished one is not. Re-read your notes for the exact contract if unsure.",
    ),
    (
        0.90,
        "You are nearly out of step budget (~90% used). Write your final artifact to the "
        "required path immediately and confirm it exists. Do not refactor, re-verify, or "
        "explore — just secure the deliverable.",
    ),
    (
        0.96,
        "Final budget warning: your steps are almost gone. Output your best working "
        "artifact to the required path right now. Anything not written to disk when the "
        "run ends will not count — stop everything else and ship it.",
    ),
)

# Fallback when the runnable config does not expose step/limit (keeps the middleware
# functional off-graph or in stripped contexts): estimate the fraction from the
# model-turn count. ~8 graph supersteps per model turn under the dcode stack, against
# the dcode default ``recursion_limit`` of 1000.
_PRESSURE_STEPS_PER_TURN = 8
_PRESSURE_FALLBACK_LIMIT = 1000


class CompletionPressureState(AgentState):
    """State schema for ``CompletionPressureMiddleware``."""

    # `PrivateStateAttr` keeps these channels out of the input/output schema while
    # they persist internally across turns (default last-value reducer).
    pressure_turns: NotRequired[Annotated[int, PrivateStateAttr]]
    pressure_tier: NotRequired[Annotated[int, PrivateStateAttr]]


class CompletionPressureMiddleware(AgentMiddleware):
    """Escalating completion nudges as the run nears the LangGraph ``recursion_limit``.

    Replaces a hard turn cap: it never terminates the run (no ``jump_to``). Instead it
    measures the run's position in its ``recursion_limit`` step budget — read live from
    ``get_config()`` (``recursion_limit`` and ``metadata.langgraph_step``) — and injects
    increasingly urgent nudges to secure a deliverable. The run ends on its own at the
    recursion limit or the harness wall-clock timeout. Lower tiers fire once when first
    crossed; the top tier repeats every turn. If the config does not expose the step
    budget, it falls back to estimating the fraction from a model-turn counter.
    """

    name = "CompletionPressureMiddleware"
    state_schema = CompletionPressureState  # type: ignore[assignment]

    @staticmethod
    def _budget_fraction() -> float | None:
        """Fraction of the recursion step budget consumed, or ``None`` if unreadable."""
        try:
            config = get_config()
        except Exception:  # noqa: BLE001  (off-graph / no active config -> use fallback)
            return None
        limit = config.get("recursion_limit")
        metadata = config.get("metadata") or {}
        step = metadata.get("langgraph_step")
        if isinstance(limit, int) and limit > 0 and isinstance(step, int) and step >= 0:
            return step / limit
        return None

    def _pressure(self, state: CompletionPressureState) -> dict[str, Any]:
        turns = state.get("pressure_turns", 0) + 1
        fraction = self._budget_fraction()
        if fraction is None:
            fraction = min(turns * _PRESSURE_STEPS_PER_TURN / _PRESSURE_FALLBACK_LIMIT, 0.999)
        tier = -1
        for index, (threshold, _text) in enumerate(_PRESSURE_TIERS):
            if fraction >= threshold:
                tier = index
        base: dict[str, Any] = {"pressure_turns": turns}
        if tier < 0:
            return base
        last = state.get("pressure_tier", -1)
        is_top = tier == len(_PRESSURE_TIERS) - 1
        if is_top or tier > last:
            return {
                **base,
                "pressure_tier": max(tier, last),
                "messages": [HumanMessage(content=_PRESSURE_TIERS[tier][1])],
            }
        return {**base, "pressure_tier": last}

    def before_model(
        self,
        state: CompletionPressureState,
        runtime: Runtime[Any],  # noqa: ARG002  (part of the hook signature)
    ) -> dict[str, Any]:
        """Inject an escalating completion nudge based on recursion-budget usage."""
        return self._pressure(state)

    async def abefore_model(
        self,
        state: CompletionPressureState,
        runtime: Runtime[Any],  # noqa: ARG002  (part of the hook signature)
    ) -> dict[str, Any]:
        """Async variant of ``before_model``."""
        return self._pressure(state)


# Injected as a HumanMessage on the first model turn. Nemotron under-weights standing
# system-prompt guidance (the "reproduce identifiers verbatim" rule was ignored even in
# a prominent base) and did not act on the same discipline appended to tool results
# either — so this front-loads it as an in-conversation message before the first
# action, forcing an explicit "extract the exact identifiers, then use them" step.
# General wording (no task-specific example) so it does not overfit.
_SPEC_FIDELITY_NUDGE = (
    "STOP — before writing any code, file, or command, do this first and treat it as "
    "mandatory, not optional:\n\n"
    "1. Re-read the task and copy out, verbatim and character-for-character, EVERY "
    "identifier it names: message / field / RPC / class / function / variable names, "
    "file paths and filenames, ports, and output formats. Record them (e.g. in a notes "
    "file) as the authoritative contract you will build against.\n"
    "2. Build strictly by copying from that list. Every name you produce must match the "
    "task's spelling EXACTLY — not close, exact. One wrong character means you have "
    "built a different interface than the one asked for: anything that relies on the "
    "task's names will fail to find yours, no matter how well your code otherwise runs.\n"
    "3. NEVER normalize, unify, rename, abbreviate, pluralize, re-case, or otherwise "
    '"tidy" the task\'s names — even when they look inconsistent, redundant, or wrong to '
    "you. If the task spells two related things differently, that is deliberate; "
    "reproduce each exactly as written. The task's wording is the authority, not your "
    "sense of what is consistent or correct. When unsure, copy the task's text — never "
    "retype an identifier from memory or paraphrase it."
)


class SpecFidelityState(AgentState):
    """State schema for ``SpecFidelityMiddleware``."""

    # `PrivateStateAttr` keeps the once-flag out of the input/output schema while it
    # persists internally across turns (default last-value reducer).
    spec_fidelity_injected: NotRequired[Annotated[bool, PrivateStateAttr]]


class SpecFidelityMiddleware(AgentMiddleware):
    """Inject a one-time spec-fidelity reminder as a `HumanMessage` on the first turn.

    Nemotron under-weights standing system-prompt guidance — the "reproduce every
    identifier verbatim" rule was ignored even when stated prominently — and did not
    act on the same discipline appended to `write_file`/`edit_file`/`execute` results.
    This delivers it as an in-conversation `HumanMessage` before the first model action,
    front-loading the "extract the exact identifiers, then use them verbatim" step.
    Fires once per run via a `PrivateStateAttr`.
    """

    name = "SpecFidelityMiddleware"
    state_schema = SpecFidelityState  # type: ignore[assignment]

    def _maybe_inject(self, state: SpecFidelityState) -> dict[str, Any]:
        if state.get("spec_fidelity_injected"):
            return {}
        return {
            "messages": [HumanMessage(content=_SPEC_FIDELITY_NUDGE)],
            "spec_fidelity_injected": True,
        }

    def before_model(
        self,
        state: SpecFidelityState,
        runtime: Runtime[Any],  # noqa: ARG002  (part of the hook signature)
    ) -> dict[str, Any]:
        """Inject the spec-fidelity reminder on the first turn only."""
        return self._maybe_inject(state)

    async def abefore_model(
        self,
        state: SpecFidelityState,
        runtime: Runtime[Any],  # noqa: ARG002  (part of the hook signature)
    ) -> dict[str, Any]:
        """Async variant of ``before_model``."""
        return self._maybe_inject(state)


def _build_extra_middleware() -> list[AgentMiddleware]:
    """Build fresh middleware instances for each assembled stack.

    Used as the profile's ``extra_middleware`` factory so each stack (main agent,
    general-purpose subagent, declarative subagents) gets its own instances rather
    than sharing per-run state — `CompletionPressureMiddleware` and `RambleMiddleware`
    (the latter pulled from the GLM-5.2 profile) track per-run nudge state.
    """
    return [
        ReadFileContinuationNoticeMiddleware(),
        ToolRetryMiddleware(
            max_retries=1,
            tools=list(_FILESYSTEM_TOOLS),
            on_failure="continue",
            initial_delay=0.0,
            backoff_factor=1.0,
            max_delay=0.0,
            jitter=False,
        ),
        NemotronToolCallShim(),
        SpecFidelityMiddleware(),
        NemotronTextToolCallParser(),
        CompletionPressureMiddleware(),
        RambleMiddleware(),
        VerifyBeforeFinalizeMiddleware(),
        StallBreakerMiddleware(),
    ]


# Replaces the SDK's generic "deep agent" BASE persona (which carries assistant-chat
# guidance — Clarifying Requests, Progress Updates, follow-up defaults — irrelevant to
# a headless coding benchmark) with a lean coding base. Sits between the caller
# instructions (USER) and the tuned SUFFIX, and reinforces spec-fidelity (the exact-
# identifier discipline) at that mid-prompt position rather than only at the top.
_BASE_SYSTEM_PROMPT = """You are an autonomous software engineer working in a sandbox with shell and filesystem tools. Work in a tight loop: understand the task, implement, then verify against the task's own wording before finishing — your first attempt is rarely right, so iterate rather than declaring done.

Match the task's contract exactly. Every identifier, field name, file path, and output format must be reproduced character-for-character as the task states it — copy them verbatim. Never rename, normalize, or "tidy" a naming inconsistency you notice: if the task calls one field `value` and another `val`, use exactly those. A schema that looks inconsistent to you is still the schema.

Read a file before editing it, and match the surrounding style."""


def register() -> None:
    """Register the built-in Nemotron 3 Ultra harness profile (base + suffix + middleware)."""
    profile = HarnessProfile(
        base_system_prompt=_BASE_SYSTEM_PROMPT,
        system_prompt_suffix=_SYSTEM_PROMPT_SUFFIX,
        extra_middleware=_build_extra_middleware,
    )
    for spec in _NEMOTRON_ULTRA_MODEL_SPECS:
        _register_harness_profile_impl(spec, profile)
