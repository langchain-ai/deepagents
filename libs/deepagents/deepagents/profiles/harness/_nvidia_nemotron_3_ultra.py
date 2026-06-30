"""Built-in NVIDIA Nemotron 3 Ultra harness profile.

Registers a `HarnessProfile` for NVIDIA Nemotron 3 Ultra
(`nvidia/nemotron-3-ultra-550b-a55b`) that pairs a behavior-shaping
`system_prompt_suffix` with three middleware:

- `NemotronToolMessageShim`: `ChatNVIDIA`'s payload builder rejects a `role="tool"`
  message whose content normalizes to null (an empty string collapses to null),
  so an empty tool result crashes the run. The shim coerces empty/None tool
  content to a non-empty placeholder. It is a no-op on non-empty results.
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

from deepagents.profiles.harness._fireworks_glm_5p2_middleware import (
    FinalizeMiddleware,
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


class NemotronToolMessageShim(AgentMiddleware):
    """Coerce an empty/None ToolMessage content to a non-empty placeholder.

    Cures the Nemotron content=None crash without touching generation settings or
    any other message role. Idempotent; a no-op for non-empty tool results.
    Implements BOTH sync and async hooks: Deep Agents executes tools
    asynchronously, so a sync-only `wrap_tool_call` raises `NotImplementedError`
    and breaks every async tool call. Both paths share `_normalize`.
    """

    name = "NemotronToolMessageShim"

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
        return self._normalize(handler(request))

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        return self._normalize(await handler(request))


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
    back to the model. Fires at most once per run via a `PrivateStateAttr` flag, and is
    backstopped by `FinalizeMiddleware`'s hard turn cap, so it cannot loop forever. This
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
  complete setup — is expected, not a violation. Once your output is computed and
  cross-checked, record it and stop; don't launch another long run just to
  re-confirm a result you've already validated.
</verification_discipline>

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
</work_in_batches>

<completion>
Do only what the task requires, then stop and give a concise final answer.
</completion>

<clarification>
When a data-analysis request is underspecified, ask for both missing dimensions before proceeding:
1. the data source or data itself, such as a file path, pasted data, dataset, or database connection;
2. the analysis goal or type, such as summary statistics, trends, anomalies, comparisons, segments, forecasts, or a specific question to answer.
</clarification>

<minimal_followups>
Ask only for missing information. Do not ask for details the user already supplied; treat explicit cadence, recipients, data, or destination as known and avoid re-asking for them.
</minimal_followups>

<context_compaction>
If a long conversation switches to a completely unrelated new task and the compact_conversation tool is available, call compact_conversation before starting the new task.
</context_compaction>

<final_answer_completeness>
After tool calls succeed, the final answer must report the concrete result, not just that the task is done. Include the key entity, action, identifier, title, recipient, service, status, or value that answers the user's request. If the user asked multiple questions, answer each one from its matching tool output; do not substitute an entity from a different subtask.
</final_answer_completeness>

<followup_defaults>
Ask follow-up questions only for information needed to proceed safely or correctly. Do not re-ask for constraints the user already gave.

When the user asks broadly for a summary, report, analysis, review, or search over available items, assume the broad scope unless they mention a filter; ask only about output format, level of detail, destination, or unavailable access.

When the user asks for a recurring task and gives a recurrence such as daily, weekly, monthly, or every N days, treat that as enough cadence and do not ask for day, time, timezone, or cadence again unless scheduling cannot proceed without it.

When advising on an operational workflow, support process, automation, or routing setup, ask what product, domain, or workflow is being supported if that context is missing.
</followup_defaults>"""
"""Text appended to the assembled base system prompt."""


def _build_extra_middleware() -> list[AgentMiddleware]:
    """Build fresh middleware instances for each assembled stack.

    Used as the profile's ``extra_middleware`` factory so each stack (main agent,
    general-purpose subagent, declarative subagents) gets its own instances rather
    than sharing per-run state — `FinalizeMiddleware` and `RambleMiddleware` (pulled
    from the GLM-5.2 profile) track per-run nudge state.
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
        NemotronToolMessageShim(),
        NemotronTextToolCallParser(),
        FinalizeMiddleware(),
        RambleMiddleware(),
        VerifyBeforeFinalizeMiddleware(),
        StallBreakerMiddleware(),
    ]


def register() -> None:
    """Register the built-in Nemotron 3 Ultra harness profile (suffix + middleware)."""
    profile = HarnessProfile(
        system_prompt_suffix=_SYSTEM_PROMPT_SUFFIX,
        extra_middleware=_build_extra_middleware,
    )
    for spec in _NEMOTRON_ULTRA_MODEL_SPECS:
        _register_harness_profile_impl(spec, profile)
