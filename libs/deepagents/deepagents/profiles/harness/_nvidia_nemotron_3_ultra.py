"""Built-in NVIDIA Nemotron 3 Ultra harness profile.

Registers a `HarnessProfile` for NVIDIA Nemotron 3 Ultra
(`nvidia/nemotron-3-ultra-550b-a55b`) that pairs a behavior-shaping
`system_prompt_suffix` with three middleware.

Unlike the other built-in profiles (which are prompt-suffix only), this one also
ships middleware. The suffix and middleware were tuned and measured *together* on
the Deep Agents behavioral eval suite and have not been measured apart, so they
are kept as one bundle:

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

from typing import TYPE_CHECKING

from langchain.agents.middleware import ToolRetryMiddleware
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import ToolMessage

from deepagents.profiles.harness.harness_profiles import (
    HarnessProfile,
    _register_harness_profile_impl,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from typing import Any

    from langchain.agents.middleware.types import ToolCallRequest
    from langgraph.types import Command

_NEMOTRON_ULTRA_MODEL_SPECS: tuple[str, ...] = (
    "NVIDIA:nvidia/nemotron-3-ultra-550b-a55b",
    "nvidia:nvidia/nemotron-3-ultra-550b-a55b",
)
"""Model specs that receive the Nemotron 3 Ultra harness profile.

`ChatNVIDIA`'s LangSmith provider is `NVIDIA` (capitalized), so a constructed
instance resolves to `NVIDIA:<model id>`; an `init_chat_model("nvidia:...")` spec
resolves to `nvidia:<model id>`. Both are registered so the profile applies
regardless of how the model was provided.
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


def _message_text(content: str | list[Any] | None) -> str:
    """Return readable text from common LangChain message content shapes."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content)


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
        content = _message_text(result.content)
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
            f"\n\n[read_file returned {limit} lines starting at offset {offset} — the "
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


_SYSTEM_PROMPT_SUFFIX: str = """<approach>
Plan briefly before acting. When several reads or lookups are independent, issue them as parallel tool calls rather than one at a time.
</approach>

<grounding>
Verify state with tools instead of recalling it: read a file before describing it, run a check before claiming a result. After each tool result, reflect briefly before choosing the next action.
</grounding>

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


def register() -> None:
    """Register the built-in Nemotron 3 Ultra harness profile (suffix + middleware)."""
    profile = HarnessProfile(
        system_prompt_suffix=_SYSTEM_PROMPT_SUFFIX,
        extra_middleware=[
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
        ],
    )
    for spec in _NEMOTRON_ULTRA_MODEL_SPECS:
        _register_harness_profile_impl(spec, profile)
