"""Built-in NVIDIA Nemotron 3 Ultra harness profile.

The profile is scoped to Nemotron 3 Ultra model specs. It adds lightweight
prompt guidance and middleware for observed tool-calling, filesystem,
context-management, and final-answer behaviors in agentic workloads.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from typing import TYPE_CHECKING, Annotated, Any, NotRequired, cast

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
from langchain_core.messages.utils import _convert_to_openai_tool_calls

from deepagents.profiles.harness.harness_profiles import (
    HarnessProfile,
    _register_harness_profile_impl,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain.agents.middleware.types import ModelCallResult, ModelRequest, ToolCallRequest
    from langchain_core.messages.tool import ToolCall
    from langgraph.runtime import Runtime
    from langgraph.types import Command

_NEMOTRON_ULTRA_MODEL_SPECS: tuple[str, ...] = (
    # NVIDIA's own API: `ChatNVIDIA` instances resolve to capitalized
    # `NVIDIA`, while `init_chat_model("nvidia:...")` specs use lowercase
    # `nvidia`.
    "NVIDIA:nvidia/nemotron-3-ultra-550b-a55b",
    "nvidia:nvidia/nemotron-3-ultra-550b-a55b",
    # Nemotron 3 Ultra as served by other providers.
    "baseten:nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B",
    "fireworks:accounts/fireworks/models/nemotron-3-ultra-nvfp4",
    "fireworks:accounts/fireworks/models/nemotron-3-ultra-bf16",
    "openrouter:nvidia/nemotron-3-ultra-550b-a55b",
    "nebius:nvidia/Nemotron-3-Ultra-550b-a55b",
    "together:nvidia/nemotron-3-ultra-550b-a55b",
)
"""Model specs that receive the Nemotron 3 Ultra harness profile.

Registered per model, not provider-wide, so other models on these providers are
unchanged. Each key must match the provider and model identifier reported by the
chat-model client.
"""

_FILESYSTEM_TOOLS: tuple[str, ...] = ("ls", "read_file", "write_file", "edit_file", "delete", "glob", "grep")
_FILE_PATH_TOOLS: frozenset[str] = frozenset({"read_file", "write_file", "edit_file", "delete"})
_EMPTY_TOOL_PLACEHOLDER = "(empty tool result)"
_DEFAULT_READ_LIMIT = 500
_HTTP_TOO_MANY_REQUESTS = 429


def _tool_content_is_empty(content: str | list[Any] | None) -> bool:
    if content is None:
        return True
    if isinstance(content, str):
        return content == ""
    if isinstance(content, list):
        for block in content:
            if not (isinstance(block, dict) and block.get("type") == "text"):
                return False
            if block.get("text"):
                return False
        return True
    return False


class NemotronToolCallShim(AgentMiddleware):
    """Repair small Nemotron filesystem tool-call and tool-result quirks."""

    name = "NemotronToolCallShim"

    @staticmethod
    def _fix_args(request: ToolCallRequest) -> ToolCallRequest:
        tool_call = request.tool_call
        name = tool_call.get("name")
        if name not in _FILE_PATH_TOOLS:
            return request

        new_args = dict(tool_call.get("args") or {})
        changed = False
        if "path" in new_args and "file_path" not in new_args:
            new_args["file_path"] = new_args.pop("path")
            changed = True
        if name == "read_file" and "limit" not in new_args:
            new_args["limit"] = _DEFAULT_READ_LIMIT
            changed = True
        if not changed:
            return request
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
        """Repair the request, run the tool, and normalize empty results."""
        fixed = self._fix_args(request)
        return self._normalize(handler(fixed))

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Async variant of `wrap_tool_call`."""
        fixed = self._fix_args(request)
        return self._normalize(await handler(fixed))


class ReadFileContinuationNoticeMiddleware(AgentMiddleware):
    """Append a continuation notice to exactly-at-limit `read_file` results."""

    name = "ReadFileContinuationNoticeMiddleware"

    @staticmethod
    def _is_numbered_read_file_row(row: str) -> bool:
        """Return whether `row` looks like a formatted `read_file` source line.

        Matches a primary source marker (`N`) followed by the current two-space
        separator (`format_content_with_line_numbers`) or the legacy `cat -n`
        tab. Continuation rows (`N.M`) fail — the `.` follows the digits, so
        they are not counted toward the source-line limit. Kept in sync with the
        separator emitted by `format_content_with_line_numbers`.
        """
        return re.match(r"^\s*\d+(?:\t|\s{2,})", row) is not None

    @staticmethod
    def _annotate(
        request: ToolCallRequest,
        result: ToolMessage | Command[Any],
    ) -> ToolMessage | Command[Any]:
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
            limit = int(args.get("limit") or _DEFAULT_READ_LIMIT)
        except (TypeError, ValueError):
            limit = _DEFAULT_READ_LIMIT

        n_lines = sum(1 for row in content.split("\n") if ReadFileContinuationNoticeMiddleware._is_numbered_read_file_row(row))
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
        """Annotate `read_file` results that may have more pages."""
        return self._annotate(request, handler(request))

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Async variant of `wrap_tool_call`."""
        return self._annotate(request, await handler(request))


_RATE_LIMIT_RETRY_DELAYS: tuple[float, ...] = (4.0, 12.0)


def _is_rate_limit_exception(exc: BaseException) -> bool:
    name = type(exc).__name__.lower()
    if "ratelimit" in name or "rate_limit" in name:
        return True
    status_code = getattr(exc, "status_code", None)
    if status_code == _HTTP_TOO_MANY_REQUESTS:
        return True
    return "rate limit" in str(exc).lower() or "rate-limit" in str(exc).lower()


class ModelRateLimitRetryMiddleware(AgentMiddleware):
    """Retry transient provider 429s around model calls."""

    name = "ModelRateLimitRetryMiddleware"

    def __init__(self, retry_delays: tuple[float, ...] = _RATE_LIMIT_RETRY_DELAYS) -> None:
        self._retry_delays = retry_delays

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelCallResult],
    ) -> ModelCallResult:
        """Retry rate-limit failures with a short backoff."""
        for delay in (*self._retry_delays, None):
            try:
                return handler(request)
            except Exception as exc:
                if delay is None or not _is_rate_limit_exception(exc):
                    raise
                time.sleep(delay)

        msg = "unreachable rate-limit retry state"
        raise RuntimeError(msg)

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelCallResult]],
    ) -> ModelCallResult:
        """Async variant of `wrap_model_call`."""
        for delay in (*self._retry_delays, None):
            try:
                return await handler(request)
            except Exception as exc:
                if delay is None or not _is_rate_limit_exception(exc):
                    raise
                await asyncio.sleep(delay)

        msg = "unreachable rate-limit retry state"
        raise RuntimeError(msg)


_FUNCTION_BLOCK_RE = re.compile(r"<function=([^>\s]+)\s*>(.*?)</function>", re.DOTALL)
_PARAMETER_RE = re.compile(r"<parameter\s+name=([^>\s]+)\s*>(.*?)</parameter>", re.DOTALL)
_ALT_FUNCTION_BLOCK_RE = re.compile(r"<function>\s*(.*?)</function>\s*(?:</tool_call>)?", re.DOTALL | re.IGNORECASE)
_ALT_NAME_RE = re.compile(r"<name\s*>(.*?)</name>|<name=([^<>\s]+)</name>", re.DOTALL | re.IGNORECASE)
_ALT_PARAMETER_RE = re.compile(r"<parameter(?:\s+name=([^>\s]+))?\s*>(.*?)</parameter>", re.DOTALL | re.IGNORECASE)
_ALT_INLINE_ARG_RE = re.compile(r"^\s*<?([A-Za-z_][\w-]*)>?\s*:\s*(.*?)\s*$", re.DOTALL)
_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>(.*?)</think>\s*", re.DOTALL | re.IGNORECASE)
_JSON_TOOL_NAME_ALIASES = {"bash": "execute", "sh": "execute", "shell": "execute"}
_HARNESS_PROFILE_SUFFIX_MARKER = "<state_changes>"

_FINAL_GUARD_SOURCE = "nemotron_final_answer_guard"
_TRANSITION_NUDGE_SOURCE = "nemotron_transition_nudge"
_FOLLOWUP_GUARD_SOURCE = "nemotron_followup_guard"
_ENTITY_GUARD_SOURCE = "nemotron_entity_guard"
_ACTION_COMMIT_NUDGE_SOURCE = "nemotron_action_commit_nudge"
_TOOL_CHAIN_NUDGE_SOURCE = "nemotron_tool_chain_nudge"
_FILESYSTEM_REQUEST_NUDGE_SOURCE = "nemotron_filesystem_request_nudge"
_DOMAIN_TOOL_PREFERENCE_SOURCE = "nemotron_domain_tool_preference"
_DOMAIN_TOOL_NUDGE_SOURCE = "nemotron_domain_tool_nudge"
_BUDGET_GUARD_SOURCE = "nemotron_progress_budget"
_INTERNAL_MESSAGE_NAMES = frozenset(
    {
        _FINAL_GUARD_SOURCE,
        _TRANSITION_NUDGE_SOURCE,
        _FOLLOWUP_GUARD_SOURCE,
        _ENTITY_GUARD_SOURCE,
        _ACTION_COMMIT_NUDGE_SOURCE,
        _TOOL_CHAIN_NUDGE_SOURCE,
        _FILESYSTEM_REQUEST_NUDGE_SOURCE,
        _DOMAIN_TOOL_PREFERENCE_SOURCE,
        _DOMAIN_TOOL_NUDGE_SOURCE,
        _BUDGET_GUARD_SOURCE,
    }
)
_VERSION_LITERAL_RE = re.compile(r"\bv\d+(?:\.\d+)+(?:[-._A-Za-z0-9]*)?\b")
_MUTATION_TOOL_VERBS = frozenset(
    {
        "approve",
        "archive",
        "assign",
        "activate",
        "book",
        "cancel",
        "charge",
        "close",
        "create",
        "deactivate",
        "delete",
        "disable",
        "enable",
        "escalate",
        "grant",
        "invite",
        "notify",
        "pay",
        "post",
        "publish",
        "reject",
        "refund",
        "remove",
        "reserve",
        "revoke",
        "schedule",
        "send",
        "submit",
        "terminate",
        "transfer",
        "update",
        "upgrade",
        "write",
    }
)
_READ_ONLY_TOOL_PREFIXES = frozenset(
    {
        "count",
        "describe",
        "fetch",
        "find",
        "get",
        "list",
        "lookup",
        "read",
        "retrieve",
        "search",
    }
)
# This list mirrors Deep Agents built-in tool names so model-level heuristics do
# not treat framework scaffolding as application/domain tools.
_BUILTIN_NON_DOMAIN_TOOLS: frozenset[str] = frozenset(
    {
        *_FILESYSTEM_TOOLS,
        "compact_conversation",
        "delete",
        "execute",
        "task",
        "write_todos",
    }
)
_MUTATION_LITERAL_KEYS = frozenset({"title", "subject"})
_MIN_MUTATION_LITERAL_LENGTH = 3
_MAX_MUTATION_LITERAL_LENGTH = 80
_RECURRENCE_RE = re.compile(
    r"\b(daily|weekly|monthly|nightly|morning|evening|every\s+(?:day|week|month|"
    r"morning|night)|each\s+(?:day|week|month)|at\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\b",
    re.IGNORECASE,
)
_SCHEDULE_QUESTION_RE = re.compile(
    r"\b(day/time|timezone|time\s*zone|cadence|frequency|schedule|what time|"
    r"which day|what day|how often|when should|when do you)\b",
    re.IGNORECASE,
)
_SOURCE_SCOPE_QUESTION_RE = re.compile(
    r"\b(which|what)\s+(?:data\s+source|source|sources|scope|folders?|"
    r"inboxes|labels|senders|projects?|repositories?|systems?|services?)\b",
    re.IGNORECASE,
)
_SOURCE_SCOPE_SUPPLIED_RE = re.compile(
    r"\b(?:my|our|the|this|these|current|all)\s+"
    r"(?:sources?|folders?|inboxes|labels|senders|projects?|repositories?|"
    r"repos?|systems?|services?|workspaces?|accounts?)\b|"
    r"\b(?:from|in|under|inside|within)\s+"
    r"(?:/[\w./-]+|[\w.-]+\.[A-Za-z0-9]{1,8}|(?:the\s+)?"
    r"[\w -]{2,80}?(?:source|folder|inbox|label|sender|project|repository|"
    r"repo|system|service|workspace|account))\b",
    re.IGNORECASE,
)
_ANALYSIS_REQUEST_RE = re.compile(r"\b(?:analy[sz]e|analysis|insight|report|dashboard)\b", re.IGNORECASE)
_ANALYSIS_GOAL_RE = re.compile(r"\b(?:goal|objective|question|metric|measure|compare|trend|segment|outcome|trying to learn)\b", re.IGNORECASE)
_SUPPORT_REQUEST_RE = re.compile(r"\b(?:customer|support|ticket|question|respond|response)\b", re.IGNORECASE)
_SUPPORT_DOMAIN_RE = re.compile(r"\b(?:domain|product|service|business|industry|customers?|users?)\b", re.IGNORECASE)
_DELIVERY_CONTEXT_RE = re.compile(r"\b(?:brief|summary|summaries|report|digest|recurring|daily|weekly|calendar|monitoring)\b", re.IGNORECASE)
_DELIVERY_QUESTION_RE = re.compile(
    r"\b(?:how|where|which|what)\b.{0,80}\b"
    r"(?:receive|send|deliver|delivery|channel|email|slack|sms|notify|notification)\b",
    re.IGNORECASE,
)
_QUESTION_START_RE = re.compile(r"(?im)^\s*(?:[-*]\s*)?(?:what|which|how|where|when|who)\b")
_VAGUE_COMPLETION_RE = re.compile(
    r"^\s*(?:done|completed|all set|handled|taken care of|finished)[.!]*\s*$",
    re.IGNORECASE,
)
_EXACT_SINGLE_WORD_REPLY_RE = re.compile(
    r"\b(?:reply|respond|return|answer)\s+with\s+(?:the\s+)?(?:single\s+word|one\s+word)\s+([A-Za-z0-9_\-[\]{}]+)\b",
    re.IGNORECASE,
)
_EXACT_PHRASE_REPLY_RE = re.compile(
    r"\b(?:reply|respond|return|answer)\s+with\s+exactly\s*:?\s*[\"']?([^\"'.?!\n]{1,80})",
    re.IGNORECASE,
)
_EXACT_ONLY_REPLY_RE = re.compile(
    r"\b(?:reply|respond|return|answer)\s+with\s+([A-Za-z0-9_\-[\]{}]+)\s+only\b",
    re.IGNORECASE,
)
_ACTION_REQUEST_RE = re.compile(
    r"\b(proceed|go ahead|make it happen|do it|now|please\s+(?:cancel|book|"
    r"update|upgrade|send|display|show|retrieve|start|fill|lock|charge)|"
    r"i want to\s+(?:cancel|book|update|upgrade|send)|"
    r"can you please\s+(?:cancel|book|update|upgrade|send))\b",
    re.IGNORECASE,
)
_CHAINED_ACTION_RE = re.compile(
    r"\b(?:then|and|after(?:ward)?|after that|once)\b.{0,160}"
    r"\b(?:email|send|notify|post|message|dm|create|schedule|book|cancel|update)\b",
    re.IGNORECASE,
)
_FILE_TASK_RE = re.compile(
    r"\b(file|files|folder|folders|directory|directories|path|paths|read_file|"
    r"write_file|edit_file|grep|glob|ls|filesystem|codebase|source code)\b",
    re.IGNORECASE,
)
_FILESYSTEM_ACCESS_REQUEST_RE = re.compile(
    r"\b(?:read|open|inspect|review|summari[sz]e|analy[sz]e|process|edit)\b"
    r".{0,160}\b(?:file|files|document|transcript|log|repository|repo|"
    r"codebase|source|/[\w./-]+|[\w.-]+\.[A-Za-z0-9]{1,8})\b",
    re.IGNORECASE,
)
_TRANSITION_NUDGE_MIN_MESSAGES = 6
_COMPACT_NEW_TASK_RE = re.compile(
    r"\b(?:move on|switch(?:ing)? to|new task|different task|unrelated task|"
    r"separate task|new topic|different topic|unrelated topic)\b",
    re.IGNORECASE,
)
_COMPACT_LARGE_READ_RE = re.compile(
    r"\b(?:read|summari[sz]e|inspect|analy[sz]e|review|process)\b.{0,120}"
    r"\b(?:file|document|transcript|log|repository|codebase|source|/[\w./-]+|[\w.-]+\.[A-Za-z0-9]{1,8})\b",
    re.IGNORECASE,
)
_COMPACT_FILE_REFERENCE_RE = re.compile(r"(?:/[\w./-]+|\b[\w.-]+\.[A-Za-z0-9]{1,8}\b)")
_FOLLOW_ON_WORK_RE = re.compile(
    r"\b(?:do the same|same thing|another|next|also|continue|again)\b.{0,120}"
    r"\b(?:file|document|transcript|log|repository|repo|codebase|source|"
    r"/[\w./-]+|[\w.-]+\.[A-Za-z0-9]{1,8})\b",
    re.IGNORECASE,
)
_MAX_MODEL_CALLS = 16
_MAX_TOOL_RESULTS = 48
_MAX_REPEATED_TOOL_CALLS = 3
_MAX_REDUNDANT_FOLLOWUP_QUESTIONS = 2
_MAX_REPAIR_MODEL_CALLS = 8
_MAX_REPAIR_TOOL_RESULTS = 28
_MAX_BUDGET_RESULTS = 12
_MAX_BUDGET_RESULT_CHARS = 500
_UNINFORMATIVE_TOOL_RESULT_PREFIXES = ("no files found", "no matches found", "error:")
_READ_FILE_DESCRIPTION_OVERRIDE = """Reads a file from the filesystem.

Use this tool for text files, source files, documents, images, audio, video, and PDFs.
If the user asks to read, inspect, review, or summarize an entire/whole/full file,
keep reading paginated chunks until you reach EOF or a tool result says the offset
exceeds the file length. A result that contains exactly `limit` numbered source
lines is only one page; continue with `offset + limit` before giving a final
whole-file answer. Use smaller `limit` values for large files to allow automatic
conversation summarization to keep context manageable.

Arguments:
- `file_path`: absolute path to the file.
- `offset`: 0-indexed source line to start from; use for pagination.
- `limit`: maximum source lines to read; use for pagination.

Results are returned with line numbers. Lines longer than 5,000 characters may
be split with continuation markers. Always read a file before editing it."""


def _parsed_tool_is_available(name: str, valid_tools: set[str] | None) -> bool:
    return valid_tools is None or name in valid_tools


def _parse_text_tool_calls(content: str, valid_tools: set[str] | None = None) -> tuple[list[ToolCall], str]:
    calls: list[ToolCall] = []
    for block in _FUNCTION_BLOCK_RE.finditer(content):
        name = block.group(1).strip("\"'")
        if not _parsed_tool_is_available(name, valid_tools):
            continue
        args = {param.group(1).strip("\"'"): param.group(2).strip() for param in _PARAMETER_RE.finditer(block.group(2))}
        calls.append({"name": name, "args": args, "id": uuid.uuid4().hex, "type": "tool_call"})
    if not calls:
        return _parse_alternate_text_tool_calls(content, valid_tools)
    leftover = _FUNCTION_BLOCK_RE.sub("", content).replace("</tool_call>", "").strip()
    return calls, leftover


def _parse_alternate_text_tool_calls(content: str, valid_tools: set[str] | None) -> tuple[list[ToolCall], str]:
    if valid_tools is None:
        return [], content

    calls: list[ToolCall] = []
    for block in _ALT_FUNCTION_BLOCK_RE.finditer(content):
        body = block.group(1)
        name = _alternate_function_name(body)
        if not name or name not in valid_tools:
            continue
        args = _alternate_function_args(body)
        calls.append({"name": name, "args": args, "id": uuid.uuid4().hex, "type": "tool_call"})
    if not calls:
        return [], content
    leftover = _ALT_FUNCTION_BLOCK_RE.sub("", content).replace("</tool_call>", "").strip()
    return calls, leftover


def _alternate_function_name(body: str) -> str:
    match = _ALT_NAME_RE.search(body)
    if match is None:
        return ""
    return (match.group(1) or match.group(2) or "").strip().strip("\"'")


def _alternate_function_args(body: str) -> dict[str, str]:
    args: dict[str, str] = {}
    for match in _ALT_PARAMETER_RE.finditer(body):
        name = (match.group(1) or "").strip().strip("\"'")
        raw = match.group(2).strip()
        if name:
            args[name] = raw
            continue
        inline = _ALT_INLINE_ARG_RE.match(raw)
        if inline is not None:
            args[inline.group(1)] = inline.group(2).strip()
    return args


def _first_json_object(content: str) -> dict[str, Any] | None:
    start, end = content.find("{"), content.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(content[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _parse_json_tool_calls(content: str, valid_tools: set[str] | None = None) -> list[ToolCall]:
    obj = _first_json_object(content)
    if obj is None:
        return []
    name = obj.get("tool")
    if not isinstance(name, str) or not name.strip():
        return []
    name = _JSON_TOOL_NAME_ALIASES.get(name.strip().lower(), name.strip())
    if not _parsed_tool_is_available(name, valid_tools):
        return []
    raw_args = obj.get("args")
    if isinstance(raw_args, dict):
        args: dict[str, Any] = raw_args
    else:
        command = obj.get("cmd") or obj.get("command")
        args = {"command": command} if isinstance(command, str) else {}
    return [{"name": name, "args": args, "id": uuid.uuid4().hex, "type": "tool_call"}]


def _strip_reasoning_tags(message: AIMessage) -> AIMessage:
    content = message.content
    if not isinstance(content, str) or "</think>" not in content.lower():
        return message

    reasoning_parts = [match.group(1).strip() for match in _THINK_BLOCK_RE.finditer(content) if match.group(1).strip()]
    clean_content = _THINK_BLOCK_RE.sub("", content).strip()
    if clean_content == content:
        return message

    additional = dict(message.additional_kwargs)
    if reasoning_parts and "reasoning_content" not in additional:
        additional["reasoning_content"] = "\n\n".join(reasoning_parts)
    return message.model_copy(update={"content": clean_content, "additional_kwargs": additional})


class ChatNVIDIAMessageCompatibilityMiddleware(AgentMiddleware):
    """Mirror standard LangChain tool-call fields into ChatNVIDIA payload metadata."""

    name = "ChatNVIDIAMessageCompatibilityMiddleware"

    @staticmethod
    def _repair_message(message: AIMessage | ToolMessage) -> AIMessage | ToolMessage:
        if isinstance(message, AIMessage):
            if not message.tool_calls or "tool_calls" in message.additional_kwargs:
                return message
            additional = dict(message.additional_kwargs)
            additional["tool_calls"] = _convert_to_openai_tool_calls(message.tool_calls)
            return message.model_copy(update={"additional_kwargs": additional})

        if getattr(message, "name", None) and "name" not in message.additional_kwargs:
            additional = dict(message.additional_kwargs)
            additional["name"] = message.name
            return message.model_copy(update={"additional_kwargs": additional})
        return message

    @staticmethod
    def _repair_messages(messages: list[Any]) -> list[Any]:
        repaired: list[Any] = []
        changed = False
        for message in messages:
            if isinstance(message, AIMessage | ToolMessage):
                fixed = ChatNVIDIAMessageCompatibilityMiddleware._repair_message(message)
                changed = changed or fixed is not message
                repaired.append(fixed)
            else:
                repaired.append(message)
        return repaired if changed else messages

    @staticmethod
    def _with_repaired_messages(request: ModelRequest[Any]) -> ModelRequest[Any]:
        original = list(request.messages or [])
        messages = ChatNVIDIAMessageCompatibilityMiddleware._repair_messages(original)
        if messages is original:
            return request
        return request.override(messages=messages)

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelCallResult],
    ) -> ModelCallResult:
        """Patch request messages before ChatNVIDIA serializes them."""
        return handler(self._with_repaired_messages(request))

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelCallResult]],
    ) -> ModelCallResult:
        """Async variant of `wrap_model_call`."""
        return await handler(self._with_repaired_messages(request))


class NemotronReasoningTagCleanupMiddleware(AgentMiddleware):
    """Remove preserved `<think>` blocks from normal assistant content."""

    name = "NemotronReasoningTagCleanupMiddleware"

    @staticmethod
    def _repair_response(response: ModelResponse) -> ModelResponse:
        return ModelResponse(
            result=[_strip_reasoning_tags(message) if isinstance(message, AIMessage) else message for message in response.result],
            structured_response=response.structured_response,
        )

    @staticmethod
    def _repair(result: ModelCallResult) -> ModelCallResult:
        if isinstance(result, ExtendedModelResponse):
            return ExtendedModelResponse(
                model_response=NemotronReasoningTagCleanupMiddleware._repair_response(result.model_response),
                command=result.command,
            )
        if isinstance(result, AIMessage):
            return _strip_reasoning_tags(result)
        if isinstance(result, ModelResponse):
            return NemotronReasoningTagCleanupMiddleware._repair_response(result)
        return result

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelCallResult],
    ) -> ModelCallResult:
        """Strip reasoning tags after the model returns."""
        return self._repair(handler(request))

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelCallResult]],
    ) -> ModelCallResult:
        """Async variant of `wrap_model_call`."""
        return self._repair(await handler(request))


class NemotronTextToolCallParser(AgentMiddleware):
    """Repair tool calls emitted as text content instead of structured calls."""

    name = "NemotronTextToolCallParser"

    @staticmethod
    def _repair_message(message: AIMessage, valid_tools: set[str] | None = None) -> AIMessage:
        if message.tool_calls:
            return message
        message = _strip_reasoning_tags(message)
        content = message.content
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        else:
            text = ""
        calls, leftover = _parse_text_tool_calls(text, valid_tools)
        if not calls:
            json_calls = _parse_json_tool_calls(text, valid_tools)
            if json_calls:
                calls, leftover = json_calls, ""
        if not calls:
            return message
        return message.model_copy(update={"tool_calls": calls, "content": leftover})

    @staticmethod
    def _repair_response(response: ModelResponse, valid_tools: set[str] | None = None) -> ModelResponse:
        return ModelResponse(
            result=[
                NemotronTextToolCallParser._repair_message(message, valid_tools) if isinstance(message, AIMessage) else message
                for message in response.result
            ],
            structured_response=response.structured_response,
        )

    @staticmethod
    def _repair(result: ModelCallResult, valid_tools: set[str] | None = None) -> ModelCallResult:
        if isinstance(result, ExtendedModelResponse):
            return ExtendedModelResponse(
                model_response=NemotronTextToolCallParser._repair_response(result.model_response, valid_tools),
                command=result.command,
            )
        if isinstance(result, AIMessage):
            return NemotronTextToolCallParser._repair_message(result, valid_tools)
        if isinstance(result, ModelResponse):
            return NemotronTextToolCallParser._repair_response(result, valid_tools)
        return result

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelCallResult],
    ) -> ModelCallResult:
        """Repair text-form tool calls after the model returns."""
        return self._repair(handler(request), set(_request_tool_names(request)))

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelCallResult]],
    ) -> ModelCallResult:
        """Async variant of `wrap_model_call`."""
        return self._repair(await handler(request), set(_request_tool_names(request)))


def _message_text(message: AIMessage | HumanMessage | ToolMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return ""


def _external_human_messages(messages: list[Any]) -> list[HumanMessage]:
    return [message for message in messages if isinstance(message, HumanMessage) and getattr(message, "name", None) not in _INTERNAL_MESSAGE_NAMES]


def _last_external_human_text(messages: list[Any]) -> str:
    humans = _external_human_messages(messages)
    return _message_text(humans[-1]) if humans else ""


def _question_count(text: str) -> int:
    return max(text.count("?"), len(_QUESTION_START_RE.findall(text)))


def _normalize_requested_final_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().strip("\"'`").strip(" .!?")).strip()


def _requested_exact_final_texts(user_text: str) -> list[str]:
    requested: list[str] = []
    for regex in (_EXACT_SINGLE_WORD_REPLY_RE, _EXACT_PHRASE_REPLY_RE, _EXACT_ONLY_REPLY_RE):
        requested.extend(_normalize_requested_final_text(match.group(1)) for match in regex.finditer(user_text))
    return [text for text in dict.fromkeys(requested) if text]


def _satisfies_exact_final_answer_request(user_text: str, final_text: str) -> bool:
    requested = _requested_exact_final_texts(user_text)
    if not requested:
        return False
    normalized_final = _normalize_requested_final_text(final_text)
    return any(normalized_final == expected for expected in requested)


def _is_final_answer(message: AIMessage) -> bool:
    return getattr(message, "name", None) != _BUDGET_GUARD_SOURCE and not message.tool_calls and bool(_message_text(message).strip())


def _nudge_update(content: str, source: str, **updates: Any) -> dict[str, Any]:
    return {"messages": [HumanMessage(content=content, name=source)], **updates}


def _iter_tool_calls(messages: list[Any]) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for message in messages:
        if isinstance(message, AIMessage):
            calls.extend(message.tool_calls or [])
    return calls


def _tool_name_is_mutation(name: str) -> bool:
    if not _tool_name_is_domain(name):
        return False
    tokens = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|\d+", name.replace("_", " "))
    if not tokens:
        return False
    lowered = [token.lower() for token in tokens]
    if lowered[0] in _READ_ONLY_TOOL_PREFIXES:
        return False
    return any(token in _MUTATION_TOOL_VERBS for token in lowered)


def _tool_name_is_domain(name: str) -> bool:
    return bool(name) and name not in _BUILTIN_NON_DOMAIN_TOOLS and not name.startswith("__")


def _messages_since_last_user(messages: list[Any]) -> list[Any]:
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if isinstance(message, HumanMessage) and getattr(message, "name", None) not in _INTERNAL_MESSAGE_NAMES:
            return messages[index:]
    return messages


def _tool_call_for_message(messages: list[Any], tool_message: ToolMessage) -> ToolCall | None:
    for call in reversed(_iter_tool_calls(messages)):
        if call.get("id") == tool_message.tool_call_id:
            return call
    return None


def _tool_name(tool: object) -> str | None:
    if isinstance(tool, dict):
        tool_dict = cast("dict[str, Any]", tool)
        name = tool_dict.get("name")
        if isinstance(name, str):
            return name
        function = tool_dict.get("function")
        if isinstance(function, dict) and isinstance(function.get("name"), str):
            return function["name"]
        return None
    name = getattr(tool, "name", None)
    return name if isinstance(name, str) else None


def _request_tool_names(request: ModelRequest[Any]) -> list[str]:
    return [name for tool in request.tools or [] for name in [_tool_name(tool)] if name]


def _ai_message_count(messages: list[Any]) -> int:
    return sum(1 for message in messages if isinstance(message, AIMessage))


def _tool_message_count(messages: list[Any]) -> int:
    return sum(1 for message in messages if isinstance(message, ToolMessage))


def _tool_call_signature(call: ToolCall) -> str:
    name = call.get("name", "")
    args = call.get("args") or {}
    return f"{name}:{json.dumps(args, sort_keys=True, default=str)}"


def _max_consecutive_repeated_tool_call_count(messages: list[Any]) -> int:
    max_count = 0
    last_sig: str | None = None
    current_count = 0
    for call in _iter_tool_calls(messages):
        sig = _tool_call_signature(call)
        if sig == last_sig:
            current_count += 1
        else:
            last_sig = sig
            current_count = 1
        max_count = max(max_count, current_count)
    return max_count


def _repair_loop_risk(messages: list[Any]) -> bool:
    return _ai_message_count(messages) >= _MAX_REPAIR_MODEL_CALLS or _tool_message_count(messages) >= _MAX_REPAIR_TOOL_RESULTS


def _mutation_literal_values(tool_calls: list[ToolCall]) -> list[str]:
    literals: list[str] = []
    for call in tool_calls:
        name = call.get("name", "")
        if not _tool_name_is_mutation(name):
            continue
        args = call.get("args") or {}
        for value in args.values():
            if not isinstance(value, str):
                continue
            literals.extend(_VERSION_LITERAL_RE.findall(value))
        for key in _MUTATION_LITERAL_KEYS:
            value = args.get(key)
            if isinstance(value, str) and _MIN_MUTATION_LITERAL_LENGTH <= len(value) <= _MAX_MUTATION_LITERAL_LENGTH:
                literals.append(value)
    return list(dict.fromkeys(literals))


def _parse_tool_value(text: str) -> object:
    stripped = text.strip()
    if not stripped:
        return None
    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return stripped


def _coerce_int(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _tool_results(messages: list[Any]) -> list[tuple[ToolCall, object]]:
    calls: dict[str, ToolCall] = {}
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        for call in message.tool_calls or []:
            call_id = call.get("id")
            if isinstance(call_id, str):
                calls[call_id] = call

    results: list[tuple[ToolCall, object]] = []
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        call = calls.get(message.tool_call_id)
        if call is not None:
            results.append((call, _parse_tool_value(_message_text(message))))
    return results


def _last_domain_mutation_result(messages: list[Any]) -> tuple[ToolCall, object] | None:
    for call, value in reversed(_tool_results(messages)):
        name = call.get("name", "")
        if _tool_name_is_mutation(name):
            return call, value
    return None


def _format_budget_value(value: object) -> str:
    text = json.dumps(value, sort_keys=True, default=str) if isinstance(value, (dict, list)) else "" if value is None else str(value)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > _MAX_BUDGET_RESULT_CHARS:
        return text[: _MAX_BUDGET_RESULT_CHARS - 3].rstrip() + "..."
    return text


def _budget_result_is_informative(value: object) -> bool:
    text = _format_budget_value(value).lower()
    return bool(text) and not any(text.startswith(prefix) for prefix in _UNINFORMATIVE_TOOL_RESULT_PREFIXES)


def _budget_fallback_text(messages: list[Any], reason: str) -> str:
    results = _tool_results(messages)
    if not results:
        return f"I could not complete this reliably within the harness step budget ({reason})."

    rows: list[str] = []
    prioritized = [entry for entry in results if _budget_result_is_informative(entry[1])]
    if len(prioritized) < _MAX_BUDGET_RESULTS:
        prioritized.extend(entry for entry in results if entry not in prioritized)
    seen: set[tuple[str, str]] = set()
    for call, value in prioritized[:_MAX_BUDGET_RESULTS]:
        name = call.get("name") or "tool"
        text = _format_budget_value(value)
        key = (name, text)
        if key in seen:
            continue
        seen.add(key)
        rows.append(f"- {name}: {text}")
    return "Using the tool results gathered so far:\n" + "\n".join(rows)


class NemotronProgressBudgetMiddleware(AgentMiddleware):
    """Stop Ultra3-specific tool loops before they consume runaway context."""

    name = "NemotronProgressBudgetMiddleware"

    def __init__(
        self,
        *,
        max_model_calls: int = _MAX_MODEL_CALLS,
        max_tool_results: int = _MAX_TOOL_RESULTS,
        max_repeated_tool_calls: int = _MAX_REPEATED_TOOL_CALLS,
    ) -> None:
        self.max_model_calls = max_model_calls
        self.max_tool_results = max_tool_results
        self.max_repeated_tool_calls = max_repeated_tool_calls

    @staticmethod
    def _messages(request: ModelRequest[Any]) -> list[Any]:
        state = request.state or {}
        if isinstance(state, dict):
            messages = state.get("messages")
            if isinstance(messages, list):
                return messages
        return list(request.messages or [])

    def _exceeded_reason(self, messages: list[Any]) -> str | None:
        ai_count = _ai_message_count(messages)
        if ai_count >= self.max_model_calls:
            return f"{ai_count} model turns"
        tool_count = _tool_message_count(messages)
        if tool_count >= self.max_tool_results:
            return f"{tool_count} tool results"
        repeats = _max_consecutive_repeated_tool_call_count(messages)
        if repeats >= self.max_repeated_tool_calls:
            return f"{repeats} repeated identical tool calls"
        return None

    @staticmethod
    def _fallback(messages: list[Any], reason: str) -> AIMessage:
        return AIMessage(
            content=_budget_fallback_text(messages, reason),
            name=_BUDGET_GUARD_SOURCE,
            response_metadata={"nemotron_progress_budget_reason": reason},
        )

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelCallResult],
    ) -> ModelCallResult:
        """Short-circuit before another model call if progress budgets are exhausted."""
        messages = _messages_since_last_user(self._messages(request))
        reason = self._exceeded_reason(messages)
        if reason is not None:
            return self._fallback(messages, reason)
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelCallResult]],
    ) -> ModelCallResult:
        """Async variant of `wrap_model_call`."""
        messages = _messages_since_last_user(self._messages(request))
        reason = self._exceeded_reason(messages)
        if reason is not None:
            return self._fallback(messages, reason)
        return await handler(request)


class NemotronPolicyNudgeState(AgentState):
    """State schema for one-shot Nemotron policy nudges."""

    nemotron_transition_nudged: NotRequired[Annotated[bool, PrivateStateAttr]]
    nemotron_action_nudged: NotRequired[Annotated[list[str], PrivateStateAttr]]
    nemotron_tool_chain_nudged: NotRequired[Annotated[bool, PrivateStateAttr]]
    nemotron_domain_tool_nudged: NotRequired[Annotated[bool, PrivateStateAttr]]


class NemotronPolicyNudgeMiddleware(AgentMiddleware):
    """Inject lightweight policy nudges for common Ultra3 agent-control misses."""

    name = "NemotronPolicyNudgeMiddleware"
    state_schema = NemotronPolicyNudgeState

    @staticmethod
    def _should_prefer_domain_tools(request: ModelRequest[Any]) -> bool:
        messages = NemotronProgressBudgetMiddleware._messages(request)
        window = _messages_since_last_user(messages)
        if len(window) != 1 or not isinstance(window[0], HumanMessage):
            return False
        user_text = _message_text(window[0])
        if _FILE_TASK_RE.search(user_text):
            return False

        names = _request_tool_names(request)
        has_filesystem = any(name in _FILESYSTEM_TOOLS for name in names)
        has_domain = any(_tool_name_is_domain(name) for name in names)
        return has_filesystem and has_domain

    @staticmethod
    def _with_domain_tool_preference(request: ModelRequest[Any]) -> ModelRequest[Any]:
        names = [name for name in _request_tool_names(request) if _tool_name_is_domain(name)]
        tool_hint = ", ".join(names[:12])
        nudge = HumanMessage(
            content=(
                "This is not a file or repository-content request. Start with "
                "the task-specific non-filesystem tools instead of ls, glob, "
                "grep, or read_file. For ranking, counting, 'which', or 'most' "
                "questions, enumerate or search for candidate entities with the "
                "available domain tools, fetch the relevant details or counts, "
                "compare those observed results, and then answer."
                f" Relevant task tools include: {tool_hint}."
            ),
            name=_DOMAIN_TOOL_PREFERENCE_SOURCE,
        )
        return request.override(messages=[*request.messages, nudge])

    @staticmethod
    def _should_prefer_filesystem_tools(request: ModelRequest[Any]) -> bool:
        messages = NemotronProgressBudgetMiddleware._messages(request)
        window = _messages_since_last_user(messages)
        if len(window) != 1 or not isinstance(window[0], HumanMessage):
            return False

        user_text = _message_text(window[0])
        if not _FILESYSTEM_ACCESS_REQUEST_RE.search(user_text):
            return False

        names = set(_request_tool_names(request))
        return "read_file" in names and bool(names.intersection(_FILESYSTEM_TOOLS))

    @staticmethod
    def _with_filesystem_preference(request: ModelRequest[Any]) -> ModelRequest[Any]:
        nudge = HumanMessage(
            content=(
                "The user is asking for file or path content, and filesystem "
                "tools are available. Do not answer that you lack access before "
                "trying the tools. If the user named a file or path, first call "
                "read_file with that path and the requested pagination/limit. If "
                "that fails or the location is ambiguous, use ls or glob to locate "
                "the file, then continue reading until the request is satisfied."
            ),
            name=_FILESYSTEM_REQUEST_NUDGE_SOURCE,
        )
        return request.override(messages=[*request.messages, nudge])

    @staticmethod
    def _should_compact_on_transition(messages: list[Any]) -> bool:
        if len(messages) < _TRANSITION_NUDGE_MIN_MESSAGES:
            return False

        humans = _external_human_messages(messages)
        if not humans:
            return False

        latest = humans[-1]
        user_text = _message_text(latest)
        window = _messages_since_last_user(messages)
        if any(call.get("name") == "compact_conversation" for call in _iter_tool_calls(window)):
            return False

        has_prior_file_work = any(call.get("name") in _FILESYSTEM_TOOLS for call in _iter_tool_calls(messages[:-1]))
        starts_new_task = _COMPACT_NEW_TASK_RE.search(user_text) is not None
        asks_follow_on_file = _FOLLOW_ON_WORK_RE.search(user_text) is not None
        asks_large_file_work = _COMPACT_LARGE_READ_RE.search(user_text) is not None
        references_file = _COMPACT_FILE_REFERENCE_RE.search(user_text) is not None
        return starts_new_task or (has_prior_file_work and (asks_follow_on_file or asks_large_file_work or references_file))

    @staticmethod
    def _transition_nudge(state: NemotronPolicyNudgeState) -> dict[str, Any] | None:
        if state.get("nemotron_transition_nudged"):
            return None
        messages = list(state.get("messages") or [])
        if not NemotronPolicyNudgeMiddleware._should_compact_on_transition(messages):
            return None
        return _nudge_update(
            "This is a long conversation and the latest user request appears to "
            "start a new task or substantial follow-on file work. If "
            "compact_conversation is available, call it before starting the new "
            "work so prior context is compressed instead of carried forward "
            "verbatim.",
            _TRANSITION_NUDGE_SOURCE,
            nemotron_transition_nudged=True,
        )

    @staticmethod
    def _action_commit_nudge(state: NemotronPolicyNudgeState) -> dict[str, Any] | None:
        messages = list(state.get("messages") or [])
        if not messages or not isinstance(messages[-1], HumanMessage):
            return None
        last = messages[-1]
        if getattr(last, "name", None) in _INTERNAL_MESSAGE_NAMES:
            return None
        text = _message_text(last)
        if not _ACTION_REQUEST_RE.search(text):
            return None
        if not _iter_tool_calls(messages[:-1]):
            return None

        key = f"{len(messages)}:{text[:200]}"
        nudged = list(state.get("nemotron_action_nudged") or [])
        if key in nudged:
            return None
        return _nudge_update(
            "The user is asking you to perform an action now. If the conversation "
            "or previous tool results already provide the required identifiers, "
            "payment/source details, recipients, or parameters, call the relevant "
            "state-changing/API tool instead of replying only with policy "
            "explanation or another confirmation request. Ask exactly one "
            "missing-field question only if a required argument is still "
            "unavailable.",
            _ACTION_COMMIT_NUDGE_SOURCE,
            nemotron_action_nudged=[*nudged, key],
        )

    @staticmethod
    def _tool_chain_nudge(state: NemotronPolicyNudgeState) -> dict[str, Any] | None:
        if state.get("nemotron_tool_chain_nudged"):
            return None
        messages = list(state.get("messages") or [])
        if not messages or not isinstance(messages[-1], ToolMessage):
            return None

        window = _messages_since_last_user(messages)
        user_text = _last_external_human_text(window)
        if not _CHAINED_ACTION_RE.search(user_text):
            return None

        calls = _iter_tool_calls(window)
        if not calls or any(_tool_name_is_mutation(call.get("name", "")) for call in calls):
            return None

        return _nudge_update(
            "The user's request has a chained action after the information "
            "lookup. Use the tool results already gathered as the summary source, "
            "then call the requested state-changing tool such as email, send, "
            "notify, post, create, schedule, book, cancel, or update. Do not "
            "repeat the same lookup unless a required argument for the action is "
            "still missing.",
            _TOOL_CHAIN_NUDGE_SOURCE,
            nemotron_tool_chain_nudged=True,
        )

    @staticmethod
    def _domain_tool_completion_nudge(state: NemotronPolicyNudgeState) -> dict[str, Any] | None:
        if state.get("nemotron_domain_tool_nudged"):
            return None
        messages = list(state.get("messages") or [])
        if not messages or not isinstance(messages[-1], ToolMessage):
            return None
        last = messages[-1]
        text = _message_text(last).lower()
        if not any(text.startswith(prefix) for prefix in _UNINFORMATIVE_TOOL_RESULT_PREFIXES):
            return None

        window = _messages_since_last_user(messages)
        last_call = _tool_call_for_message(window, last)
        if last_call is None or last_call.get("name") not in _FILESYSTEM_TOOLS:
            return None

        calls = _iter_tool_calls(window)
        if not any(_tool_name_is_domain(call.get("name", "")) for call in calls):
            return None

        return _nudge_update(
            "The filesystem search did not find useful files. Continue with the "
            "available non-filesystem API/domain tools instead of grepping or "
            "listing more files. For lookup, ranking, counting, or 'most' "
            "questions, enumerate or search for candidate entities with domain "
            "tools, fetch details or counts with the matching domain tools, "
            "compare the observed results, and answer from those results.",
            _DOMAIN_TOOL_NUDGE_SOURCE,
            nemotron_domain_tool_nudged=True,
        )

    @staticmethod
    def _merge_updates(updates: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not updates:
            return None

        messages = [message for update in updates for message in update.get("messages", [])]
        state_updates = {key: value for update in updates for key, value in update.items() if key != "messages"}
        return {"messages": messages, **state_updates}

    @classmethod
    def _nudge(cls, state: NemotronPolicyNudgeState) -> dict[str, Any] | None:
        updates = [
            update
            for update in (
                cls._transition_nudge(state),
                cls._action_commit_nudge(state),
                cls._tool_chain_nudge(state),
                cls._domain_tool_completion_nudge(state),
            )
            if update is not None
        ]
        return cls._merge_updates(updates)

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelCallResult],
    ) -> ModelCallResult:
        """Add first-turn policy nudges when appropriate."""
        if self._should_prefer_domain_tools(request):
            request = self._with_domain_tool_preference(request)
        if self._should_prefer_filesystem_tools(request):
            request = self._with_filesystem_preference(request)
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelCallResult]],
    ) -> ModelCallResult:
        """Async variant of `wrap_model_call`."""
        if self._should_prefer_domain_tools(request):
            request = self._with_domain_tool_preference(request)
        if self._should_prefer_filesystem_tools(request):
            request = self._with_filesystem_preference(request)
        return await handler(request)

    def before_model(
        self,
        state: AgentState[Any],
        runtime: Runtime[Any],  # noqa: ARG002
    ) -> dict[str, Any] | None:
        """Inject one-shot policy nudges before the model acts."""
        return self._nudge(cast("NemotronPolicyNudgeState", state))

    async def abefore_model(
        self,
        state: AgentState[Any],
        runtime: Runtime[Any],  # noqa: ARG002
    ) -> dict[str, Any] | None:
        """Async variant of `before_model`."""
        return self._nudge(cast("NemotronPolicyNudgeState", state))


class FollowupDisciplineState(AgentState):
    """State schema for `FollowupDisciplineMiddleware`."""

    nemotron_followup_guard_fired: NotRequired[Annotated[bool, PrivateStateAttr]]


class FollowupDisciplineMiddleware(AgentMiddleware):
    """Send Ultra3 back once when it asks redundant follow-up questions."""

    name = "FollowupDisciplineMiddleware"
    state_schema = FollowupDisciplineState

    @staticmethod
    def _needs_rewrite(user_text: str, final_text: str) -> bool:
        user_lower = user_text.lower()
        has_recurrence = bool(_RECURRENCE_RE.search(user_text))
        asks_schedule = bool(_SCHEDULE_QUESTION_RE.search(final_text))
        asks_too_many_questions = has_recurrence and _question_count(final_text) > _MAX_REDUNDANT_FOLLOWUP_QUESTIONS
        asks_source_scope = bool(_SOURCE_SCOPE_QUESTION_RE.search(final_text))
        source_scope_supplied = _SOURCE_SCOPE_SUPPLIED_RE.search(user_text) is not None
        analysis_needs_goal = bool(_ANALYSIS_REQUEST_RE.search(user_text)) and "data" in user_lower and not _ANALYSIS_GOAL_RE.search(final_text)
        support_needs_domain = bool(_SUPPORT_REQUEST_RE.search(user_text)) and not _SUPPORT_DOMAIN_RE.search(final_text)
        delivery_context = bool(_DELIVERY_CONTEXT_RE.search(user_text))
        asks_delivery_or_source = _DELIVERY_QUESTION_RE.search(final_text) or _SOURCE_SCOPE_QUESTION_RE.search(final_text)
        recurring_needs_delivery = delivery_context and has_recurrence and not asks_delivery_or_source
        return (
            (has_recurrence and asks_schedule)
            or asks_too_many_questions
            or (asks_source_scope and source_scope_supplied)
            or analysis_needs_goal
            or support_needs_domain
            or recurring_needs_delivery
        )

    @staticmethod
    def _nudge(state: FollowupDisciplineState) -> dict[str, Any] | None:
        messages = list(state.get("messages") or [])
        if state.get("nemotron_followup_guard_fired") or _repair_loop_risk(messages) or not messages:
            return None
        last = messages[-1]
        if not isinstance(last, AIMessage) or not _is_final_answer(last):
            return None
        user_text = _last_external_human_text(messages[:-1])
        final_text = _message_text(last)
        if _satisfies_exact_final_answer_request(user_text, final_text) or not FollowupDisciplineMiddleware._needs_rewrite(user_text, final_text):
            return None
        return _nudge_update(
            "Rewrite your follow-up so it asks for the smallest useful missing "
            "information. Do not re-ask about schedule, cadence, source, or "
            "scope when those are already supplied. For vague analysis requests, "
            "ask for both the data source and the analysis goal. For support or "
            "customer-response improvement requests, ask about the product/domain "
            "and the current support surface. For recurring briefs, reports, or "
            "monitoring requests with a stated cadence, ask for the missing "
            "delivery channel or content/source detail, not the day/time again.",
            _FOLLOWUP_GUARD_SOURCE,
            jump_to="model",
            nemotron_followup_guard_fired=True,
        )

    @hook_config(can_jump_to=["model"])
    def after_agent(
        self,
        state: AgentState[Any],
        runtime: Runtime[Any],  # noqa: ARG002
    ) -> dict[str, Any] | None:
        """Loop once when the final follow-up asks redundant setup questions."""
        return self._nudge(cast("FollowupDisciplineState", state))

    @hook_config(can_jump_to=["model"])
    async def aafter_agent(
        self,
        state: AgentState[Any],
        runtime: Runtime[Any],  # noqa: ARG002
    ) -> dict[str, Any] | None:
        """Async variant of `after_agent`."""
        return self._nudge(cast("FollowupDisciplineState", state))


class EntityResolutionGuardState(AgentState):
    """State schema for `EntityResolutionGuardMiddleware`."""

    nemotron_entity_pre_nudged: NotRequired[Annotated[bool, PrivateStateAttr]]
    nemotron_entity_guard_fired: NotRequired[Annotated[bool, PrivateStateAttr]]


class EntityResolutionGuardMiddleware(AgentMiddleware):
    """Send Ultra3 back once when it finalizes with unresolved or mis-bound IDs."""

    name = "EntityResolutionGuardMiddleware"
    state_schema = EntityResolutionGuardState

    @staticmethod
    def _current_entity_ids(messages: list[Any]) -> dict[str, int]:
        current: dict[str, int] = {}
        for call, result in _tool_results(messages):
            name = call.get("name")
            if not isinstance(name, str):
                continue
            match = re.fullmatch(r"get_current_([a-z_]+)_id", name)
            if match is None:
                continue
            entity_id = _coerce_int(result)
            if entity_id is not None:
                current[match.group(1)] = entity_id
        return current

    @staticmethod
    def _relation_bindings(messages: list[Any]) -> dict[tuple[str, int, str], int]:
        bindings: dict[tuple[str, int, str], int] = {}
        for call, result in _tool_results(messages):
            name = call.get("name", "")
            if not isinstance(name, str) or re.fullmatch(r"get_[a-z_]+_(?:name|title)", name):
                continue
            match = re.fullmatch(r"get_([a-z_]+)_([a-z_]+)", name)
            if match is None:
                continue
            source, target = match.groups()
            source_id = _coerce_int((call.get("args") or {}).get(f"{source}_id"))
            target_id = _coerce_int(result)
            if source_id is not None and target_id is not None:
                bindings[(source, source_id, target)] = target_id
        return bindings

    @staticmethod
    def _display_resolved_ids(messages: list[Any]) -> set[tuple[str, int]]:
        resolved: set[tuple[str, int]] = set()
        for call in _iter_tool_calls(messages):
            name = call.get("name", "")
            match = re.fullmatch(r"get_([a-z_]+)_(?:name|title)", name)
            if match is None:
                continue
            entity = match.group(1)
            entity_id = _coerce_int((call.get("args") or {}).get(f"{entity}_id"))
            if entity_id is not None:
                resolved.add((entity, entity_id))
        return resolved

    @staticmethod
    def _missing_current_branch_resolutions(messages: list[Any], user_text: str, final_text: str) -> list[str]:
        user_lower = user_text.lower()
        current = EntityResolutionGuardMiddleware._current_entity_ids(messages)
        if not current:
            return []

        bindings = EntityResolutionGuardMiddleware._relation_bindings(messages)
        resolved = EntityResolutionGuardMiddleware._display_resolved_ids(messages)
        final_numbers = set(re.findall(r"\b\d{4,}\b", final_text))

        missing: list[str] = []
        for source, source_id in current.items():
            if f"current {source}" not in user_lower:
                continue
            candidate_targets = {
                target for relation_source, _, target in bindings if relation_source == source and re.search(rf"\b{re.escape(target)}\b", user_lower)
            }
            for target in sorted(candidate_targets):
                target_id = bindings.get((source, source_id, target))
                if target_id is None:
                    missing.append(f"{target} lookup for current {source} {source_id}")
                elif (target, target_id) not in resolved:
                    missing.append(f"{target}_id {target_id} from current {source} {source_id}")

        for (source, source_id, target), target_id in bindings.items():
            if str(target_id) in final_numbers and (target, target_id) not in resolved:
                missing.append(f"{target}_id {target_id} from {source} {source_id}")

        return list(dict.fromkeys(missing))

    @staticmethod
    def _missing_display_resolutions(messages: list[Any], user_text: str, final_text: str) -> list[str]:
        missing: list[str] = []
        resolved = EntityResolutionGuardMiddleware._display_resolved_ids(messages)
        final_numbers = set(re.findall(r"\b\d{4,}\b", final_text))
        user_lower = user_text.lower()

        for call in _iter_tool_calls(messages):
            name = call.get("name", "")
            match = re.fullmatch(r"get_([a-z_]+)_[a-z_]+", name)
            if match is None or re.fullmatch(r"get_[a-z_]+_(?:name|title)", name):
                continue
            entity = match.group(1)
            user_asks_for_entity = re.search(rf"\b(?:which|what)\s+(?:\w+\s+){{0,4}}{re.escape(entity)}\b", user_lower) is not None
            entity_reference = (
                f"that {entity}" in user_lower
                or f"{entity} with" in user_lower
                or f"{entity} whose" in user_lower
                or f"selected {entity}" in user_lower
            )
            if not user_asks_for_entity and not entity_reference:
                continue
            entity_id = _coerce_int((call.get("args") or {}).get(f"{entity}_id"))
            if entity_id is None:
                continue
            id_visible = str(entity_id) in final_numbers
            if (id_visible or entity_reference) and (entity, entity_id) not in resolved:
                missing.append(f"{entity}_id {entity_id} title/name")

        return list(dict.fromkeys(missing))

    @staticmethod
    def _resolution_steps(missing: list[str]) -> str:
        steps: list[str] = []
        for item in missing:
            relation_lookup = re.fullmatch(r"([a-z_]+) lookup for current ([a-z_]+) (\d+)", item)
            if relation_lookup is not None:
                target, source, source_id = relation_lookup.groups()
                steps.append(f"call get_{source}_{target} with {source}_id={source_id}")
                continue
            relation = re.fullmatch(r"([a-z_]+)_id (\d+) from (?:current )?([a-z_]+) (\d+)", item)
            if relation is not None:
                entity, entity_id, _, _ = relation.groups()
                steps.append(f"call get_{entity}_title or get_{entity}_name with {entity}_id={entity_id}, whichever tool exists")
                continue
            display = re.fullmatch(r"([a-z_]+)_id (\d+) title/name", item)
            if display is not None:
                entity, entity_id = display.groups()
                steps.append(f"call get_{entity}_title or get_{entity}_name with {entity}_id={entity_id}, whichever tool exists")
        return "; ".join(dict.fromkeys(steps))

    @staticmethod
    def _missing_resolutions(messages: list[Any], user_text: str, final_text: str) -> list[str]:
        missing = EntityResolutionGuardMiddleware._missing_current_branch_resolutions(messages, user_text, final_text)
        missing.extend(EntityResolutionGuardMiddleware._missing_display_resolutions(messages, user_text, final_text))
        return list(dict.fromkeys(missing))

    @staticmethod
    def _before_nudge(state: EntityResolutionGuardState) -> dict[str, Any] | None:
        if state.get("nemotron_entity_pre_nudged"):
            return None
        messages = list(state.get("messages") or [])
        if not messages or not isinstance(messages[-1], ToolMessage):
            return None
        user_text = _last_external_human_text(messages)
        missing = EntityResolutionGuardMiddleware._missing_resolutions(messages, user_text, "")
        if not missing:
            return None
        steps = EntityResolutionGuardMiddleware._resolution_steps(missing)
        return _nudge_update(
            "Before answering, resolve each current-entity or ID branch with its "
            "own lookup result instead of reusing another branch's entity. "
            f"Missing resolution(s): {', '.join(missing)}. Required next "
            f"lookup(s): {steps}.",
            _ENTITY_GUARD_SOURCE,
            nemotron_entity_pre_nudged=True,
        )

    @staticmethod
    def _nudge(state: EntityResolutionGuardState) -> dict[str, Any] | None:
        if state.get("nemotron_entity_guard_fired"):
            return None
        messages = list(state.get("messages") or [])
        if _repair_loop_risk(messages) or not messages:
            return None
        last = messages[-1]
        if not isinstance(last, AIMessage) or not _is_final_answer(last):
            return None
        user_text = _last_external_human_text(messages[:-1])
        final_text = _message_text(last)
        missing = EntityResolutionGuardMiddleware._missing_resolutions(messages[:-1], user_text, final_text)
        if not missing:
            return None
        steps = EntityResolutionGuardMiddleware._resolution_steps(missing)
        return _nudge_update(
            "Your final answer is using or mixing opaque entity IDs before "
            "resolving them to user-facing details. Keep each branch bound to "
            f"the ID that produced it. Resolve these before answering: {', '.join(missing)}. "
            "If a matching name/details lookup tool is available, call it now, "
            f"then answer from that result. Required next lookup(s): {steps}. "
            "Do not reuse a name or details from a different entity or question "
            "branch.",
            _ENTITY_GUARD_SOURCE,
            jump_to="model",
            nemotron_entity_guard_fired=True,
        )

    def before_model(
        self,
        state: AgentState[Any],
        runtime: Runtime[Any],  # noqa: ARG002
    ) -> dict[str, Any] | None:
        """Inject entity-branch guidance before the model finalizes."""
        return self._before_nudge(cast("EntityResolutionGuardState", state))

    async def abefore_model(
        self,
        state: AgentState[Any],
        runtime: Runtime[Any],  # noqa: ARG002
    ) -> dict[str, Any] | None:
        """Async variant of `before_model`."""
        return self._before_nudge(cast("EntityResolutionGuardState", state))

    @hook_config(can_jump_to=["model"])
    def after_agent(
        self,
        state: AgentState[Any],
        runtime: Runtime[Any],  # noqa: ARG002
    ) -> dict[str, Any] | None:
        """Loop once when final text needs ID resolution or branch rebinding."""
        return self._nudge(cast("EntityResolutionGuardState", state))

    @hook_config(can_jump_to=["model"])
    async def aafter_agent(
        self,
        state: AgentState[Any],
        runtime: Runtime[Any],  # noqa: ARG002
    ) -> dict[str, Any] | None:
        """Async variant of `after_agent`."""
        return self._nudge(cast("EntityResolutionGuardState", state))


class FinalAnswerGuardState(AgentState):
    """State schema for `FinalAnswerGuardMiddleware`."""

    nemotron_final_guard_fired: NotRequired[Annotated[bool, PrivateStateAttr]]


class FinalAnswerGuardMiddleware(AgentMiddleware):
    """Send Ultra3 back once when a final answer drops obvious required details."""

    name = "FinalAnswerGuardMiddleware"
    state_schema = FinalAnswerGuardState

    @staticmethod
    def _literal_nudge(tool_calls: list[ToolCall], final_text_lower: str) -> dict[str, Any] | None:
        missing_literals = [literal for literal in _mutation_literal_values(tool_calls) if literal.lower() not in final_text_lower]
        if missing_literals:
            return _nudge_update(
                "Your final answer omitted exact literal value(s) from the "
                f"completed tool action: {', '.join(missing_literals)}. Answer "
                "again and include each literal exactly, along with the concrete "
                "result.",
                _FINAL_GUARD_SOURCE,
                jump_to="model",
                nemotron_final_guard_fired=True,
            )
        return None

    @staticmethod
    def _mutation_result_nudge(mutation_result: tuple[ToolCall, object] | None, final_text: str) -> dict[str, Any] | None:
        if mutation_result is not None:
            call, value = mutation_result
            result_summary = _format_budget_value(value)
            if result_summary and _VAGUE_COMPLETION_RE.fullmatch(final_text.strip()):
                return _nudge_update(
                    "Your final answer should communicate the concrete outcome "
                    "of the completed state-changing tool call. Latest mutation "
                    f"tool: {call.get('name', 'tool')}. Observed result: "
                    f"{result_summary}. Answer again from that result, including "
                    "what changed and any important status, amount, date/time, "
                    "identifier, or remaining caveat present in the tool result.",
                    _FINAL_GUARD_SOURCE,
                    jump_to="model",
                    nemotron_final_guard_fired=True,
                )
        return None

    @staticmethod
    def _nudge(state: FinalAnswerGuardState) -> dict[str, Any] | None:
        messages = list(state.get("messages") or [])
        if state.get("nemotron_final_guard_fired") or _repair_loop_risk(messages) or not messages:
            return None
        last = messages[-1]
        if not isinstance(last, AIMessage) or not _is_final_answer(last):
            return None

        final_text = _message_text(last)
        final_text_lower = final_text.lower()
        user_text = _last_external_human_text(messages[:-1])
        if _satisfies_exact_final_answer_request(user_text, final_text):
            return None
        tool_calls = _iter_tool_calls(messages[:-1])

        return FinalAnswerGuardMiddleware._literal_nudge(
            tool_calls,
            final_text_lower,
        ) or FinalAnswerGuardMiddleware._mutation_result_nudge(
            _last_domain_mutation_result(messages[:-1]),
            final_text,
        )

    @hook_config(can_jump_to=["model"])
    def after_agent(
        self,
        state: AgentState[Any],
        runtime: Runtime[Any],  # noqa: ARG002
    ) -> dict[str, Any] | None:
        """Loop once when the final answer misses concrete tool-derived details."""
        return self._nudge(cast("FinalAnswerGuardState", state))

    @hook_config(can_jump_to=["model"])
    async def aafter_agent(
        self,
        state: AgentState[Any],
        runtime: Runtime[Any],  # noqa: ARG002
    ) -> dict[str, Any] | None:
        """Async variant of `after_agent`."""
        return self._nudge(cast("FinalAnswerGuardState", state))


_SYSTEM_PROMPT_SUFFIX: str = """\
<approach>
Plan briefly before acting. When several reads or lookups are independent, issue them as parallel tool calls rather than one at a time.
</approach>

<grounding>
Verify state with tools instead of recalling it. Read files before describing
them, use lookup tools for identifiers, and use mutation tools before saying a
requested change is done.
</grounding>

<loop_control>
If a tool call fails, read the error and change the call before retrying; never
re-issue the same failing call unchanged. If a command times out or the same
error repeats, reduce the input, add a termination condition, or switch
approaches before trying again.
</loop_control>

<tool_selection>
Use filesystem tools only for file, path, repository-content, or document
questions. For API, operational, business-object, or other domain questions,
prefer the task-specific non-filesystem tools. For ranking, counting, "which",
or "most" questions over domain entities, enumerate or search candidate
entities with domain tools, fetch the relevant details or counts with matching
domain tools, compare the observed tool results, and answer from that
comparison.
</tool_selection>

<state_changes>
If the user asks to book, cancel, update, send, notify, create, or otherwise
change external state, the change is complete only after the relevant tool call
succeeds. Do not merely describe the intended action or ask the user to assume it
happened. After a successful mutation, use the tool result as the source of truth
for the final answer.
</state_changes>

<final_answer_completeness>
After tool calls succeed, the final answer must include the concrete result, not
just "done". Preserve short exact literals that identify the completed action,
especially versions, titles, and subjects from the user's request or successful
mutation tool arguments/results. If you used an opaque entity ID and an obvious
name or detail lookup tool is available, resolve the ID to human-readable
details before answering. If the user asked multiple questions, answer each one
from its matching tool output; do not substitute an entity from another subtask.
</final_answer_completeness>

<followup_defaults>
Ask follow-up questions only for information needed to proceed safely or
correctly. Do not re-ask for constraints the user already gave. For broad
analysis requests, ask for both the data source and the analysis goal before
using tools. For recurring reports, summaries, monitoring, or support workflows,
treat a stated cadence as sufficient and ask only for missing content, source,
threshold, delivery, or domain details needed to perform the task.
</followup_defaults>

<context_compaction>
If a long conversation switches to a completely unrelated new task and the
compact_conversation tool is available, call compact_conversation before starting
the new task. Also call compact_conversation before reading or summarizing a
large new file after a long conversation.
</context_compaction>"""


def _build_extra_middleware() -> list[AgentMiddleware]:
    return [
        NemotronProgressBudgetMiddleware(),
        NemotronPolicyNudgeMiddleware(),
        NemotronToolCallShim(),
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
        ModelRateLimitRetryMiddleware(),
        ChatNVIDIAMessageCompatibilityMiddleware(),
        NemotronReasoningTagCleanupMiddleware(),
        NemotronTextToolCallParser(),
        FollowupDisciplineMiddleware(),
        EntityResolutionGuardMiddleware(),
        FinalAnswerGuardMiddleware(),
    ]


def register() -> None:
    """Register the built-in Nemotron 3 Ultra harness profile."""
    profile = HarnessProfile(
        system_prompt_suffix=_SYSTEM_PROMPT_SUFFIX,
        tool_description_overrides={"read_file": _READ_FILE_DESCRIPTION_OVERRIDE},
        extra_middleware=_build_extra_middleware,
    )
    for spec in _NEMOTRON_ULTRA_MODEL_SPECS:
        _register_harness_profile_impl(spec, profile)
