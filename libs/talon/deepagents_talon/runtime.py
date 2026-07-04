"""Deep Agents runtime used by the Talon host.

Talon is an experimental runtime and is subject to change or removal at any time.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import os
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeGuard, cast

from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend
from deepagents.middleware.summarization import (
    SummarizationToolMiddleware,
    create_summarization_tool_middleware,
)
from deepagents.profiles.provider.provider_profiles import apply_provider_profile
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from deepagents_code.tools import fetch_url, web_search
from deepagents_talon.cron import CronJobStore, CronOrigin, CronTools
from deepagents_talon.interfaces import (
    AgentRequest,
    AgentResult,
    ToolApprovalDecision,
    ToolApprovalHandler,
    ToolApprovalRequest,
)
from deepagents_talon.observability import log_event, stable_log_ref

if TYPE_CHECKING:
    from deepagents import AsyncSubAgent, CompiledSubAgent, SubAgent
    from deepagents.backends.protocol import BackendProtocol
    from langchain.agents.middleware import InterruptOnConfig
    from langchain.agents.middleware.types import AgentMiddleware
    from langchain_core.language_models import BaseChatModel
    from langchain_core.tools import BaseTool
    from langgraph.types import Checkpointer

logger = logging.getLogger(__name__)

DEFAULT_RECURSION_LIMIT = 500
DEFAULT_MAX_RETRIES = 3
DEFAULT_MAX_CONTINUATIONS = 3
DEFAULT_MAX_APPROVAL_ROUNDS = 50
CONTEXT_SIZE_ENV_KEY = "DEEPAGENTS_TALON_CONTEXT_SIZE"
INTERRUPT_ON_TOOLS_ENV_KEY = "DEEPAGENTS_TALON_INTERRUPT_ON_TOOLS"
RECURSION_LIMIT_ENV_KEY = "DEEPAGENTS_TALON_RECURSION_LIMIT"
_WORKSPACE_ENV = "DEEPAGENTS_TALON_WORKSPACE"
_SAFE_BACKEND_PATH = "/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin"
ModelContent = str | list[dict[str, object]]

_BAD_REQUEST_STATUS_CODE = 400
_RETRYABLE_STATUS_CODES = frozenset({408, 409, 413, 429, 500, 502, 503, 504})
_BACKEND_ENV_ALLOWED_KEYS = frozenset(
    {
        "CI",
        "CLICOLOR",
        "CLICOLOR_FORCE",
        "COLORTERM",
        "FORCE_COLOR",
        "HOME",
        "LANG",
        "LOGNAME",
        "NO_COLOR",
        "SHELL",
        "TEMP",
        "TERM",
        "TMP",
        "TMPDIR",
        "TZ",
        "USER",
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "XDG_RUNTIME_DIR",
        "XDG_STATE_HOME",
    }
)
_BACKEND_ENV_ALLOWED_PREFIXES = ("LC_",)
_BACKEND_ENV_HIJACK_KEYS = frozenset(
    {
        "BASH_ENV",
        "DYLD_INSERT_LIBRARIES",
        "DYLD_LIBRARY_PATH",
        "ENV",
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
        "PYTHONHOME",
        "PYTHONPATH",
        "ZDOTDIR",
    }
)
_BACKEND_ENV_SECRET_MARKERS = (
    "APIKEY",
    "API_KEY",
    "AUTHORIZATION",
    "BEARER",
    "CREDENTIAL",
    "OAUTH",
    "PASSWORD",
    "SECRET",
    "TOKEN",
)
_RETRYABLE_BAD_REQUEST_MARKERS = (
    "failed to parse",
    "tool_call",
    "tool call",
    "context length",
    "context window",
    "context limit",
    "maximum context",
    "max context",
    "input too long",
    "request too large",
)
_RETRYABLE_MESSAGE_MARKERS = (
    *_RETRYABLE_BAD_REQUEST_MARKERS,
    "connection aborted",
    "connection closed",
    "connection lost",
    "connection refused",
    "connection reset",
    "connection timed out",
    "read timeout",
    "timed out",
    "temporarily unavailable",
    "temporary failure",
)

_CONTINUATION_NUDGE = (
    "Your action budget was exhausted mid-task. Continue working and complete the task. "
    "If you have already finished, provide your final answer now."
)
_FORCE_SUMMARY_PROMPT = (
    "You ran out of actions. Provide a concise summary of everything you have "
    "accomplished so far. Do not call any more tools."
)
_CRON_AUTO_DENY_MESSAGE = (
    "Tool approval is unavailable for scheduled runs; skipped the gated tool call."
)
_CHANNEL_AUTO_DENY_MESSAGE = (
    "Tool approval is unavailable on this channel; skipped the gated tool call."
)

_CRON_ORIGIN: contextvars.ContextVar[CronOrigin | None] = contextvars.ContextVar(
    "talon_cron_origin",
    default=None,
)


class EchoAgentRuntime:
    """Small placeholder runtime for host bootstrapping and tests."""

    async def start(self) -> None:
        """Initialize the placeholder runtime."""

    async def stop(self) -> None:
        """Release placeholder runtime resources."""

    async def invoke(self, request: AgentRequest) -> AgentResult:
        """Return the request text as a trivial agent response.

        Args:
            request: Agent request supplied by the Talon host.

        Returns:
            Echo response tagged as placeholder runtime output.
        """
        return AgentResult(text=request.text)


@dataclass(frozen=True, slots=True)
class RuntimeAgentComponents:
    """Runtime components used when rebuilding a `DeepAgentRuntime` graph.

    Args:
        model: Chat model identifier for `create_deep_agent`.
        tools: Runtime tools exposed to the agent.
        system_prompt: Optional system prompt.
        subagents: Optional subagent specs available to the main agent.
        skills: Optional explicit skill source paths.
        middleware: Optional middleware to pass through to `create_deep_agent`.
        interrupt_on: Optional human-in-the-loop tool approval configuration.
    """

    model: str
    tools: Sequence[BaseTool | Callable[..., object]] = ()
    system_prompt: str | None = None
    subagents: Sequence[SubAgent | CompiledSubAgent | AsyncSubAgent] | None = None
    skills: Sequence[str] | None = None
    middleware: Sequence[AgentMiddleware[Any, Any, Any]] = ()
    interrupt_on: Mapping[str, bool | InterruptOnConfig] | None = None


@dataclass(frozen=True, slots=True)
class _ApprovalAuditContext:
    interrupt_id: str
    conversation_ref: str
    trigger: str
    action_count: int
    action_names: tuple[str, ...]


class DeepAgentRuntime:
    """Deep Agents-backed runtime for Talon.

    Args:
        model: Chat model identifier for `create_deep_agent`.
        tools: Runtime tools exposed to the agent in addition to web and cron tools.
        system_prompt: Optional system prompt. When omitted and `assistant_dir`
            is supplied, `AGENTS.md` is loaded from that directory.
        subagents: Optional subagent specs available to the main agent.
        assistant_dir: Materialized assistant directory containing `AGENTS.md`,
            `skills/`, and optional manifest memory metadata.
        cron_store: Optional cron store. When supplied, cron management tools
            are scoped to the current request origin and exposed to the agent.
        backend: Filesystem/execution backend. Defaults to local shell execution.
        skills: Optional explicit skill source paths. When omitted, sources are
            loaded from `assistant_dir/skills` and skill directory environment vars.
        middleware: Optional middleware to pass through to `create_deep_agent`.
        interrupt_on: Optional human-in-the-loop tool approval configuration
            to pass through to `create_deep_agent`.
        memory: Optional explicit memory file paths. When omitted, paths are
            loaded from manifest metadata, memory path environment vars, or an
            assistant-local memory file.
        checkpointer: Optional LangGraph checkpointer. Defaults to in-memory
            checkpointing so turns in the same conversation share chat history.
        include_web_tools: Whether to include fetch/search/request tools.
        recursion_limit: Per-invocation graph recursion limit.
        max_retries: Retries for transient provider, parse, context-limit, and
            transport errors.
        max_continuations: Number of continuation nudges after empty responses.
    """

    def __init__(  # noqa: PLR0913  # runtime construction mirrors graph wiring knobs
        self,
        *,
        model: str,
        tools: Sequence[BaseTool | Callable[..., object]] = (),
        system_prompt: str | None = None,
        subagents: Sequence[SubAgent | CompiledSubAgent | AsyncSubAgent] | None = None,
        assistant_dir: Path | None = None,
        cron_store: CronJobStore | None = None,
        backend: BackendProtocol | None = None,
        skills: Sequence[str] | None = None,
        middleware: Sequence[AgentMiddleware[Any, Any, Any]] = (),
        interrupt_on: Mapping[str, bool | InterruptOnConfig] | None = None,
        memory: Sequence[str] | None = None,
        checkpointer: Checkpointer | None = None,
        include_web_tools: bool = True,
        recursion_limit: int = DEFAULT_RECURSION_LIMIT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        max_continuations: int = DEFAULT_MAX_CONTINUATIONS,
        env: Mapping[str, str] | None = None,
    ) -> None:
        """Initialize without constructing the graph."""
        values = os.environ if env is None else env
        resolved_recursion_limit = _recursion_limit_from_env(values, recursion_limit)
        if resolved_recursion_limit <= 0:
            msg = "recursion_limit must be positive"
            raise ValueError(msg)
        if max_retries < 1:
            msg = "max_retries must be at least 1"
            raise ValueError(msg)
        if max_continuations < 0:
            msg = "max_continuations cannot be negative"
            raise ValueError(msg)

        self.model = model
        self.tools = tuple(tools)
        self.system_prompt = system_prompt
        self.subagents = tuple(subagents) if subagents is not None else None
        self.assistant_dir = assistant_dir
        self.cron_store = cron_store
        self.env = dict(os.environ if env is None else env)
        self.backend = backend if backend is not None else _default_backend(self.env)
        self.skills = tuple(skills) if skills is not None else None
        self.middleware = tuple(middleware)
        self.interrupt_on = interrupt_on_with_env_overlay(interrupt_on, self.env)
        self.memory = tuple(memory) if memory is not None else None
        self.checkpointer = checkpointer if checkpointer is not None else InMemorySaver()
        self.include_web_tools = include_web_tools
        self.recursion_limit = resolved_recursion_limit
        self.max_retries = max_retries
        self.max_continuations = max_continuations
        self._graph: object | None = None

    async def start(self) -> None:
        """Construct the Deep Agents graph."""
        tools = self._build_tools()
        context_size = _context_size_from_env(self.env)
        model = _resolve_model_from_env(self.model, self.env, context_size=context_size)
        middleware = list(self.middleware)
        if context_size is not None and not _has_summarization_tool_middleware(middleware):
            middleware.append(create_summarization_tool_middleware(model, self.backend))
        self._graph = create_deep_agent(
            model=model,
            tools=tools,
            system_prompt=self._resolve_system_prompt(),
            subagents=list(self.subagents) if self.subagents is not None else None,
            backend=self.backend,
            skills=self._resolve_skills(),
            middleware=middleware,
            interrupt_on=self.interrupt_on,
            memory=self._resolve_memory(),
            checkpointer=self.checkpointer,
        )

    async def stop(self) -> None:
        """Release runtime resources."""
        self._graph = None
        cleanup = getattr(self.checkpointer, "close", None)
        if callable(cleanup):
            result = cleanup()
            if isinstance(result, Awaitable):
                await result

    async def invoke(self, request: AgentRequest) -> AgentResult:
        """Invoke the Deep Agents graph for one Talon request.

        Args:
            request: Agent request supplied by the Talon host.

        Returns:
            Final assistant text from the graph.

        Raises:
            RuntimeError: If the runtime has not been started.
        """
        if self._graph is None:
            msg = "DeepAgentRuntime must be started before invoke"
            raise RuntimeError(msg)

        token = _CRON_ORIGIN.set(_cron_origin_from_request(request))
        try:
            text = await self._invoke_until_text(request)
        finally:
            _CRON_ORIGIN.reset(token)
        return AgentResult(text=text)

    def _build_tools(self) -> list[BaseTool | Callable[..., object]]:
        tools: list[BaseTool | Callable[..., object]] = []
        if self.include_web_tools:
            tools.extend([fetch_url, web_search])
        if self.cron_store is not None:
            cron = CronTools(store=self.cron_store, origin=_current_cron_origin)
            tools.extend(cron.as_langchain_tools())
        tools.extend(self.tools)
        return tools

    async def _invoke_until_text(self, request: AgentRequest) -> str:
        state = await self._invoke_until_unblocked(
            _request_model_content(request),
            request,
        )
        text = _last_text(state)
        if text:
            return text

        for attempt in range(self.max_continuations):
            logger.warning(
                "Agent returned no text for conversation %s; sending continuation nudge %d/%d",
                request.conversation_id,
                attempt + 1,
                self.max_continuations,
            )
            state = await self._invoke_until_unblocked(
                _CONTINUATION_NUDGE,
                request,
            )
            text = _last_text(state)
            if text:
                return text

        state = await self._invoke_until_unblocked(
            _FORCE_SUMMARY_PROMPT,
            request,
        )
        return _last_text(state)

    async def _invoke_with_retries(self, content: ModelContent, conversation_id: str) -> object:
        return await self._invoke_payload_with_retries(
            {"messages": [{"role": "user", "content": content}]},
            conversation_id,
        )

    async def _resume_with_retries(self, command: Command, conversation_id: str) -> object:
        return await self._invoke_payload_with_retries(command, conversation_id)

    async def _invoke_payload_with_retries(self, payload: object, conversation_id: str) -> object:
        invoke = self._graph_invoke()
        config = {
            "recursion_limit": self.recursion_limit,
            "configurable": {"thread_id": conversation_id},
        }
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                return await invoke(payload, config=config)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if not _is_retryable(exc) or attempt + 1 >= self.max_retries:
                    raise
                last_exc = exc
                backoff = min(2**attempt, 10)
                logger.warning(
                    "Retryable agent error in conversation %s; retrying in %ds: %s",
                    conversation_id,
                    backoff,
                    exc,
                )
                await asyncio.sleep(backoff)
        if last_exc is not None:
            raise last_exc
        msg = "agent invocation retry loop exited unexpectedly"
        raise RuntimeError(msg)

    def _graph_invoke(self) -> Callable[..., Awaitable[object]]:
        ainvoke = getattr(self._graph, "ainvoke", None)
        if not callable(ainvoke):
            msg = "Deep Agents graph does not expose async invocation"
            raise TypeError(msg)
        return cast("Callable[..., Awaitable[object]]", ainvoke)

    async def _invoke_until_unblocked(
        self,
        content: ModelContent,
        request: AgentRequest,
    ) -> object:
        state = await self._invoke_with_retries(content, request.conversation_id)
        for _ in range(DEFAULT_MAX_APPROVAL_ROUNDS):
            interrupts = _interrupts_from_state(state)
            if not interrupts:
                return state
            resume = await self._build_approval_resume(request, interrupts)
            state = await self._resume_with_retries(resume, request.conversation_id)
        msg = "agent hit tool approval interrupt limit"
        raise RuntimeError(msg)

    async def _build_approval_resume(
        self,
        request: AgentRequest,
        interrupts: Sequence[object],
    ) -> Command:
        payload: dict[str, dict[str, list[dict[str, str]]]] = {}
        for interrupt in interrupts:
            interrupt_id = _interrupt_id(interrupt)
            if interrupt_id is None:
                logger.warning("Received tool approval interrupt without an id")
                continue
            action_requests = _action_requests_from_interrupt(interrupt)
            decision, reject_message, _resolution = await _approval_decision(
                request,
                interrupt_id,
                action_requests,
            )
            payload[interrupt_id] = {
                "decisions": _decision_payload(
                    decision,
                    count=max(len(action_requests), 1),
                    reject_message=reject_message,
                )
            }
        if not payload:
            msg = "agent returned approval interrupts without resumable ids"
            raise RuntimeError(msg)
        return Command(resume=payload)

    def _resolve_system_prompt(self) -> str | None:
        if self.system_prompt is not None:
            return self.system_prompt
        if self.assistant_dir is None:
            return None
        path = self.assistant_dir / "AGENTS.md"
        try:
            if path.is_file():
                return path.read_text(encoding="utf-8")
        except OSError:
            logger.warning("Could not read Talon system prompt from %s", path, exc_info=True)
        return None

    def _resolve_skills(self) -> list[str] | None:
        if self.skills is not None:
            return list(self.skills) or None
        sources: list[str] = []
        if self.assistant_dir is not None:
            skills_dir = self.assistant_dir / "skills"
            try:
                skills_dir.mkdir(parents=True, exist_ok=True)
                sources.append(str(skills_dir))
            except OSError:
                logger.warning("Could not create Talon skills dir %s", skills_dir, exc_info=True)

        for path in _split_path_env(
            self.env.get("DEEPAGENTS_TALON_SKILLS_DIRS") or self.env.get("SKILLS_DIRS"),
        ):
            if path not in sources:
                sources.append(path)
        return sources or None

    def _resolve_memory(self) -> list[str] | None:
        if self.memory is not None:
            return list(self.memory) or None
        paths = _split_path_env(
            self.env.get("DEEPAGENTS_TALON_MEMORY_PATHS") or self.env.get("AGENT_MEMORY_PATHS"),
        )
        if not paths and self.assistant_dir is not None:
            paths.extend(_manifest_memory_paths(self.assistant_dir))
        if not paths and self.assistant_dir is not None:
            paths.append(str(self.assistant_dir / "memory" / "AGENTS.md"))
        prepared = [_prepare_memory_path(path) for path in paths]
        return [path for path in prepared if path is not None] or None


def _interrupts_from_state(state: object) -> tuple[object, ...]:
    if not isinstance(state, Mapping):
        return ()
    data = cast("Mapping[str, object]", state)
    interrupts = data.get("__interrupt__")
    if not isinstance(interrupts, Sequence) or isinstance(interrupts, (str, bytes, bytearray)):
        return ()
    return tuple(interrupts)


def _interrupt_id(interrupt: object) -> str | None:
    value = getattr(interrupt, "id", None)
    return value if isinstance(value, str) and value else None


def _action_requests_from_interrupt(interrupt: object) -> tuple[Mapping[str, object], ...]:
    value = getattr(interrupt, "value", None)
    if not isinstance(value, Mapping):
        logger.warning("Received malformed tool approval interrupt: missing value mapping")
        return ()
    data = cast("Mapping[str, object]", value)
    requests = data.get("action_requests")
    if not isinstance(requests, Sequence) or isinstance(requests, (str, bytes, bytearray)):
        logger.warning("Received malformed tool approval interrupt: missing action_requests")
        return ()

    parsed: list[Mapping[str, object]] = []
    for item in requests:
        if isinstance(item, Mapping):
            parsed.append(cast("Mapping[str, object]", item))
        else:
            logger.warning("Ignoring malformed tool approval action request: %r", item)
    return tuple(parsed)


async def _approval_decision(
    request: AgentRequest,
    interrupt_id: str,
    action_requests: Sequence[Mapping[str, object]],
) -> tuple[ToolApprovalDecision, str | None, str]:
    audit = _approval_audit_context(request, interrupt_id, action_requests)
    _log_approval_interrupt(audit)

    if request.metadata.get("trigger") == "cron":
        logger.warning(
            "Auto-denying %d tool approval request(s) for cron conversation %s",
            len(action_requests),
            audit.conversation_ref,
        )
        _log_approval_resolution(audit, decision="reject", resolution="cron_auto_deny")
        return "reject", _CRON_AUTO_DENY_MESSAGE, "cron_auto_deny"

    handler = _approval_handler_from_request(request)
    if handler is None:
        logger.warning(
            "Auto-denying %d tool approval request(s) for conversation %s without approval handler",
            len(action_requests),
            audit.conversation_ref,
        )
        _log_approval_resolution(audit, decision="reject", resolution="channel_auto_deny")
        return "reject", _CHANNEL_AUTO_DENY_MESSAGE, "channel_auto_deny"

    decision = await handler(
        ToolApprovalRequest(
            conversation_id=request.conversation_id,
            interrupt_id=interrupt_id,
            action_requests=tuple(action_requests),
        )
    )
    if decision == "approve":
        _log_approval_resolution(audit, decision="approve", resolution="operator")
        return "approve", None, "operator"
    _log_approval_resolution(audit, decision="reject", resolution="operator")
    return "reject", "Denied by operator.", "operator"


def _approval_audit_context(
    request: AgentRequest,
    interrupt_id: str,
    action_requests: Sequence[Mapping[str, object]],
) -> _ApprovalAuditContext:
    trigger = request.metadata.get("trigger")
    trigger_name = trigger if isinstance(trigger, str) and trigger else "channel"
    return _ApprovalAuditContext(
        interrupt_id=interrupt_id,
        conversation_ref=stable_log_ref(request.conversation_id),
        trigger=trigger_name,
        action_count=len(action_requests),
        action_names=_approval_action_names(action_requests),
    )


def _log_approval_interrupt(audit: _ApprovalAuditContext) -> None:
    log_event(
        logger,
        "tool_approval.interrupt",
        action_count=audit.action_count,
        action_names=audit.action_names,
        conversation_ref=audit.conversation_ref,
        interrupt_id=audit.interrupt_id,
        trigger=audit.trigger,
    )


def _log_approval_resolution(
    audit: _ApprovalAuditContext,
    *,
    decision: ToolApprovalDecision,
    resolution: str,
) -> None:
    log_event(
        logger,
        "tool_approval.resolved",
        action_count=audit.action_count,
        action_names=audit.action_names,
        conversation_ref=audit.conversation_ref,
        decision="approved" if decision == "approve" else "denied",
        interrupt_id=audit.interrupt_id,
        resolution=resolution,
        trigger=audit.trigger,
    )


def _approval_action_names(
    action_requests: Sequence[Mapping[str, object]],
) -> tuple[str, ...]:
    names: list[str] = []
    for action in action_requests:
        name = action.get("name")
        names.append(name if isinstance(name, str) and name else "unknown")
    return tuple(names)


def _approval_handler_from_request(request: AgentRequest) -> ToolApprovalHandler | None:
    return request.approval_handler


def _decision_payload(
    decision: ToolApprovalDecision,
    *,
    count: int,
    reject_message: str | None,
) -> list[dict[str, str]]:
    if decision == "approve":
        return [{"type": "approve"} for _ in range(count)]
    if reject_message:
        return [{"type": "reject", "message": reject_message} for _ in range(count)]
    return [{"type": "reject"} for _ in range(count)]


def interrupt_on_with_env_overlay(
    interrupt_on: Mapping[str, bool | InterruptOnConfig] | None,
    env: Mapping[str, str],
) -> dict[str, bool | InterruptOnConfig] | None:
    """Merge Talon's local tool approval env overlay into an `interrupt_on` mapping.

    Args:
        interrupt_on: Base human-in-the-loop tool approval configuration.
        env: Environment values to inspect for Talon approval overrides.

    Returns:
        Merged approval configuration, or `None` when neither source configures
        approval.
    """
    overlay = _interrupt_on_tools_from_env(env)
    if interrupt_on is None and not overlay:
        return None

    merged: dict[str, bool | InterruptOnConfig] = {}
    if interrupt_on is not None:
        merged.update(interrupt_on)
    merged.update(overlay)
    return merged


def _interrupt_on_tools_from_env(env: Mapping[str, str]) -> dict[str, bool]:
    raw = env.get(INTERRUPT_ON_TOOLS_ENV_KEY)
    if raw is None or not raw.strip():
        return {}
    return {name: True for name in (part.strip() for part in raw.split(",")) if name}


def _default_backend(env: Mapping[str, str] | None) -> LocalShellBackend:
    values = os.environ if env is None else env
    root = values.get(_WORKSPACE_ENV) or None
    return LocalShellBackend(
        root_dir=root,
        virtual_mode=False,
        env=_backend_child_env(values),
        inherit_env=False,
    )


def _backend_child_env(env: Mapping[str, str]) -> dict[str, str]:
    values = {
        key: value
        for key, value in env.items()
        if _is_allowed_backend_env_key(key) and not _is_scrubbed_backend_env_key(key)
    }
    values["PATH"] = _SAFE_BACKEND_PATH
    return values


def _is_allowed_backend_env_key(key: str) -> bool:
    return key in _BACKEND_ENV_ALLOWED_KEYS or key.startswith(_BACKEND_ENV_ALLOWED_PREFIXES)


def _is_scrubbed_backend_env_key(key: str) -> bool:
    return (
        key in _BACKEND_ENV_HIJACK_KEYS
        or key.startswith(("LANGSMITH_", "LANGCHAIN_"))
        or any(marker in key for marker in _BACKEND_ENV_SECRET_MARKERS)
    )


def _resolve_model_from_env(
    model: str,
    env: Mapping[str, str],
    *,
    context_size: int | None = None,
) -> str | BaseChatModel:
    base_url = env.get("OPENAI_BASE_URL")
    if context_size is None and (not base_url or not _is_openai_model(model)):
        return model

    init_kwargs = apply_provider_profile(model)
    if base_url and _is_openai_model(model):
        init_kwargs["base_url"] = base_url

    resolved = init_chat_model(model, **init_kwargs)
    if context_size is not None:
        _apply_context_size(resolved, context_size)
    return resolved


def _context_size_from_env(env: Mapping[str, str]) -> int | None:
    return _positive_int_from_env(env, CONTEXT_SIZE_ENV_KEY)


def _recursion_limit_from_env(env: Mapping[str, str], fallback: int) -> int:
    """Resolve the recursion limit from the environment with a code fallback.

    The `DEEPAGENTS_TALON_RECURSION_LIMIT` env var, when set, overrides the
    caller-supplied value so operators can tune the graph recursion limit
    without changing code. Falls back to the caller value when unset.
    """
    resolved = _positive_int_from_env(env, RECURSION_LIMIT_ENV_KEY)
    return resolved if resolved is not None else fallback


def _positive_int_from_env(env: Mapping[str, str], key: str) -> int | None:
    raw = env.get(key)
    if raw is None or not raw.strip():
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        msg = f"{key} must be a positive integer"
        raise ValueError(msg) from exc
    if value <= 0:
        msg = f"{key} must be a positive integer"
        raise ValueError(msg)
    return value


def _has_summarization_tool_middleware(
    middleware: Sequence[AgentMiddleware[Any, Any, Any]],
) -> bool:
    return any(isinstance(item, SummarizationToolMiddleware) for item in middleware)


def _apply_context_size(model: BaseChatModel, context_size: int) -> None:
    profile = getattr(model, "profile", None)
    merged = (
        {**profile, "max_input_tokens": context_size}
        if isinstance(profile, dict)
        else {"max_input_tokens": context_size}
    )
    try:
        cast("Any", model).profile = merged
    except (AttributeError, TypeError, ValueError) as exc:
        msg = f"Could not apply {CONTEXT_SIZE_ENV_KEY} to model profile"
        raise ValueError(msg) from exc


def _is_openai_model(model: str) -> bool:
    return model.startswith("openai:")


def _current_cron_origin() -> CronOrigin:
    origin = _CRON_ORIGIN.get()
    if origin is None:
        msg = "cron tools must be called from within a Talon conversation"
        raise RuntimeError(msg)
    return origin


def _cron_origin_from_request(request: AgentRequest) -> CronOrigin:
    channel = request.metadata.get("channel")
    message_id = request.metadata.get("message_id")
    origin_conversation_id = request.metadata.get("origin_conversation_id")
    return CronOrigin(
        conversation_id=(
            origin_conversation_id
            if isinstance(origin_conversation_id, str) and origin_conversation_id
            else request.conversation_id
        ),
        channel=channel if isinstance(channel, str) else None,
        message_id=message_id if isinstance(message_id, str) else None,
    )


def _request_model_content(request: AgentRequest) -> ModelContent:
    content = request.metadata.get("model_content")
    if _is_model_content(content):
        return content
    return request.text


def _is_model_content(value: object) -> TypeGuard[list[dict[str, object]]]:
    return isinstance(value, list) and all(isinstance(item, dict) for item in value)


def _split_path_env(raw: str | None) -> list[str]:
    if not raw:
        return []
    separator = ";" if ";" in raw else os.pathsep
    return [str(Path(part).expanduser()) for part in raw.split(separator) if part.strip()]


def _manifest_memory_paths(assistant_dir: Path) -> list[str]:
    path = assistant_dir / "manifest.json"
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        logger.warning("Could not read Talon manifest memory paths from %s", path, exc_info=True)
        return []
    if not isinstance(data, dict):
        return []
    memory = data.get("memory")
    raw = memory.get("paths") if isinstance(memory, dict) else data.get("memory_paths")
    if not isinstance(raw, list):
        return []

    paths: list[str] = []
    for item in raw:
        if not isinstance(item, str) or not item:
            continue
        candidate = Path(item).expanduser()
        if not candidate.is_absolute():
            candidate = assistant_dir / candidate
        paths.append(str(candidate))
    return paths


def _prepare_memory_path(raw: str) -> str | None:
    path = Path(raw).expanduser()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.touch()
        return str(path)
    except OSError:
        logger.warning("Could not prepare Talon memory file %s", path, exc_info=True)
        return None


def _status_code(exc: BaseException) -> int | None:
    for source in (exc, getattr(exc, "response", None)):
        if source is None:
            continue
        for attr in ("status_code", "status"):
            value = getattr(source, attr, None)
            if isinstance(value, int):
                return value
    if isinstance(exc, BaseExceptionGroup):
        for item in exc.exceptions:
            value = _status_code(item)
            if value is not None:
                return value
    return None


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True

    text = str(exc).lower()
    status_code = _status_code(exc)
    if status_code in _RETRYABLE_STATUS_CODES:
        return True
    if status_code == _BAD_REQUEST_STATUS_CODE:
        return _contains_marker(text, _RETRYABLE_BAD_REQUEST_MARKERS)
    return _contains_marker(text, _RETRYABLE_MESSAGE_MARKERS)


def _contains_marker(text: str, markers: Sequence[str]) -> bool:
    return any(marker in text for marker in markers)


def _last_text(state: object) -> str:
    if not isinstance(state, Mapping):
        return ""
    data = cast("Mapping[str, object]", state)
    messages = data.get("messages")
    if not isinstance(messages, list) or not messages:
        return ""
    last = messages[-1]
    if isinstance(last, Mapping):
        content = cast("Mapping[str, object]", last).get("content", "")
    else:
        content = getattr(last, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(_content_block_text(block) for block in content).strip()
    return ""


def _content_block_text(block: object) -> str:
    if isinstance(block, str):
        return block
    if isinstance(block, Mapping):
        data = cast("Mapping[str, object]", block)
        text = data.get("text")
        if isinstance(text, str):
            return text
    return ""
