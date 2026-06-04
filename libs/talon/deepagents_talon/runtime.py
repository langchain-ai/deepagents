"""Deep Agents runtime used by the Talon host."""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import os
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, TypeGuard, cast

from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend
from langgraph.checkpoint.memory import InMemorySaver

from deepagents_talon.cron import CronJobStore, CronOrigin, CronTools
from deepagents_talon.interfaces import AgentRequest, AgentResult
from deepagents_talon.tools import build_web_tools

if TYPE_CHECKING:
    from deepagents.backends.protocol import BackendProtocol
    from langchain_core.tools import BaseTool
    from langgraph.types import Checkpointer

logger = logging.getLogger(__name__)

DEFAULT_RECURSION_LIMIT = 150
DEFAULT_MAX_RETRIES = 3
DEFAULT_MAX_CONTINUATIONS = 3
ModelContent = str | list[dict[str, object]]

_CONTINUATION_NUDGE = (
    "Your action budget was exhausted mid-task. Continue working and complete the task. "
    "If you have already finished, provide your final answer now."
)
_FORCE_SUMMARY_PROMPT = (
    "You ran out of actions. Provide a concise summary of everything you have "
    "accomplished so far. Do not call any more tools."
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


class DeepAgentRuntime:
    """Deep Agents-backed runtime for Talon.

    Args:
        model: Chat model identifier for `create_deep_agent`.
        tools: Runtime tools exposed to the agent in addition to web and cron tools.
        system_prompt: Optional system prompt. When omitted and `assistant_dir`
            is supplied, `AGENTS.md` is loaded from that directory.
        assistant_dir: Materialized assistant directory containing `AGENTS.md`,
            `skills/`, and optional manifest memory metadata.
        cron_store: Optional cron store. When supplied, cron management tools
            are scoped to the current request origin and exposed to the agent.
        backend: Filesystem/execution backend. Defaults to local shell execution.
        skills: Optional explicit skill source paths. When omitted, sources are
            loaded from `assistant_dir/skills` and skill directory environment vars.
        memory: Optional explicit memory file paths. When omitted, paths are
            loaded from manifest metadata, memory path environment vars, or an
            assistant-local memory file.
        checkpointer: Optional LangGraph checkpointer. Defaults to in-memory
            checkpointing so turns in the same conversation share chat history.
        include_web_tools: Whether to include fetch/search/request tools.
        recursion_limit: Per-invocation graph recursion limit.
        max_retries: Retries for transient parse, context, and connection errors.
        max_continuations: Number of continuation nudges after empty responses.
    """

    def __init__(  # noqa: PLR0913  # runtime construction mirrors graph wiring knobs
        self,
        *,
        model: str,
        tools: Sequence[BaseTool | Callable[..., object]] = (),
        system_prompt: str | None = None,
        assistant_dir: Path | None = None,
        cron_store: CronJobStore | None = None,
        backend: BackendProtocol | None = None,
        skills: Sequence[str] | None = None,
        memory: Sequence[str] | None = None,
        checkpointer: Checkpointer | None = None,
        include_web_tools: bool = True,
        recursion_limit: int = DEFAULT_RECURSION_LIMIT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        max_continuations: int = DEFAULT_MAX_CONTINUATIONS,
        env: Mapping[str, str] | None = None,
    ) -> None:
        """Initialize without constructing the graph."""
        if recursion_limit <= 0:
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
        self.assistant_dir = assistant_dir
        self.cron_store = cron_store
        self.backend = backend if backend is not None else _default_backend()
        self.skills = tuple(skills) if skills is not None else None
        self.memory = tuple(memory) if memory is not None else None
        self.checkpointer = checkpointer if checkpointer is not None else InMemorySaver()
        self.include_web_tools = include_web_tools
        self.recursion_limit = recursion_limit
        self.max_retries = max_retries
        self.max_continuations = max_continuations
        self.env = dict(os.environ if env is None else env)
        self._graph: object | None = None

    async def start(self) -> None:
        """Construct the Deep Agents graph."""
        tools = self._build_tools()
        self._graph = create_deep_agent(
            model=self.model,
            tools=tools,
            system_prompt=self._resolve_system_prompt(),
            backend=self.backend,
            skills=self._resolve_skills(),
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
            tools.extend(build_web_tools())
        if self.cron_store is not None:
            cron = CronTools(store=self.cron_store, origin=_current_cron_origin)
            tools.extend(cron.as_langchain_tools())
        tools.extend(self.tools)
        return tools

    async def _invoke_until_text(self, request: AgentRequest) -> str:
        state = await self._invoke_with_retries(
            _request_model_content(request),
            request.conversation_id,
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
            state = await self._invoke_with_retries(
                _CONTINUATION_NUDGE,
                request.conversation_id,
            )
            text = _last_text(state)
            if text:
                return text

        state = await self._invoke_with_retries(
            _FORCE_SUMMARY_PROMPT,
            request.conversation_id,
        )
        return _last_text(state)

    async def _invoke_with_retries(self, content: ModelContent, conversation_id: str) -> object:
        ainvoke = getattr(self._graph, "ainvoke", None)
        if not callable(ainvoke):
            msg = "Deep Agents graph does not expose async invocation"
            raise TypeError(msg)

        invoke = cast("Callable[..., Awaitable[object]]", ainvoke)
        config = {
            "recursion_limit": self.recursion_limit,
            "configurable": {"thread_id": conversation_id},
        }
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                return await invoke(
                    {"messages": [{"role": "user", "content": content}]},
                    config=config,
                )
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


def _default_backend() -> LocalShellBackend:
    return LocalShellBackend(virtual_mode=False, inherit_env=True)


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


def _is_retryable(exc: Exception) -> bool:
    text = str(exc).lower()
    status_code = getattr(exc, "status_code", None)
    return (
        status_code in {400, 408, 409, 413, 429, 500, 502, 503, 504}
        or "failed to parse" in text
        or "tool_call" in text
        or "context" in text
        or "connection" in text
        or "timeout" in text
        or "temporarily" in text
    )


def _last_text(state: object) -> str:
    if not isinstance(state, Mapping):
        return ""
    data = cast("Mapping[str, object]", state)
    messages = data.get("messages")
    if not isinstance(messages, list) or not messages:
        return ""
    content = getattr(messages[-1], "content", "")
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
