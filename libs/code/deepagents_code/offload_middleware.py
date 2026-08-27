"""CLI-specific conversation compaction middleware."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, Protocol, cast
from uuid import NAMESPACE_URL, uuid4, uuid5
from weakref import WeakValueDictionary

from deepagents.backends.protocol import FILE_NOT_FOUND
from deepagents.middleware.summarization import (
    SummarizationState,
    SummarizationToolMiddleware,
    create_summarization_middleware,
    create_summarization_tool_middleware,
)
from langchain.tools import (
    ToolRuntime,  # noqa: TC002  # inspected for runtime injection
)
from langchain_core.exceptions import ContextOverflowError
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import StructuredTool
from langgraph.config import get_config
from langgraph.types import Command  # noqa: TC002  # inspected for tool schema
from typing_extensions import TypedDict

from deepagents_code._cli_context import CLIContextSchema
from deepagents_code.cost_tracking import CostState
from deepagents_code.hooks.models.domain import (
    CompactTrigger,
    HookEvent,
    PreCompactDecision,
    PreCompactEvent,
)
from deepagents_code.hooks.server_middleware import (
    _DEFAULT_DEADLINE,
    _PRE_TOOL_STATE_KEY,
    HookTransportInterruptError,
    _event_enabled,
    _hook_context,
    _invoke_hook,
    _require_decision,
    _session_gate,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from deepagents.backends.composite import CompositeBackend
    from deepagents.backends.protocol import (
        BackendProtocol,
        EditResult,
        FileDownloadResponse,
        WriteResult,
    )
    from deepagents.middleware.summarization import SummarizationMiddleware
    from langchain.agents.middleware.types import (
        ExtendedModelResponse,
        ModelRequest,
        ModelResponse,
    )
    from langchain_core.messages import AnyMessage
    from langgraph.runtime import Runtime

    from deepagents_code.hooks.server_middleware import ServerHooksMiddleware

logger = logging.getLogger(__name__)


class _RetryingModelInvoker(Runnable[LanguageModelInput, AIMessage]):
    """Apply dcode's retry budget to one auxiliary model runnable."""

    def __init__(self, model: BaseChatModel) -> None:
        self._model = model

    def invoke(
        self,
        input: LanguageModelInput,  # noqa: A002  # `Runnable` keyword contract
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AIMessage:
        """Invoke the model with dcode-owned retries.

        Returns:
            The model response.
        """
        from deepagents_code.model_retry import retry_model_call

        return retry_model_call(
            self._model,
            lambda: self._model.invoke(input, config=config, **kwargs),
        )

    async def ainvoke(
        self,
        input: LanguageModelInput,  # noqa: A002  # `Runnable` keyword contract
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AIMessage:
        """Asynchronously invoke the model with dcode-owned retries.

        Returns:
            The model response.
        """
        from deepagents_code.model_retry import aretry_model_call

        return await aretry_model_call(
            self._model,
            lambda: self._model.ainvoke(input, config=config, **kwargs),
        )


def _install_summary_model_retries(summarization: SummarizationMiddleware) -> None:
    """Replace LangChain's generic summary retries with dcode's exact policy.

    Raises:
        AttributeError: If the SDK no longer exposes the summary-model slot.
    """
    helper = summarization._lc_helper
    # Assigning to a renamed slot would create a new attribute nothing reads,
    # leaving LangChain's unconditional three-attempt `with_retry` in place --
    # a silent no-op that defeats `--max-retries 0` and reports nothing.
    if not hasattr(helper, "_summary_model"):
        msg = (
            f"{type(helper).__name__} exposes no '_summary_model' slot, so dcode "
            "cannot own compaction summarization retries. The SDK's "
            "summarization internals have changed."
        )
        raise AttributeError(msg)
    helper._summary_model = _RetryingModelInvoker(summarization.model)


class _OffloadState(CostState, SummarizationState, total=False):
    """Checkpoint channels server-owned forced compaction reads and writes.

    `_summarization_event` is inherited from `SummarizationState` rather than
    re-declared so it keeps the SDK's own annotation (`NotRequired`, `| None`,
    and the `PrivateStateAttr` marker) instead of a divergent copy that claimed
    the value is always present.

    `total=False` describes the inherited shape only -- this body declares no
    keys of its own, so the modifier is deliberate documentation rather than a
    constraint on anything written here.
    """


type OffloadStatus = Literal["compacted", "empty", "noop", "denied", "failed"]
"""Outcome of one offload attempt. Aliased so the result type and the private
`_result` factory cannot drift apart."""


class OffloadResult(TypedDict):
    """Typed result emitted by the server-owned offload operation."""

    status: OffloadStatus
    messages_offloaded: int
    messages_kept: int
    tokens_before: int
    tokens_after: int
    archive_path: str | None
    archive_ephemeral: bool
    error: str | None


class OffloadStateUpdate(TypedDict, total=False):
    """The only checkpoint channels a server-owned operation may write.

    Naming the permitted channels makes the load-bearing invariant -- that this
    route can never write `messages` -- a property of the type rather than a
    single string check performed after the summarizer has already been billed.
    The runtime check in `offload_api` stays as a backstop for the `Any`-typed
    values the summarization SDK hands back.
    """

    _summarization_event: dict[str, Any]
    _summarization_session_id: str
    _session_cost_usd: float


class OffloadExecution(NamedTuple):
    """State update and typed result produced by one server operation."""

    update: OffloadStateUpdate
    result: OffloadResult
    archive: _PendingArchive | None = None


_archive_locks: WeakValueDictionary[str, asyncio.Lock] = WeakValueDictionary()


def _archive_lock(session_id: str) -> asyncio.Lock:
    """Return the process-local lock serializing one archive's read/write cycle."""
    lock = _archive_locks.get(session_id)
    if lock is None:
        lock = asyncio.Lock()
        _archive_locks[session_id] = lock
    return lock


class _PendingArchive(NamedTuple):
    """Archive append deferred until the checkpoint summary is reserved."""

    summarization: SummarizationMiddleware
    backend: BackendProtocol
    messages: list[AnyMessage]
    session_id: str
    summary: str
    state_cutoff: int

    def update(self, file_path: str | None) -> dict[str, Any]:
        """Build the summary update with the archive's settled path.

        Returns:
            State update containing the summary, cutoff, and archive path.
        """
        return CLICompactionMiddleware._forced_compaction_update(
            self.summarization,
            self.summary,
            file_path,
            self.state_cutoff,
            self.session_id,
        )

    async def _previous_content(self, path: str) -> tuple[bool, str]:
        """Read the archive snapshot needed to undo an uncommitted append.

        Returns:
            Whether the archive existed and its prior UTF-8 content.

        Raises:
            RuntimeError: If the backend cannot return the archive snapshot.
        """
        responses = await self.backend.adownload_files([path])
        if not responses:
            msg = f"archive backend returned no response for {path}"
            raise RuntimeError(msg)
        response = responses[0]
        if response.error == FILE_NOT_FOUND:
            return False, ""
        if response.error is not None:
            msg = f"archive read failed for {path}: {response.error}"
            raise RuntimeError(msg)
        content = response.content or b""
        return True, content.decode("utf-8")

    async def write(self) -> _ArchiveAppend | None:
        """Append staged messages and retain enough state for rollback.

        Returns:
            The reversible append, or `None` when the SDK could not write it.
        """
        path = self.summarization._get_history_path(self.session_id)
        existed, previous = await self._previous_content(path)
        guard = cast("BackendProtocol", _ArchiveReadGuard(self.backend))
        written_path = await self.summarization._aoffload_to_backend(
            guard, self.messages, self.session_id
        )
        append = _ArchiveAppend(self.backend, path, existed, previous)
        if written_path is None:
            await append.rollback()
            return None
        return append


class _ArchiveAppend(NamedTuple):
    """Completed archive append that can be restored until checkpointed."""

    backend: BackendProtocol
    path: str
    existed: bool
    previous: str

    async def rollback(self) -> None:
        """Restore the exact archive snapshot from before the append.

        Raises:
            RuntimeError: If the backend cannot restore the snapshot.
        """
        result = (
            await self.backend.awrite(self.path, self.previous)
            if self.existed
            else await self.backend.adelete(self.path)
        )
        if result.error is not None:
            msg = f"archive rollback failed for {self.path}: {result.error}"
            raise RuntimeError(msg)


class _ForcedCompactionPlan(NamedTuple):
    """Checkpoint update plus its not-yet-written archive append."""

    summarization: SummarizationMiddleware
    summary: str
    state_cutoff: int
    archive: _PendingArchive

    def update(self, file_path: str | None) -> dict[str, Any]:
        """Build the summary update with the archive's settled path.

        Returns:
            State update containing the summary, cutoff, and archive path.
        """
        return self.archive.update(file_path)


def unchanged_offload_result(
    status: OffloadStatus,
    *,
    messages: int,
    tokens: int,
    error: str | None = None,
) -> OffloadResult:
    """Build a result for an operation that did not compact state.

    Module-level so the HTTP boundary can report an unchanged outcome without
    resolving the server runtime first -- an empty thread has nothing to
    compact, and building the agent only to describe that is both wasteful and
    a way for a construction failure to turn "nothing to do" into a 500.

    Args:
        status: A non-compacting outcome.
        messages: Messages left in the conversation.
        tokens: Context estimate, unchanged by definition.
        error: Reason, required for `denied` and `failed`.

    Returns:
        Typed result containing unchanged context statistics.

    Raises:
        ValueError: If a refusal carries no reason.
    """
    if status in {"denied", "failed"} and not error:
        # `error` is `str | None` on every status because the wire shape is one
        # flat object, so the checker cannot make "a refusal has a reason" a
        # compile-time fact. Enforce it at the single construction point
        # instead: a reasonless refusal renders as the client's generic "the
        # server rejected the operation", which tells the user nothing.
        msg = f"An offload {status!r} result must carry a reason."
        raise ValueError(msg)
    return {
        "status": status,
        "messages_offloaded": 0,
        "messages_kept": messages,
        "tokens_before": tokens,
        "tokens_after": tokens,
        "archive_path": None,
        "archive_ephemeral": False,
        "error": error,
    }


class OffloadCompleteResponse(TypedDict):
    """Wire response for an attempt that finished without needing the client."""

    status: Literal["complete"]
    result: OffloadResult


class OffloadInterruptResponse(TypedDict):
    """Wire response carrying a hook request the client must fulfill.

    `request` stays a plain mapping on purpose: the client transports it back
    without inspecting it, and only `hooks.interrupt` owns its shape.
    """

    status: Literal["interrupt"]
    request: dict[str, Any]


type OffloadResponse = OffloadCompleteResponse | OffloadInterruptResponse
"""One round of the offload operation protocol.

Tagged on `status` so the producer (`offload_api._execute_offload`) and the
consumer (`RemoteAgent.aoffload`) are checked against one definition instead of
two independently hand-written `isinstance` ladders. The client still validates
at runtime -- this crosses HTTP, so the type is a contract, not a guarantee.
"""


_OFFLOAD_OPERATION_ATTR = "_dcode_offload_operation"


def attach_offload_operation(
    backend: CompositeBackend,
    operation: OffloadOperation,
) -> None:
    """Publish the operation on the backend shared with the server runtime.

    Args:
        backend: Composite backend owned by the agent server.
        operation: Offload implementation bound to that backend.

    Raises:
        ValueError: If compaction writes through a different backend.
    """
    # The SDK requires `backend` in its constructor, so a real summarization
    # middleware always has one; `None` here means a test double, which is
    # allowed through rather than asserted against.
    bound = getattr(operation._compaction._summarization, "_backend", None)
    if bound is not None and bound is not backend:
        msg = "Offload operation must use the agent's composite backend"
        raise ValueError(msg)
    setattr(backend, _OFFLOAD_OPERATION_ATTR, operation)


def offload_operation_from(backend: CompositeBackend) -> OffloadOperation | None:
    """Return the server operation published on `backend`, when available."""
    operation = getattr(backend, _OFFLOAD_OPERATION_ATTR, None)
    return operation if isinstance(operation, OffloadOperation) else None


def _event_cutoff(event: object) -> int:
    """Return the absolute cutoff index carried by a `_summarization_event`.

    Args:
        event: A `_summarization_event` mapping (as persisted in state), or
            `None`.

    Returns:
        The `cutoff_index`, or `0` when the event is missing or malformed.
    """
    if isinstance(event, dict):
        cutoff = event.get("cutoff_index")
        # `bool` is excluded explicitly: it passes `isinstance(_, int)`, so a
        # malformed `cutoff_index: true` would otherwise read as cutoff 1 and
        # silently shift the offloaded/kept counts by one message. The HTTP
        # boundary rejects bools for `model_context_limit` for the same reason.
        if isinstance(cutoff, int) and not isinstance(cutoff, bool):
            return cutoff
    return 0


class _AutoCompactionBlockedError(Exception):
    """Carry a blocked provider overflow past the SDK fallback handler."""

    def __init__(self, overflow: ContextOverflowError) -> None:
        super().__init__(str(overflow))
        self.overflow = overflow


class RuntimeModelConfig(NamedTuple):
    """Active model configuration read from a tool runtime.

    A named tuple rather than a bare 4-tuple so the two structurally identical
    `dict` slots (`model_params`, `profile_overrides`) are addressed by name at
    both the construction sites (keyword args) and the read site (attribute
    access) — a silent positional transposition the type checker would not catch
    is thereby avoided. Positional construction/unpacking is still possible and
    would defeat this, so call sites must keep using names.
    """

    model_spec: str | None
    model_params: dict[str, Any]
    profile_overrides: dict[str, Any]
    context_limit: int | None


class _HasRunContext(Protocol):
    """Anything carrying a per-run context object.

    The compaction helpers read `context` and nothing else, so they accept both
    the `ToolRuntime` injected into the tool and the plain LangGraph `Runtime`
    the server operation receives -- which is not a `ToolRuntime`. Stating
    the dependency this narrowly means a helper that starts touching, say,
    `tool_call_id` fails to type-check instead of breaking the server operation
    at runtime.
    """

    @property
    def context(self) -> object:
        """The run's context object.

        Typed as `object` rather than `Any`: every consumer narrows the shape
        with `isinstance` before touching it, so `object` type-checks the same
        code while still rejecting an unnarrowed attribute access.
        """
        ...


def _runtime_model_config(runtime: _HasRunContext) -> RuntimeModelConfig:
    """Read the active model configuration from a run context carrier.

    Args:
        runtime: Runtime carrying the current `CLIContext`.

    Returns:
        The active model specification, invocation parameters, profile
            overrides, and effective context-window limit.
    """
    context = runtime.context
    if isinstance(context, CLIContextSchema):
        return RuntimeModelConfig(
            model_spec=context.model,
            model_params=context.model_params,
            profile_overrides=context.profile_overrides,
            context_limit=context.model_context_limit,
        )
    if isinstance(context, dict):
        # The remote boundary delivers the context as JSON, so the keys are
        # strings; the values stay unknown and are narrowed individually below.
        fields = cast("dict[str, Any]", context)
        model = fields.get("model")
        params = fields.get("model_params")
        profile_overrides = fields.get("profile_overrides")
        context_limit = fields.get("model_context_limit")
        return RuntimeModelConfig(
            model_spec=model if isinstance(model, str) else None,
            model_params=dict(params) if isinstance(params, dict) else {},
            profile_overrides=(
                dict(profile_overrides) if isinstance(profile_overrides, dict) else {}
            ),
            context_limit=context_limit if isinstance(context_limit, int) else None,
        )
    return RuntimeModelConfig(
        model_spec=None, model_params={}, profile_overrides={}, context_limit=None
    )


class _ArchiveReadGuard:
    """Prevent an archive write after its prerequisite read fails.

    The SDK archive helper treats any unsuccessful read like a missing file and
    follows it with a truncating `write`. This narrow backend adapter preserves
    the SDK formatting and append behavior while making that fallback fail closed.
    """

    def __init__(self, backend: BackendProtocol) -> None:
        self._backend = backend
        self._read_failed = False

    def _record_response_errors(
        self, responses: list[FileDownloadResponse]
    ) -> list[FileDownloadResponse]:
        """Record read errors other than an expected missing archive.

        Args:
            responses: Backend download responses to inspect.

        Returns:
            The unchanged backend download responses.
        """
        if any(
            response.error is not None and response.error != FILE_NOT_FOUND
            for response in responses
        ):
            self._read_failed = True
        return responses

    def _ensure_read_succeeded(self) -> None:
        """Raise when a prior archive read failed in this operation.

        Raises:
            RuntimeError: If the prerequisite archive read failed.
        """
        if self._read_failed:
            msg = "archive read failed; refusing to overwrite existing history"
            raise RuntimeError(msg)

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Delegate a synchronous read while recording failures.

        Args:
            paths: Backend paths to read.

        Returns:
            The backend download responses.
        """
        try:
            responses = self._backend.download_files(paths)
        except Exception:
            self._read_failed = True
            raise
        return self._record_response_errors(responses)

    async def adownload_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Delegate an asynchronous read while recording failures.

        Args:
            paths: Backend paths to read.

        Returns:
            The backend download responses.
        """
        try:
            responses = await self._backend.adownload_files(paths)
        except Exception:
            self._read_failed = True
            raise
        return self._record_response_errors(responses)

    def write(self, file_path: str, content: str) -> WriteResult:
        """Write only when the prerequisite archive read succeeded.

        Args:
            file_path: Backend path to write.
            content: Complete archive content.

        Returns:
            The backend write result.
        """
        self._ensure_read_succeeded()
        return self._backend.write(file_path, content)

    async def awrite(self, file_path: str, content: str) -> WriteResult:
        """Asynchronously write only after a successful archive read.

        Args:
            file_path: Backend path to write.
            content: Complete archive content.

        Returns:
            The backend write result.
        """
        self._ensure_read_succeeded()
        return await self._backend.awrite(file_path, content)

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """Edit only when the prerequisite archive read did not raise.

        Args:
            file_path: Backend path to edit.
            old_string: Existing archive content.
            new_string: Archive content with the new section appended.
            replace_all: Whether to replace every match.

        Returns:
            The backend edit result.
        """
        self._ensure_read_succeeded()
        return self._backend.edit(
            file_path, old_string, new_string, replace_all=replace_all
        )

    async def aedit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """Asynchronously edit only after a successful archive read.

        Args:
            file_path: Backend path to edit.
            old_string: Existing archive content.
            new_string: Archive content with the new section appended.
            replace_all: Whether to replace every match.

        Returns:
            The backend edit result.
        """
        self._ensure_read_succeeded()
        return await self._backend.aedit(
            file_path, old_string, new_string, replace_all=replace_all
        )


class CLICompactionMiddleware(SummarizationToolMiddleware):
    """Add hook-aware automatic and server-owned compaction for dcode.

    The SDK tool's normal, model-initiated behavior remains unchanged.
    `_aplan_forced_compaction_update` is the state-only planning entry point
    used by the server-owned `/offload` operation.
    """

    def __init__(
        self,
        summarization: SummarizationMiddleware,
        *,
        system_prompt: str | None = None,
        cli_max_retries: int | None = None,
    ) -> None:
        """Initialize the CLI compaction middleware.

        Args:
            summarization: Summarization engine used by every compaction path.
            system_prompt: Optional prompt fragment advertising the compact tool.
            cli_max_retries: Explicit `--max-retries` value to retain when
                rebuilding a runtime-selected model for `/offload`.
        """
        super().__init__(summarization, system_prompt=system_prompt)
        self._cli_max_retries = cli_max_retries

    @property
    def name(self) -> str:
        """Replace the SDK auto-summarizer while retaining the compact tool."""
        return self._summarization.name

    @staticmethod
    def _auto_compaction_id(request: ModelRequest) -> str:
        """Return a stable identity for one model-input snapshot."""
        messages = request.messages
        last = messages[-1]
        identity = (
            last.id or hashlib.sha256(last.model_dump_json().encode()).hexdigest()
        )
        return f"{len(messages)}:{identity}"

    def _pre_auto_compact(self, request: ModelRequest) -> bool:
        """Run `PreCompact` before automatic summarization.

        Returns:
            Whether summarization may continue.
        """
        from langgraph.config import get_config

        runtime = request.runtime
        gate = _session_gate(runtime.context)
        if not _event_enabled(gate, HookEvent.PRE_COMPACT):
            return True
        try:
            config = get_config()
        except RuntimeError:
            config = None
        decision = _invoke_hook(
            _hook_context(runtime.context, config, Path.cwd()),
            PreCompactEvent(event=HookEvent.PRE_COMPACT, trigger=CompactTrigger.AUTO),
            gate=gate,
            config=config,
            deadline=_DEFAULT_DEADLINE,
            logical_event_id=self._auto_compaction_id(request),
        )
        return _require_decision(decision, PreCompactDecision).continue_processing

    def _auto_compaction_request(self, request: ModelRequest) -> ModelRequest | None:
        """Return the prepared request when threshold compaction will run."""
        summarization = self._summarization
        messages = summarization._get_effective_messages(request)
        tokens = summarization._count_tokens(
            messages, request.system_message, request.tools
        )
        messages, modified = summarization._truncate_args(messages, tokens)
        if modified:
            tokens = summarization._count_tokens(
                messages, request.system_message, request.tools
            )
        if (
            not summarization._should_summarize(messages, tokens)
            or summarization._determine_cutoff_index(messages) <= 0
        ):
            return None
        return request.override(messages=messages)

    def _pre_overflow_compact(
        self,
        request: ModelRequest,
        overflow: ContextOverflowError,
    ) -> None:
        """Gate a provider-overflow fallback before it compacts.

        Raises:
            _AutoCompactionBlockedError: If the hook blocks compaction.
        """
        if self._summarization._determine_cutoff_index(
            request.messages
        ) > 0 and not self._pre_auto_compact(request):
            raise _AutoCompactionBlockedError(overflow) from overflow

    def wrap_model_call(  # ty: ignore[invalid-method-override]  # delegates auto summarizer
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse | ExtendedModelResponse:
        """Run `PreCompact` before synchronous automatic summarization.

        Returns:
            The model response.
        """
        call_model = partial(super().wrap_model_call, handler=handler)
        prepared = self._auto_compaction_request(request)
        if prepared is not None:
            if not self._pre_auto_compact(prepared):
                return call_model(prepared)
            return self._summarization.wrap_model_call(request, call_model)

        overflow_gated = False

        def gated_handler(next_request: ModelRequest) -> ModelResponse:
            nonlocal overflow_gated
            try:
                return call_model(next_request)
            except ContextOverflowError as overflow:
                if not overflow_gated:
                    overflow_gated = True
                    self._pre_overflow_compact(next_request, overflow)
                raise

        try:
            return self._summarization.wrap_model_call(request, gated_handler)
        except _AutoCompactionBlockedError as blocked:
            raise blocked.overflow from None

    async def _awrap_with_archive_lock(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse | ExtendedModelResponse:
        """Serialize an automatic archive append with other compactions.

        Returns:
            The wrapped model response.
        """
        session_id = self._summarization._get_session_id(request.state)
        async with _archive_lock(session_id):
            return await self._summarization.awrap_model_call(request, handler)

    async def awrap_model_call(  # ty: ignore[invalid-method-override]  # delegates auto summarizer
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse | ExtendedModelResponse:
        """Run `PreCompact` before asynchronous automatic summarization.

        Returns:
            The model response.
        """
        call_model = partial(super().awrap_model_call, handler=handler)
        prepared = self._auto_compaction_request(request)
        if prepared is not None:
            if not self._pre_auto_compact(prepared):
                return await call_model(prepared)
            return await self._awrap_with_archive_lock(request, call_model)

        overflow_gated = False

        async def gated_handler(next_request: ModelRequest) -> ModelResponse:
            nonlocal overflow_gated
            try:
                return await call_model(next_request)
            except ContextOverflowError as overflow:
                if not overflow_gated:
                    overflow_gated = True
                    self._pre_overflow_compact(next_request, overflow)
                raise

        try:
            return await self._awrap_with_archive_lock(request, gated_handler)
        except _AutoCompactionBlockedError as blocked:
            raise blocked.overflow from None

    async def _arun_compact(self, runtime: ToolRuntime[Any, Any]) -> Command:
        """Serialize a model-initiated archive append with other compactions.

        Returns:
            The compact tool's state command.
        """
        session_id = self._summarization._get_session_id(runtime.state)
        async with _archive_lock(session_id):
            return await super()._arun_compact(runtime)

    def _create_compact_tool(self) -> StructuredTool:
        """Create the CLI variant of `compact_conversation`.

        Returns:
            The model-initiated compaction tool.
        """
        middleware = self

        def sync_compact(runtime: ToolRuntime[Any, Any]) -> Command:
            return middleware._run_compact(runtime)

        async def async_compact(runtime: ToolRuntime[Any, Any]) -> Command:
            return await middleware._arun_compact(runtime)

        return StructuredTool.from_function(
            name="compact_conversation",
            description=(
                "Compact the conversation by summarizing older messages into "
                "a concise summary. Use this proactively when the conversation "
                "is getting long to free up context window space."
            ),
            func=sync_compact,
            coroutine=async_compact,
        )

    def _summarization_for_runtime(
        self, runtime: _HasRunContext
    ) -> SummarizationMiddleware:
        """Build a summarizer for the active runtime model when overridden.

        Args:
            runtime: Runtime carrying the current `CLIContext`.

        Returns:
            The startup summarizer when no runtime model is selected, otherwise
                a model-aware summarizer using the same configured backend.
        """
        config = _runtime_model_config(runtime)
        if not config.model_spec:
            return self._summarization

        from deepagents_code.config import create_model

        model = create_model(
            config.model_spec,
            extra_kwargs=config.model_params or None,
            profile_overrides=config.profile_overrides or None,
            cli_max_retries=self._cli_max_retries,
        ).model
        context_limit = config.context_limit
        if context_limit is not None:
            profile = getattr(model, "profile", None)
            native = (
                profile.get("max_input_tokens") if isinstance(profile, dict) else None
            )
            if native != context_limit:
                merged = (
                    {**profile, "max_input_tokens": context_limit}
                    if isinstance(profile, dict)
                    else {"max_input_tokens": context_limit}
                )
                try:
                    model.profile = merged  # ty: ignore[invalid-assignment]
                except (AttributeError, TypeError, ValueError):
                    logger.warning(
                        "Could not apply runtime context limit %d to the offload "
                        "model profile; using its resolved profile",
                        context_limit,
                        exc_info=True,
                    )
        # Never pass the `_ArchiveReadGuard` wrapper to the constructor: the SDK
        # resolves the archive prefix once in `__init__` via
        # `backend.artifacts_root if isinstance(backend, CompositeBackend)`, and
        # the guard is not a `CompositeBackend`, so that check would fall back
        # to a `/` prefix. The archive write would then miss the
        # `conversation_history` route and land in the default backend --
        # silently writing into the user's project tree. An `artifacts_root`
        # passthrough on the guard would not help; the `isinstance` is what
        # fails.
        #
        # This is a forward-looking constraint on the *constructor argument*,
        # not a bug being fixed: the previous code also passed the real
        # composite backend here and only swapped `_backend` for the guard
        # afterwards, so the prefix was correct then too. `_PendingArchive`
        # applies the guard separately when writing.
        summarization = create_summarization_middleware(
            model, self._summarization._backend
        )
        _install_summary_model_retries(summarization)
        return summarization

    async def _aplan_forced_compaction_update(
        self, state: _OffloadState, runtime: _HasRunContext
    ) -> _ForcedCompactionPlan | None:
        """Summarize forced-compaction history without writing its archive.

        Unlike the tool paths, this raises on failure instead of returning a
        `ToolMessage`: the server operation has no tool node to carry one, so it
        converts the exception into a client-visible error. Any summarizer,
        or planning failure therefore propagates out of this method rather than
        being folded into the return value.

        Args:
            state: Checkpointed conversation and prior summarization event.
            runtime: Run context carrier used to select the summarizer model.

        Returns:
            The checkpoint/archive plan, or `None` when nothing can be compacted.

        Raises:
            ValueError: If called directly with no messages. The owning server
                operation handles an empty thread before reaching this helper.
        """
        summarization = await asyncio.to_thread(
            self._summarization_for_runtime, runtime
        )
        messages = state.get("messages", [])
        event = state.get("_summarization_event")
        if not messages:
            msg = "Offload compaction requires checkpointed conversation messages."
            raise ValueError(msg)
        effective = summarization._apply_event_to_messages(messages, event)
        cutoff = summarization._determine_cutoff_index(effective)
        if cutoff == 0:
            return None
        # Resolved once and threaded into the update below: the SDK call is the
        # relative-to-absolute conversion, and computing it twice would let the
        # value checked here drift from the value committed.
        state_cutoff = summarization._compute_state_cutoff(event, cutoff)
        if state_cutoff <= _event_cutoff(event):
            # Degenerate chained compaction: everything eligible is already
            # behind the prior event's cutoff, so only the previous summary
            # would be re-summarized. Committing would spend a model call to
            # replace the in-context summary with a lossier summary-of-a-summary
            # and drop the prior `file_path` from the event -- while the client,
            # which keys its report on the *absolute* cutoff advancing, still
            # reported "nothing to offload". Stop before the model call so the
            # report and the state agree.
            return None
        to_summarize, _ = summarization._partition_messages(effective, cutoff)
        summary = await summarization._acreate_summary(to_summarize)
        session_id = summarization._get_session_id(state)
        archive = _PendingArchive(
            summarization,
            self._summarization._backend,
            to_summarize,
            session_id,
            summary,
            state_cutoff,
        )
        return _ForcedCompactionPlan(summarization, summary, state_cutoff, archive)

    async def arun_forced_compaction_update(
        self, state: _OffloadState, runtime: _HasRunContext
    ) -> dict[str, Any] | None:
        """Run forced compaction and persist its archive immediately.

        The HTTP operation uses `_aplan_forced_compaction_update` directly so
        it can reserve the checkpoint before this side effect. Direct callers
        retain the historical all-in-one behavior.

        Returns:
            The completed state update, or `None` when nothing can be compacted.
        """
        plan = await self._aplan_forced_compaction_update(state, runtime)
        if plan is None:
            return None
        try:
            async with _archive_lock(plan.archive.session_id):
                append = await plan.archive.write()
        except Exception:
            logger.exception("/offload archive append failed")
            append = None
        if append is None:
            # `_aoffload_to_backend` catches every write failure and returns
            # `None`, which also swallows `_ArchiveReadGuard`'s deliberate
            # "refusing to overwrite existing history" `RuntimeError`. Its own
            # log names neither the thread nor this call site, so record one
            # here that does.
            #
            # Not raised: the compaction is still useful (the summary is
            # in-context and the raw messages remain in the checkpoint), and the
            # client reports the missing archive to the user as an error rather
            # than a success. Escalating here would change that policy, not just
            # its observability.
            logger.error(
                "/offload compacted %d messages but the archive write failed; "
                "those messages are not recoverable from storage",
                len(plan.archive.messages),
            )
        return plan.update(append.path if append is not None else None)

    @staticmethod
    def _forced_compaction_update(
        summarization: SummarizationMiddleware,
        summary: str,
        file_path: str | None,
        state_cutoff: int,
        session_id: str,
    ) -> dict[str, Any]:
        """Build the state-only result used by the server `/offload` operation.

        The returned dict carries the `_summarization_event` payload plus the
        session id, so it is not itself a `SummarizationEvent` and cannot be
        annotated as one. That the event's `summary_message` really is the
        `HumanMessage` the channel expects is therefore enforced at runtime by
        the `isinstance` check below, not by the type checker.

        Args:
            summarization: SDK summarization middleware building the message.
            summary: Generated summary text.
            file_path: Archive path, or `None` when the write failed.
            state_cutoff: **Absolute** cutoff index, already converted from the
                relative one by `_compute_state_cutoff`. Taken pre-resolved
                rather than converted here so the caller's no-advance check and
                the committed value cannot disagree.
            session_id: The id that named the history file. Persisted under
                `_summarization_session_id` (mirroring the SDK's compact and
                auto-summarize paths) so a later offload appends to the same
                archive instead of minting a fresh file.

        Returns:
            The summarization state update.

        Raises:
            TypeError: If the summarizer's first message is not the
                `HumanMessage` the event schema declares.
        """
        summary_message = summarization._build_new_messages_with_path(
            summary, file_path
        )[0]
        if not isinstance(summary_message, HumanMessage):
            # `_build_new_messages_with_path` is annotated `list[AnyMessage]`
            # but documents (and the SDK's own call site assumes, with a type
            # suppression) that element 0 is the summary `HumanMessage`. Check
            # rather than suppress: the node turns this into a visible
            # "Compaction failed" instead of checkpointing an event whose
            # `summary_message` violates its own schema.
            msg = (
                "Summarizer returned a "
                f"{type(summary_message).__name__} summary message; expected "
                "HumanMessage."
            )
            raise TypeError(msg)
        return {
            "_summarization_event": {
                # Absolute, not relative: a second `/offload` on the same thread
                # reads this back as its base.
                "cutoff_index": state_cutoff,
                "summary_message": summary_message,
                "file_path": file_path,
            },
            "_summarization_session_id": session_id,
        }


def _create_cli_compaction_middleware(
    model: str | BaseChatModel,
    backend: BackendProtocol,
    *,
    cli_max_retries: int | None = None,
) -> CLICompactionMiddleware:
    """Create the dcode compaction middleware from the SDK configuration.

    Args:
        model: Startup model or model specification.
        backend: Agent backend used for archive persistence.
        cli_max_retries: Explicit `--max-retries` value for runtime rebuilds.

    Returns:
        CLI compaction middleware with the SDK's model-aware defaults.
    """
    sdk_middleware = create_summarization_tool_middleware(model, backend)
    _install_summary_model_retries(sdk_middleware._summarization)
    return CLICompactionMiddleware(
        sdk_middleware._summarization,
        system_prompt=sdk_middleware.system_prompt,
        cli_max_retries=cli_max_retries,
    )


_OFFLOAD_CALL_NAMESPACE = uuid5(NAMESPACE_URL, "https://deepagents/offload/forced-call")
"""Namespace for deriving the `/offload` hook dispatch's forced tool-call id."""


def _forced_offload_call_id() -> str:
    """Return the tool-call id the `/offload` hook dispatch runs against.

    Two requirements pull in opposite directions, and both are load-bearing:

    *Stable across resumes.* `ServerHooksMiddleware` folds this id into its hook
    `invocation_id`, and answering a hook request re-executes the operation
    **from the top** rather than resuming mid-coroutine. An id minted fresh here
    would therefore differ between the request and the resume, and
    `parse_hook_resume_value` rejects a mismatched invocation id as fatal ("the
    client answered a different request") -- which would break `/offload` for
    exactly those users who have a `PreCompact`/`PreToolUse` hook configured,
    and make the client's whole fulfill/resume loop unreachable.

    *Distinct across attempts.* The client memoizes fulfillments by
    `(snapshot_id, invocation_id)` for the session, and the hook `prompt_id`
    only rotates on user-prompt submit. A constant would make two `/offload`s
    within one turn collide and replay the first attempt's decision -- including
    a denial -- instead of re-running the user's hook.

    `configurable.checkpoint_ns` satisfies both, and
    `offload_api._execute_offload` is the invariant's owner: it derives the
    namespace as `dcode_offload:{operation_id}` from the client's per-attempt
    `operation_id`, which `RemoteAgent.aoffload` mints once and reuses across
    every resume round of that attempt. Changing either that namespace format or
    the client's reuse of `operation_id` breaks hook resume, silently, for hook
    users only.

    Returns:
        An id stable across this attempt's resume rounds and distinct from every
            other attempt's.
    """
    try:
        config = get_config()
    except RuntimeError:
        # No runnable context at all -- a direct call outside a graph run.
        # Nothing can interrupt or resume such a call, so uniqueness is the only
        # property left to preserve, and the `uuid4()` fallback is correct.
        return f"offload-precompact-{uuid4()}"
    configurable = config.get("configurable")
    namespace = (
        configurable.get("checkpoint_ns") if isinstance(configurable, dict) else None
    )
    if not namespace:
        # A runnable context *without* a usable `checkpoint_ns` is a different
        # situation entirely, and a silent fallback here is the failure mode
        # this function exists to prevent: the id would differ between the
        # request and the resume, `parse_hook_resume_value` would reject the
        # mismatch as fatal, and `/offload` would die with "the client answered
        # a different request" -- but only for users with hooks configured, and
        # with nothing in the logs pointing here. Say so loudly.
        logger.warning(
            "Deriving the /offload hook call id inside a run but "
            "`configurable.checkpoint_ns` is %r; falling back to a random id. "
            "Configured PreCompact/PreToolUse hooks will fail to resume this "
            "run. This usually means LangGraph moved or renamed the key.",
            namespace,
        )
        return f"offload-precompact-{uuid4()}"
    return f"offload-precompact-{uuid5(_OFFLOAD_CALL_NAMESPACE, namespace)}"


class OffloadOperation:
    """Compact checkpoint state behind dcode's server-owned HTTP boundary."""

    def __init__(
        self,
        compaction: CLICompactionMiddleware,
        hooks: ServerHooksMiddleware,
    ) -> None:
        """Initialize the operation with the agent's own policy and hooks.

        Args:
            compaction: Compaction middleware bound to the agent backend.
            hooks: Server hook middleware used by the interactive graph.
        """
        self._compaction = compaction
        self._hooks = hooks

    _result = staticmethod(unchanged_offload_result)

    async def _run_hooks(
        self, runtime: Runtime[CLIContextSchema]
    ) -> tuple[Literal["denied", "failed"], str] | None:
        """Dispatch the forced call through `PreCompact` and `PreToolUse`.

        Returns:
            Failure status and detail, or `None` when hooks allow compaction.

        Raises:
            HookTransportInterruptError: If the client must fulfill a hook request.
        """
        try:
            forced_call_id = _forced_offload_call_id()
            hook_update = await self._hooks.aafter_model(
                cast(
                    "Any",
                    {
                        "messages": [
                            AIMessage(
                                content="",
                                tool_calls=[
                                    {
                                        "name": "compact_conversation",
                                        "args": {"force": True},
                                        "id": forced_call_id,
                                    }
                                ],
                            )
                        ]
                    },
                ),
                cast("Runtime[Any]", runtime),
            )
        except HookTransportInterruptError:
            raise
        except Exception as exc:
            logger.exception("/offload hook dispatch failed")
            return "failed", f"Offload hooks failed: {type(exc).__name__}: {exc}"

        # Fail closed on a missing channel rather than defaulting to allow.
        # `_after_model` always returns this key, so its absence means the
        # channel, the id derivation, or the outcome shape drifted -- and a
        # `.get(..., {})` chain would read a user's *denial* as "no outcome" and
        # compact straight through it, with no log.
        outcomes = hook_update.get(_PRE_TOOL_STATE_KEY)
        if not isinstance(outcomes, dict):
            logger.error(
                "Compaction hooks returned no %s channel for /offload; refusing "
                "rather than treating a possible denial as an allow",
                _PRE_TOOL_STATE_KEY,
            )
            return "failed", (
                "Could not read the compaction hook decision; offload refused."
            )
        outcome = outcomes.get(forced_call_id) or {}
        if outcome.get("behavior") == "deny":
            reason = outcome.get("reason") or "Blocked by a compaction hook"
            return "denied", str(reason)
        if outcome.get("context"):
            logger.warning(
                "Discarding PreToolUse additionalContext for the /offload "
                "operation; no tool result or model turn exists to carry it"
            )
        return None

    async def execute(
        self,
        state: _OffloadState,
        runtime: Runtime[CLIContextSchema],
    ) -> OffloadExecution:
        """Run one offload against server-read checkpoint state.

        Returns:
            State update for the server to persist and the typed client result.

        Raises:
            HookTransportInterruptError: If the client must fulfill a hook request.
        """
        from langchain_core.messages.utils import count_tokens_approximately

        messages = list(state.get("messages", []))
        event = state.get("_summarization_event")
        effective = self._compaction._summarization._apply_event_to_messages(
            messages, event
        )
        tokens_before = count_tokens_approximately(effective)
        if not messages:
            result = self._result("empty", messages=0, tokens=0)
            return OffloadExecution({}, result)

        hook_failure = await self._run_hooks(runtime)
        if hook_failure is not None:
            status, error = hook_failure
            result = self._result(
                status,
                messages=max(0, len(messages) - _event_cutoff(event)),
                tokens=tokens_before,
                error=error,
            )
            return OffloadExecution({}, result)

        try:
            plan = await self._compaction._aplan_forced_compaction_update(
                state, runtime
            )
        except HookTransportInterruptError:
            raise
        except Exception as exc:
            logger.exception("forced /offload compaction failed")
            result = self._result(
                "failed",
                messages=max(0, len(messages) - _event_cutoff(event)),
                tokens=tokens_before,
                error=f"Compaction failed: {type(exc).__name__}: {exc}",
            )
            return OffloadExecution({}, result)

        if plan is None:
            result = self._result(
                "noop",
                messages=max(0, len(messages) - _event_cutoff(event)),
                tokens=tokens_before,
            )
            return OffloadExecution({}, result)

        update = plan.update(None)
        new_event = update["_summarization_event"]
        new_cutoff = _event_cutoff(new_event)
        prior_cutoff = _event_cutoff(event)
        effective_after = self._compaction._summarization._apply_event_to_messages(
            messages, new_event
        )
        file_path = new_event.get("file_path")
        from deepagents_code.offload import offload_storage_is_ephemeral

        result: OffloadResult = {
            "status": "compacted",
            "messages_offloaded": max(0, new_cutoff - prior_cutoff),
            "messages_kept": max(0, len(messages) - new_cutoff),
            "tokens_before": tokens_before,
            "tokens_after": count_tokens_approximately(effective_after),
            "archive_path": file_path if isinstance(file_path, str) else None,
            "archive_ephemeral": offload_storage_is_ephemeral(),
            "error": None,
        }
        # Forward only the permitted channels rather than the SDK's whole update
        # dict, so a future SDK change that adds keys (`messages` above all)
        # cannot reach the checkpoint write through this operation.
        return OffloadExecution(
            {
                "_summarization_event": new_event,
                "_summarization_session_id": update["_summarization_session_id"],
            },
            result,
            plan.archive,
        )
