"""CLI-specific conversation compaction middleware."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from threading import Lock
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
from langchain.agents.middleware.summarization import (
    _get_approximate_token_counter,  # noqa: PLC2701  # keep summary trimming aligned with LangChain
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

from deepagents_code._cli_context import (
    INHERIT_SUMMARIZATION_MODEL,
    CLIContextSchema,
)
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
    from collections.abc import AsyncIterator, Awaitable, Callable

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

_SUMMARY_TRIM_FALLBACK = 4_000
"""Conservative history budget when a summary model exposes no context limit."""


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


def _require_helper_slot(helper: object, name: str, purpose: str) -> None:
    """Fail loudly when an SDK summarization slot dcode overwrites is gone.

    Every write dcode makes into `_lc_helper` replaces behavior the SDK would
    otherwise supply. Assigning to a renamed slot would create a new attribute
    nothing reads, leaving the SDK's own behavior in place -- a silent no-op
    that reports nothing. Checking first turns that into a loud failure.

    Args:
        helper: The SDK summarization helper being patched.
        name: Attribute dcode is about to overwrite.
        purpose: What dcode loses if the slot is gone, phrased to follow
            "so dcode cannot ...".

    Raises:
        AttributeError: If the SDK no longer exposes the named slot.
    """
    if hasattr(helper, name):
        return
    msg = (
        f"{type(helper).__name__} exposes no {name!r} slot, so dcode cannot "
        f"{purpose}. The SDK's summarization internals have changed."
    )
    raise AttributeError(msg)


def _install_summary_model_retries(
    summarization: SummarizationMiddleware,
    model: BaseChatModel | None = None,
) -> None:
    """Replace LangChain's generic summary retries with dcode's exact policy.

    Also selects the model summaries are generated with, which is the only
    place a `--summarization-model` override takes effect.

    Args:
        summarization: Middleware whose summary-model slot is replaced.
        model: Model to generate summaries with. `None` reuses
            `summarization.model`, the model driving thresholds and counting.
    """
    helper = summarization._lc_helper
    # A renamed slot would leave LangChain's unconditional three-attempt
    # `with_retry` in place, silently defeating `--max-retries 0`.
    _require_helper_slot(
        helper, "_summary_model", "own compaction summarization retries"
    )
    helper._summary_model = _RetryingModelInvoker(
        model if model is not None else summarization.model
    )


def _install_summary_trim_limit(
    summarization: SummarizationMiddleware, model: BaseChatModel
) -> None:
    """Bound the history sent to a dedicated summary model.

    Set after construction rather than through the
    `create_summarization_middleware` kwarg because the summary model is built
    lazily, long after the middleware exists. Raises `AttributeError` through
    `_require_helper_slot` if the slot is gone.

    Args:
        summarization: Middleware whose trim budget is replaced.
        model: The model summaries are generated with.
    """
    helper = summarization._lc_helper
    # dcode leaves trimming off by default: `create_summarization_middleware`
    # passes `trim_tokens_to_summarize=None`, overriding LangChain's 4000, and
    # the helper then skips trimming entirely. So a renamed slot would send the
    # whole conversation to a model picked for being smaller than the main one.
    _require_helper_slot(
        helper,
        "trim_tokens_to_summarize",
        "bound the history sent to a dedicated summary model",
    )
    helper.trim_tokens_to_summarize = _summary_trim_limit(model)


def _summary_trim_limit(model: BaseChatModel) -> int:
    """Return the token budget for history sent to the summary model.

    Reserves a fifth of the model's input window for the summary prompt,
    message serialization overhead, and the imprecision of the approximate
    token counter that measures against this budget -- the provider limit is
    hard, the measurement is not.

    Args:
        model: The model summaries are generated with.

    Returns:
        The summary-history budget, falling back to
            `_SUMMARY_TRIM_FALLBACK` when the model exposes no usable positive
            `max_input_tokens`.
    """
    profile = getattr(model, "profile", None)
    if not isinstance(profile, dict):
        return _SUMMARY_TRIM_FALLBACK
    context_limit = profile.get("max_input_tokens")
    # `bool` is an `int` subclass: `max_input_tokens=True` would otherwise
    # reserve its way down to a one-token summary budget.
    if (
        not isinstance(context_limit, int)
        or isinstance(context_limit, bool)
        or context_limit <= 0
    ):
        return _SUMMARY_TRIM_FALLBACK
    return max(1, context_limit * 4 // 5)


def _install_summary_token_counter(
    summarization: SummarizationMiddleware, model: BaseChatModel
) -> None:
    """Count dedicated-summary input with that model's provider tuning.

    Raises `AttributeError` through `_require_helper_slot` if a slot is gone.

    Args:
        summarization: Middleware whose summary token counters are replaced.
        model: The model summaries are generated with.
    """
    helper = summarization._lc_helper
    # A renamed slot would measure the summary budget with the main model's
    # tuning while spending it against the summary model's window.
    for slot in ("token_counter", "_partial_token_counter"):
        _require_helper_slot(
            helper, slot, "count dedicated-summary input for the right provider"
        )
    counter = _get_approximate_token_counter(model)
    helper.token_counter = counter
    helper._partial_token_counter = partial(
        counter,
        use_usage_metadata_scaling=False,  # ty: ignore[unknown-argument]
    )


def _is_blocking_error(exc: BaseException) -> bool:
    """Whether `exc` is the event-loop blocking guard (`blockbuster`).

    `BlockingError` marks a defect in the caller, not a bad model spec or a
    provider failure, so the degrade-to-main-model fallbacks re-raise it rather
    than absorb it. Matched by class name, as elsewhere in the package, because
    the guard is a test-only dependency that must not be imported here.

    Returns:
        Whether any class in the exception's MRO is named `BlockingError`.
    """
    return any(cls.__name__ == "BlockingError" for cls in type(exc).__mro__)


class _LazySummaryModel:
    """Configure a dedicated model immediately before summary generation."""

    def __init__(
        self,
        summarization: SummarizationMiddleware,
        model_spec: str,
        cli_max_retries: int | None,
    ) -> None:
        self._summarization = summarization
        self._model_spec = model_spec
        self._cli_max_retries = cli_max_retries
        self._lock = Lock()
        self._invocation_lock = Lock()
        self._configured = False
        self._warned = False
        helper = summarization._lc_helper
        # Main-model summary tuning, captured before any override installs so
        # `_degrade_to_main_model` can uninstall the override.
        self._main_summary_tuning = (
            helper.token_counter,
            helper._partial_token_counter,
            helper.trim_tokens_to_summarize,
        )
        self._create_summary = helper._create_summary
        self._acreate_summary = helper._acreate_summary

    def _configure(self) -> None:
        """Install the model once, degrading to the main model on failure.

        A summary model that cannot be built is a broken optimization, not a
        broken session: raising here would fail the turn that triggered
        compaction and would also break `/compact`, the tool the user reaches
        for when the context is full. So the summary runs on the main model
        instead and the next compaction retries the override.
        """
        with self._lock:
            if self._configured:
                return
            from deepagents_code.config import create_model

            try:
                model = create_model(
                    self._model_spec,
                    cli_max_retries=self._cli_max_retries,
                ).model
            except Exception as exc:
                # BlockingError means this ran on the server event loop; that is
                # a defect in the caller, not a bad model spec, so let it out.
                if _is_blocking_error(exc):
                    raise
                _install_summary_model_retries(self._summarization)
                if not self._warned:
                    # Warned once per summarizer: compaction is rare, but a
                    # broken spec would otherwise log on every attempt.
                    self._warned = True
                    logger.warning(
                        "Could not build the summarization model %r; compaction "
                        "summaries use the main agent model instead. Run "
                        "`/summarization-model clear` to stop trying it.",
                        self._model_spec,
                        exc_info=True,
                    )
                return
            _install_summary_model_retries(self._summarization, model)
            _install_summary_token_counter(self._summarization, model)
            _install_summary_trim_limit(self._summarization, model)
            self._configured = True

    def _degrade_to_main_model(self, exc: Exception) -> None:
        """Restore main-model summary generation after an invocation failure.

        A dedicated model that builds but cannot generate (an unknown model ID,
        say, or missing provider access) is the same broken optimization as one
        that cannot build: the summary runs on the main model instead, and
        `_configured` resets so the next compaction retries the override.

        Args:
            exc: The failure being logged. Passed explicitly because the async
                path runs this on a worker thread, where `sys.exc_info()` is
                thread-local and would log `NoneType: None`.
        """
        with self._lock:
            _install_summary_model_retries(self._summarization)
            helper = self._summarization._lc_helper
            (
                helper.token_counter,
                helper._partial_token_counter,
                helper.trim_tokens_to_summarize,
            ) = self._main_summary_tuning
            self._configured = False
            if not self._warned:
                # Shared with the build-failure warning: either failure mode
                # means the override is suspect, and one warning per summarizer
                # is enough.
                self._warned = True
                logger.warning(
                    "The summarization model %r failed to generate a summary; "
                    "compaction summaries use the main agent model instead. Run "
                    "`/summarization-model clear` to stop trying it.",
                    self._model_spec,
                    exc_info=exc,
                )

    def create_summary(self, messages: list[AnyMessage]) -> str:
        """Generate a synchronous summary after lazy configuration.

        Returns:
            The generated summary.
        """
        with self._invocation_lock:
            self._configure()
            try:
                return self._create_summary(messages)
            except Exception as exc:
                # `_configured` means the override, not the main model, was behind
                # the failed call; a main-model failure has nothing to fall back to.
                if _is_blocking_error(exc) or not self._configured:
                    raise
                self._degrade_to_main_model(exc)
            return self._create_summary(messages)

    @asynccontextmanager
    async def _async_invocation_lock(self) -> AsyncIterator[None]:
        """Hold the shared invocation lock without blocking the event loop."""
        acquisition = asyncio.create_task(
            asyncio.to_thread(self._invocation_lock.acquire)
        )
        try:
            await asyncio.shield(acquisition)
        except BaseException:
            await acquisition
            self._invocation_lock.release()
            raise
        try:
            yield
        finally:
            self._invocation_lock.release()

    async def acreate_summary(self, messages: list[AnyMessage]) -> str:
        """Generate an asynchronous summary after lazy configuration.

        Returns:
            The generated summary.
        """
        async with self._async_invocation_lock():
            await asyncio.to_thread(self._configure)
            try:
                return await self._acreate_summary(messages)
            except Exception as exc:
                # `_configured` means the override, not the main model, was behind
                # the failed call; a main-model failure has nothing to fall back to.
                if _is_blocking_error(exc) or not self._configured:
                    raise
                # Locked like `_configure`, so it stays off the event loop.
                await asyncio.to_thread(self._degrade_to_main_model, exc)
            return await self._acreate_summary(messages)


def _install_lazy_summary_model(
    summarization: SummarizationMiddleware,
    model_spec: str,
    cli_max_retries: int | None,
) -> None:
    """Defer dedicated model construction until a summary is requested.

    Raises `AttributeError` through `_require_helper_slot` if a hook is gone.

    Args:
        summarization: Middleware whose summary hooks are wrapped.
        model_spec: Spec to resolve when a summary is first requested.
        cli_max_retries: Explicit `--max-retries` value, or `None`.
    """
    for slot in ("_create_summary", "_acreate_summary"):
        _require_helper_slot(
            summarization._lc_helper, slot, "defer summary-model construction"
        )
    lazy = _LazySummaryModel(summarization, model_spec, cli_max_retries)
    # LangChain annotates these private attributes as unbound method shapes;
    # instance-local bound wrappers are intentional so only this summarizer is lazy.
    summarization._lc_helper._create_summary = (  # ty: ignore[invalid-assignment]
        lazy.create_summary
    )
    summarization._lc_helper._acreate_summary = (  # ty: ignore[invalid-assignment]
        lazy.acreate_summary
    )


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

    A named tuple rather than a bare tuple so the structurally identical slots
    — the two `str | None` model specs and the two `dict` overrides — are
    addressed by name at both the construction sites (keyword args) and the
    read site (attribute access) — a silent positional transposition the type
    checker would not catch is thereby avoided. Positional
    construction/unpacking is still possible and would defeat this, so call
    sites must keep using names.
    """

    model_spec: str | None
    summarization_model_spec: str | None
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
    # `from_payload` narrows both the in-process schema and the remote JSON
    # dict, so this stays in step with every other context consumer.
    context = CLIContextSchema.from_payload(runtime.context)
    if context is None:
        return RuntimeModelConfig(
            model_spec=None,
            summarization_model_spec=None,
            model_params={},
            profile_overrides={},
            context_limit=None,
        )
    return RuntimeModelConfig(
        model_spec=context.model,
        summarization_model_spec=context.summarization_model,
        model_params=context.model_params,
        profile_overrides=context.profile_overrides,
        context_limit=context.model_context_limit,
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
        summarization_model_spec: str | None = None,
    ) -> None:
        """Initialize the CLI compaction middleware.

        Args:
            summarization: Summarization engine used by every compaction path.
            system_prompt: Optional prompt fragment advertising the compact tool.
            cli_max_retries: Explicit `--max-retries` value to retain when
                rebuilding a runtime-selected model for `/offload`.
            summarization_model_spec: Startup model spec to use only for summaries.
        """
        super().__init__(summarization, system_prompt=system_prompt)
        self._cli_max_retries = cli_max_retries
        self._summarization_model_spec = summarization_model_spec
        # One-entry memo for `_summarization_for_runtime`. Every model call
        # consults it, but the runtime model configuration only changes on
        # `/model` or `/summarization-model`, so a single slot hits almost
        # always and keeps `create_model`'s credential reads and client
        # construction off the per-turn path. It also lets one
        # `_LazySummaryModel` survive across compactions, which is what its
        # `_configured` and `_warned` memos are sized for.
        self._runtime_summarization: tuple[str, SummarizationMiddleware] | None = None

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

    @staticmethod
    def _auto_compaction_request(
        request: ModelRequest,
        summarization: SummarizationMiddleware,
    ) -> ModelRequest | None:
        """Return the prepared request when threshold compaction will run.

        Args:
            request: The pending model request.
            summarization: Request-local summarizer for this runtime. Passed in
                rather than read from `self._summarization` so a summarization
                override cannot be bypassed; `@staticmethod` makes that
                structural rather than a convention.

        Returns:
            The prepared request, or `None` when threshold compaction will not
                run.
        """
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
        summarization: SummarizationMiddleware,
    ) -> None:
        """Gate a provider-overflow fallback before it compacts.

        Args:
            request: The request that overflowed.
            overflow: The provider overflow being recovered from.
            summarization: Request-local summarizer for this runtime, passed in
                so an override cannot be bypassed.

        Raises:
            _AutoCompactionBlockedError: If the hook blocks compaction.
        """
        if summarization._determine_cutoff_index(
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
        summarization = self._summarization_for_runtime(request.runtime)
        prepared = self._auto_compaction_request(request, summarization)
        if prepared is not None:
            if not self._pre_auto_compact(prepared):
                return call_model(prepared)
            return summarization.wrap_model_call(request, call_model)

        overflow_gated = False

        def gated_handler(next_request: ModelRequest) -> ModelResponse:
            nonlocal overflow_gated
            try:
                return call_model(next_request)
            except ContextOverflowError as overflow:
                if not overflow_gated:
                    overflow_gated = True
                    self._pre_overflow_compact(next_request, overflow, summarization)
                raise

        try:
            return summarization.wrap_model_call(request, gated_handler)
        except _AutoCompactionBlockedError as blocked:
            raise blocked.overflow from None

    @staticmethod
    async def _awrap_with_archive_lock(
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
        summarization: SummarizationMiddleware,
    ) -> ModelResponse | ExtendedModelResponse:
        """Serialize an automatic archive append with other compactions.

        Args:
            request: The pending model request.
            handler: The downstream model handler.
            summarization: Request-local summarizer for this runtime. Passed in
                rather than read from `self._summarization` so a summarization
                override cannot be bypassed; `@staticmethod` makes that
                structural rather than a convention.

        Returns:
            The wrapped model response.
        """
        session_id = summarization._get_session_id(request.state)
        async with _archive_lock(session_id):
            return await summarization.awrap_model_call(request, handler)

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
        summarization = await asyncio.to_thread(
            self._summarization_for_runtime, request.runtime
        )
        prepared = self._auto_compaction_request(request, summarization)
        if prepared is not None:
            if not self._pre_auto_compact(prepared):
                return await call_model(prepared)
            return await self._awrap_with_archive_lock(
                request, call_model, summarization
            )

        overflow_gated = False

        async def gated_handler(next_request: ModelRequest) -> ModelResponse:
            nonlocal overflow_gated
            try:
                return await call_model(next_request)
            except ContextOverflowError as overflow:
                if not overflow_gated:
                    overflow_gated = True
                    self._pre_overflow_compact(next_request, overflow, summarization)
                raise

        try:
            return await self._awrap_with_archive_lock(
                request, gated_handler, summarization
            )
        except _AutoCompactionBlockedError as blocked:
            raise blocked.overflow from None

    def _run_compact(self, runtime: ToolRuntime[Any, Any]) -> Command:
        """Run model-initiated compaction with a request-local summarizer.

        Takes no archive lock, unlike `_arun_compact`: `_archive_lock` is an
        asyncio lock, so only the async path can serialize. This matches the
        SDK's sync behavior.

        Args:
            runtime: The tool runtime for this compaction.

        Returns:
            The compact tool's state command.
        """
        tool = self._compact_host(self._summarization_for_runtime(runtime))
        return tool._run_compact(runtime)

    async def _arun_compact(self, runtime: ToolRuntime[Any, Any]) -> Command:
        """Serialize model-initiated compaction with a request-local summarizer.

        Args:
            runtime: The tool runtime for this compaction.

        Returns:
            The compact tool's state command.
        """
        summarization = await asyncio.to_thread(
            self._summarization_for_runtime, runtime
        )
        session_id = summarization._get_session_id(runtime.state)
        tool = self._compact_host(summarization)
        async with _archive_lock(session_id):
            return await tool._arun_compact(runtime)

    @staticmethod
    def _compact_host(
        summarization: SummarizationMiddleware,
    ) -> SummarizationToolMiddleware:
        """Host the SDK's compact implementation over a chosen summarizer.

        The SDK's `_run_compact` reads `self._summarization`, so a throwaway
        host is the only way to point it at the request-local summarizer.

        Args:
            summarization: Summarizer the compaction must run on.

        Returns:
            A middleware instance used only for its compact implementation.
                `system_prompt` is `None` because only `wrap_model_call` reads
                it and this instance never wraps a call.
        """
        return SummarizationToolMiddleware(summarization, system_prompt=None)

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
        """Build a summarizer when the runtime overrides either model.

        Args:
            runtime: Runtime carrying the current `CLIContext`.

        Returns:
            The startup summarizer when neither a runtime model nor a
                summarization model is selected. Otherwise a summarizer over
                the same configured backend whose thresholds and token counting
                track the main model, while summary generation uses the
                summarization override when one is set -- so the two can be
                different models.
        """
        config = _runtime_model_config(runtime)
        summary_model_spec = config.summarization_model_spec
        if summary_model_spec == INHERIT_SUMMARIZATION_MODEL:
            summary_model_spec = None
        elif summary_model_spec is None:
            summary_model_spec = self._summarization_model_spec
        if not config.model_spec and not summary_model_spec:
            return self._summarization

        cache_key = json.dumps(
            [
                config.model_spec,
                summary_model_spec,
                config.model_params,
                config.profile_overrides,
                config.context_limit,
            ],
            sort_keys=True,
            default=repr,
        )
        cached = self._runtime_summarization
        if cached is not None and cached[0] == cache_key:
            return cached[1]

        from deepagents_code.config import create_model

        if config.model_spec:
            model = create_model(
                config.model_spec,
                extra_kwargs=config.model_params or None,
                profile_overrides=config.profile_overrides or None,
                cli_max_retries=self._cli_max_retries,
            ).model
        else:
            model = self._summarization.model
        # Only a freshly built model may have its profile mutated. On the
        # `else` branch above, `model` is the shared startup summarizer's own
        # instance, so applying a per-request limit here would leak into every
        # later turn. The limit is also derived from that same profile, so
        # applying it there would be a no-op anyway.
        context_limit = config.context_limit if config.model_spec else None
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
        if summary_model_spec:
            _install_lazy_summary_model(
                summarization,
                summary_model_spec,
                self._cli_max_retries,
            )
        else:
            _install_summary_model_retries(summarization)
        self._runtime_summarization = (cache_key, summarization)
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
    summarization_model_spec: str | None = None,
) -> CLICompactionMiddleware:
    """Create the dcode compaction middleware from the SDK configuration.

    Args:
        model: Startup model or model specification.
        backend: Agent backend used for archive persistence.
        cli_max_retries: Explicit `--max-retries` value for runtime rebuilds.
        summarization_model_spec: Startup model spec to use only for summaries.

    Returns:
        CLI compaction middleware with the SDK's model-aware defaults.
    """
    sdk_middleware = create_summarization_tool_middleware(model, backend)
    _install_summary_model_retries(sdk_middleware._summarization)
    return CLICompactionMiddleware(
        sdk_middleware._summarization,
        system_prompt=sdk_middleware.system_prompt,
        cli_max_retries=cli_max_retries,
        summarization_model_spec=summarization_model_spec,
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
