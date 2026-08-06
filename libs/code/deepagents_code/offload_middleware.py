"""CLI-specific conversation compaction middleware."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, NamedTuple, Protocol, TypedDict, cast
from uuid import uuid4

from deepagents.backends.protocol import FILE_NOT_FOUND
from deepagents.middleware.summarization import (
    SummarizationEvent,
    SummarizationState,
    SummarizationToolMiddleware,
    create_summarization_middleware,
    create_summarization_tool_middleware,
)
from langchain.tools import (
    ToolRuntime,  # noqa: TC002  # inspected for runtime injection
)
from langchain_core.exceptions import ContextOverflowError
from langchain_core.messages import AIMessage, AnyMessage, ToolMessage
from langchain_core.tools import InjectedToolArg, StructuredTool
from langgraph.errors import GraphBubbleUp
from langgraph.graph.message import add_messages
from langgraph.types import Command

from deepagents_code._cli_context import CLIContextSchema
from deepagents_code.cost_tracking import CostState, CostTrackingMiddleware
from deepagents_code.hooks.models.domain import (
    CompactTrigger,
    HookEvent,
    PreCompactDecision,
    PreCompactEvent,
)
from deepagents_code.hooks.server_middleware import (
    _DEFAULT_DEADLINE,
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
    from langchain.chat_models import BaseChatModel
    from langgraph.graph.state import CompiledStateGraph
    from langgraph.prebuilt.tool_node import ToolCallRequest
    from langgraph.runtime import Runtime

    from deepagents_code.hooks.server_middleware import ServerHooksMiddleware

logger = logging.getLogger(__name__)


class _OffloadState(CostState, SummarizationState, total=False):
    """Checkpoint channels the explicit forced-compaction graph reads and writes.

    `_summarization_event` is inherited from `SummarizationState` rather than
    re-declared so it keeps the SDK's own annotation (`NotRequired`, `| None`,
    and the `PrivateStateAttr` marker) instead of a divergent copy that claimed
    the value is always present.

    Note this schema does *not* by itself keep any channel out of the graph's
    input: `PrivateStateAttr` / `OmitFromInput` are honored by `create_agent`,
    not by a raw `StateGraph`, so every declared channel would otherwise be
    client-writable. `_OffloadInput` is what actually restricts the surface.
    """


class _OffloadInput(TypedDict, total=False):
    """The only channel a caller may write when starting an `/offload` run.

    Passed as `StateGraph(input_schema=...)` so the restriction is enforced by
    the graph rather than by the client's discipline in what it sends. Replaying
    any other channel is actively harmful: `_session_cost_usd` reduces with
    `operator.add`, so echoing back the checkpointed total would double the
    thread's recorded spend on every `/offload`, and `_session_cost_transfers`
    (`operator.or_`) would resurrect settled subagent transfers. Writable
    `_summarization_event` would additionally let a local caller set the
    compaction cutoff directly (see THREAT_MODEL TB10).

    The node still reads the full `_OffloadState` from the checkpoint; only the
    run *input* is narrowed.
    """

    messages: Annotated[list[AnyMessage], add_messages]


class OffloadServerResources(NamedTuple):
    """Middleware the `/offload` operation graph shares with the agent graph."""

    compaction: CLICompactionMiddleware
    """Compaction implementation bound to the agent's composite backend."""

    hooks: ServerHooksMiddleware
    """Lifecycle middleware carrying the `PreCompact`/`PreToolUse` boundary.

    Not optional: `create_cli_agent` mounts this middleware unconditionally, and
    an absent instance would make the operation graph skip the only hook gate
    `/offload` still crosses.
    """


_OFFLOAD_RESOURCES_ATTR = "_cli_offload_resources"
"""Attribute carrying `OffloadServerResources` on the composite backend.

`create_cli_agent` builds this middleware but returns only the agent and the
backend, and every call site unpacks that pair positionally (3 in
`deepagents_code`, 61 in `tests`), so widening the return would churn the whole
test suite for one server-only consumer. The backend is the one object both the
agent graph and the separately-resolved `offload` graph already hold, which
makes it the carrier. Reaching it through `attach_offload_resources` /
`offload_resources_from` instead of a bare `getattr` keeps the attribute name in
one place and the unchecked write behind a single type assertion.
"""


def attach_offload_resources(
    backend: CompositeBackend, resources: OffloadServerResources
) -> None:
    """Publish the operation graph's middleware on the shared backend.

    Args:
        backend: Composite backend returned alongside the agent graph.
        resources: Middleware instances the `offload` graph must reuse.
    """
    setattr(backend, _OFFLOAD_RESOURCES_ATTR, resources)


def offload_resources_from(backend: CompositeBackend) -> OffloadServerResources | None:
    """Read back the middleware published by `attach_offload_resources`.

    Args:
        backend: Composite backend returned alongside the agent graph.

    Returns:
        The published resources, or `None` when the backend carries none (so
            the caller can fail with its own message rather than an
            `AttributeError`).
    """
    resources = getattr(backend, _OFFLOAD_RESOURCES_ATTR, None)
    return resources if isinstance(resources, OffloadServerResources) else None


COMPACTION_FAILURE_PREFIX = "Compaction failed"
"""Stable prefix for forced-compaction failure tool messages.

The seeded driver (local in-process agents) drives the tool and can only observe
the resulting `ToolMessage` text across the LangGraph server boundary, so it keys
failure detection on this prefix. Owning the literal here means the producers
(`_forced_compact_error` and the operation graph's node, which reuses the prefix
in the `RuntimeError` it raises) and both consumers
(`app._drive_legacy_seeded_compaction` live-stream detection and
`app._find_compaction_failure` committed-state scan) reference one constant
instead of re-hardcoding the wording independently.

Note: this value is deliberately identical to the leading text of the SDK's own
model-initiated compaction-failure message, so a failure emitted by either path
is recognized. Because the scan is bounded to messages produced by the current
`/offload` attempt, a stale failure from an unrelated prior turn is not matched.
Only the *prefix position* is load-bearing; wording after it is free to change.
"""

_OFFLOAD_SEED_ID_PREFIX = "offload-seed-"


class _AutoCompactionBlockedError(Exception):
    """Carry a blocked provider overflow past the SDK fallback handler."""

    def __init__(self, overflow: ContextOverflowError) -> None:
        super().__init__(str(overflow))
        self.overflow = overflow


def _offload_seed_message_id(tool_call_id: str) -> str:
    """Return the stable message ID for a forced `/offload` tool call.

    Args:
        tool_call_id: The seeded `compact_conversation` tool call ID.

    Returns:
        The synthetic assistant message ID associated with the tool call.
    """
    return f"{_OFFLOAD_SEED_ID_PREFIX}{tool_call_id}"


def _without_offload_seed(messages: list[Any], tool_call_id: str) -> list[Any]:
    """Exclude the synthetic `/offload` seed from retention calculations.

    Args:
        messages: Effective conversation messages including the forced tool call.
        tool_call_id: The seeded `compact_conversation` tool call ID.

    Returns:
        Conversation messages without the matching synthetic assistant message.
    """
    if not tool_call_id:
        return messages
    seed_id = _offload_seed_message_id(tool_call_id)
    return [
        message
        for message in messages
        if (
            message.get("id")
            if isinstance(message, dict)
            else getattr(message, "id", None)
        )
        != seed_id
    ]


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
    the operation graph's node receives -- which is not a `ToolRuntime`. Stating
    the dependency this narrowly means a helper that starts touching, say,
    `tool_call_id` fails to type-check instead of breaking the operation graph
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


def _offload_tool_call_id(context: object) -> str | None:
    """Read the sole tool-call ID authorized for an `/offload` run.

    Args:
        context: Runtime context supplied to the agent graph.

    Returns:
        The authorized tool-call ID, or `None` during an ordinary agent run.
    """
    value = (
        context.offload_tool_call_id
        if isinstance(context, CLIContextSchema)
        else context.get("offload_tool_call_id")
        if isinstance(context, dict)
        else None
    )
    return value if isinstance(value, str) and value else None


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
    """Add hook-aware automatic and explicit forced compaction for dcode.

    The SDK tool's normal, model-initiated behavior remains unchanged. The
    private `force` input is used only by the user-initiated `/offload` path,
    which must compact whenever messages exceed the retention window even when
    the conversation has not reached the SDK's proactive eligibility gate.
    """

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
            return await self._summarization.awrap_model_call(request, call_model)

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
            return await self._summarization.awrap_model_call(request, gated_handler)
        except _AutoCompactionBlockedError as blocked:
            raise blocked.overflow from None

    @staticmethod
    def _offload_rejection(request: ToolCallRequest) -> ToolMessage | None:
        """Reject every tool except the exact call seeded by `/offload`.

        Args:
            request: Tool call about to be executed by the graph's tool node.

        Returns:
            An error result for an unauthorized `/offload` tool call, otherwise
                `None` for an ordinary run or the exact seeded compaction call.
        """
        expected_id = _offload_tool_call_id(request.runtime.context)
        if expected_id is None:
            return None

        tool_call = request.tool_call
        args = tool_call.get("args")
        messages = request.state.get("messages", [])
        last_message = messages[-1] if messages else None
        last_message_id = (
            last_message.get("id")
            if isinstance(last_message, dict)
            else getattr(last_message, "id", None)
        )
        is_seeded_compaction = (
            tool_call.get("id") == expected_id
            and tool_call.get("name") == "compact_conversation"
            and isinstance(args, dict)
            and args.get("force") is True
            and last_message_id == _offload_seed_message_id(expected_id)
        )
        if is_seeded_compaction:
            return None

        return ToolMessage(
            content=(
                "Not executed: /offload only authorizes its seeded "
                "conversation compaction call."
            ),
            name=tool_call.get("name"),
            tool_call_id=tool_call["id"],
            status="error",
        )

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Apply the `/offload` per-run tool guard before synchronous tools.

        Args:
            request: Tool call about to be executed.
            handler: The remaining middleware/tool execution chain.

        Returns:
            The guarded rejection or the downstream tool result.
        """
        if (rejection := self._offload_rejection(request)) is not None:
            return rejection
        return handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Apply the `/offload` per-run tool guard before asynchronous tools.

        Args:
            request: Tool call about to be executed.
            handler: The remaining middleware/tool execution chain.

        Returns:
            The guarded rejection or the downstream tool result.
        """
        if (rejection := self._offload_rejection(request)) is not None:
            return rejection
        return await handler(request)

    def _create_compact_tool(self) -> StructuredTool:
        """Create the CLI variant of `compact_conversation`.

        Returns:
            A tool that accepts the `/offload`-only `force` flag.
        """
        middleware = self

        # `force` is annotated `InjectedToolArg` so it is stripped from the
        # schema the model sees. ToolNode also strips the seeded value before
        # invocation, so forced mode is selected from the trusted runtime
        # context after `_offload_rejection` validates the raw tool call.
        def sync_compact(
            runtime: ToolRuntime[Any, Any],
            force: Annotated[bool, InjectedToolArg] = False,
        ) -> Command:
            del force
            if _offload_tool_call_id(runtime.context) != runtime.tool_call_id:
                return middleware._run_compact(runtime)
            return middleware._run_forced_compact(runtime)

        async def async_compact(
            runtime: ToolRuntime[Any, Any],
            force: Annotated[bool, InjectedToolArg] = False,
        ) -> Command:
            del force
            if _offload_tool_call_id(runtime.context) != runtime.tool_call_id:
                return await middleware._arun_compact(runtime)
            return await middleware._arun_forced_compact(runtime)

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

    def _guarded_backend(self) -> BackendProtocol:
        """Wrap the configured backend with fail-closed archive append behavior.

        Returns:
            A backend adapter that refuses writes after raised archive reads.
        """
        return cast("BackendProtocol", _ArchiveReadGuard(self._summarization._backend))

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
        # Reuse the original composite backend, not the `_ArchiveReadGuard`
        # wrapper: the SDK resolves the archive prefix once in its constructor
        # via `backend.artifacts_root if isinstance(backend, CompositeBackend)`,
        # and the guard is not a `CompositeBackend`, so that check falls back to
        # a `/` prefix. The archive write would then miss the
        # `conversation_history` route and land in the default backend --
        # silently writing into the user's project tree. Adding an
        # `artifacts_root` passthrough to the guard would not help; the
        # `isinstance` is what fails. The offload call sites apply the guard
        # separately when writing (see `_guarded_backend` call sites).
        return create_summarization_middleware(model, self._summarization._backend)

    def _run_forced_compact(self, runtime: ToolRuntime) -> Command:
        """Synchronously compact without the SDK eligibility gate.

        This deliberately mirrors the SDK's own `_run_compact` step sequence
        (apply prior event, determine cutoff, partition, summarize, offload,
        build result) minus the eligibility gate. Because it is a fork rather
        than an override, it must be kept in parity when the SDK's compaction
        flow changes; the closest-fitting SDK-side fix (a `force=` seam on
        `_run_compact`) is out of scope for this PR, which is confined to
        Deep Agents Code. `test_forced_compact_matches_sdk_summarizer_calls`
        guards the summarizer-method call set against drift, but only by
        *existence*: it catches a renamed or removed dependency, not a changed
        signature nor a new step added to `_run_compact` (e.g. if the SDK later
        moved inline-media offload into the gated path). Two known consequences
        of that today: this fork does not call `_offload_inline_media` (only the
        auto `wrap_model_call` path does), so inline base64 media in compacted
        messages is not offloaded to referenceable paths and is dropped from the
        XML archive -- pre-existing SDK tool-path behavior, not introduced here.

        Returns:
            The compaction state update or an error tool message.
        """
        try:
            summarization = self._summarization_for_runtime(runtime)
            messages = runtime.state.get("messages", [])
            event = runtime.state.get("_summarization_event")
            effective = summarization._apply_event_to_messages(messages, event)
            effective = _without_offload_seed(effective, runtime.tool_call_id or "")
            cutoff = summarization._determine_cutoff_index(effective)
            if cutoff == 0:
                return self._nothing_to_compact(runtime.tool_call_id or "")
            to_summarize, _ = summarization._partition_messages(effective, cutoff)
            summary = summarization._create_summary(to_summarize)
            file_path = summarization._offload_to_backend(
                self._guarded_backend(), to_summarize
            )
            return self._build_compact_result(
                runtime, to_summarize, summary, file_path, event, cutoff
            )
        except Exception as exc:  # tool errors must surface as ToolMessages
            logger.exception("forced compact_conversation failed")
            return self._forced_compact_error(runtime.tool_call_id or "", exc)

    async def _arun_forced_compact(self, runtime: ToolRuntime) -> Command:
        """Asynchronously compact without the SDK eligibility gate.

        Returns:
            The compaction state update or an error tool message.
        """
        try:
            summarization = await asyncio.to_thread(
                self._summarization_for_runtime, runtime
            )
            messages = runtime.state.get("messages", [])
            event = runtime.state.get("_summarization_event")
            effective = summarization._apply_event_to_messages(messages, event)
            effective = _without_offload_seed(effective, runtime.tool_call_id or "")
            cutoff = summarization._determine_cutoff_index(effective)
            if cutoff == 0:
                return self._nothing_to_compact(runtime.tool_call_id or "")
            to_summarize, _ = summarization._partition_messages(effective, cutoff)
            summary = await summarization._acreate_summary(to_summarize)
            file_path = await summarization._aoffload_to_backend(
                self._guarded_backend(), to_summarize
            )
            return self._build_compact_result(
                runtime, to_summarize, summary, file_path, event, cutoff
            )
        except Exception as exc:  # tool errors must surface as ToolMessages
            logger.exception("forced compact_conversation failed")
            return self._forced_compact_error(runtime.tool_call_id or "", exc)

    async def arun_forced_compaction_update(
        self, state: _OffloadState, runtime: _HasRunContext
    ) -> dict[str, object] | None:
        """Run forced compaction as a server operation without a tool message.

        Unlike the tool paths, this raises on failure instead of returning a
        `ToolMessage`: the operation graph has no tool node to carry one, so its
        node converts the exception into a client-visible error.

        Args:
            state: Checkpointed conversation and prior summarization event.
            runtime: Run context carrier used to select the summarizer model.

        Returns:
            The state update, or `None` when nothing can be compacted.
        """
        summarization = await asyncio.to_thread(
            self._summarization_for_runtime, runtime
        )
        messages = state.get("messages", [])
        event = state.get("_summarization_event")
        effective = summarization._apply_event_to_messages(messages, event)
        cutoff = summarization._determine_cutoff_index(effective)
        if cutoff == 0:
            return None
        to_summarize, _ = summarization._partition_messages(effective, cutoff)
        summary = await summarization._acreate_summary(to_summarize)
        file_path = await summarization._aoffload_to_backend(
            self._guarded_backend(), to_summarize
        )
        return self._forced_compaction_update(
            summarization, summary, file_path, event, cutoff
        )

    @staticmethod
    def _forced_compaction_update(
        summarization: SummarizationMiddleware,
        summary: str,
        file_path: str | None,
        event: SummarizationEvent | None,
        cutoff: int,
    ) -> dict[str, object]:
        """Build the state-only result used by the dedicated `/offload` graph.

        Returns:
            The summarization-event state update.
        """
        summary_message = summarization._build_new_messages_with_path(
            summary, file_path
        )[0]
        return {
            "_summarization_event": {
                "cutoff_index": summarization._compute_state_cutoff(event, cutoff),
                "summary_message": summary_message,
                "file_path": file_path,
            }
        }

    @staticmethod
    def _forced_compact_error(tool_call_id: str, exc: Exception) -> Command:
        """Build a forced-compaction failure result with a stable prefix.

        Owned by dcode so the `/offload` client can detect failures via
        `COMPACTION_FAILURE_PREFIX`. The tool must return a `ToolMessage` rather
        than raise, so the model (and the client) see the failure as ordinary
        tool output.

        The message is intentionally generic about *where* the failure occurred:
        the guarded body spans cutoff determination, summary generation, the
        archive write, and result building, so it does not assert a specific
        stage (and does not claim nothing was written — an archive may have been
        persisted before a later step failed). It states only what is always
        true on this path: the summarization event was not committed, so the
        effective conversation is unchanged.

        Args:
            tool_call_id: The originating tool call ID.
            exc: The exception raised while compacting.

        Returns:
            A `Command` whose `ToolMessage` content starts with
                `COMPACTION_FAILURE_PREFIX`.
        """
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=(
                            f"{COMPACTION_FAILURE_PREFIX}: an error occurred "
                            f"during compaction ({type(exc).__name__}: {exc}). "
                            "Your conversation is unchanged."
                        ),
                        tool_call_id=tool_call_id,
                    )
                ],
            }
        )


def _create_cli_compaction_middleware(
    model: str | BaseChatModel,
    backend: BackendProtocol,
) -> CLICompactionMiddleware:
    """Create the dcode compaction middleware from the SDK configuration.

    Args:
        model: Startup model or model specification.
        backend: Agent backend used for archive persistence.

    Returns:
        CLI compaction middleware with the SDK's model-aware defaults.
    """
    sdk_middleware = create_summarization_tool_middleware(model, backend)
    return CLICompactionMiddleware(
        sdk_middleware._summarization,
        system_prompt=sdk_middleware.system_prompt,
    )


def create_forced_compaction_graph(
    middleware: CLICompactionMiddleware,
    *,
    hooks_middleware: ServerHooksMiddleware | None,
) -> CompiledStateGraph[Any, CLIContextSchema, Any, Any]:
    """Create the dedicated server graph used by the `/offload` command.

    This operation updates only the persisted summarization event. Unlike the
    model-facing `compact_conversation` tool, it does not manufacture an
    assistant tool call or a tool result: the slash command itself is the
    explicit user authorization boundary.

    Args:
        middleware: Compaction implementation configured with the agent's
            composite backend.
        hooks_middleware: Server lifecycle middleware shared with the
            interactive graph, or an explicit `None` to run with no hook gate.
            It is invoked against an in-memory forced tool call so `PreCompact`
            (and `PreToolUse`, which the same hook boundary dispatches for the
            forced call) retains its normal authorization boundary without
            persisting synthetic conversation messages. Required rather than
            defaulted so skipping the gate cannot happen by omission.

    Returns:
        A checkpointable graph that performs one forced compaction attempt.
    """
    from langgraph.graph import END, START, StateGraph

    cost_tracking: CostTrackingMiddleware[CLIContextSchema] = CostTrackingMiddleware()

    async def force_compact(
        state: _OffloadState, runtime: Runtime[CLIContextSchema]
    ) -> dict[str, object]:
        if hooks_middleware is not None:
            # A fresh id per invocation. `ServerHooksMiddleware` derives its hook
            # `invocation_id` from this value together with the thread, hook
            # snapshot, and prompt id, then memoizes fulfillments by that id.
            # Because the prompt id only rotates on user-prompt submit, a
            # constant here would make two `/offload`s within one turn collide
            # and replay the first run's decision -- including a denial --
            # instead of re-running the user's hook.
            forced_call_id = f"offload-precompact-{uuid4()}"
            try:
                hook_update = await hooks_middleware.aafter_model(
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
            except GraphBubbleUp:
                # Hook approval requests pause the operation graph through this
                # control-flow exception. The caller streams and fulfills it.
                raise
            except Exception as exc:
                logger.exception("/offload hook dispatch failed")
                # Same `RuntimeError` re-raise rationale as the compaction
                # failure below, but worded so a hook-layer failure is not read
                # as a compaction failure. Nothing has been written yet here.
                msg = (
                    f"Offload hooks failed: {type(exc).__name__}: {exc}. "
                    "Your conversation is unchanged."
                )
                raise RuntimeError(msg) from exc
            outcomes = hook_update.get("_hooks_pre_tool_outcomes", {})
            outcome = outcomes.get(forced_call_id, {})
            if outcome.get("behavior") == "deny":
                # Returning `{}` here would be indistinguishable from "nothing
                # old enough to compact", so the client would report a hook
                # veto as "the conversation is already compact". Raise instead
                # so the reason reaches the user. Deliberately not re-wrapped by
                # the handler below: this message is already user-facing.
                reason = outcome.get("reason") or "Blocked by a compaction hook"
                raise RuntimeError(str(reason))
        try:
            update = await middleware.arun_forced_compaction_update(state, runtime)
        except Exception as exc:
            logger.exception("forced /offload compaction failed")
            # Re-raise as `RuntimeError`: the server serializes exceptions for
            # the client and preserves the message only for an allowlist of
            # builtin types, replacing every other one with "An internal error
            # occurred" -- which is what an `OSError` from the archive write or
            # a provider SDK error would otherwise become.
            #
            # Deliberately no cost drain on this path: `aafter_agent` returns an
            # update the raise would discard, and its drain is destructive, so
            # the summarizer's spend would be lost outright. Left undrained it
            # is charged on the next turn's first step instead.
            msg = (
                f"{COMPACTION_FAILURE_PREFIX}: {type(exc).__name__}: {exc}. "
                "Your conversation is unchanged."
            )
            raise RuntimeError(msg) from exc
        # Summary generation invokes a model outside the normal agent loop, so
        # drain it here rather than leaving this run's checkpoint incomplete.
        #
        # A drain failure must not propagate: `update` already reflects an
        # archive section written to the backend, so raising here would discard
        # it and tell the user their conversation is unchanged while leaving an
        # orphaned section no `_summarization_event` references. Undrained spend
        # is merely charged on the next turn, which is the same outcome as the
        # failure path above.
        try:
            cost_update = await cost_tracking.aafter_agent(state, runtime)
        except Exception:
            logger.exception(
                "Failed to drain summary cost after /offload; the spend is "
                "charged on the next turn instead"
            )
            cost_update = None
        return {**(update or {}), **(cost_update or {})}

    graph = StateGraph(
        cast("Any", _OffloadState),
        context_schema=CLIContextSchema,
        input_schema=cast("Any", _OffloadInput),
    )
    graph.add_node("force_compact", force_compact)
    graph.add_edge(START, "force_compact")
    graph.add_edge("force_compact", END)
    return graph.compile()
