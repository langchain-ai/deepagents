"""CLI-specific conversation compaction middleware."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Annotated, Any, NamedTuple, cast

from deepagents.backends.protocol import FILE_NOT_FOUND
from deepagents.middleware.summarization import (
    SummarizationToolMiddleware,
    create_summarization_middleware,
    create_summarization_tool_middleware,
)
from langchain.tools import (
    ToolRuntime,  # noqa: TC002  # inspected for runtime injection
)
from langchain_core.messages import ToolMessage, get_buffer_string
from langchain_core.tools import InjectedToolArg, StructuredTool
from langgraph.types import Command

from deepagents_code._cli_context import CLIContextSchema

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from deepagents.backends.protocol import (
        BACKEND_TYPES,
        BackendProtocol,
        EditResult,
        FileDownloadResponse,
        WriteResult,
    )
    from deepagents.middleware.summarization import SummarizationMiddleware
    from langchain.chat_models import BaseChatModel
    from langchain_core.messages import AnyMessage
    from langgraph.prebuilt.tool_node import ToolCallRequest

logger = logging.getLogger(__name__)


COMPACTION_FAILURE_PREFIX = "Compaction failed"
"""Stable prefix for forced-compaction failure tool messages.

`/offload` drives the tool server-side and can only observe the resulting
`ToolMessage` text across the LangGraph server boundary, so it keys failure
detection on this prefix. Owning the literal here means the producer
(`_forced_compact_error`) and both consumers (`app._drive_server_side_compaction`
live-stream detection and `app._find_compaction_failure` committed-state scan)
reference one constant instead of re-hardcoding the wording independently.

Note: this value is deliberately identical to the leading text of the SDK's own
model-initiated compaction-failure message, so a failure emitted by either path
is recognized. Because the scan is bounded to messages produced by the current
`/offload` attempt, a stale failure from an unrelated prior turn is not matched.
Only the *prefix position* is load-bearing; wording after it is free to change.
"""

_OFFLOAD_SEED_ID_PREFIX = "offload-seed-"


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


def _runtime_model_config(runtime: ToolRuntime) -> RuntimeModelConfig:
    """Read the active model configuration from a tool runtime.

    Args:
        runtime: Runtime injected into the compaction tool.

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
        model = context.get("model")
        params = context.get("model_params")
        profile_overrides = context.get("profile_overrides")
        context_limit = context.get("model_context_limit")
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


def _summary_input(
    summarization: SummarizationMiddleware,
    messages: list[AnyMessage],
) -> tuple[str | None, str | None]:
    """Prepare the SDK summary prompt without swallowing model exceptions.

    Args:
        summarization: Deep Agents summarization helper.
        messages: Messages selected for summarization.

    Returns:
        An immediate summary and `None`, or `None` and the model prompt.
    """
    if not messages:
        return "No previous conversation history.", None
    trimmed = summarization._lc_helper._trim_messages_for_summary(messages)
    if not trimmed:
        return "Previous conversation was too long to summarize.", None
    formatted = get_buffer_string(trimmed, format="xml")
    prompt = summarization._lc_helper.summary_prompt.format(messages=formatted).rstrip()
    return None, prompt


def _create_summary_with_retry(
    summarization: SummarizationMiddleware,
    messages: list[AnyMessage],
) -> str:
    """Invoke the summary model while preserving failures for retry policy.

    Returns:
        The generated or immediate summary text.

    Raises:
        RuntimeError: If summary input preparation produces an invalid result.
    """
    immediate, prompt = _summary_input(summarization, messages)
    if immediate is not None:
        return immediate
    if prompt is None:
        msg = "Unexpected: summary input produced neither text nor a prompt"
        raise RuntimeError(msg)
    response = summarization.model.invoke(
        prompt,
        config={"metadata": {"lc_source": "summarization"}},
    )
    return response.text.strip()


async def _acreate_summary_with_retry(
    summarization: SummarizationMiddleware,
    messages: list[AnyMessage],
) -> str:
    """Asynchronously invoke the summary model without swallowing failures.

    Returns:
        The generated or immediate summary text.

    Raises:
        RuntimeError: If summary input preparation produces an invalid result.
    """
    immediate, prompt = _summary_input(summarization, messages)
    if immediate is not None:
        return immediate
    if prompt is None:
        msg = "Unexpected: summary input produced neither text nor a prompt"
        raise RuntimeError(msg)
    response = await summarization.model.ainvoke(
        prompt,
        config={"metadata": {"lc_source": "summarization"}},
    )
    return response.text.strip()


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
    """Add explicit forced compaction and runtime model selection for dcode.

    The SDK tool's normal, model-initiated behavior remains unchanged. The
    private `force` input is used only by the user-initiated `/offload` path,
    which must compact whenever messages exceed the retention window even when
    the conversation has not reached the SDK's proactive eligibility gate.
    """

    _model_retry_fallback: int

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

    def _resolve_backend(self, runtime: ToolRuntime) -> BackendProtocol:
        """Resolve the backend with fail-closed archive append behavior.

        Args:
            runtime: Runtime used to resolve backend factories.

        Returns:
            A backend adapter that refuses writes after raised archive reads.
        """
        backend = super()._resolve_backend(runtime)
        return cast("BackendProtocol", _ArchiveReadGuard(backend))

    def _is_runtime_compaction_eligible(
        self,
        summarization: SummarizationMiddleware,
        messages: list[AnyMessage],
    ) -> bool:
        """Evaluate the gate against the resolved runtime summarizer.

        Returns:
            Whether the active model's thresholds allow compaction.
        """
        if summarization is self._summarization:
            return self._is_eligible_for_compaction(messages)
        runtime_tool = SummarizationToolMiddleware(summarization, system_prompt=None)
        return runtime_tool._is_eligible_for_compaction(messages)

    def _summarization_for_runtime(
        self, runtime: ToolRuntime
    ) -> SummarizationMiddleware:
        """Build a summarizer for the active runtime model when overridden.

        Args:
            runtime: Runtime carrying the current `CLIContext`.

        Returns:
            The startup summarizer when no runtime model is selected, otherwise
                a model-aware summarizer using the same resolved backend.
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
        backend = self._resolve_backend(runtime)
        return create_summarization_middleware(model, backend)

    def _run_compact(self, runtime: ToolRuntime) -> Command:
        """Run ordinary synchronous compaction with model retries.

        Returns:
            The compaction state update or an error tool message.
        """
        tool_call_id = runtime.tool_call_id or ""
        try:
            summarization = self._summarization_for_runtime(runtime)
        except Exception as exc:
            logger.exception("compact_conversation model resolution failed")
            return self._compact_error(tool_call_id, exc)
        messages = runtime.state.get("messages", [])
        event = runtime.state.get("_summarization_event")
        effective = summarization._apply_event_to_messages(messages, event)

        if not self._is_runtime_compaction_eligible(summarization, effective):
            return self._nothing_to_compact(tool_call_id)
        cutoff = summarization._determine_cutoff_index(effective)
        if cutoff == 0:
            return self._nothing_to_compact(tool_call_id)

        try:
            to_summarize, _ = summarization._partition_messages(effective, cutoff)
            from deepagents_code.model_retry import (
                DEFAULT_MODEL_RETRIES,
                CodeModelRetryMiddleware,
                _runtime_model_retry_override,
            )

            retry = CodeModelRetryMiddleware(
                max_retries=getattr(
                    self, "_model_retry_fallback", DEFAULT_MODEL_RETRIES
                )
            )
            writer = getattr(runtime, "stream_writer", None)
            summary = retry.run_with_retry(
                summarization.model,
                lambda: _create_summary_with_retry(summarization, to_summarize),
                writer=writer,
                max_retries=_runtime_model_retry_override(runtime),
            )
            backend = self._resolve_backend(runtime)
            file_path = summarization._offload_to_backend(backend, to_summarize)
        except Exception as exc:  # tool must return a ToolMessage, not raise
            logger.exception("compact_conversation tool failed")
            return self._compact_error(tool_call_id, exc)

        return self._build_compact_result(
            runtime, to_summarize, summary, file_path, event, cutoff
        )

    async def _arun_compact(self, runtime: ToolRuntime) -> Command:
        """Run ordinary asynchronous compaction with model retries.

        Returns:
            The compaction state update or an error tool message.
        """
        tool_call_id = runtime.tool_call_id or ""
        try:
            summarization = await asyncio.to_thread(
                self._summarization_for_runtime, runtime
            )
        except Exception as exc:
            logger.exception("compact_conversation model resolution failed")
            return self._compact_error(tool_call_id, exc)
        messages = runtime.state.get("messages", [])
        event = runtime.state.get("_summarization_event")
        effective = summarization._apply_event_to_messages(messages, event)

        if not self._is_runtime_compaction_eligible(summarization, effective):
            return self._nothing_to_compact(tool_call_id)
        cutoff = summarization._determine_cutoff_index(effective)
        if cutoff == 0:
            return self._nothing_to_compact(tool_call_id)

        try:
            to_summarize, _ = summarization._partition_messages(effective, cutoff)
            from deepagents_code.model_retry import (
                DEFAULT_MODEL_RETRIES,
                CodeModelRetryMiddleware,
                _runtime_model_retry_override,
            )

            retry = CodeModelRetryMiddleware(
                max_retries=getattr(
                    self, "_model_retry_fallback", DEFAULT_MODEL_RETRIES
                )
            )
            writer = getattr(runtime, "stream_writer", None)
            summary = await retry.arun_with_retry(
                summarization.model,
                lambda: _acreate_summary_with_retry(summarization, to_summarize),
                writer=writer,
                max_retries=_runtime_model_retry_override(runtime),
            )
            backend = self._resolve_backend(runtime)
            file_path = await summarization._aoffload_to_backend(backend, to_summarize)
        except Exception as exc:  # tool must return a ToolMessage, not raise
            logger.exception("compact_conversation tool failed")
            return self._compact_error(tool_call_id, exc)

        return self._build_compact_result(
            runtime, to_summarize, summary, file_path, event, cutoff
        )

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
        tool_call_id = runtime.tool_call_id or ""
        try:
            summarization = self._summarization_for_runtime(runtime)
            messages = runtime.state.get("messages", [])
            event = runtime.state.get("_summarization_event")
            effective = summarization._apply_event_to_messages(messages, event)
            effective = _without_offload_seed(effective, tool_call_id)
            cutoff = summarization._determine_cutoff_index(effective)
            if cutoff == 0:
                return self._nothing_to_compact(tool_call_id)

            to_summarize, _ = summarization._partition_messages(effective, cutoff)
            from deepagents_code.model_retry import (
                DEFAULT_MODEL_RETRIES,
                CodeModelRetryMiddleware,
                _runtime_model_retry_override,
            )

            retry = CodeModelRetryMiddleware(
                max_retries=getattr(
                    self, "_model_retry_fallback", DEFAULT_MODEL_RETRIES
                )
            )
            writer = getattr(runtime, "stream_writer", None)
            summary = retry.run_with_retry(
                summarization.model,
                lambda: _create_summary_with_retry(summarization, to_summarize),
                writer=writer,
                max_retries=_runtime_model_retry_override(runtime),
            )
            backend = self._resolve_backend(runtime)
            file_path = summarization._offload_to_backend(backend, to_summarize)
            # The inherited `_build_compact_result` produces the same event and
            # tool message as the SDK's gated path via model-independent helpers
            # (string formatting + a staticmethod), so the runtime-selected
            # summarizer is not needed to build it. Kept inside the `try` so a
            # failure here still returns a ToolMessage rather than raising.
            return self._build_compact_result(
                runtime, to_summarize, summary, file_path, event, cutoff
            )
        except Exception as exc:  # tool errors must surface as ToolMessages
            logger.exception("forced compact_conversation failed")
            return self._forced_compact_error(tool_call_id, exc)

    async def _arun_forced_compact(self, runtime: ToolRuntime) -> Command:
        """Asynchronously compact without the SDK eligibility gate.

        Returns:
            The compaction state update or an error tool message.
        """
        tool_call_id = runtime.tool_call_id or ""
        try:
            summarization = await asyncio.to_thread(
                self._summarization_for_runtime, runtime
            )
            messages = runtime.state.get("messages", [])
            event = runtime.state.get("_summarization_event")
            effective = summarization._apply_event_to_messages(messages, event)
            effective = _without_offload_seed(effective, tool_call_id)
            cutoff = summarization._determine_cutoff_index(effective)
            if cutoff == 0:
                return self._nothing_to_compact(tool_call_id)

            to_summarize, _ = summarization._partition_messages(effective, cutoff)
            from deepagents_code.model_retry import (
                DEFAULT_MODEL_RETRIES,
                CodeModelRetryMiddleware,
                _runtime_model_retry_override,
            )

            retry = CodeModelRetryMiddleware(
                max_retries=getattr(
                    self, "_model_retry_fallback", DEFAULT_MODEL_RETRIES
                )
            )
            writer = getattr(runtime, "stream_writer", None)
            summary = await retry.arun_with_retry(
                summarization.model,
                lambda: _acreate_summary_with_retry(summarization, to_summarize),
                writer=writer,
                max_retries=_runtime_model_retry_override(runtime),
            )
            backend = self._resolve_backend(runtime)
            file_path = await summarization._aoffload_to_backend(backend, to_summarize)
            # See `_run_forced_compact` for why the inherited builder is reused
            # and why it stays inside the `try`.
            return self._build_compact_result(
                runtime, to_summarize, summary, file_path, event, cutoff
            )
        except Exception as exc:  # tool errors must surface as ToolMessages
            logger.exception("forced compact_conversation failed")
            return self._forced_compact_error(tool_call_id, exc)

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
    backend: BACKEND_TYPES,
    *,
    model_retries: int | None = None,
) -> CLICompactionMiddleware:
    """Create the dcode compaction middleware from the SDK configuration.

    Args:
        model: Startup model or model specification.
        backend: Agent backend used for archive persistence.
        model_retries: Caller-provided retry budget for unannotated models.

    Returns:
        CLI compaction middleware with the SDK's model-aware defaults.
    """
    sdk_middleware = create_summarization_tool_middleware(model, backend)
    middleware = CLICompactionMiddleware(
        sdk_middleware._summarization,
        system_prompt=sdk_middleware.system_prompt,
    )
    if model_retries is not None:
        middleware._model_retry_fallback = model_retries
    return middleware
