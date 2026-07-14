"""CLI-specific conversation compaction middleware."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Annotated, Any

from deepagents.middleware.summarization import (
    SummarizationToolMiddleware,
    create_summarization_middleware,
    create_summarization_tool_middleware,
)
from langchain.tools import (
    ToolRuntime,  # noqa: TC002  # inspected for runtime injection
)
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolArg, StructuredTool
from langgraph.types import Command

from deepagents_code._cli_context import CLIContextSchema

if TYPE_CHECKING:
    from deepagents.backends.protocol import BACKEND_TYPES
    from deepagents.middleware.summarization import SummarizationMiddleware
    from langchain.chat_models import BaseChatModel

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


def _runtime_model_config(
    runtime: ToolRuntime,
) -> tuple[str | None, dict[str, Any], dict[str, Any], int | None]:
    """Read the active model configuration from a tool runtime.

    Args:
        runtime: Runtime injected into the compaction tool.

    Returns:
        The active model specification, invocation parameters, profile
            overrides, and effective context-window limit.
    """
    context = runtime.context
    if isinstance(context, CLIContextSchema):
        return (
            context.model,
            context.model_params,
            context.profile_overrides,
            context.model_context_limit,
        )
    if isinstance(context, dict):
        model = context.get("model")
        params = context.get("model_params")
        profile_overrides = context.get("profile_overrides")
        context_limit = context.get("model_context_limit")
        return (
            model if isinstance(model, str) else None,
            dict(params) if isinstance(params, dict) else {},
            dict(profile_overrides) if isinstance(profile_overrides, dict) else {},
            context_limit if isinstance(context_limit, int) else None,
        )
    return None, {}, {}, None


class CLICompactionMiddleware(SummarizationToolMiddleware):
    """Add explicit forced compaction and runtime model selection for dcode.

    The SDK tool's normal, model-initiated behavior remains unchanged. The
    private `force` input is used only by the user-initiated `/offload` path,
    which must compact whenever messages exceed the retention window even when
    the conversation has not reached the SDK's proactive eligibility gate.
    """

    def _create_compact_tool(self) -> StructuredTool:
        """Create the CLI variant of `compact_conversation`.

        Returns:
            A tool that accepts the `/offload`-only `force` flag.
        """
        middleware = self

        # `force` is annotated `InjectedToolArg` so it is stripped from the
        # schema the model sees: the model can only reach the normal, gated
        # path. `/offload` seeds the tool call with `force=True` directly, and
        # an injected value supplied on the call still reaches the function.
        def sync_compact(
            runtime: ToolRuntime,
            force: Annotated[bool, InjectedToolArg] = False,
        ) -> Command:
            if not force:
                return middleware._run_compact(runtime)
            return middleware._run_forced_compact(runtime)

        async def async_compact(
            runtime: ToolRuntime,
            force: Annotated[bool, InjectedToolArg] = False,
        ) -> Command:
            if not force:
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
        model_spec, model_params, profile_overrides, context_limit = (
            _runtime_model_config(runtime)
        )
        if not model_spec:
            return self._summarization

        from deepagents_code.config import create_model

        model = create_model(
            model_spec,
            extra_kwargs=model_params or None,
            profile_overrides=profile_overrides or None,
        ).model
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

    def _run_forced_compact(self, runtime: ToolRuntime) -> Command:
        """Synchronously compact without the SDK eligibility gate.

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
            summary = summarization._create_summary(to_summarize)
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
            summary = await summarization._acreate_summary(to_summarize)
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
