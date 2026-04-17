"""Codex-specific compaction middleware using the OpenAI Responses `/compact` endpoint.

This middleware is a drop-in replacement for `SummarizationMiddleware` when the
active harness profile sets ``use_codex_compaction=True`` (currently the Codex
family — ``gpt-5.2-codex``, ``gpt-5.3-codex``).

## What is compaction?

Instead of generating an LLM-authored text summary of older turns (the default
``SummarizationMiddleware`` behavior), compaction calls OpenAI's
``/responses/compact`` endpoint, which returns an opaque ``encrypted_content``
item. That item round-trips through the Responses API on subsequent turns and
preserves procedural detail (tool-call sequencing, reasoning, phase metadata)
that ad-hoc text summarization loses.

Trade-off: the compaction item is opaque and only usable within OpenAI's
Responses API. If the API call fails for any reason, this middleware falls
back transparently to LLM-based summarization via an internal
``_DeepAgentsSummarizationMiddleware`` instance.

## Composition, not duplication

Rather than re-implement trigger thresholds, arg truncation, offload,
partitioning, and thread-ID resolution, this class **composes** a
``_DeepAgentsSummarizationMiddleware`` (via ``create_summarization_middleware``)
and delegates to it for everything except the ``/compact`` call itself. This
preserves Deep Agents-level behaviors (especially tool-call arg truncation)
that would otherwise be silently lost on Codex.

## State contract

Adds ``codex_compaction_item`` to agent state — a dict persisted across turns
via the standard LangGraph update mechanism:

```python
{
    "cutoff_index": int,  # absolute index in state messages
    "output": list[dict],  # /compact output items (model_dump'd)
    "file_path": str | None,  # backend path for offloaded history
}
```

The ``_summarization_event`` key from the inner middleware is orthogonal. If a
thread switches profiles mid-run, one of the two keys may become stale but
neither is removed — both remain safe no-ops when inapplicable.

## Phase metadata

Phase round-tripping lives entirely in ``langchain-openai`` ``>=1.1.12``; this
middleware does not touch the field. The drift-guard unit test in
``test_codex_compaction.py`` confirms that ``_construct_responses_api_input``
continues to lift ``block["phase"]`` onto the outbound input item.
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Annotated, Any, NotRequired

from langchain.agents.middleware.types import AgentMiddleware, AgentState, ExtendedModelResponse, PrivateStateAttr
from langchain_core.messages import AIMessage, AnyMessage, BaseMessage
from langgraph.types import Command
from typing_extensions import TypedDict

from deepagents.middleware.summarization import (
    _DeepAgentsSummarizationMiddleware,
    create_summarization_middleware,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain.agents.middleware.types import ModelRequest, ModelResponse
    from langchain_core.language_models import BaseChatModel
    from openai import AsyncOpenAI

    from deepagents.backends.protocol import BACKEND_TYPES

logger = logging.getLogger(__name__)

_COMPACT_TIMEOUT_SECONDS: float = 180.0
"""Per-call timeout for ``/compact`` requests, in seconds.

Compacting a large context window is a real model invocation and can take a
while, but a runaway call must not stall a user turn. 180s is generous enough
to cover large (~200k token) windows on Codex while still failing fast enough
to trip the LLM-summarization fallback if the endpoint hangs.
"""


class _CodexCompactionEvent(TypedDict):
    """Event payload persisted in state after a successful `/compact` call.

    Attributes:
        cutoff_index: Absolute index in state messages where compaction
            occurred. Messages before this index are superseded by the
            synthetic compaction head message on effective-list reconstruction.
        output: The ``/compact`` endpoint's output items, serialized via
            ``model_dump(exclude_none=True, mode="json")``. These round-trip
            back to Responses input items via
            ``_construct_responses_api_input``.
        file_path: Backend path where the pre-compaction messages were
            offloaded, or ``None`` if offload failed.
    """

    cutoff_index: int
    output: list[dict[str, Any]]
    file_path: str | None


class _CodexCompactionState(AgentState):
    """State extension adding the `/compact` output payload.

    The value is the most recent compaction event; older events are superseded
    rather than accumulated. Messages prior to ``cutoff_index`` are offloaded
    to the backend and no longer part of the effective conversation.
    """

    codex_compaction_item: Annotated[NotRequired[_CodexCompactionEvent | None], PrivateStateAttr]


class CodexCompactionMiddleware(AgentMiddleware):
    """Compaction middleware using OpenAI Responses `/compact`.

    Composes a ``_DeepAgentsSummarizationMiddleware`` for all non-compaction
    concerns (arg truncation, offload, partitioning, thread ID, fallback).

    Args:
        model: Resolved chat model instance. Must be an OpenAI model that
            supports the ``/compact`` endpoint (currently the Codex family).
            The model is used both to extract the OpenAI model name for the
            ``/compact`` request and as the summarization model for the
            fallback path.
        backend: Backend or factory for offloading conversation history
            before compaction. Forwarded to the inner summarization middleware.
        client: Optional pre-configured ``AsyncOpenAI`` client. If ``None``,
            a default client is lazily created on first use, reading
            credentials from the environment.
    """

    state_schema = _CodexCompactionState

    def __init__(
        self,
        model: BaseChatModel,
        backend: BACKEND_TYPES,
        *,
        client: AsyncOpenAI | None = None,
    ) -> None:
        """Initialize the compaction middleware.

        Args:
            model: Resolved OpenAI chat model instance (see class docstring).
            backend: Backend or factory for offloading older turns.
            client: Optional pre-configured `AsyncOpenAI` client; one is
                lazily constructed if absent.
        """
        self._model = model
        self._inner: _DeepAgentsSummarizationMiddleware = create_summarization_middleware(model, backend)
        self._client = client

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_client(self) -> AsyncOpenAI:
        """Return the cached `AsyncOpenAI` client, constructing one if needed."""
        if self._client is None:
            from openai import AsyncOpenAI  # noqa: PLC0415

            self._client = AsyncOpenAI()
        return self._client

    def _resolve_model_name(self) -> str:
        """Extract the OpenAI model identifier from the resolved chat model.

        Raises:
            RuntimeError: If no model name can be discovered. This is a
                configuration error — compaction requires an OpenAI model.
        """
        for attr in ("model_name", "model"):
            value = getattr(self._model, attr, None)
            if isinstance(value, str) and value:
                return value
        msg = (
            "CodexCompactionMiddleware could not determine the OpenAI model "
            f"name from {type(self._model).__name__}. Configure the profile "
            "with a ChatOpenAI instance."
        )
        raise RuntimeError(msg)

    @staticmethod
    def _build_compaction_message(output_items: list[dict[str, Any]]) -> AIMessage:
        """Wrap raw `/compact` output items in an `AIMessage` for the message list.

        The blocks round-trip unchanged through ``_construct_responses_api_input``
        because the Responses layer in langchain-openai explicitly supports
        ``compaction`` / ``reasoning`` / etc. content-block types.

        Args:
            output_items: List of ``ResponseOutputItem.model_dump()`` dicts
                returned by the ``/compact`` call.

        Returns:
            A synthetic `AIMessage` that stands in for all compacted turns.
        """
        return AIMessage(
            content=output_items,
            additional_kwargs={"lc_source": "codex_compaction"},
        )

    def _effective_messages(
        self,
        messages: list[AnyMessage],
        event: _CodexCompactionEvent | None,
    ) -> list[AnyMessage]:
        """Reconstruct the effective message list from state + compaction event.

        When a prior compaction event exists, the effective conversation is
        the synthetic compaction message followed by all state messages from
        ``cutoff_index`` onward. Mirrors
        ``_DeepAgentsSummarizationMiddleware._apply_event_to_messages``.

        Args:
            messages: Full message list from state.
            event: The most recent `codex_compaction_item`, or ``None``.

        Returns:
            The effective conversation the model should see on this turn.
        """
        if event is None:
            return list(messages)
        try:
            cutoff_idx = event["cutoff_index"]
            output_items = event["output"]
        except KeyError as exc:
            logger.warning("Malformed codex_compaction_item (missing keys): %s", exc)
            return list(messages)
        compaction_msg = self._build_compaction_message(output_items)
        if cutoff_idx > len(messages):
            logger.warning(
                "Compaction cutoff_index %d exceeds message count %d",
                cutoff_idx,
                len(messages),
            )
            return [compaction_msg]
        return [compaction_msg, *messages[cutoff_idx:]]

    @staticmethod
    def _compute_state_cutoff(
        event: _CodexCompactionEvent | None,
        effective_cutoff: int,
    ) -> int:
        """Translate an effective-list cutoff to an absolute state index.

        Same arithmetic as the summarization middleware: when a prior
        compaction event exists, the effective list starts with one
        synthesized message at index 0, so we subtract one to avoid
        double-counting it.
        """
        if event is None:
            return effective_cutoff
        prior_cutoff = event.get("cutoff_index")
        if not isinstance(prior_cutoff, int):
            return effective_cutoff
        return prior_cutoff + effective_cutoff - 1

    async def _do_compact(
        self,
        to_compact: list[AnyMessage],
    ) -> list[dict[str, Any]]:
        """Invoke the Responses `/compact` endpoint and return serialized output.

        When a prior compaction event exists, the synthetic compaction head is
        already at ``to_compact[0]`` (see ``_effective_messages``). Its content
        is the prior ``/compact`` output, and
        ``_construct_responses_api_input`` passes each compaction / reasoning
        content block through unchanged — so the prior items are emitted into
        ``input_items`` exactly once via the normal round-trip. Do not also
        prepend ``prior_event["output"]`` here; doing so duplicates the
        compaction item in the outbound request.

        We deliberately do not forward ``request.system_message`` as the
        endpoint's ``instructions`` parameter. ``/compact`` summarizes prior
        conversation state; it should not be handed a task-framing system
        prompt that could steer the compaction away from a faithful transcript.
        The system message is still sent on the subsequent ``/responses`` call
        where it belongs.

        Args:
            to_compact: Messages to send to `/compact` (the "old" partition
                of the current effective message list, including the
                synthetic compaction head if one was present).

        Returns:
            The ``/compact`` response's ``output`` list, each item serialized
                via ``model_dump(exclude_none=True, mode="json")``.

        Raises:
            Any exception from the OpenAI SDK is surfaced for the caller to
            handle (typically triggers fallback to summarization).
        """
        from langchain_openai.chat_models.base import _construct_responses_api_input  # noqa: PLC0415

        input_items = _construct_responses_api_input(to_compact)

        client = self._get_client()
        model_name = self._resolve_model_name()
        result = await client.responses.compact(
            model=model_name,
            input=input_items,
            timeout=_COMPACT_TIMEOUT_SECONDS,
        )
        return [item.model_dump(exclude_none=True, mode="json") for item in result.output]

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse | ExtendedModelResponse:
        """Apply compaction-aware message handling around the model call.

        1. Reconstruct effective messages from any prior compaction event.
        2. Delegate tool-call arg truncation to the inner middleware.
        3. If under the compaction trigger, forward unchanged.
        4. If over the trigger, call `/compact`:
            - On success, rebuild state with a new compaction event and a
                synthetic head message.
            - On failure, delegate the full turn to the inner summarization
                middleware (LLM-based fallback).

        Args:
            request: Incoming model request from LangGraph.
            handler: Downstream handler to invoke with the (possibly modified)
                request.

        Returns:
            A plain `ModelResponse` when no compaction happens this turn, or
                an `ExtendedModelResponse` whose command updates
                ``codex_compaction_item`` in state.
        """
        inner = self._inner

        prior_event: _CodexCompactionEvent | None = request.state.get("codex_compaction_item")
        effective = self._effective_messages(request.messages, prior_event)

        truncated, _ = inner._truncate_args(effective, request.system_message, request.tools)

        counted = [request.system_message, *truncated] if request.system_message is not None else truncated
        try:
            total_tokens = inner.token_counter(counted, tools=request.tools)  # ty: ignore[unknown-argument]
        except TypeError:
            total_tokens = inner.token_counter(counted)

        if not inner._should_summarize(truncated, total_tokens):
            return await handler(request.override(messages=truncated))

        effective_cutoff = inner._determine_cutoff_index(truncated)
        if effective_cutoff <= 0:
            return await handler(request.override(messages=truncated))

        to_compact, preserved = inner._partition_messages(truncated, effective_cutoff)

        backend = inner._get_backend(request.state, request.runtime)

        # Sequence compact before offload: if ``/compact`` fails, we delegate
        # the whole turn to ``inner.awrap_model_call`` which performs its own
        # offload. Running the wrapper's offload in parallel (or before the
        # fallback) would append the same pre-compaction window to the
        # history file twice.
        try:
            compact_output_items = await self._do_compact(to_compact)
        except Exception:
            logger.exception(
                "Codex /compact call failed; falling back to LLM summarization",
            )
            return await inner.awrap_model_call(request, handler)

        file_path = await inner._aoffload_to_backend(backend, to_compact)

        if file_path is None:
            msg = "Offloading conversation history to backend failed during Codex compaction. Older messages will not be recoverable."
            logger.error(msg)
            warnings.warn(msg, stacklevel=2)

        state_cutoff_index = self._compute_state_cutoff(prior_event, effective_cutoff)
        new_event: _CodexCompactionEvent = {
            "cutoff_index": state_cutoff_index,
            "output": compact_output_items,
            "file_path": file_path,
        }

        compaction_head = self._build_compaction_message(compact_output_items)
        modified_messages: list[BaseMessage] = [compaction_head, *preserved]
        response = await handler(request.override(messages=modified_messages))

        return ExtendedModelResponse(
            model_response=response,
            command=Command(update={"codex_compaction_item": new_event}),
        )

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse | ExtendedModelResponse:
        """Synchronous entry point — unsupported for compaction.

        The `/compact` endpoint is async-only via the OpenAI SDK. Sync callers
        should not be common on the Codex path, but for safety we fall back
        entirely to the inner summarization middleware's sync path.

        Args:
            request: Incoming model request.
            handler: Downstream handler.

        Returns:
            Whatever the inner summarization middleware returns.
        """
        return self._inner.wrap_model_call(request, handler)
