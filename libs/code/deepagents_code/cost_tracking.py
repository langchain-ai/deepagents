"""Estimate and persist cumulative model cost for each thread.

The graph owns the durable total. `CostTrackingMiddleware` is the only writer of
`_session_cost_usd`, so each cost update rides the model checkpoint and works for
local, headless, and remote graph execution without a client-side state update.
The client is a reader: it renders the streamed total and never maintains its own
lifetime figure.

Coverage is not limited to the agent's own model node. Offload/summarization and
the Auto mode classifier invoke a model directly, outside `after_model`, and
subagents run their own graph. `_SessionCostRecorder` — a callback handler
installed process-wide for every model request (see `_install_recorder`) —
collects one record per *completed* request, keyed by thread, and
`CostTrackingMiddleware` drains and prices those records on the main agent's
checkpoint path. New side invokes are covered with no extra wiring.

The recorder only collects; the middleware alone prices and writes. The agent's
own response is still priced from state, but only when the recorder did not
already charge that message ID, so a request is never counted twice. That
fallback keeps main-agent cost correct even for a model that never fires
callbacks.

Nested agents first checkpoint their own spend on the same private channel.
That makes a completed model call durable before a later tool approval can
interrupt the subgraph. When the subagent finishes, its middleware transfers
the accumulated delta through an owner-scoped state entry. The subagent tool
checkpoints that entry on the parent graph even when a sibling interrupts, while
the private total itself remains isolated between graphs.

Every caller uses `estimate_cost`, the only function that imports or calls
`genai-prices`. The import is lazy so the package and its bundled pricing data
stay off the CLI startup path. Unsupported models and malformed usage return
`None`; pricing must never interrupt a model turn.
"""

from __future__ import annotations

import logging
import math
import operator
import threading
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, NotRequired, TypedDict

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ContextT,
    OmitFromInput,
    PrivateStateAttr,
)
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.runnables.config import ensure_config
from langgraph.types import Overwrite

from deepagents_code.resume_state import ResumeState

if TYPE_CHECKING:
    from uuid import UUID

    from langchain_core.outputs import LLMResult
    from langgraph.runtime import Runtime

logger = logging.getLogger(__name__)

SESSION_COST_EVENT_TYPE = "session_cost"
"""Custom-stream event type carrying the thread's absolute cumulative cost.

Emitted by the durable writer so the status bar can track spend live without
re-pricing anything. The payload is `{"type": ..., "total": <usd>}`; `total` is
the full thread lifetime estimate, never a delta, so a client that misses an
event still converges on the next one.
"""

_PROVIDER_ALIASES: dict[str, str] = {
    "azure_openai": "azure",
    "bedrock": "aws",
    "google_genai": "google",
    "google_vertexai": "google",
    "mistralai": "mistral",
    "xai": "x-ai",
}
"""Map LangChain provider names to the identifiers used by `genai-prices`."""

_UNPRICEABLE_PROVIDERS: frozenset[str] = frozenset({"openai_codex"})
"""Providers whose access model is not equivalent to per-token API billing."""

_CONFIGURED_PROVIDER_METADATA_KEY = "deepagents_code_configured_provider"
"""Model metadata key preserving the provider selected by `create_model`."""

_CHECKPOINT_NAMESPACE_METADATA_KEY = "langgraph_checkpoint_ns"
"""Callback metadata key identifying the graph node that made a request."""


def _set_configured_provider_metadata(model: object, provider: str) -> None:
    """Attach the configured provider to every request made by a model.

    LangChain provider integrations can report a generic backend in response
    metadata: Azure and the Codex subscription model both report `openai`.
    Model metadata reaches `on_chat_model_start`, so recording the configured
    provider there preserves the distinction for main, side, and nested calls.

    Args:
        model: Chat model whose callback metadata should carry the provider.
        provider: Provider selected while constructing the model.
    """
    if not provider:
        return
    try:
        current = getattr(model, "metadata", None)
        metadata = dict(current) if isinstance(current, Mapping) else {}
        metadata[_CONFIGURED_PROVIDER_METADATA_KEY] = provider
        model.metadata = metadata  # ty: ignore[unresolved-attribute]
    except Exception:
        # Cost estimation is best-effort and must never make a usable model fail
        # construction. The response metadata and main-message fallback still
        # cover providers whose model object rejects metadata assignment.
        logger.debug(
            "Could not attach configured provider metadata to %s",
            type(model).__name__,
            exc_info=True,
        )


def _resolve_pricing_provider(
    provider: object,
    fallback_provider: str,
    *,
    prefer_fallback_provider: bool = True,
) -> str:
    """Resolve response metadata without losing a configured provider alias.

    Args:
        provider: Provider named by the response, if any.
        fallback_provider: Provider configured for the completed request.
        prefer_fallback_provider: Whether a configured alias or non-API provider
            should replace response metadata. Disable this when the fallback
            belongs to a parent request rather than this specific model call.

    Returns:
        The provider identifier to use for pricing.
    """
    explicit_provider = provider if isinstance(provider, str) and provider else ""
    resolved_provider = explicit_provider or fallback_provider
    if explicit_provider and not prefer_fallback_provider:
        return explicit_provider
    fallback_provider_key = fallback_provider.strip().lower()
    if fallback_provider_key in _PROVIDER_ALIASES or (
        fallback_provider_key in _UNPRICEABLE_PROVIDERS
    ):
        return fallback_provider
    return resolved_provider


def _token_count(value: object) -> int:
    """Return a non-negative integer token count for a metadata value."""
    return (
        value
        if isinstance(value, int) and not isinstance(value, bool) and value > 0
        else 0
    )


def _cache_write_tokens(details: Mapping[str, Any]) -> int:
    """Return cache-write tokens from LangChain `input_token_details`.

    LangChain Anthropic zeroes the generic `cache_creation` field when the
    response includes a TTL breakdown (`ephemeral_5m_input_tokens` /
    `ephemeral_1h_input_tokens`). Sum those detailed fields when present so
    tokens are priced as cache writes rather than ordinary input. Fall back to
    `cache_creation` or the `cache_write` alias used by some other providers.
    `genai-prices` exposes a single cache-write rate, so 5-minute and 1-hour
    writes share that catalog price.
    """
    detailed = _token_count(details.get("ephemeral_5m_input_tokens")) + _token_count(
        details.get("ephemeral_1h_input_tokens")
    )
    if detailed:
        return detailed
    return _token_count(details.get("cache_creation") or details.get("cache_write"))


def estimate_cost(
    usage_metadata: Mapping[str, Any] | None,
    model_name: str,
    provider: str = "",
) -> float | None:
    """Estimate one model request's cost in USD from LangChain usage metadata.

    LangChain's `input_tokens` is the full input count, including cache reads and
    writes. `genai-prices` receives that inclusive total plus the two cache
    buckets; it subtracts the cache buckets before applying the ordinary input
    rate, then prices reads and writes separately so tokens are not double-counted.

    Args:
        usage_metadata: The request's LangChain `usage_metadata` mapping.
        model_name: Model identifier used for the request.
        provider: LangChain provider identifier. An empty value lets
            `genai-prices` infer the provider from `model_name`.

    Returns:
        Estimated cost in USD, or `None` when usage or pricing is unavailable.
    """
    model_ref = model_name.strip()
    provider_key = provider.strip().lower()
    if not usage_metadata or not model_ref:
        return None
    if provider_key in _UNPRICEABLE_PROVIDERS:
        logger.debug(
            "Cost estimate unavailable for non-API provider=%r model=%r",
            provider,
            model_ref,
        )
        return None

    input_tokens = _token_count(usage_metadata.get("input_tokens"))
    output_tokens = _token_count(usage_metadata.get("output_tokens"))
    if not input_tokens and not output_tokens:
        # `total_tokens` combines input and output, which normally have different
        # rates. Without the split there is no defensible estimate.
        return None

    details = usage_metadata.get("input_token_details")
    if isinstance(details, Mapping):
        cache_read_tokens = _token_count(details.get("cache_read"))
        cache_write_tokens = _cache_write_tokens(details)
    else:
        cache_read_tokens = 0
        cache_write_tokens = 0

    # Provider metadata can occasionally report cache parts that exceed the
    # inclusive input total. Clamp the parts so pricing still produces a safe
    # estimate instead of failing the model turn with negative uncached input.
    cache_read_tokens = min(cache_read_tokens, input_tokens)
    cache_write_tokens = min(cache_write_tokens, input_tokens - cache_read_tokens)

    provider_id = _PROVIDER_ALIASES.get(provider_key, provider_key) or None
    try:
        from genai_prices import Usage, calc_price

        price = calc_price(
            Usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_tokens=cache_read_tokens or None,
                cache_write_tokens=cache_write_tokens or None,
            ),
            model_ref=model_ref,
            provider_id=provider_id,
        )
        cost_usd = float(price.total_price)
    except Exception:
        logger.debug(
            "Cost estimate unavailable for model=%r provider=%r",
            model_ref,
            provider,
            exc_info=True,
        )
        return None

    return cost_usd if math.isfinite(cost_usd) and cost_usd >= 0 else None


def resolve_message_model(
    message: object,
    *,
    fallback_model: str = "",
    fallback_provider: str = "",
    prefer_fallback_provider: bool = True,
) -> tuple[str, str]:
    """Resolve the model and provider attached to a streamed model message.

    Args:
        message: An AI message or chunk with optional `response_metadata`.
        fallback_model: Model to use when message metadata does not name one.
        fallback_provider: Provider to use when message metadata does not name one.
            Known provider aliases and non-API providers override generic
            response metadata.
        prefer_fallback_provider: Whether those configured providers should
            replace response metadata. Disable this when the fallback describes
            a parent request rather than the message itself.

    Returns:
        The `(model_name, provider)` pair used for pricing.
    """
    metadata = getattr(message, "response_metadata", None)
    if not isinstance(metadata, Mapping):
        metadata = {}
    model_name = metadata.get("model_name") or metadata.get("model") or fallback_model
    resolved_model = model_name if isinstance(model_name, str) else fallback_model
    provider = metadata.get("model_provider") or metadata.get("provider")
    resolved_provider = _resolve_pricing_provider(
        provider,
        fallback_provider,
        prefer_fallback_provider=prefer_fallback_provider,
    )
    return resolved_model, resolved_provider


_MAX_TRACKED_THREADS = 64
"""Threads held in the recorder at once.

The middleware drains a thread on every model step, so a live thread holds at
most one step of records. Extra entries mean model calls nobody drained — a run
that died mid-turn, or a process (such as the client) that prices nothing. The
cap bounds that residue instead of letting it grow for the process lifetime.
"""

_MAX_RECORDS_PER_THREAD = 1_024
"""Undrained records kept for one thread before the oldest are dropped."""

_MAX_INFLIGHT_REQUESTS = 4_096
"""Started requests tracked at once, in case one neither ends nor errors."""


@dataclass(frozen=True, slots=True)
class _ModelCallRecord:
    """One completed model request awaiting pricing."""

    message_id: str | None
    """Response message ID, used to avoid charging a request twice."""

    usage_metadata: Mapping[str, Any]
    """The request's LangChain `usage_metadata`."""

    model_name: str
    """Model named by the response, or `""` when it named none."""

    provider: str
    """Provider named by the response, or `""` when it named none."""

    scope: str = ""
    """Checkpoint namespace of the graph that owns this request."""


@dataclass(frozen=True, slots=True)
class _ModelCallContext:
    """Request metadata retained until its model callback completes."""

    thread_id: str
    """Thread that owns the request's eventual cost."""

    configured_provider: str
    """Provider selected for this request, or `""` when unavailable."""

    scope: str
    """Checkpoint namespace of the graph that owns this request."""


def _parent_checkpoint_scope(namespace: object) -> str:
    """Return the graph namespace containing a checkpointed node.

    LangGraph appends the current node and task ID to the graph's checkpoint
    namespace. Removing that final segment gives every model and middleware
    node in one graph invocation the same scope, while parallel subagents keep
    the distinct tool-task prefixes LangGraph assigns them.
    """
    if not isinstance(namespace, str) or not namespace:
        return ""
    return namespace.rpartition("|")[0]


def _owning_checkpoint_scope(scope: str) -> str:
    """Return the parent graph that owns a completed nested transfer."""
    parts = scope.split("|") if scope else []
    while parts and parts[-1].isdigit():
        parts.pop()
    if parts:
        parts.pop()
    while parts and parts[-1].isdigit():
        parts.pop()
    return "|".join(parts)


class _SessionCostRecorder(BaseCallbackHandler):
    """Collect completed model requests per thread for the graph to price.

    Installed for every model request in the process, so it sees the agent model
    node, subagent graphs, and direct side invokes (offload/summarization, the
    Auto classifier) alike. It deliberately does no pricing: `estimate_cost`
    imports `genai-prices` on first use, and this handler runs inline on the
    event loop, which the server guards against blocking calls. The middleware
    prices drained records from a worker thread instead.
    """

    run_inline = True
    """Run in the calling context: dict bookkeeping only, no I/O to offload.

    Running inline also keeps `on_chat_model_start` in the invoking context, so
    the ambient config is available as a fallback source of the thread ID.
    """

    def __init__(self) -> None:
        """Initialize empty per-run and per-thread state."""
        self._lock = threading.Lock()
        self._run_contexts: OrderedDict[UUID, _ModelCallContext] = OrderedDict()
        self._records: OrderedDict[str, list[_ModelCallRecord]] = OrderedDict()

    def _start(self, run_id: UUID, metadata: Mapping[str, Any] | None) -> None:
        configurable = ensure_config().get("configurable") or {}
        thread_id = metadata.get("thread_id") if metadata is not None else None
        if not isinstance(thread_id, str) or not thread_id:
            # A caller that passes its own `metadata` can replace the ambient
            # metadata LangGraph populated, dropping `thread_id`. The ambient
            # config survives that merge, so read the thread from there.
            thread_id = configurable.get("thread_id")
        if not isinstance(thread_id, str) or not thread_id:
            # Without a thread there is nothing to attribute the cost to. The
            # middleware still prices the agent's own response from state.
            return
        provider = (
            metadata.get(_CONFIGURED_PROVIDER_METADATA_KEY)
            if metadata is not None
            else None
        )
        configured_provider = provider if isinstance(provider, str) and provider else ""
        namespace = (
            metadata.get(_CHECKPOINT_NAMESPACE_METADATA_KEY)
            if metadata is not None
            else None
        )
        if not isinstance(namespace, str) or not namespace:
            namespace = configurable.get("checkpoint_ns")
        with self._lock:
            self._run_contexts[run_id] = _ModelCallContext(
                thread_id=thread_id,
                configured_provider=configured_provider,
                scope=_parent_checkpoint_scope(namespace),
            )
            # A request that neither completes nor errors (a cancelled turn)
            # leaves its entry behind, so evict the oldest rather than growing.
            while len(self._run_contexts) > _MAX_INFLIGHT_REQUESTS:
                self._run_contexts.popitem(last=False)

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],  # noqa: ARG002  # Callback interface.
        messages: list[list[Any]],  # noqa: ARG002  # Callback interface.
        *,
        run_id: UUID,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,  # noqa: ARG002  # Callback interface.
    ) -> None:
        """Remember which thread a starting chat-model request belongs to."""
        self._start(run_id, metadata)

    def on_llm_start(
        self,
        serialized: dict[str, Any],  # noqa: ARG002  # Callback interface.
        prompts: list[str],  # noqa: ARG002  # Callback interface.
        *,
        run_id: UUID,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,  # noqa: ARG002  # Callback interface.
    ) -> None:
        """Remember the thread for a starting completion-model request."""
        self._start(run_id, metadata)

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        **kwargs: Any,  # noqa: ARG002  # Callback interface.
    ) -> None:
        """Record the completed request's usage for its thread."""
        with self._lock:
            context = self._run_contexts.pop(run_id, None)
        if context is None:
            return
        try:
            record = _record_from_response(
                response,
                configured_provider=context.configured_provider,
                scope=context.scope,
            )
        except Exception:
            logger.debug("Could not read usage from a model response", exc_info=True)
            return
        if record is None:
            return
        with self._lock:
            while context.thread_id not in self._records and (
                len(self._records) >= _MAX_TRACKED_THREADS
            ):
                self._records.popitem(last=False)
                logger.debug("Dropped undrained cost records for an inactive thread")
            records = self._records.setdefault(context.thread_id, [])
            self._records.move_to_end(context.thread_id)
            records.append(record)
            if len(records) > _MAX_RECORDS_PER_THREAD:
                del records[:-_MAX_RECORDS_PER_THREAD]
                logger.debug("Dropped the oldest undrained cost records for a thread")

    def on_llm_error(
        self,
        error: BaseException,  # noqa: ARG002  # Callback interface.
        *,
        run_id: UUID,
        **kwargs: Any,  # noqa: ARG002  # Callback interface.
    ) -> None:
        """Forget a failed request so its run entry cannot leak."""
        with self._lock:
            self._run_contexts.pop(run_id, None)

    def drain(
        self,
        thread_id: str,
        *,
        scope: str | None = None,
    ) -> list[_ModelCallRecord]:
        """Remove and return the records collected for one thread.

        Args:
            thread_id: Thread whose completed requests should be priced.
            scope: When provided, claim only requests from this graph invocation.

        Returns:
            Records for every request completed since the previous drain.
        """
        with self._lock:
            records = self._records.get(thread_id, [])
            if scope is None:
                return self._records.pop(thread_id, [])
            claimed = [record for record in records if record.scope == scope]
            remaining = [record for record in records if record.scope != scope]
            if remaining:
                self._records[thread_id] = remaining
            else:
                self._records.pop(thread_id, None)
            return claimed


def _record_from_response(
    response: LLMResult,
    *,
    configured_provider: str = "",
    scope: str = "",
) -> _ModelCallRecord | None:
    """Build a pricing record from a completed model response.

    Args:
        response: Completed LangChain model response containing usage metadata.
        configured_provider: Provider selected for this specific request.
        scope: Checkpoint namespace of the graph that made the request.

    Returns:
        The record, or `None` when the response carries no usage to price.
    """
    message: object | None = None
    usage_metadata: object = None
    for generations in response.generations:
        for generation in generations:
            candidate = getattr(generation, "message", None)
            candidate_usage = getattr(candidate, "usage_metadata", None)
            # A response with several candidates reports usage for the request as
            # a whole, on whichever generation the provider attached it to.
            if candidate is not None and candidate_usage:
                message = candidate
                usage_metadata = candidate_usage
    if message is None or not isinstance(usage_metadata, Mapping):
        return None
    model_name, provider = resolve_message_model(
        message,
        fallback_provider=configured_provider,
    )
    message_id = getattr(message, "id", None)
    return _ModelCallRecord(
        message_id=message_id if isinstance(message_id, str) and message_id else None,
        usage_metadata=dict(usage_metadata),
        model_name=model_name,
        provider=provider,
        scope=scope,
    )


_RECORDER = _SessionCostRecorder()
"""The process-wide recorder that every model request reports to."""

_RECORDER_VAR: ContextVar[_SessionCostRecorder | None] = ContextVar(
    "deepagents_code_session_cost_recorder",
    default=_RECORDER,
)
"""Context variable holding the recorder for LangChain's configure hooks.

The default value is the recorder itself, so no caller has to install anything:
`_configure` reads this variable while building every callback manager and
attaches the handler it finds. A context that sets the variable to `None` opts
out for the duration, which is how tests isolate the recorder.
"""


def _install_recorder() -> None:
    """Attach the recorder to every model request made in this process.

    `register_configure_hook` is the same mechanism LangSmith tracing uses. The
    hook is registered non-inheritable so the handler is attached freshly per
    request from the context variable rather than propagating through child runs
    as an inherited callback.
    """
    from langchain_core.tracers.context import register_configure_hook

    register_configure_hook(_RECORDER_VAR, inheritable=False)


_install_recorder()


def _drain_recorded_costs(
    thread_id: str | None,
    *,
    scope: str | None = None,
) -> list[_ModelCallRecord]:
    """Return the active recorder's pending records for a thread.

    Args:
        thread_id: Thread being priced, or `None` when the run has no thread.
        scope: When provided, claim only records owned by this graph invocation.

    Returns:
        Records to price, or an empty list when there is nothing to drain.
    """
    recorder = _RECORDER_VAR.get()
    if recorder is None or not thread_id:
        return []
    return recorder.drain(thread_id, scope=scope)


class _CostTransfer(TypedDict):
    """One completed nested total addressed to its owning parent graph."""

    owner_scope: str
    cost_usd: float


class CostState(ResumeState):
    """Agent state extended with the cumulative thread-cost channel."""

    _session_cost_usd: Annotated[NotRequired[float], PrivateStateAttr, operator.add]
    """Cumulative estimated USD cost for all priceable calls in this thread.

    Kept schema-private so cost is not part of public graph input/output, while
    still using an additive reducer so each drain contributes only the spend it
    priced and no write has to read-modify-write the running total.
    `operator.add` is last so LangGraph still detects the reducer.
    """

    _session_cost_transfers: Annotated[
        NotRequired[dict[str, _CostTransfer]],
        OmitFromInput,
        operator.or_,
    ]
    """Completed nested totals waiting for their owning parent graph.

    Each entry maps the completed graph's checkpoint scope to its parent scope
    and total. The map reducer lets parallel subagents hand off independently.
    `OmitFromInput` prevents a parent's pending entries from seeding a child,
    while leaving them in subagent output so the task tool can checkpoint the
    transfer on the owning parent graph.
    """


def _checkpointed_model_spec(state: CostState) -> tuple[str, str]:
    """Return the `(model, provider)` checkpointed for the completed request.

    Returns:
        The model name and provider from `_model_spec`, either of which may be
        `""` when the spec is absent or names only a model.
    """
    model_spec = state.get("_model_spec")
    if not isinstance(model_spec, str) or not model_spec:
        return "", ""
    provider, separator, model_name = model_spec.partition(":")
    if not separator:
        return provider, ""
    return model_name, provider


def _pricing_target(
    model_name: str,
    provider: str,
    fallback: tuple[str, str],
) -> tuple[str, str]:
    """Return the model and provider to price one request with.

    Args:
        model_name: Model the response named, or `""`.
        provider: Provider the response named, or `""`.
        fallback: The `(model, provider)` checkpointed for the request.

    Returns:
        The resolved pair, falling back to the checkpointed spec and then to the
            configured CLI model.

            Either value may still be `""`, which `estimate_cost` treats
            as unpriceable.
    """
    resolved_model = model_name or fallback[0]
    resolved_provider = provider or fallback[1]
    if not resolved_model:
        from deepagents_code.config import settings

        resolved_model = settings.model_name or ""
        resolved_provider = resolved_provider or settings.model_provider or ""
    return resolved_model, resolved_provider


def _thread_id(runtime: Runtime[ContextT]) -> str | None:
    """Return the thread whose recorded model calls this run should price.

    Returns:
        The execution's thread ID, or `None` when the run has no thread (an
        uncheckpointed invoke), in which case nothing is drained.
    """
    execution_info = getattr(runtime, "execution_info", None)
    thread_id = getattr(execution_info, "thread_id", None)
    return thread_id if isinstance(thread_id, str) and thread_id else None


def _checkpoint_scope(runtime: Runtime[ContextT]) -> str:
    """Return the checkpoint scope shared by this graph's middleware nodes."""
    execution_info = getattr(runtime, "execution_info", None)
    return _parent_checkpoint_scope(getattr(execution_info, "checkpoint_ns", None))


def _latest_ai_message(messages: Sequence[Any]) -> AIMessage | None:
    """Return the most recent AI message, if any.

    Returns:
        The last `AIMessage` in the sequence, or `None`.
    """
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return message
    return None


class CostTrackingMiddleware(AgentMiddleware[CostState, ContextT]):
    """Own the thread's cumulative `_session_cost_usd` checkpoint value.

    The main agent owns the thread total. Nested instances checkpoint local
    deltas before an interrupt can pause their graph, then transfer the completed
    subagent total through state for its owning parent graph to checkpoint.
    """

    state_schema = CostState

    def __init__(self, *, nested: bool = False) -> None:
        """Initialize cost tracking.

        Args:
            nested: When `True`, this instance belongs to a subagent. It
                checkpoints local spend and transfers the completed delta to the
                owning parent graph through state.
        """
        super().__init__()
        self._nested = nested

    def before_agent(  # ty: ignore[invalid-method-override]
        self,
        state: CostState,  # noqa: ARG002
        runtime: Runtime[ContextT],  # noqa: ARG002
    ) -> dict[str, Any] | None:
        """Zero the cost channel before a nested agent run.

        Returns:
            An overwrite of `_session_cost_usd` to `0.0` for a nested instance,
            otherwise `None`.
        """
        if not self._nested:
            return None
        return {"_session_cost_usd": Overwrite(0.0)}

    async def abefore_agent(  # ty: ignore[invalid-method-override]
        self,
        state: CostState,
        runtime: Runtime[ContextT],
    ) -> dict[str, Any] | None:
        """Async variant of `before_agent`.

        Returns:
            The same state update as `before_agent`.
        """
        return self.before_agent(state, runtime)

    def after_model(  # ty: ignore[invalid-method-override]
        self,
        state: CostState,
        runtime: Runtime[ContextT],
    ) -> dict[str, Any] | None:
        """Charge every model request completed since the previous checkpoint.

        Args:
            state: Current state containing messages and prior session cost.
            runtime: LangGraph runtime used to identify the thread and to stream
                the new total to the client.

        Returns:
            The additive cost delta, or `None` when there is nothing priceable
            to add. Returning `None` leaves the prior checkpoint value unchanged.
        """
        return self._charge(state, runtime, price_latest_message=True)

    def after_agent(  # ty: ignore[invalid-method-override]
        self,
        state: CostState,
        runtime: Runtime[ContextT],
    ) -> dict[str, Any] | None:
        """Charge model requests that completed after the last model step.

        Work outside the agent loop can spend once `after_model` has run for the
        final step: `ReliableRubricMiddleware.aafter_agent` runs a whole grading
        agent, and `after_agent` hooks run in reverse stack order, so this one
        drains after it. Anything that still spends later is charged on the next
        turn's first step rather than lost, but draining here keeps the turn's
        own checkpoint complete. A nested instance also transfers its cumulative
        local delta after the subagent completes.

        Args:
            state: Final state for the completed agent run.
            runtime: LangGraph runtime used to identify the thread and to stream
                the new total to the client.

        Returns:
            The additive cost delta and/or a nested transfer update, or `None`
            when nothing was left to charge or transfer.
        """
        update = self._charge(state, runtime, price_latest_message=False)
        if self._nested:
            prior_usd = state.get("_session_cost_usd")
            if not isinstance(prior_usd, int | float) or not math.isfinite(prior_usd):
                prior_usd = 0.0
            delta_usd = update.get("_session_cost_usd", 0.0) if update else 0.0
            total_usd = max(float(prior_usd), 0.0) + delta_usd
            scope = _checkpoint_scope(runtime)
            if scope and total_usd > 0:
                transfers = dict(state.get("_session_cost_transfers") or {})
                if update:
                    pending = update.get("_session_cost_transfers")
                    if isinstance(pending, Overwrite):
                        transfers = dict(pending.value)
                transfers[scope] = {
                    "owner_scope": _owning_checkpoint_scope(scope),
                    "cost_usd": total_usd,
                }
                if update is None:
                    update = {}
                update["_session_cost_transfers"] = Overwrite(transfers)
        return update

    def _charge(
        self,
        state: CostState,
        runtime: Runtime[ContextT],
        *,
        price_latest_message: bool,
    ) -> dict[str, Any] | None:
        """Price drained requests and return the additive delta.

        Args:
            state: Current agent state.
            runtime: LangGraph runtime for the running step.
            price_latest_message: Whether to also price the latest AI message
                when the recorder did not charge it. Only the model step that
                produced the message does this; `after_agent` would otherwise
                re-price a message an earlier drain already charged.

        Returns:
            Updates for the cost delta and claimed transfers, or `None` when
            neither changed.
        """
        thread_id = _thread_id(runtime)
        fallback = _checkpointed_model_spec(state)
        message = (
            _latest_ai_message(state.get("messages") or [])
            if price_latest_message
            else None
        )
        main_message_id = message.id if message is not None else None
        delta_usd = 0.0
        transfers = state.get("_session_cost_transfers") or {}
        remaining_transfers = dict(transfers)
        owner_scope = _checkpoint_scope(runtime)
        claimed_transfer = False
        for source_scope, transfer in transfers.items():
            if (
                isinstance(source_scope, str)
                and isinstance(transfer, Mapping)
                and transfer.get("owner_scope") == owner_scope
                and isinstance(transfer.get("cost_usd"), int | float)
                and math.isfinite(transfer["cost_usd"])
                and transfer["cost_usd"] > 0
            ):
                delta_usd += float(transfer["cost_usd"])
                remaining_transfers.pop(source_scope, None)
                claimed_transfer = True
        charged_message_ids: set[str] = set()
        charged_count = 0
        scope = _checkpoint_scope(runtime) if self._nested else None
        for record in _drain_recorded_costs(thread_id, scope=scope):
            # `_model_spec` describes the main response, while the recorder also
            # contains subagent and side-model calls. Correct generic provider
            # metadata only for the record that joins to that main response.
            # Every other record keeps the provider it reported.
            provider = record.provider
            if main_message_id is not None and record.message_id == main_message_id:
                provider = _resolve_pricing_provider(provider, fallback[1])
            cost_usd = estimate_cost(
                record.usage_metadata,
                *_pricing_target(record.model_name, provider, fallback),
            )
            if cost_usd is None:
                continue
            delta_usd += cost_usd
            charged_count += 1
            if record.message_id is not None:
                charged_message_ids.add(record.message_id)

        if price_latest_message:
            # A model that never fires callbacks (or a request the recorder
            # could not attribute to this thread) leaves the agent's own
            # response uncharged, so price it from state. Joining on message ID
            # keeps a request the recorder already charged from being added
            # twice; only successfully priced records are in the charged set.
            # An unidentified response cannot be joined, so treat any charged
            # request as covering it: undercounting one request beats charging
            # the same one twice.
            already_charged = (
                message.id in charged_message_ids
                if message is not None and message.id is not None
                else charged_count > 0
            )
            if message is not None and not already_charged:
                model_name, provider = resolve_message_model(
                    message,
                    fallback_model=fallback[0],
                    fallback_provider=fallback[1],
                )
                cost_usd = estimate_cost(
                    getattr(message, "usage_metadata", None),
                    *_pricing_target(model_name, provider, fallback),
                )
                if cost_usd is not None:
                    delta_usd += cost_usd

        if delta_usd <= 0 and not claimed_transfer:
            return None
        if not self._nested and delta_usd > 0:
            self._emit_total(state, runtime, delta_usd)
        update: dict[str, Any] = {}
        if claimed_transfer:
            update["_session_cost_transfers"] = Overwrite(remaining_transfers)
        if delta_usd > 0:
            update["_session_cost_usd"] = delta_usd
        return update

    @staticmethod
    def _emit_total(
        state: CostState,
        runtime: Runtime[ContextT],
        delta_usd: float,
    ) -> None:
        """Stream the thread's new absolute total for the live status bar.

        The channel is schema-private, so it does not reach the client on the
        state stream. Sending the absolute total (rather than the delta the
        checkpoint stores) lets the client set the displayed figure outright and
        stay correct even if it misses an event.

        Args:
            state: State carrying the total this delta applies to.
            runtime: LangGraph runtime providing the custom stream writer.
            delta_usd: Cost just charged to the channel.
        """
        writer = getattr(runtime, "stream_writer", None)
        if not callable(writer):
            return
        prior_usd = state.get("_session_cost_usd")
        if not isinstance(prior_usd, int | float) or not math.isfinite(prior_usd):
            prior_usd = 0.0
        try:
            writer(
                {
                    "type": SESSION_COST_EVENT_TYPE,
                    "total": max(float(prior_usd), 0.0) + delta_usd,
                }
            )
        except Exception:
            logger.debug("Could not emit the session cost event", exc_info=True)
