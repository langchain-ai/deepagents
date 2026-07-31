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
re-pricing anything. The payload is `{"type": ..., "total": <usd>, "thread_id":
<id>, "pricing_ok": <bool>}`; `total` is the full thread lifetime estimate,
never a delta, so a client that misses an event still converges on the next one.
`thread_id` lets a client that has since switched threads discard a total
belonging to the previous one. `pricing_ok` reports whether price data loaded in
the process that actually did the pricing, which is the only way a client can
tell a broken remote install from models with no published rates.
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


def _cache_write_counts(details: Mapping[str, Any]) -> tuple[int, int, int]:
    """Return generic-only, five-minute, and one-hour cache-write counts.

    LangChain Anthropic zeroes the generic `cache_creation` field when the
    response includes a TTL breakdown, so preserve that breakdown. Only the
    one-hour bucket earns anything by being split out: `genai-prices` publishes a
    premium one-hour rate, while five-minute writes have no rate of their own and
    fall back to the generic cache-write rate. Forwarding the five-minute count
    is therefore inert today and kept for when a rate appears.

    Falls back to `cache_creation`, then to a `cache_write` alias that no bundled
    LangChain integration currently emits, kept only as a defensive spelling.

    Returns:
        The generic-only, five-minute, and one-hour counts. A TTL breakdown wins
            outright, so the generic slot is mutually exclusive with the other
            two -- `estimate_cost` depends on that when deriving the aggregate
            it forwards.
    """
    five_minute = _token_count(details.get("ephemeral_5m_input_tokens"))
    one_hour = _token_count(details.get("ephemeral_1h_input_tokens"))
    if five_minute or one_hour:
        return 0, five_minute, one_hour
    generic = _token_count(details.get("cache_creation")) or _token_count(
        details.get("cache_write")
    )
    return generic, 0, 0


def _clamp_cache_counts(
    input_tokens: int,
    cache_read: int,
    cache_writes: tuple[int, int, int],
) -> tuple[int, tuple[int, int, int]]:
    """Clamp disjoint cache buckets to the inclusive input-token total.

    "Disjoint" is a precondition owed by the caller, not something checked here:
    `_cache_write_counts` guarantees the generic slot is mutually exclusive with
    the TTL slots, so draining all three from one shared budget cannot
    double-count. A non-exclusive tuple would over-consume the budget.

    Buckets drain in tuple order, so the one-hour bucket -- the only one with a
    premium rate -- is the first to be starved and the estimate is biased low.
    That is a deliberate tie-break, not an accident: clamping only fires when the
    provider's own numbers are self-inconsistent, so there is no way to know
    which bucket was misreported, and undercounting is the safer error.

    Args:
        input_tokens: Inclusive input-token count, covering reads and writes.
        cache_read: Cache-read token count.
        cache_writes: Generic-only, five-minute, and one-hour write counts.

    Returns:
        The clamped read count and write-count tuple.
    """
    clamped_read = min(cache_read, input_tokens)
    remaining = input_tokens - clamped_read
    clamped_writes: list[int] = []
    for count in cache_writes:
        clamped = min(count, remaining)
        clamped_writes.append(clamped)
        remaining -= clamped
    return clamped_read, (
        clamped_writes[0],
        clamped_writes[1],
        clamped_writes[2],
    )


def _clamped_detail(
    value: object,
    total: int,
    *,
    field: str,
    model_ref: str,
    provider: str,
) -> int:
    """Return a detail token count clamped to the total that contains it.

    `genai-prices` treats each detail bucket as a subset of its aggregate and
    rejects one that exceeds it, which would drop the whole request from the
    session estimate. Clamping keeps it priced. As with the cache buckets, this
    only ever fires on self-inconsistent provider numbers and it changes what the
    user is billed, so it is reported rather than applied silently.

    Args:
        value: Raw metadata value for the detail bucket.
        total: Inclusive total the bucket has to fit inside.
        field: Detail name, for the log message only.
        model_ref: Model being priced, for the log message only.
        provider: Provider being priced, for the log message only.

    Returns:
        The clamped count.
    """
    count = _token_count(value)
    if count <= total:
        return count
    logger.warning(
        "Detail token count exceeds the total containing it; clamping for "
        "pricing. model=%r provider=%r field=%s reported=%d clamped=%d",
        model_ref,
        provider,
        field,
        count,
        total,
    )
    return total


_PRICING_UNAVAILABLE = False
"""Whether importing `genai-prices` failed on the most recent attempt.

Cleared by a later success. Latching it would report a transient failure as a
broken install for the rest of the process, telling the user to reinstall while
costs compute normally.
"""

_PRICING_CONTRACT_BROKEN = False
"""Whether `Usage` rejected the normalized fields supplied by this package.

Construction only receives fixed keyword names and normalized non-negative
counts, so a failure there indicates an incompatible pricing API and affects
every request. Failures from `calc_price` itself can instead depend on the
request's decomposition and must not poison process-wide health. Cleared by a
later successful construction for the same reason `_PRICING_UNAVAILABLE` is.
"""

_AUDIO_CACHE_OVERLAP_REPORTED = False
"""Whether the audio/cache overlap understatement has been reported.

Not cleared: the condition is a property of the catalog and of what LangChain
reports, so it holds for the rest of the process once seen. Reporting it per
request would fire on every audio request for the models that price the
intersection, which is noise rather than news.
"""


def pricing_data_available() -> bool:
    """Report whether `genai-prices` is currently able to price a request.

    Note that `True` also covers "not yet attempted", so callers must only
    consult this after pricing has been tried -- otherwise a session that has
    priced nothing reads as healthy.

    Returns:
        `False` when the most recent import failed, or when the most recent
            construction of `Usage` rejected this package's normalized field
            schema. Either way callers can explain an empty cost figure as a
            broken pricing install rather than as models that have no published
            rates.
    """
    return not (_PRICING_UNAVAILABLE or _PRICING_CONTRACT_BROKEN)


def _load_pricing() -> tuple[Any, Any] | None:
    """Import `genai-prices` lazily, tracking whether it is currently loadable.

    A failed import means no request can ever be priced, which is a different
    problem from a model the catalog does not cover -- and one the user can act
    on. Keeping it separate from the usage-schema guard below stops a broken
    install from being reported as a pricing-coverage gap.

    The pair is deliberately re-imported rather than cached: `sys.modules` makes
    that cheap, and holding a reference would pin the first-seen `calc_price`,
    defeating any later patch of it.

    Returns:
        The `(Usage, calc_price)` pair, or `None` when the package is
            unavailable.
    """
    global _PRICING_UNAVAILABLE  # noqa: PLW0603
    try:
        from genai_prices import Usage, calc_price
    except Exception:
        if not _PRICING_UNAVAILABLE:
            # Once per process: this repeats on every request otherwise.
            logger.warning(
                "Could not load genai-prices; cost estimates are unavailable "
                "for this session.",
                exc_info=True,
            )
        _PRICING_UNAVAILABLE = True
        return None
    # A success clears the flag: the earlier failure was transient, and leaving
    # it set would keep telling the user to reinstall a working package.
    _PRICING_UNAVAILABLE = False
    return Usage, calc_price


def estimate_cost(
    usage_metadata: Mapping[str, Any] | None,
    model_name: str,
    provider: str = "",
) -> float | None:
    """Estimate one model request's cost in USD from LangChain usage metadata.

    LangChain's `input_tokens` is the full input count, including cache reads and
    writes. `genai-prices` receives that inclusive total plus the cache, modality,
    and reasoning details; it subtracts each detail bucket from the total that
    contains it before applying rates, so tokens are not double-counted. Only
    buckets the matched model actually publishes a rate for are broken out --
    forwarding one the catalog does not price leaves those tokens in the ordinary
    input or output total rather than pricing them separately.

    Args:
        usage_metadata: The request's LangChain `usage_metadata` mapping.
        model_name: Model identifier used for the request.
        provider: LangChain provider identifier. An empty value lets
            `genai-prices` infer the provider from `model_name`.

    Returns:
        Estimated cost in USD, or `None` when usage or pricing is unavailable.
    """
    global _AUDIO_CACHE_OVERLAP_REPORTED, _PRICING_CONTRACT_BROKEN  # noqa: PLW0603
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
        logger.debug(
            "Usage reports only a combined token total, which cannot be priced: "
            "model=%r provider=%r",
            model_ref,
            provider,
        )
        return None

    input_details = usage_metadata.get("input_token_details")
    if isinstance(input_details, Mapping):
        cache_read_tokens = _token_count(input_details.get("cache_read"))
        cache_writes = _cache_write_counts(input_details)
        input_audio_tokens = _clamped_detail(
            input_details.get("audio"),
            input_tokens,
            field="input audio",
            model_ref=model_ref,
            provider=provider,
        )
    else:
        cache_read_tokens = 0
        cache_writes = (0, 0, 0)
        input_audio_tokens = 0

    output_details = usage_metadata.get("output_token_details")
    if isinstance(output_details, Mapping):
        output_reasoning_tokens = _token_count(output_details.get("reasoning"))
        # Unlike other integrations, langchain-perplexity reports completion
        # tokens as the output total and exposes reasoning as an extra bucket.
        # `genai-prices` expects the total to include every detailed bucket.
        if provider_key == "perplexity":
            output_tokens += output_reasoning_tokens
        # Clamped against the output total independently, not against each
        # other's remainder: audio and reasoning tokens overlap rather than
        # partition the output, so their sum legitimately can exceed it.
        output_audio_tokens = _clamped_detail(
            output_details.get("audio"),
            output_tokens,
            field="output audio",
            model_ref=model_ref,
            provider=provider,
        )
        output_reasoning_tokens = _clamped_detail(
            output_reasoning_tokens,
            output_tokens,
            field="output reasoning",
            model_ref=model_ref,
            provider=provider,
        )
    else:
        output_audio_tokens = 0
        output_reasoning_tokens = 0

    # Provider metadata can occasionally report cache parts that exceed the
    # inclusive input total. `calc_price` rejects a negative uncached input, so
    # without clamping the whole request would be dropped from the total rather
    # than estimated. Clamping only ever fires on self-inconsistent provider
    # numbers, so say so -- a silent clamp can materially undercount.
    original_cache_read = cache_read_tokens
    original_cache_writes = cache_writes
    cache_read_tokens, cache_writes = _clamp_cache_counts(
        input_tokens, cache_read_tokens, cache_writes
    )
    if (
        cache_read_tokens != original_cache_read
        or cache_writes != original_cache_writes
    ):
        logger.warning(
            "Cache token counts exceed the inclusive input total; clamping for "
            "pricing. model=%r provider=%r input=%d cache_read=%d->%d "
            "cache_write generic=%d->%d 5m=%d->%d 1h=%d->%d",
            model_ref,
            provider,
            input_tokens,
            original_cache_read,
            cache_read_tokens,
            # Reported alongside the clamped values, and per bucket rather than
            # summed: which bucket lost tokens decides how much cost was forgone,
            # since only the one-hour rate carries a premium.
            *(
                count
                for pair in zip(original_cache_writes, cache_writes, strict=True)
                for count in pair
            ),
        )
    generic_cache_write_tokens, cache_write_5m_tokens, cache_write_1h_tokens = (
        cache_writes
    )
    # `genai-prices` treats the TTL buckets as descendants of the generic one and
    # refuses a TTL count without its ancestor ("reported descendant usage keys
    # ... require explicit cache_write_tokens"), so this aggregate is required
    # rather than redundant. The `or` is safe only because `_cache_write_counts`
    # returns the generic slot as mutually exclusive with the TTL slots; were
    # both ever populated, this would forward the generic count alone and drop
    # the TTL tokens.
    cache_write_tokens = generic_cache_write_tokens or (
        cache_write_5m_tokens + cache_write_1h_tokens
    )

    # LangChain does not expose the intersection of audio and cached tokens. For
    # a model whose catalog entry prices an audio/cache-read or audio/cache-write
    # intersection, reporting both aggregates makes `genai-prices` demand the
    # missing intersection count and reject the request, dropping it from the
    # total entirely. Forgoing the audio split keeps the request priced, at the
    # cost of billing those tokens at the ordinary input rate.
    if input_audio_tokens and (cache_read_tokens or any(cache_writes)):
        if not _AUDIO_CACHE_OVERLAP_REPORTED:
            # Once per process: for a model that prices the intersection this is
            # structural rather than anomalous, so it would otherwise repeat on
            # every request and train the reader to ignore the log.
            _AUDIO_CACHE_OVERLAP_REPORTED = True
            logger.warning(
                "Pricing audio input at the ordinary input rate because the "
                "catalog wants an audio/cache intersection, which "
                "LangChain does not report. This understates requests that mix "
                "audio with prompt caching. model=%r provider=%r audio=%d",
                model_ref,
                provider,
                input_audio_tokens,
            )
        input_audio_tokens = 0

    pricing = _load_pricing()
    if pricing is None:
        return None
    usage_type, calc_price = pricing

    provider_id = _PROVIDER_ALIASES.get(provider_key, provider_key) or None
    try:
        usage = usage_type(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens or None,
            cache_write_tokens=cache_write_tokens or None,
            cache_write_5m_tokens=cache_write_5m_tokens or None,
            cache_write_1h_tokens=cache_write_1h_tokens or None,
            input_audio_tokens=input_audio_tokens or None,
            output_audio_tokens=output_audio_tokens or None,
            output_reasoning_tokens=output_reasoning_tokens or None,
        )
    except Exception:
        # These fixed fields and normalized counts are valid for every request.
        # Their rejection therefore proves an installed API incompatibility,
        # unlike request-specific decomposition failures from `calc_price`.
        if not _PRICING_CONTRACT_BROKEN:
            logger.warning(
                "genai-prices rejected the usage schema, so no request can be "
                "priced; cost estimates are unavailable for this session. "
                "model=%r provider=%r",
                model_ref,
                provider,
                exc_info=True,
            )
        _PRICING_CONTRACT_BROKEN = True
        return None
    _PRICING_CONTRACT_BROKEN = False

    try:
        price = calc_price(
            usage,
            model_ref=model_ref,
            provider_id=provider_id,
        )
        cost_usd = float(price.total_price)
    except LookupError:
        # The catalog publishes no rates for this model or provider. Ordinary,
        # not actionable, and specifically not a broken install -- so it must
        # leave `_PRICING_CONTRACT_BROKEN` alone.
        logger.debug(
            "Cost estimate unavailable for model=%r provider=%r",
            model_ref,
            provider,
            exc_info=True,
        )
        return None
    except Exception:
        # Decomposition failures can depend on this request's overlapping detail
        # totals. Other ordinary requests may still price successfully, so leave
        # process-wide health alone.
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
            # Register a thread-less marker anyway: `on_llm_end` otherwise
            # cannot tell this ordinary case from an entry the in-flight cap
            # evicted, which is the one that really does lose spend.
            with self._lock:
                self._run_contexts[run_id] = _ModelCallContext(
                    thread_id="",
                    configured_provider="",
                    scope="",
                )
                while len(self._run_contexts) > _MAX_INFLIGHT_REQUESTS:
                    self._run_contexts.popitem(last=False)
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
            # Every start registers a context, including thread-less ones, so
            # a missing entry means `_MAX_INFLIGHT_REQUESTS` evicted it -- the
            # request outlived that many newer starts -- and real spend is
            # being dropped. Distinct from the thread-less case below, which is
            # ordinary and costs nothing.
            logger.warning(
                "No start context for a completed request; it outlived %d newer "
                "requests and its cost is dropped from the session total.",
                _MAX_INFLIGHT_REQUESTS,
            )
            return
        if not context.thread_id:
            # Ordinary: an uncheckpointed invoke has no thread to attribute to.
            # The middleware still prices the agent's own response from state.
            logger.debug(
                "Completed request has no thread to attribute its cost to; the "
                "middleware prices the main response from state instead."
            )
            return
        try:
            record = _record_from_response(
                response,
                configured_provider=context.configured_provider,
                scope=context.scope,
            )
        except Exception:
            # This is the sole entry point for every priced request, so a
            # systematic failure here silently zeroes the whole session.
            logger.warning(
                "Could not read usage from a model response; its cost is "
                "dropped from the session total.",
                exc_info=True,
            )
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
                dropped = len(records) - _MAX_RECORDS_PER_THREAD
                del records[:-_MAX_RECORDS_PER_THREAD]
                # In a pricing process the middleware drains every model step,
                # so reaching this cap means spend is being permanently lost.
                logger.warning(
                    "Dropped %d undrained cost record(s) for an active thread; "
                    "their cost is missing from the session total.",
                    dropped,
                )

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

    def restore(self, thread_id: str, records: list[_ModelCallRecord]) -> None:
        """Return drained records to the recorder after a failed pricing pass.

        `drain` removes what it returns, so a caller that fails partway through
        pricing would otherwise destroy real spend. Restoring puts the records
        back ahead of anything recorded since, preserving completion order so
        the retry prices them in the order they arrived.

        Args:
            thread_id: Thread the records were drained from.
            records: Records to put back. An empty list is a no-op.
        """
        if not records:
            return
        with self._lock:
            existing = self._records.get(thread_id)
            if existing is None:
                while len(self._records) >= _MAX_TRACKED_THREADS:
                    self._records.popitem(last=False)
                    logger.debug(
                        "Dropped undrained cost records for an inactive thread"
                    )
                existing = []
                self._records[thread_id] = existing
            existing[:0] = records
            if len(existing) > _MAX_RECORDS_PER_THREAD:
                dropped = len(existing) - _MAX_RECORDS_PER_THREAD
                del existing[:dropped]
                logger.warning(
                    "Dropped %d restored cost record(s) to keep the active "
                    "thread bounded; their cost is missing from the session total.",
                    dropped,
                )
            self._records.move_to_end(thread_id)


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


def _restore_recorded_costs(
    thread_id: str | None,
    records: list[_ModelCallRecord],
) -> bool:
    """Put drained records back after pricing failed.

    Args:
        thread_id: Thread the records were drained from.
        records: Records to return to the recorder.

    Returns:
        `True` when the records were handed back, `False` when there is no
            recorder to hand them to and the spend is therefore lost.
    """
    if not records:
        return True
    recorder = _RECORDER_VAR.get()
    if recorder is None or not thread_id:
        return False
    recorder.restore(thread_id, records)
    return True


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
        # Each middleware hook is its own graph node, so an exception here would
        # fail the user's turn. Cost tracking is never worth that. `_charge`
        # returns its drained records to the recorder before propagating, so
        # skipping this step's charge defers the spend rather than losing it.
        try:
            return self._charge(state, runtime, price_latest_message=True)
        except Exception:
            logger.warning("Cost tracking failed to charge a model step", exc_info=True)
            return None

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
        # See `after_model`: a raise here would fail the turn, and `_charge`
        # has already handed its records back. A nested instance that cannot
        # transfer loses only its own subagent total.
        try:
            return self._after_agent_update(state, runtime)
        except Exception:
            logger.warning(
                "Cost tracking failed to charge the completed agent run",
                exc_info=True,
            )
            return None

    def _after_agent_update(
        self,
        state: CostState,
        runtime: Runtime[ContextT],
    ) -> dict[str, Any] | None:
        """Drain late records and, for a nested run, stage the parent transfer.

        Args:
            state: Final state for the completed agent run.
            runtime: LangGraph runtime used to identify the thread and scope.

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
        pricing_attempted = False
        scope = _checkpoint_scope(runtime) if self._nested else None
        # `drain` removes what it returns, so anything that raises below would
        # destroy real spend. Hold the records and hand them back on failure so
        # the next drain genuinely can re-price them.
        drained = _drain_recorded_costs(thread_id, scope=scope)
        try:
            for record in drained:
                # `_model_spec` describes the main response, while the recorder
                # also contains subagent and side-model calls. Correct generic
                # provider metadata only for the record that joins to that main
                # response. Every other record keeps the provider it reported.
                provider = record.provider
                if main_message_id is not None and record.message_id == main_message_id:
                    provider = _resolve_pricing_provider(provider, fallback[1])
                pricing_attempted = True
                cost_usd = estimate_cost(
                    record.usage_metadata,
                    *_pricing_target(record.model_name, provider, fallback),
                )
                if cost_usd is None:
                    # Silently omitting this leaves the total quietly short, so
                    # leave a breadcrumb naming what could not be priced.
                    logger.debug(
                        "Dropping an unpriceable request from the session total: "
                        "model=%r provider=%r",
                        record.model_name,
                        provider,
                    )
                    continue
                delta_usd += cost_usd
                charged_count += 1
                if record.message_id is not None:
                    charged_message_ids.add(record.message_id)
            if price_latest_message:
                # A model that never fires callbacks (or a request the recorder
                # could not attribute to this thread) leaves the agent's own
                # response uncharged, so price it from state. Joining on message
                # ID keeps a request the recorder already charged from being
                # added twice; only successfully priced records are in the
                # charged set. An unidentified response cannot be joined, so
                # treat any charged request as covering it: undercounting one
                # request beats charging the same one twice.
                already_charged = (
                    message.id in charged_message_ids
                    if message is not None and message.id is not None
                    else charged_count > 0
                )
                if (
                    message is not None
                    and message.id is None
                    and already_charged
                    and logger.isEnabledFor(logging.DEBUG)
                ):
                    logger.debug(
                        "Not pricing an unidentified main response from state; a "
                        "drained record may already cover it."
                    )
                if message is not None and not already_charged:
                    model_name, provider = resolve_message_model(
                        message,
                        fallback_model=fallback[0],
                        fallback_provider=fallback[1],
                    )
                    pricing_attempted = True
                    cost_usd = estimate_cost(
                        getattr(message, "usage_metadata", None),
                        *_pricing_target(model_name, provider, fallback),
                    )
                    if cost_usd is not None:
                        delta_usd += cost_usd

            if not self._nested and (delta_usd > 0 or pricing_attempted):
                pricing_ok = pricing_data_available()
                if delta_usd > 0 or not pricing_ok:
                    self._emit_total(
                        state,
                        runtime,
                        delta_usd,
                        pricing_ok=pricing_ok,
                    )
            if delta_usd <= 0 and not claimed_transfer:
                return None
            update: dict[str, Any] = {}
            if claimed_transfer:
                update["_session_cost_transfers"] = Overwrite(remaining_transfers)
            if delta_usd > 0:
                update["_session_cost_usd"] = delta_usd
        except BaseException:
            # `BaseException`, not `Exception`: a cancelled turn raises
            # `CancelledError`, which would otherwise discard the records
            # silently -- the hook wrappers below do not catch it either.
            # Everything from the drain to the return is covered, because a
            # failure anywhere in it means the delta never reaches the
            # checkpoint, so the records must survive to be re-priced.
            if _restore_recorded_costs(thread_id, drained):
                logger.warning(
                    "Pricing a drained batch failed; returned %d record(s) to "
                    "the recorder to be re-priced on the next drain.",
                    len(drained),
                    exc_info=True,
                )
            else:
                logger.warning(
                    "Pricing a drained batch failed and there is no recorder to "
                    "return %d record(s) to; their cost is lost from the "
                    "session total.",
                    len(drained),
                    exc_info=True,
                )
            raise
        return update

    @staticmethod
    def _emit_total(
        state: CostState,
        runtime: Runtime[ContextT],
        delta_usd: float,
        *,
        pricing_ok: bool,
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
            pricing_ok: Whether price data loaded in the pricing process.
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
                    "thread_id": _thread_id(runtime) or "",
                    # Pricing runs here, which in a remote deployment is not the
                    # client's process. Without this the client can only inspect
                    # its own install and would blame the user's model choice
                    # for a broken package on the server.
                    "pricing_ok": pricing_ok,
                }
            )
        except Exception:
            logger.debug("Could not emit the session cost event", exc_info=True)
