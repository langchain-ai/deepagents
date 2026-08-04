"""Lightweight session statistics, token formatting, and usage-table rendering.

Holds `SessionStats`/`ModelStats`, the `format_token_count` formatter, and
`print_usage_table` (which imports `rich.table` lazily). The module is
intentionally kept free of heavy top-level dependencies (no pydantic, no
config, no widget imports) so that `app.py` can import `SessionStats` and
`format_token_count` at module level without pulling in the full
`textual_adapter` dependency tree.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal, cast

from deepagents_code.formatting import format_duration

if TYPE_CHECKING:
    from rich.console import Console

logger = logging.getLogger(__name__)

SpinnerStatus = (
    Literal[
        "Thinking",
        "Offloading",
        "Loading thread",
        "Drafting acceptance criteria",
    ]
    | None
)
"""Valid spinner display states, or `None` to hide."""

UsageKind = Literal["assistant", "subagent", "offload", "auto"]
"""Billing/display class for a model request."""

USAGE_KIND_ORDER: tuple[UsageKind, ...] = (
    "assistant",
    "subagent",
    "offload",
    "auto",
)
"""Stable display order for per-type cost breakdowns."""

USAGE_KIND_LABELS: dict[UsageKind, str] = {
    "assistant": "Assistant",
    "subagent": "Subagents",
    "offload": "Offload",
    "auto": "Auto mode",
}
"""User-facing labels for `UsageKind` values."""


def classify_usage_kind(
    *,
    is_main_agent: bool,
    metadata: Mapping[str, Any] | None = None,
) -> UsageKind:
    """Classify a streamed model request for cost and usage breakdowns.

    Args:
        is_main_agent: Whether the stream namespace is the top-level agent.
        metadata: LangChain callback/stream metadata for the chunk.

    Returns:
        The usage kind used in `/cost` and session stats.
    """
    if not is_main_agent:
        return "subagent"
    source = metadata.get("lc_source") if metadata is not None else None
    if source == "summarization":
        return "offload"
    if source == "auto_mode_classifier":
        return "auto"
    return "assistant"


@dataclass
class ModelStats:
    """Token stats for a single model within a session."""

    request_count: int = 0
    """Number of LLM API requests made to this model."""

    input_tokens: int = 0
    """Cumulative input tokens sent to this model."""

    output_tokens: int = 0
    """Cumulative output tokens received from this model."""

    cost_usd: float = 0.0
    """Cumulative estimated USD cost for priceable requests to this model."""

    priced_request_count: int = 0
    """Requests with a cost estimate, including estimates of literal zero."""

    provider: str = ""
    """Provider that served this model (e.g. `openai`), or `""` when unknown."""

    model_name: str = ""
    """Model name displayed in usage output."""


@dataclass
class KindStats:
    """Token and cost stats for one `UsageKind` bucket."""

    request_count: int = 0
    """Number of LLM API requests in this bucket."""

    input_tokens: int = 0
    """Cumulative input tokens in this bucket."""

    output_tokens: int = 0
    """Cumulative output tokens in this bucket."""

    cost_usd: float = 0.0
    """Cumulative estimated USD cost for priceable requests in this bucket."""

    priced_request_count: int = 0
    """Requests with a cost estimate, including estimates of literal zero."""


@dataclass(frozen=True, slots=True)
class RecordedUsage:
    """Usage returned after recording one streamed model message."""

    input_tokens: int
    """Input-token delta contributed by this message."""

    output_tokens: int
    """Output-token delta contributed by this message."""

    cost_usd: float | None
    """Cost contributed by this message, or `None` when pricing was unavailable.

    Negative when the message re-prices its request downward -- a corrected
    prompt count, or a model that turned out to cost less than the fallback the
    earlier chunks were priced against.
    """

    request_tokens: int
    """Running token total for the request after applying this message."""


ModelStatsKey = tuple[str, str]
"""Per-model dict key: the `(provider, model_name)` pair.

Pairing the provider with the model name keeps the same model served by
different providers (e.g. `gpt-5.5` via `openai` vs `azure`) in separate rows
instead of collapsing them. The key is always built from the same values stored
on the corresponding `ModelStats`, so key and fields never diverge.
"""


@dataclass
class SessionStats:
    """Stats accumulated over a single agent turn (or full session)."""

    request_count: int = 0
    """Total LLM API requests made.

    One completed API request counts once, however many stream chunks carried
    its usage: `record_message_usage` revises a request in place rather than
    recording each chunk separately.
    """

    input_tokens: int = 0
    """Cumulative input tokens across all LLM requests."""

    output_tokens: int = 0
    """Cumulative output tokens across all LLM requests."""

    total_cost_usd: float = 0.0
    """Cumulative estimated USD cost across priceable LLM requests."""

    priced_request_count: int = 0
    """Requests with a cost estimate, including estimates of literal zero."""

    wall_time_seconds: float = 0.0
    """Wall-clock duration from stream start to end."""

    per_model: dict[ModelStatsKey, ModelStats] = field(default_factory=dict)
    """Per-model breakdown keyed by `(provider, model_name)`.

    Populated only when `record_request` receives a non-empty `model_name`. Empty
    dict means no named-model requests were recorded; `print_usage_table` omits
    the model table in that case and shows only the wall-time line (if applicable).
    """

    per_kind: dict[UsageKind, KindStats] = field(default_factory=dict)
    """Per-type breakdown for assistant, nested, and hidden model spend."""

    def record_request(
        self,
        model_name: str,
        input_toks: int,
        output_toks: int,
        provider: str = "",
        *,
        cost_usd: float | None = None,
        kind: UsageKind = "assistant",
    ) -> None:
        """Accumulate usage for one completed LLM request.

        Updates session totals plus the per-model and per-type breakdowns.

        Args:
            model_name: The model that served this request.

                Combined with `provider` to form the per-model key. Pass
                an empty string to skip the per-model breakdown for this request.
            input_toks: Input tokens for this request.
            output_toks: Output tokens for this request.
            provider: Provider that served the model (e.g. `openai`).

                Combined with `model_name` to form the per-model key, so
                the same model served by different providers is
                tracked separately.
            cost_usd: Estimated request cost, or `None` when no estimate exists.

                Missing estimates leave monetary totals unchanged.
            kind: Request class used for `/cost` type breakdowns.
        """
        self.request_count += 1
        self.input_tokens += input_toks
        self.output_tokens += output_toks
        if cost_usd is not None:
            self.total_cost_usd += cost_usd
            self.priced_request_count += 1
        kind_entry = self.per_kind.setdefault(kind, KindStats())
        kind_entry.request_count += 1
        kind_entry.input_tokens += input_toks
        kind_entry.output_tokens += output_toks
        if cost_usd is not None:
            kind_entry.cost_usd += cost_usd
            kind_entry.priced_request_count += 1
        if model_name:
            key = (provider, model_name)
            entry = self.per_model.setdefault(
                key,
                ModelStats(provider=provider, model_name=model_name),
            )
            entry.request_count += 1
            entry.input_tokens += input_toks
            entry.output_tokens += output_toks
            if cost_usd is not None:
                entry.cost_usd += cost_usd
                entry.priced_request_count += 1

    def retract_request(self, recorded: RecordedRequest) -> None:
        """Reverse the `record_request` call that produced *recorded*.

        A request whose usage arrives across several stream chunks is recorded
        as soon as the first chunk lands, so the display can react, and then
        re-recorded with its running totals as later chunks arrive. Retracting
        the previous version first keeps that one API call counted once, with
        one per-model row, instead of once per chunk.

        Takes the ledger entry rather than loose values so the retraction cannot
        drift from what was recorded: a mismatch would desync the session totals
        from the per-kind and per-model breakdowns silently.

        Args:
            recorded: Ledger entry describing the contribution to reverse.
        """
        model_name = recorded.model_name
        provider = recorded.provider
        input_toks = recorded.input_tokens
        output_toks = recorded.output_tokens
        cost_usd = recorded.cost_usd
        kind = recorded.kind

        self.request_count -= 1
        self.input_tokens -= input_toks
        self.output_tokens -= output_toks
        if cost_usd is not None:
            self.total_cost_usd -= cost_usd
            self.priced_request_count -= 1
        kind_entry = self.per_kind.get(kind)
        if kind_entry is not None:
            kind_entry.request_count -= 1
            kind_entry.input_tokens -= input_toks
            kind_entry.output_tokens -= output_toks
            if cost_usd is not None:
                kind_entry.cost_usd -= cost_usd
                kind_entry.priced_request_count -= 1
            if kind_entry.request_count <= 0:
                # Mirrors the per-model eviction below: a kind whose only
                # request moved elsewhere would otherwise leave an all-zero row
                # in the `/cost` type breakdown.
                del self.per_kind[kind]
        if model_name:
            entry = self.per_model.get((provider, model_name))
            if entry is not None:
                entry.request_count -= 1
                entry.input_tokens -= input_toks
                entry.output_tokens -= output_toks
                if cost_usd is not None:
                    entry.cost_usd -= cost_usd
                    entry.priced_request_count -= 1
                if entry.request_count <= 0:
                    # The chunk-revision path can move a request to a different
                    # model once the final chunk names one; drop the row it
                    # vacated so the breakdown does not show an empty entry.
                    del self.per_model[provider, model_name]

    def merge(self, other: SessionStats) -> None:
        """Merge another `SessionStats` into this one (mutates *self*).

        Used to accumulate per-turn stats into a session-level total.

        Args:
            other: The stats to fold in.
        """
        self.request_count += other.request_count
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.total_cost_usd += other.total_cost_usd
        self.priced_request_count += other.priced_request_count
        self.wall_time_seconds += other.wall_time_seconds
        for key, ms in other.per_model.items():
            entry = self.per_model.setdefault(
                key,
                ModelStats(provider=ms.provider, model_name=ms.model_name),
            )
            entry.request_count += ms.request_count
            entry.input_tokens += ms.input_tokens
            entry.output_tokens += ms.output_tokens
            entry.cost_usd += ms.cost_usd
            entry.priced_request_count += ms.priced_request_count
        for kind, kind_stats in other.per_kind.items():
            entry = self.per_kind.setdefault(kind, KindStats())
            entry.request_count += kind_stats.request_count
            entry.input_tokens += kind_stats.input_tokens
            entry.output_tokens += kind_stats.output_tokens
            entry.cost_usd += kind_stats.cost_usd
            entry.priced_request_count += kind_stats.priced_request_count


@dataclass(frozen=True, slots=True)
class RecordedRequest:
    """What a stream consumer last recorded for one request.

    Held so a later chunk of the same request can retract that exact
    contribution and re-record the running totals, keeping one API call to one
    request and one per-model row.
    """

    model_name: str
    """Model the contribution was recorded under."""

    provider: str
    """Provider the contribution was recorded under."""

    kind: UsageKind
    """Type bucket the contribution was recorded under."""

    input_tokens: int
    """Running input tokens recorded so far for the request."""

    output_tokens: int
    """Running output tokens recorded so far for the request."""

    cost_usd: float | None
    """Estimate for the whole request so far, or `None` when unpriceable."""

    usage_metadata: Mapping[str, Any]
    """Merged usage for the request so far.

    Kept so the request can be re-priced as a whole once a later chunk reveals
    the model. Summing per-chunk estimates would freeze the rates that applied
    when the model was still unknown.
    """

    finalized: bool
    """Whether the request's usage is complete and can no longer be added to.

    Set either by a completed (non-chunk) message, which carries the request's
    whole usage, or by `finalize_recorded_requests` at the end of a stream
    round. A later chunk for a finalized request is a replay, not a revision.
    """


def finalize_recorded_requests(
    recorded_requests: dict[str, RecordedRequest],
) -> None:
    """Close every request in a ledger that outlives its stream round.

    A chunked request is left open between chunks so later ones can revise it.
    That is only correct *within* one round: when a consumer reuses its ledger
    across HITL resume passes, the replayed chunks of an already-recorded
    request would otherwise merge a second time and double its tokens and cost.
    Closing the ledger at each round boundary makes the replay indistinguishable
    from the stray-chunk case `record_message_usage` already rejects.

    Args:
        recorded_requests: Ledger to close. Mutated in place.
    """
    for request_id, recorded in recorded_requests.items():
        if not recorded.finalized:
            recorded_requests[request_id] = replace(recorded, finalized=True)


def _names_a_model(message: object) -> bool:
    """Report whether a message's own metadata names the model that served it.

    Google attaches `model_name` only to the chunk carrying `finish_reason`, so
    a chunk that does not name one must keep the model already recorded rather
    than reverting the request to the caller's fallback.

    Args:
        message: Streamed model message or chunk.

    Returns:
        `True` when response metadata names a model.
    """
    metadata = getattr(message, "response_metadata", None)
    if not isinstance(metadata, Mapping):
        return False
    return bool(metadata.get("model_name") or metadata.get("model"))


def _positive_int(value: object) -> int:
    """Return a token count as a non-negative int, or `0` when unusable."""
    return (
        value
        if isinstance(value, int) and not isinstance(value, bool) and value > 0
        else 0
    )


def _carries_token_counts(usage: Mapping[str, Any]) -> bool:
    """Report whether usage states any token count, including a negative one.

    Google's final chunk can report a *negative* `input_tokens` to correct an
    over-counted prompt, which normalizes to zero but is still a real revision
    of the request.

    Args:
        usage: A message's `usage_metadata`.

    Returns:
        `True` when any top-level token field holds a non-zero integer.
    """
    return any(
        isinstance(value, int) and not isinstance(value, bool) and value != 0
        for value in (
            usage.get("input_tokens"),
            usage.get("output_tokens"),
            usage.get("total_tokens"),
        )
    )


def _display_token_counts(usage: Mapping[str, Any]) -> tuple[int, int]:
    """Return the `(input, output)` token counts to display for some usage.

    Args:
        usage: A message's `usage_metadata`, or a request's merged usage.

    Returns:
        Non-negative input and output counts. When neither is reported but a
            total is, the total is attributed to input so the request is not
            shown as having used nothing.
    """
    input_count = _positive_int(usage.get("input_tokens"))
    output_count = _positive_int(usage.get("output_tokens"))
    if not input_count and not output_count:
        return _positive_int(usage.get("total_tokens")), 0
    return input_count, output_count


def _merge_usage(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Combine two usage metadata mappings for the same request.

    Uses LangChain's own recursive adder so nested token details (cache reads,
    cache writes, reasoning tokens) are summed rather than dropped -- those
    buckets carry their own rates, so losing them would misprice the request.

    Summing assumes each chunk reports an *incremental* usage delta. That is
    LangChain's contract, not just ours: `AIMessageChunk.__add__` adds the usage
    of chunks sharing an ID, so a provider that repeated cumulative totals per
    chunk would already double-count in the framework's own aggregation.

    Args:
        left: Usage accumulated for the request so far.
        right: Usage reported by the message being folded in.

    Returns:
        The combined usage, or `left` unchanged if the two cannot be added.
    """
    try:
        # Imported inside the guard: a langchain-core release that renames or
        # moves `add_usage` would otherwise raise straight into the stream loop,
        # which does not wrap this call.
        from langchain_core.messages.ai import add_usage

        return dict(add_usage(cast("Any", dict(left)), cast("Any", dict(right))))
    except Exception:
        # Malformed provider usage must not break accounting for the request.
        # `left` is returned unchanged, so this chunk's usage is dropped rather
        # than mis-merged -- log it, because the resulting total is short.
        logger.warning(
            "Could not merge streamed usage metadata; this chunk's tokens and "
            "cost are missing from the request.",
            exc_info=True,
        )
        return left


def _cost_delta(
    previous_cost_usd: float | None,
    cost_usd: float | None,
    *,
    model_name: str,
    previous_model_name: str,
) -> float | None:
    """Return the provisional-display delta between two estimates of a request.

    Re-pricing can move a request from priceable to unpriceable, when the model
    it finally names has no published rates but the caller's fallback did. The
    accumulator drops the old estimate, so reporting `None` would leave the
    caller's provisional display holding a cost nothing backs any more. Report
    the retraction as a negative delta instead.

    Args:
        previous_cost_usd: Estimate last recorded for the request.
        cost_usd: Estimate now recorded for it.
        model_name: Model the request is now priced under, for the log.
        previous_model_name: Model it was priced under before, for the log.

    Returns:
        The signed change to apply, or `None` when the request has never had a
            priceable estimate and so contributes nothing to the display.
    """
    if cost_usd is None:
        if previous_cost_usd is None:
            return None
        logger.warning(
            "Re-filing a request under the model it named made it unpriceable; "
            "dropping %.6f USD from the session total. from=%r to=%r",
            previous_cost_usd,
            previous_model_name,
            model_name,
        )
        return -previous_cost_usd
    return cost_usd - (previous_cost_usd or 0.0)


def _resolve_usage_model(
    message: object,
    *,
    fallback_model: str,
    fallback_provider: str,
    request_metadata: Mapping[str, Any] | None,
    kind: UsageKind,
) -> tuple[str, str]:
    """Resolve the `(model, provider)` a streamed message should be priced as.

    Args:
        message: Streamed model message or chunk.
        fallback_model: Model to use when response metadata does not name one.
        fallback_provider: Provider to use when response metadata omits it.
        request_metadata: Stream metadata for this specific request, if any.
        kind: Request class used by the type breakdown.

    Returns:
        The `(model_name, provider)` pair to record and price under.
    """
    from deepagents_code.cost_tracking import (
        _CONFIGURED_PROVIDER_METADATA_KEY,
        resolve_message_model,
    )

    configured_provider = (
        request_metadata.get(_CONFIGURED_PROVIDER_METADATA_KEY)
        if request_metadata is not None
        else None
    )
    has_request_provider = isinstance(configured_provider, str) and bool(
        configured_provider
    )
    return resolve_message_model(
        message,
        fallback_model=fallback_model,
        fallback_provider=(
            configured_provider if has_request_provider else fallback_provider
        ),
        # Request-specific metadata safely corrects generic provider responses.
        # Without it, only the main request may use the parent fallback; hidden
        # calls can be cross-provider and must keep their explicit response value.
        prefer_fallback_provider=has_request_provider or kind == "assistant",
    )


def _move_request_to_named_model(
    stats: SessionStats,
    message: object,
    previous: RecordedRequest,
    *,
    recorded_requests: dict[str, RecordedRequest],
    request_id: str,
    fallback_model: str,
    fallback_provider: str,
    request_metadata: Mapping[str, Any] | None,
    kind: UsageKind,
) -> float | None:
    """Re-file an already-recorded request under the model it finally named.

    Called for a message that names a model but carries no usable tokens of its
    own, which is how Google can close a stream. Token totals are unchanged, but
    the request is re-priced under the newly named model: the estimate it
    carried was computed against the caller's fallback model.

    Args:
        stats: Accumulator holding the request.
        message: The model-naming message.
        previous: What was last recorded for this request.
        recorded_requests: Ledger to update in place.
        request_id: Message ID keying the ledger entry.
        fallback_model: Model to use when response metadata does not name one.
        fallback_provider: Provider to use when response metadata omits it.
        request_metadata: Stream metadata for this specific request, if any.
        kind: Request class used by the type breakdown.

    Returns:
        The signed cost change the re-pricing applied, for the caller's
            provisional display, or `None` when it did not change.
    """
    model_name, provider = _resolve_usage_model(
        message,
        fallback_model=fallback_model,
        fallback_provider=fallback_provider,
        request_metadata=request_metadata,
        kind=kind,
    )
    if (model_name, provider) == (previous.model_name, previous.provider):
        return None

    from deepagents_code.cost_tracking import estimate_cost

    cost_usd = estimate_cost(previous.usage_metadata, model_name, provider)
    stats.retract_request(previous)
    stats.record_request(
        model_name,
        previous.input_tokens,
        previous.output_tokens,
        provider,
        cost_usd=cost_usd,
        kind=previous.kind,
    )
    recorded_requests[request_id] = RecordedRequest(
        model_name=model_name,
        provider=provider,
        kind=previous.kind,
        input_tokens=previous.input_tokens,
        output_tokens=previous.output_tokens,
        cost_usd=cost_usd,
        usage_metadata=previous.usage_metadata,
        finalized=previous.finalized,
    )
    return _cost_delta(
        previous.cost_usd,
        cost_usd,
        model_name=model_name,
        previous_model_name=previous.model_name,
    )


def record_message_usage(
    stats: SessionStats,
    message: object,
    *,
    fallback_model: str = "",
    fallback_provider: str = "",
    request_metadata: Mapping[str, Any] | None = None,
    kind: UsageKind = "assistant",
    recorded_requests: dict[str, RecordedRequest] | None = None,
) -> RecordedUsage | None:
    """Record usage attached to one streamed model message.

    A request is entered in `recorded_requests` only once usable token metadata
    has been recorded for it. Callers retain that ledger across stream rounds so
    one API call stays one row, however its usage arrives.

    A streamed chunk and a completed message report usage differently, so they
    are handled differently. A completed `AIMessage` carries the request's
    *whole* usage and is idempotent: replaying it must not count twice. A chunk
    carries whatever the provider chose to emit at that point in the stream --
    Anthropic and OpenAI attach the full usage to one chunk, while Google emits
    an incremental delta on every chunk, which the consumer is expected to sum.
    Skipping a chunk whose ID was already seen would drop every Google chunk
    after the first, losing most of the request's output tokens and cost.

    Summing them as separate requests would be wrong in the other direction, so
    a later chunk instead *revises* the request already recorded: its earlier
    contribution is retracted and re-recorded with the running totals. One API
    call therefore counts once, with one per-model row, no matter how many
    chunks carried its usage. Google also names the model only on its final
    chunk, so the model is upgraded when a message supplies one and otherwise
    left alone -- without that, one call would straddle a fallback-model row and
    a real-model row.

    Args:
        stats: Accumulator that receives the request.
        message: Streamed model message or chunk.
        fallback_model: Model to use when response metadata does not name one.
        fallback_provider: Provider to use when response metadata omits it.
        request_metadata: Stream metadata identifying the provider configured for
            this specific request, when available.
        kind: Request class used by the type breakdown.
        recorded_requests: Ledger of requests this stream consumer has already
            recorded, keyed by message ID. Mutated in place.

    Returns:
        The tokens and cost *this message* contributed, or `None` when it has no
            usable usage metadata or was already recorded in full. A message
            that only re-files an already-recorded request under a newly named
            model reports zero tokens and the signed cost change, because
            `stats` moved even though no new tokens arrived.
    """
    usage = getattr(message, "usage_metadata", None)
    if not isinstance(usage, Mapping) or not usage:
        return None

    # Imported here, not at module scope: this module is deliberately free of
    # heavy top-level dependencies (see the module docstring).
    from langchain_core.messages import AIMessageChunk

    if recorded_requests is None:
        recorded_requests = {}
    message_id = getattr(message, "id", None)
    request_id = message_id if isinstance(message_id, str) and message_id else None
    is_chunk = isinstance(message, AIMessageChunk)
    if request_id is not None and request_id in recorded_requests and not is_chunk:
        # A completed message repeats the whole request. Whether the request was
        # built from chunks or from an identical earlier replay, it is already
        # accounted for.
        return None

    input_count, output_count = _display_token_counts(usage)
    previous = recorded_requests.get(request_id) if request_id is not None else None
    if previous is not None and previous.finalized:
        # A completed message already supplied the whole request; a stray later
        # chunk cannot add to it.
        return None

    # A chunk carrying only a negative correction normalizes to zero counts, but
    # it still revises a request already recorded, so let it through.
    revises_previous = previous is not None and _carries_token_counts(usage)
    if not input_count and not output_count and not revises_previous:
        # Google's model-naming chunk can carry a zero-token delta, so a
        # request whose earlier chunks fell back to the configured model
        # would otherwise be stranded on the wrong per-model row.
        if previous is not None and _names_a_model(message):
            reprice_delta = _move_request_to_named_model(
                stats,
                message,
                previous,
                recorded_requests=recorded_requests,
                request_id=str(request_id),
                fallback_model=fallback_model,
                fallback_provider=fallback_provider,
                request_metadata=request_metadata,
                kind=kind,
            )
            if reprice_delta is not None:
                # `stats` changed even though no tokens arrived, so reporting
                # nothing would leave the caller's provisional display holding
                # the estimate this re-pricing just replaced.
                return RecordedUsage(
                    input_tokens=0,
                    output_tokens=0,
                    cost_usd=reprice_delta,
                    request_tokens=previous.input_tokens + previous.output_tokens,
                )
        return None

    from deepagents_code.cost_tracking import estimate_cost

    model_name, provider = _resolve_usage_model(
        message,
        fallback_model=fallback_model,
        fallback_provider=fallback_provider,
        request_metadata=request_metadata,
        kind=kind,
    )
    accumulated_usage: Mapping[str, Any] = usage

    if previous is not None:
        # Roll this chunk into the request already recorded rather than adding a
        # second one. The model is only upgraded when this message actually
        # named one, so the intermediate chunks' fallback does not overwrite it.
        stats.retract_request(previous)
        if not _names_a_model(message):
            model_name, provider = previous.model_name, previous.provider
        accumulated_usage = _merge_usage(previous.usage_metadata, usage)
        # Re-derive the displayed counts from the merged usage rather than
        # adding this message's own. Google's final chunk can carry a *negative*
        # input-token correction -- it reports a lower cumulative prompt count
        # that the provider treats as ground truth -- and per-message
        # normalization floors that at zero. Reading both the counts and the
        # cost off the same merged usage keeps them from disagreeing.
        input_count, output_count = _display_token_counts(accumulated_usage)

    # Price the request's whole accumulated usage under the model currently
    # known for it, rather than summing what each chunk cost when it arrived.
    # Early chunks are priced against the caller's fallback model, so keeping
    # their estimates would bill part of the request at the wrong rates -- or,
    # when the fallback is unpriceable, leave a priceable request showing no
    # cost at all.
    cost_usd = estimate_cost(accumulated_usage, model_name, provider)

    stats.record_request(
        model_name,
        input_count,
        output_count,
        provider,
        cost_usd=cost_usd,
        kind=kind,
    )
    if request_id is not None:
        recorded_requests[request_id] = RecordedRequest(
            model_name=model_name,
            provider=provider,
            kind=kind,
            input_tokens=input_count,
            output_tokens=output_count,
            cost_usd=cost_usd,
            usage_metadata=accumulated_usage,
            finalized=not is_chunk,
        )
    # Provisional pricing needs this message's delta, while the context display
    # needs the request's running token total after folding the message in.
    return RecordedUsage(
        input_tokens=input_count - (previous.input_tokens if previous else 0),
        output_tokens=output_count - (previous.output_tokens if previous else 0),
        cost_usd=_cost_delta(
            previous.cost_usd if previous else None,
            cost_usd,
            model_name=model_name,
            previous_model_name=previous.model_name if previous else model_name,
        ),
        request_tokens=input_count + output_count,
    )


def format_token_count(count: int) -> str:
    """Format a token count into a human-readable short string.

    Args:
        count: Number of tokens.

    Returns:
        Formatted string like `'12.5K'`, `'1.2M'`, or `'500'`.
    """
    if count >= 1_000_000:  # noqa: PLR2004
        return f"{count / 1_000_000:.1f}M"
    if count >= 1000:  # noqa: PLR2004
        return f"{count / 1000:.1f}K"
    return str(count)


def format_cost(cost_usd: float) -> str:
    """Format an estimated USD cost for compact display.

    Args:
        cost_usd: Estimated cost in US dollars.

    Returns:
        A string such as `'$0.42'`; positive sub-cent values use `'<$0.01'`.
    """
    if cost_usd <= 0:
        return "$0.00"
    if cost_usd < 0.01:  # noqa: PLR2004  # Display floor for sub-cent estimates.
        return "<$0.01"
    return f"${cost_usd:.2f}"


def _recorded_cost(cost_usd: float, priced_request_count: int) -> str:
    """Format a cost cell, distinguishing unpriced requests from zero cost.

    Returns:
        Formatted cost, or an em dash when no request was priceable.
    """
    return format_cost(cost_usd) if priced_request_count else "—"


def print_usage_table(
    stats: SessionStats,
    wall_time: float,
    console: Console,
) -> None:
    """Print a model-usage stats table to a Rich console.

    Each row shows the serving provider alongside the model name. When the
    session spans multiple models each gets its own row with a totals row
    appended; single-model sessions show one row.

    Args:
        stats: Cumulative session stats.
        wall_time: Total wall-clock time in seconds.
        console: Rich console for output.
    """
    from rich.table import Table

    has_time = wall_time >= 0.1  # noqa: PLR2004
    if not (stats.request_count or stats.input_tokens or has_time):
        return

    if stats.per_model:
        multi_model = len(stats.per_model) > 1

        table = Table(
            show_header=True,
            header_style="bold",
            box=None,
            padding=(0, 2, 0, 0),
            show_edge=False,
        )
        table.add_column("Provider", style="dim")
        table.add_column("Model", style="dim")
        table.add_column("Reqs", justify="right", style="dim")
        table.add_column("InputTok", justify="right", style="dim")
        table.add_column("OutputTok", justify="right", style="dim")
        table.add_column("Cost", justify="right", style="dim")

        if multi_model:
            for ms in stats.per_model.values():
                table.add_row(
                    ms.provider,
                    ms.model_name,
                    str(ms.request_count),
                    format_token_count(ms.input_tokens),
                    format_token_count(ms.output_tokens),
                    _recorded_cost(ms.cost_usd, ms.priced_request_count),
                )
            table.add_row(
                "",
                "Total",
                str(stats.request_count),
                format_token_count(stats.input_tokens),
                format_token_count(stats.output_tokens),
                _recorded_cost(stats.total_cost_usd, stats.priced_request_count),
            )
        else:
            ms = next(iter(stats.per_model.values()))
            table.add_row(
                ms.provider,
                ms.model_name,
                str(stats.request_count),
                format_token_count(stats.input_tokens),
                format_token_count(stats.output_tokens),
                _recorded_cost(stats.total_cost_usd, stats.priced_request_count),
            )

        console.print()
        console.print("[bold]Usage Stats[/bold]")
        console.print(table)
    if has_time:
        console.print()
        console.print(
            f"Agent active  {format_duration(wall_time)}",
            style="dim",
            highlight=False,
        )
