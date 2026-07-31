"""Lightweight session statistics, token formatting, and usage-table rendering.

Holds `SessionStats`/`ModelStats`, the `format_token_count` formatter, and
`print_usage_table` (which imports `rich.table` lazily). The module is
intentionally kept free of heavy top-level dependencies (no pydantic, no
config, no widget imports) so that `app.py` can import `SessionStats` and
`format_token_count` at module level without pulling in the full
`textual_adapter` dependency tree.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from deepagents_code.formatting import format_duration

if TYPE_CHECKING:
    from rich.console import Console

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
    """Usage captured from one completed streamed model request."""

    input_tokens: int
    """Input tokens reported for the request."""

    output_tokens: int
    """Output tokens reported for the request."""

    cost_usd: float | None
    """Estimated request cost, or `None` when pricing was unavailable."""


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

    Each chunk with `usage_metadata` counts as one completed request.
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
        cost_usd: float | None = None,
        *,
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


def record_message_usage(
    stats: SessionStats,
    message: object,
    *,
    fallback_model: str = "",
    fallback_provider: str = "",
    request_metadata: Mapping[str, Any] | None = None,
    kind: UsageKind = "assistant",
    seen_message_ids: set[str] | None = None,
) -> RecordedUsage | None:
    """Record usage attached to one streamed model message.

    Request IDs are marked only after usable token metadata is recorded. A
    resumed graph stream can replay a completed message, so callers may retain
    `seen_message_ids` across stream rounds to keep that request out of the
    breakdown a second time.

    Args:
        stats: Accumulator that receives the request.
        message: Streamed model message or chunk.
        fallback_model: Model to use when response metadata does not name one.
        fallback_provider: Provider to use when response metadata omits it.
        request_metadata: Stream metadata identifying the provider configured for
            this specific request, when available.
        kind: Request class used by the type breakdown.
        seen_message_ids: Request IDs already recorded by this stream consumer.

    Returns:
        Captured token and cost data, or `None` when the message has no usable
            usage metadata or was already recorded.
    """
    usage = getattr(message, "usage_metadata", None)
    if not isinstance(usage, Mapping) or not usage:
        return None

    message_id = getattr(message, "id", None)
    request_id = message_id if isinstance(message_id, str) and message_id else None
    if (
        request_id is not None
        and seen_message_ids is not None
        and request_id in seen_message_ids
    ):
        return None

    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    total_tokens = usage.get("total_tokens", 0)
    input_count = (
        input_tokens
        if isinstance(input_tokens, int)
        and not isinstance(input_tokens, bool)
        and input_tokens > 0
        else 0
    )
    output_count = (
        output_tokens
        if isinstance(output_tokens, int)
        and not isinstance(output_tokens, bool)
        and output_tokens > 0
        else 0
    )
    total_count = (
        total_tokens
        if isinstance(total_tokens, int)
        and not isinstance(total_tokens, bool)
        and total_tokens > 0
        else 0
    )
    if not input_count and not output_count:
        if not total_count:
            return None
        input_count = total_count

    from deepagents_code.cost_tracking import (
        _CONFIGURED_PROVIDER_METADATA_KEY,
        estimate_cost,
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
    provider_fallback = (
        configured_provider if has_request_provider else fallback_provider
    )
    model_name, provider = resolve_message_model(
        message,
        fallback_model=fallback_model,
        fallback_provider=provider_fallback,
        # Request-specific metadata safely corrects generic provider responses.
        # Without it, only the main request may use the parent fallback; hidden
        # calls can be cross-provider and must keep their explicit response value.
        prefer_fallback_provider=has_request_provider or kind == "assistant",
    )
    cost_usd = estimate_cost(usage, model_name, provider)
    stats.record_request(
        model_name,
        input_count,
        output_count,
        provider,
        cost_usd=cost_usd,
        kind=kind,
    )
    if request_id is not None and seen_message_ids is not None:
        seen_message_ids.add(request_id)
    return RecordedUsage(input_count, output_count, cost_usd)


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
