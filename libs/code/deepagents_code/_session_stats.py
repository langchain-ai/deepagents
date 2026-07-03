"""Lightweight session statistics and token formatting utilities.

This module is intentionally kept free of heavy dependencies (no pydantic, no
config, no widget imports) so that `app.py` can import `SessionStats` and
`format_token_count` at module level without pulling in the full
`textual_adapter` dependency tree.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

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


@dataclass
class ModelStats:
    """Token stats for a single model within a session."""

    request_count: int = 0
    """Number of LLM API requests made to this model."""

    input_tokens: int = 0
    """Cumulative input tokens sent to this model."""

    output_tokens: int = 0
    """Cumulative output tokens received from this model."""

    cost: float = 0.0
    """Cumulative estimated USD cost for requests served by this model."""

    provider: str = ""
    """Provider that served this model (e.g. `openai`), or `""` when unknown."""

    model_name: str = ""
    """Model name displayed in usage output."""


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

    cost: float = 0.0
    """Cumulative estimated USD cost across all LLM requests.

    Populated only when `record_request` receives a non-`None` `cost` (i.e. the
    model was found in the bundled price data). Left at `0.0` otherwise.
    """

    wall_time_seconds: float = 0.0
    """Wall-clock duration from stream start to end."""

    per_model: dict[ModelStatsKey, ModelStats] = field(default_factory=dict)
    """Per-model breakdown keyed by `(provider, model_name)`.

    Populated only when `record_request` receives a non-empty `model_name`. Empty
    dict means no named-model requests were recorded; `print_usage_table` omits
    the model table in that case and shows only the wall-time line (if applicable).
    """

    def record_request(
        self,
        model_name: str,
        input_toks: int,
        output_toks: int,
        provider: str = "",
        cost: float | None = None,
    ) -> None:
        """Accumulate token counts (and optional cost) for one LLM request.

        Updates both the session totals and the per-model breakdown.

        Args:
            model_name: The model that served this request. Combined with
                `provider` to form the per-model key. Pass an empty string to
                skip the per-model breakdown for this request.
            input_toks: Input tokens for this request.
            output_toks: Output tokens for this request.
            provider: Provider that served the model (e.g. `openai`). Combined
                with `model_name` to form the per-model key, so the same model
                served by different providers is tracked separately.
            cost: Estimated USD cost for this request, or `None` when pricing is
                unavailable (unknown model/provider). `None` leaves the cost
                totals unchanged; `0.0` is a valid recorded cost.
        """
        self.request_count += 1
        self.input_tokens += input_toks
        self.output_tokens += output_toks
        if cost is not None:
            self.cost += cost
        if model_name:
            key = (provider, model_name)
            entry = self.per_model.setdefault(
                key,
                ModelStats(provider=provider, model_name=model_name),
            )
            entry.request_count += 1
            entry.input_tokens += input_toks
            entry.output_tokens += output_toks
            if cost is not None:
                entry.cost += cost

    def merge(self, other: SessionStats) -> None:
        """Merge another `SessionStats` into this one (mutates *self*).

        Used to accumulate per-turn stats into a session-level total.

        Args:
            other: The stats to fold in.
        """
        self.request_count += other.request_count
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cost += other.cost
        self.wall_time_seconds += other.wall_time_seconds
        for key, ms in other.per_model.items():
            entry = self.per_model.setdefault(
                key,
                ModelStats(provider=ms.provider, model_name=ms.model_name),
            )
            entry.request_count += ms.request_count
            entry.input_tokens += ms.input_tokens
            entry.output_tokens += ms.output_tokens
            entry.cost += ms.cost


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
