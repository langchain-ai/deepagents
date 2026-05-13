"""Lightweight session statistics and token formatting utilities.

This module is intentionally kept free of heavy dependencies (no pydantic, no
config, no widget imports) so that `app.py` can import `SessionStats` and
`format_token_count` at module level without pulling in the full
`textual_adapter` dependency tree.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

SpinnerStatus = Literal["Thinking", "Offloading"] | None
"""Valid spinner display states, or `None` to hide."""


@dataclass
class ModelStats:
    """Token stats for a single model within a session.

    Attributes:
        request_count: Number of LLM API requests made to this model.
        input_tokens: Cumulative input tokens sent to this model.
        output_tokens: Cumulative output tokens received from this model.
    """

    request_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class SessionStats:
    """Stats accumulated over a single agent turn (or full session).

    Attributes:
        request_count: Total LLM API requests made (each chunk with
            usage_metadata counts as one completed request).
        input_tokens: Cumulative input tokens across all LLM requests.
        output_tokens: Cumulative output tokens across all LLM requests.
        wall_time_seconds: Wall-clock duration from stream start to end.
        per_model: Per-model breakdown keyed by model name.
            Populated only when `record_request` receives a non-empty
            `model_name`. Empty dict means no named-model requests were
            recorded; `print_usage_table` omits the model table in that case and
            shows only the wall-time line (if applicable).
    """

    request_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    wall_time_seconds: float = 0.0
    per_model: dict[str, ModelStats] = field(default_factory=dict)

    def record_request(
        self,
        model_name: str,
        input_toks: int,
        output_toks: int,
    ) -> None:
        """Accumulate token counts for one completed LLM request.

        Updates both the session totals and the per-model breakdown.

        Args:
            model_name: The model that served this request (used as the
                per-model key). Pass an empty string to skip the per-model
                breakdown for this request.
            input_toks: Input tokens for this request.
            output_toks: Output tokens for this request.
        """
        self.request_count += 1
        self.input_tokens += input_toks
        self.output_tokens += output_toks
        if model_name:
            entry = self.per_model.setdefault(model_name, ModelStats())
            entry.request_count += 1
            entry.input_tokens += input_toks
            entry.output_tokens += output_toks

    def merge(self, other: SessionStats) -> None:
        """Merge another `SessionStats` into this one (mutates *self*).

        Used to accumulate per-turn stats into a session-level total.

        Args:
            other: The stats to fold in.
        """
        self.request_count += other.request_count
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.wall_time_seconds += other.wall_time_seconds
        for model, ms in other.per_model.items():
            entry = self.per_model.setdefault(model, ModelStats())
            entry.request_count += ms.request_count
            entry.input_tokens += ms.input_tokens
            entry.output_tokens += ms.output_tokens


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
