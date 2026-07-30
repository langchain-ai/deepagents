"""Shared event contract for expensive uncached model requests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

_SECONDS_PER_HOUR = 3600

EXPENSIVE_REQUEST_EVENT_TYPE = "expensive_model_request"
EXPENSIVE_REQUEST_TOKEN_THRESHOLD = 500_000
PROMPT_CACHE_TTL_SECONDS: dict[str, int] = {
    "anthropic": 5 * 60,
    "openai": 24 * 60 * 60,
}

PromptCacheProvider = Literal["anthropic", "openai"]


@dataclass(frozen=True)
class ExpensiveRequestWarning:
    """Validated warning details shared by interactive and headless clients."""

    provider: PromptCacheProvider
    input_tokens: int

    @property
    def cache_ttl_seconds(self) -> int:
        """Configured prompt-cache TTL for this provider."""
        return PROMPT_CACHE_TTL_SECONDS[self.provider]


def parse_expensive_request_warning(data: object) -> ExpensiveRequestWarning | None:
    """Validate a custom-stream payload as an expensive-request warning.

    Returns:
        Validated warning details, or `None` when the payload does not match.
    """
    if not isinstance(data, dict) or data.get("type") != EXPENSIVE_REQUEST_EVENT_TYPE:
        return None
    provider = data.get("provider")
    input_tokens = data.get("input_tokens")
    if provider not in PROMPT_CACHE_TTL_SECONDS:
        return None
    if (
        not isinstance(input_tokens, int)
        or isinstance(input_tokens, bool)
        or input_tokens <= EXPENSIVE_REQUEST_TOKEN_THRESHOLD
    ):
        return None
    return ExpensiveRequestWarning(
        provider=cast("PromptCacheProvider", provider),
        input_tokens=input_tokens,
    )


def format_expensive_request_warning(warning: ExpensiveRequestWarning) -> str:
    """Format an expensive-request warning for a user-facing notification.

    Returns:
        Notification text with the provider, token estimate, and cache TTL.
    """
    ttl = warning.cache_ttl_seconds
    ttl_label = (
        f"{ttl // _SECONDS_PER_HOUR}-hour"
        if ttl >= _SECONDS_PER_HOUR
        else f"{ttl // 60}-minute"
    )
    provider = warning.provider.capitalize()
    return (
        "Expensive request: about to send approximately "
        f"{warning.input_tokens:,} input tokens to {provider} outside its "
        f"{ttl_label} prompt cache TTL."
    )
