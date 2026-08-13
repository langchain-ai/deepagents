"""Provider policy and pricing helpers for cold prompt-cache warnings."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal
from urllib.parse import urlparse

CacheConfidence = Literal["expired", "may_be_cold"]
CacheWriteBucket = Literal["generic", "5m", "1h"]

_OPENAI_MODEL_VERSION = re.compile(r"^gpt-(?P<major>\d+)(?:\.(?P<minor>\d+))?")

_ANTHROPIC_MINIMUM_TOKENS: tuple[tuple[str, int], ...] = (
    ("claude-opus-5", 512),
    ("claude-fable-5", 512),
    ("claude-mythos-5", 512),
    ("claude-opus-4-7", 2048),
    ("claude-mythos-preview", 2048),
    ("claude-haiku-3-5", 2048),
    ("claude-opus-4-6", 4096),
    ("claude-opus-4-5", 4096),
    ("claude-haiku-4-5", 4096),
)
"""Prefixes for Claude models whose cache minimum differs from 1,024 tokens.

From the per-model minimums in Anthropic's prompt-caching docs. Order matters:
the first matching prefix wins, so `claude-mythos-preview` must precede a bare
`claude-mythos` entry if one is ever added.
"""

_ANTHROPIC_DEFAULT_MINIMUM_TOKENS = 1024
"""Cache minimum for the Claude models the table above does not name."""


@dataclass(frozen=True, slots=True)
class PromptCachePolicy:
    """Prompt-cache behavior needed to decide and price a warning."""

    provider_name: str
    window_seconds: int
    confidence: CacheConfidence
    minimum_tokens: int
    write_bucket: CacheWriteBucket


@dataclass(frozen=True, slots=True)
class RewarmEstimate:
    """Estimated input cost for a cold prefix and its warm-cache delta."""

    cold_cost_usd: float
    incremental_cost_usd: float


def _official_endpoint(base_url: str | None, hostname: str) -> bool:
    """Return whether an optional endpoint targets the provider's official API."""
    if not base_url:
        return True
    try:
        parsed = urlparse(base_url)
    except ValueError:
        return False
    return parsed.scheme in {"http", "https"} and parsed.hostname == hostname


def _openai_uses_thirty_minute_cache(model_name: str) -> bool:
    """Return whether an OpenAI model belongs to the GPT-5.6-or-newer family."""
    match = _OPENAI_MODEL_VERSION.match(model_name.lower())
    if match is None:
        return False
    major = int(match.group("major"))
    minor = int(match.group("minor") or 0)
    return (major, minor) >= (5, 6)


def _anthropic_minimum_tokens(model_name: str) -> int:
    """Return the documented minimum cacheable prefix for a Claude model."""
    normalized = model_name.lower()
    return next(
        (
            minimum
            for prefix, minimum in _ANTHROPIC_MINIMUM_TOKENS
            if normalized.startswith(prefix)
        ),
        _ANTHROPIC_DEFAULT_MINIMUM_TOKENS,
    )


def resolve_prompt_cache_policy(
    model_spec: str,
    model_params: dict[str, Any] | None = None,
    *,
    base_url: str | None = None,
) -> PromptCachePolicy | None:
    """Resolve a documented cache policy for one effective model invocation.

    Returns:
        Matching policy, or `None` when retention cannot be resolved safely.
    """
    if ":" not in model_spec:
        return None
    provider, model_name = model_spec.split(":", 1)
    provider = provider.strip().lower()
    model_name = model_name.strip()
    if not model_name:
        return None
    params = model_params or {}

    if provider == "anthropic":
        if not _official_endpoint(base_url, "api.anthropic.com"):
            return None
        minimum = _anthropic_minimum_tokens(model_name)
        cache_control = params.get("cache_control")
        ttl = cache_control.get("ttl") if isinstance(cache_control, dict) else None
        if ttl == "1h":
            return PromptCachePolicy("Anthropic", 3600, "expired", minimum, "1h")
        return PromptCachePolicy("Anthropic", 300, "expired", minimum, "5m")

    if provider != "openai" or not _official_endpoint(base_url, "api.openai.com"):
        return None
    if _openai_uses_thirty_minute_cache(model_name):
        # 30 minutes is the documented guaranteed minimum for GPT-5.6+; past
        # the window the prefix can only be treated as expired.
        return PromptCachePolicy("OpenAI", 1800, "expired", 1024, "generic")

    retention = params.get("prompt_cache_retention")
    # `in_memory` and `24h` are documented maximums ("up to one hour", "a
    # maximum, not a guarantee"): entries may be evicted earlier, so a warning
    # is only defensible once the maximum has passed -- and even then the entry
    # may linger, so the confidence stays "may_be_cold".
    if retention == "in_memory":
        return PromptCachePolicy("OpenAI", 3600, "may_be_cold", 1024, "generic")
    if retention == "24h":
        return PromptCachePolicy("OpenAI", 86400, "may_be_cold", 1024, "generic")
    return None


def estimate_rewarm_cost(
    context_tokens: int,
    model_spec: str,
    policy: PromptCachePolicy,
) -> RewarmEstimate | None:
    """Estimate cold input spend and the incremental cost over a cache hit.

    Returns:
        Price estimate, or `None` when usage cannot be priced defensibly.
    """
    if context_tokens < policy.minimum_tokens or ":" not in model_spec:
        return None
    provider, model_name = model_spec.split(":", 1)
    if not provider or not model_name:
        return None

    warm_usage: dict[str, Any] = {
        "input_tokens": context_tokens,
        "output_tokens": 0,
        "total_tokens": context_tokens,
        "input_token_details": {"cache_read": context_tokens},
    }
    detail_key = {
        "generic": "cache_write",
        "5m": "ephemeral_5m_input_tokens",
        "1h": "ephemeral_1h_input_tokens",
    }[policy.write_bucket]
    cold_usage: dict[str, Any] = {
        "input_tokens": context_tokens,
        "output_tokens": 0,
        "total_tokens": context_tokens,
        "input_token_details": {detail_key: context_tokens},
    }

    from deepagents_code.cost_tracking import estimate_cost

    warm_cost = estimate_cost(warm_usage, model_name, provider)
    cold_cost = estimate_cost(cold_usage, model_name, provider)
    if warm_cost is None or cold_cost is None:
        return None
    if not math.isfinite(warm_cost) or not math.isfinite(cold_cost):
        return None
    return RewarmEstimate(
        cold_cost_usd=max(cold_cost, 0.0),
        incremental_cost_usd=max(cold_cost - warm_cost, 0.0),
    )


def parse_cache_timestamp(value: object) -> datetime | None:
    """Parse a persisted UTC timestamp, rejecting malformed or naive values.

    Returns:
        UTC datetime, or `None` when the value is unusable.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(UTC)


def format_cache_age(seconds: float) -> str:
    """Format elapsed cache age for compact modal copy.

    Returns:
        Compact hours/minutes label.
    """
    seconds = max(int(seconds), 0)
    hours, remainder = divmod(seconds, 3600)
    minutes = remainder // 60
    if hours:
        return f"{hours}h {minutes}m" if minutes else f"{hours}h"
    return f"{minutes}m"


def format_cache_window(seconds: int) -> str:
    """Format a provider cache window for compact modal copy.

    Returns:
        Compact hours or minutes label.
    """
    if seconds % 3600 == 0:
        return f"{seconds // 3600}h"
    return f"{seconds // 60}m"
