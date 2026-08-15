"""Provider policy and pricing helpers for cold prompt-cache warnings."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal, assert_never
from urllib.parse import urlparse

CacheConfidence = Literal["expired", "may_be_cold"]
"""How firmly a passed retention window implies the cached prefix is gone.

`expired` is used when the window is a documented *maximum* (Anthropic's TTL,
OpenAI's `prompt_cache_retention` ceilings): once it passes, the entry is gone.
`may_be_cold` is used when the window is a documented *minimum* (GPT-5.6+),
where the provider is only guaranteed to have kept the entry that long and may
well have kept it longer.
"""

CacheWriteBucket = Literal["generic", "generic_write", "5m"]
"""Which pricing treatment a cold (cache-writing) request receives.

`5m` prices Anthropic's five-minute ephemeral write premium. `generic_write`
tags the miss as a cache write: GPT-5.6+ bills writes at 1.25x the uncached
input rate, and omitting the detail would price the miss at plain input.
`generic` covers misses with no write surcharge, which is how OpenAI priced
prompt caching before GPT-5.6.

`generic_write` is assigned by *model-name version* (see
`_openai_uses_thirty_minute_cache`), not by inspecting the catalog. Every
current `gpt-5.6-*` entry does publish a write rate, but that coupling is not
enforced: a future 5.6-family model with no cache rates would be tagged
`generic_write` and priced at plain input by `estimate_cost`, which drops
detail keys the catalog cannot price. The `-pro` variants are the precedent --
`gpt-5.4-pro` and `gpt-5.5-pro` publish no cache rates at all.
"""

ColdCacheReason = Literal["idle", "identity_changed", "age_unknown"]
"""Why a turn is treated as facing a cold prompt cache.

`idle` means the last request is older than the policy's retention window.
`identity_changed` means the model or its cache-affecting params differ from
the last successful turn, so the cached prefix cannot be reused regardless of
age. `age_unknown` means there is no usable record of when this thread last
reached the model -- a checkpoint written before cold-cache tracking existed,
or one whose timestamp could not be parsed -- so the cache cannot be assumed
warm. Each maps to distinct modal copy; they are not interchangeable, and in
particular `age_unknown` must not be reported as `identity_changed`, which
would claim a model change that never happened.
"""

_OPENAI_MODEL_VERSION = re.compile(r"^gpt-(?P<major>\d+)(?:\.(?P<minor>\d+))?")

_ANTHROPIC_MINIMUM_TOKENS: tuple[tuple[str, int], ...] = (
    ("claude-opus-5", 512),
    ("claude-fable-5", 512),
    ("claude-mythos-5", 512),
    ("claude-opus-4-7", 2048),
    ("claude-mythos-preview", 2048),
    ("claude-3-5-haiku", 2048),
    ("claude-opus-4-6", 4096),
    ("claude-opus-4-5", 4096),
    ("claude-haiku-4-5", 4096),
)
"""Prefixes for Claude models whose cache minimum differs from 1,024 tokens.

From the per-model minimums in Anthropic's prompt-caching docs:
https://docs.claude.com/en/docs/build-with-claude/prompt-caching

Prefixes must match real model ids. Haiku 3.5 predates the family-then-version
naming and ships as `claude-3-5-haiku-*`, so a `claude-haiku-3-5` prefix would
never match and would silently fall through to the 1,024 default. Order
matters: the first matching prefix wins, so a more specific prefix must
precede any shorter prefix of it.
"""

_ANTHROPIC_DEFAULT_MINIMUM_TOKENS = 1024
"""Cache minimum for the Claude models the table above does not name."""

_OPENAI_MINIMUM_TOKENS = 1024
"""Minimum cacheable prefix OpenAI documents, independent of Anthropic's."""

_ANTHROPIC_MIDDLEWARE_TTL_SECONDS = 300
"""Retention implied by the `cache_control` this stack actually sends.

`AnthropicPromptCachingMiddleware` runs *inside* `ConfigurableModelMiddleware`
and unconditionally rewrites `model_settings["cache_control"]` with its own
`ttl` (5m, the middleware default this stack never overrides). A user-supplied
`cache_control.ttl` in `model_params` is therefore overwritten before the
request leaves the process, so honoring it here would promise an hour of
retention the API never agreed to and suppress the warning for 55 minutes of a
dead cache. Read the middleware's effective TTL before reintroducing a longer
window.
"""


@dataclass(frozen=True, slots=True, kw_only=True)
class PromptCachePolicy:
    """Prompt-cache behavior needed to decide and price a warning.

    Keyword-only because `window_seconds` and `minimum_tokens` are adjacent
    bare ints: positionally, transposing them builds a plausible-looking policy
    that silently misprices and mis-gates.
    """

    provider_name: str
    """Display name of the provider whose endpoint was validated."""

    window_seconds: int
    """Retention window, past which `confidence` describes what is known."""

    confidence: CacheConfidence
    """Whether `window_seconds` is a documented maximum or minimum."""

    minimum_tokens: int
    """Smallest prefix the provider will cache at all."""

    write_bucket: CacheWriteBucket
    """Pricing treatment for the cold request; see `estimate_rewarm_cost`."""


@dataclass(frozen=True, slots=True, kw_only=True)
class RewarmEstimate:
    """Estimated input cost for a cold prefix and its warm-cache delta.

    Both figures are USD, non-negative, and finite; `incremental_cost_usd` is
    the part of `cold_cost_usd` that a cache hit would have avoided, so it
    never exceeds it.

    Keyword-only for the same reason as `PromptCachePolicy`: these are adjacent
    bare floats, and transposing them positionally yields copy that reads fine
    ("may cost up to ~$0.02 ... roughly ~$3.40 more than a warm cache hit")
    while being arithmetically impossible.
    """

    cold_cost_usd: float
    """Input spend to send the prefix uncached."""

    incremental_cost_usd: float
    """How much of `cold_cost_usd` a warm cache would have saved."""

    def __post_init__(self) -> None:
        """Enforce the documented ordering and non-negativity invariants.

        Raises:
            ValueError: When either figure is negative or the delta exceeds the
                total it is a part of.
        """
        if self.cold_cost_usd < 0 or self.incremental_cost_usd < 0:
            msg = (
                f"RewarmEstimate costs must be non-negative, got "
                f"cold={self.cold_cost_usd!r}, "
                f"incremental={self.incremental_cost_usd!r}"
            )
            raise ValueError(msg)
        if self.incremental_cost_usd > self.cold_cost_usd:
            msg = (
                f"RewarmEstimate incremental cost {self.incremental_cost_usd!r} "
                f"cannot exceed the cold cost {self.cold_cost_usd!r}"
            )
            raise ValueError(msg)


def debug_stand_in_policy() -> PromptCachePolicy:
    """Build the placeholder policy used by `DEEPAGENTS_CODE_DEBUG_COLD_CACHE`.

    Keeps the modal reachable on providers with no documented cache policy.
    Lives here rather than in the caller so the Anthropic window and minimum
    stay tied to `_ANTHROPIC_MIDDLEWARE_TTL_SECONDS` and
    `_ANTHROPIC_DEFAULT_MINIMUM_TOKENS` instead of being re-hardcoded, which
    would silently drift the moment either constant is revised.

    The provider name is deliberately Anthropic's: under the debug flag the
    modal may therefore cite Anthropic retention while a different provider is
    active. The figures are illustrative in that mode, not real estimates.

    Returns:
        Stand-in policy shaped like Anthropic's.
    """
    return PromptCachePolicy(
        provider_name="Anthropic",
        window_seconds=_ANTHROPIC_MIDDLEWARE_TTL_SECONDS,
        confidence="expired",
        minimum_tokens=_ANTHROPIC_DEFAULT_MINIMUM_TOKENS,
        write_bucket="5m",
    )


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
        # Deliberately ignores `params["cache_control"]`: see
        # `_ANTHROPIC_MIDDLEWARE_TTL_SECONDS` for why a user-set `ttl` never
        # reaches the wire.
        return PromptCachePolicy(
            provider_name="Anthropic",
            window_seconds=_ANTHROPIC_MIDDLEWARE_TTL_SECONDS,
            confidence="expired",
            minimum_tokens=_anthropic_minimum_tokens(model_name),
            write_bucket="5m",
        )

    if provider != "openai" or not _official_endpoint(base_url, "api.openai.com"):
        return None
    if _openai_uses_thirty_minute_cache(model_name):
        # 30 minutes is the documented guaranteed *minimum* for GPT-5.6+, but
        # OpenAI may retain the prefix longer, so past the window it can only
        # be treated as possibly cold.
        # https://platform.openai.com/docs/guides/prompt-caching
        return PromptCachePolicy(
            provider_name="OpenAI",
            window_seconds=1800,
            confidence="may_be_cold",
            minimum_tokens=_OPENAI_MINIMUM_TOKENS,
            write_bucket="generic_write",
        )

    # `in_memory` and `24h` are documented *maximums* ("up to one hour", "a
    # maximum, not a guarantee"): entries may be evicted earlier, so a warning
    # is only defensible once the maximum has passed -- at which point the
    # entry is gone rather than merely doubtful.
    # https://platform.openai.com/docs/guides/prompt-caching
    retention = params.get("prompt_cache_retention")
    retention_windows = {"in_memory": 3600, "24h": 86400}
    window = retention_windows.get(retention) if isinstance(retention, str) else None
    if window is None:
        return None
    return PromptCachePolicy(
        provider_name="OpenAI",
        window_seconds=window,
        confidence="expired",
        minimum_tokens=_OPENAI_MINIMUM_TOKENS,
        write_bucket="generic",
    )


def estimate_rewarm_cost(
    context_tokens: int,
    model_spec: str,
    policy: PromptCachePolicy,
) -> RewarmEstimate | None:
    """Estimate cold input spend and the incremental cost over a cache hit.

    Prices are derived by running two synthetic usage payloads through the
    ordinary `estimate_cost` path -- one billed as a full cache read, one as a
    cold request -- so the catalog stays the single source of truth. Outputs
    are zeroed on both sides so the delta is input-only.

    Returns:
        Price estimate, or `None` when the prefix is below the provider's cache
            minimum or the usage cannot be priced defensibly.
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
    cold_usage: dict[str, Any] = {
        "input_tokens": context_tokens,
        "output_tokens": 0,
        "total_tokens": context_tokens,
    }
    match policy.write_bucket:
        case "5m":
            # Anthropic bills a write premium over the ordinary input rate; the
            # detail key must stay in sync with
            # `cost_tracking._cache_write_counts`.
            cold_usage["input_token_details"] = {
                "ephemeral_5m_input_tokens": context_tokens
            }
        case "generic_write":
            # GPT-5.6+ bills a miss as a cache write at 1.25x the input rate,
            # so the write detail must reach `estimate_cost`; omitting it would
            # price the miss at plain input. `cache_write` is the generic alias
            # `_cache_write_counts` reads -- it is load-bearing here, not a
            # defensive spelling.
            cold_usage["input_token_details"] = {"cache_write": context_tokens}
        case "generic":
            # No cache-write detail at all: pre-5.6 OpenAI bills a miss at the
            # ordinary input rate, so tagging those tokens as a cache write
            # would apply a premium the provider never charges and overstate
            # both the displayed cost and the threshold comparison.
            pass
        case _:  # pragma: no cover - exhaustiveness guard
            assert_never(policy.write_bucket)

    from deepagents_code.cost_tracking import estimate_cost

    warm_cost = estimate_cost(warm_usage, model_name, provider)
    cold_cost = estimate_cost(cold_usage, model_name, provider)
    if warm_cost is None or cold_cost is None:
        return None
    if not math.isfinite(warm_cost) or not math.isfinite(cold_cost):
        return None
    # Clamp both sides before subtracting. Clamping only the total would let a
    # negative `cold_cost` produce an incremental figure larger than the cold
    # figure it is supposed to be a part of.
    cold = max(cold_cost, 0.0)
    warm = max(warm_cost, 0.0)
    return RewarmEstimate(
        cold_cost_usd=cold,
        incremental_cost_usd=max(cold - warm, 0.0),
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
