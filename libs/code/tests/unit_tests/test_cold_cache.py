"""Tests for cold prompt-cache policy and pricing helpers."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import pytest

from deepagents_code.cold_cache import (
    CacheConfidence,
    CacheWriteBucket,
    PromptCachePolicy,
    estimate_rewarm_cost,
    format_cache_age,
    format_cache_window,
    parse_cache_timestamp,
    resolve_prompt_cache_policy,
)


def _policy(
    provider_name: str,
    window_seconds: int,
    confidence: CacheConfidence,
    minimum_tokens: int,
    write_bucket: CacheWriteBucket,
) -> PromptCachePolicy:
    """Build a policy positionally to keep expectations readable in tests."""
    return PromptCachePolicy(
        provider_name=provider_name,
        window_seconds=window_seconds,
        confidence=confidence,
        minimum_tokens=minimum_tokens,
        write_bucket=write_bucket,
    )


def test_anthropic_policy_ignores_user_supplied_cache_control_ttl() -> None:
    """A user `ttl` never reaches the wire, so it must not widen the window.

    `AnthropicPromptCachingMiddleware` runs inside `ConfigurableModelMiddleware`
    and overwrites `model_settings["cache_control"]` with its own 5m TTL.
    Honoring the user's `1h` here would suppress the warning for 55 minutes of
    a cache that died at five.
    """
    default = resolve_prompt_cache_policy("anthropic:claude-sonnet-4-6")
    with_ttl = resolve_prompt_cache_policy(
        "anthropic:claude-sonnet-4-6",
        {"cache_control": {"type": "ephemeral", "ttl": "1h"}},
    )

    assert default == _policy("Anthropic", 300, "expired", 1024, "5m")
    assert with_ttl == default


@pytest.mark.parametrize(
    ("model", "minimum"),
    [
        ("claude-opus-5", 512),
        ("claude-fable-5", 512),
        ("claude-mythos-5", 512),
        ("claude-opus-4-8", 1024),
        ("claude-sonnet-5", 1024),
        ("claude-sonnet-4-6", 1024),
        ("claude-opus-4-7", 2048),
        ("claude-mythos-preview", 2048),
        ("claude-opus-4-6", 4096),
        ("claude-opus-4-5", 4096),
        ("claude-haiku-4-5", 4096),
    ],
)
def test_resolves_anthropic_per_model_minimums(model: str, minimum: int) -> None:
    policy = resolve_prompt_cache_policy(f"anthropic:{model}")

    assert policy is not None
    assert policy.minimum_tokens == minimum


@pytest.mark.parametrize("model", ["gpt-5.6", "gpt-5.6-pro", "gpt-6"])
def test_resolves_current_openai_minimum_retention(model: str) -> None:
    policy = resolve_prompt_cache_policy(f"openai:{model}")

    # 30 minutes is the documented guaranteed minimum, but OpenAI may retain
    # the prefix longer, so past the window it may still be warm.
    assert policy == _policy("OpenAI", 1800, "may_be_cold", 1024, "generic")


def test_resolves_explicit_older_openai_retention() -> None:
    in_memory = resolve_prompt_cache_policy(
        "openai:gpt-5.5",
        {"prompt_cache_retention": "in_memory"},
    )
    extended = resolve_prompt_cache_policy(
        "openai:gpt-5.5",
        {"prompt_cache_retention": "24h"},
    )

    # Both windows are documented maximums ("up to one hour", "a maximum, not
    # a guarantee"), so once the window passes the entry is gone -- unlike the
    # GPT-5.6+ minimum, which the provider may exceed.
    assert in_memory == _policy("OpenAI", 3600, "expired", 1024, "generic")
    assert extended == _policy("OpenAI", 86400, "expired", 1024, "generic")


def test_ignores_non_string_openai_retention() -> None:
    assert (
        resolve_prompt_cache_policy("openai:gpt-5.5", {"prompt_cache_retention": 3600})
        is None
    )


def test_estimate_rewarm_cost_respects_per_model_anthropic_minimum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 4,096-token floor rejects 3,000 tokens on Haiku but not on Opus 5."""
    monkeypatch.setattr(
        "deepagents_code.cost_tracking.estimate_cost", lambda *_args: 1.0
    )
    haiku = resolve_prompt_cache_policy("anthropic:claude-haiku-4-5")
    opus = resolve_prompt_cache_policy("anthropic:claude-opus-5")

    assert haiku is not None
    assert opus is not None
    assert estimate_rewarm_cost(3000, "anthropic:claude-haiku-4-5", haiku) is None
    assert estimate_rewarm_cost(3000, "anthropic:claude-opus-5", opus) is not None


def test_skips_unresolved_or_custom_provider_policies() -> None:
    assert resolve_prompt_cache_policy("openai:gpt-5.5") is None
    assert resolve_prompt_cache_policy("google_genai:gemini-3.6-flash") is None
    assert (
        resolve_prompt_cache_policy(
            "openai:gpt-5.6",
            base_url="https://gateway.example.com/v1",
        )
        is None
    )
    assert (
        resolve_prompt_cache_policy(
            "anthropic:claude-sonnet-4-6",
            base_url="https://gateway.example.com",
        )
        is None
    )


@pytest.mark.parametrize(
    ("bucket", "cold_details"),
    [
        # OpenAI bills a miss at the plain input rate, so the cold side carries
        # no cache-write detail at all -- tagging one applies a write premium
        # the provider never charges.
        ("generic", None),
        ("5m", {"ephemeral_5m_input_tokens": 50_000}),
    ],
)
def test_estimate_rewarm_cost_uses_policy_bucket(
    monkeypatch: pytest.MonkeyPatch,
    bucket: CacheWriteBucket,
    cold_details: dict[str, int] | None,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_estimate(
        usage: dict[str, Any],
        model_name: str,
        provider: str,
    ) -> float:
        calls.append(usage)
        assert model_name == "model"
        assert provider == "anthropic"
        details = usage.get("input_token_details", {})
        return 0.1 if "cache_read" in details else 1.25

    monkeypatch.setattr("deepagents_code.cost_tracking.estimate_cost", fake_estimate)
    policy = _policy("Anthropic", 300, "expired", 1024, bucket)

    estimate = estimate_rewarm_cost(50_000, "anthropic:model", policy)

    assert estimate is not None
    assert estimate.cold_cost_usd == pytest.approx(1.25)
    assert estimate.incremental_cost_usd == pytest.approx(1.15)
    assert calls[0]["input_tokens"] == 50_000
    assert calls[0]["input_token_details"] == {"cache_read": 50_000}
    assert calls[1]["input_tokens"] == 50_000
    assert calls[1].get("input_token_details") == cold_details


def test_generic_bucket_prices_a_miss_at_the_plain_input_rate() -> None:
    """An OpenAI cold turn must cost exactly the uncached input price.

    Guards the real pricing path rather than a fake: the catalog applies a
    cache-write premium to some models, and forwarding a write key here
    overstated both the displayed cost and the threshold comparison.
    """
    from deepagents_code.cost_tracking import estimate_cost

    policy = resolve_prompt_cache_policy("openai:gpt-5.6-terra")
    assert policy is not None

    estimate = estimate_rewarm_cost(100_000, "openai:gpt-5.6-terra", policy)
    plain_input = estimate_cost(
        {"input_tokens": 100_000, "output_tokens": 0, "total_tokens": 100_000},
        "gpt-5.6-terra",
        "openai",
    )

    assert estimate is not None
    assert plain_input is not None
    assert estimate.cold_cost_usd == pytest.approx(plain_input)


def test_estimate_rewarm_cost_requires_cacheable_and_priceable_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _policy("OpenAI", 1800, "may_be_cold", 1024, "generic")
    assert estimate_rewarm_cost(100, "openai:gpt-5.6", policy) is None

    monkeypatch.setattr(
        "deepagents_code.cost_tracking.estimate_cost",
        lambda *_args: None,
    )
    assert estimate_rewarm_cost(5000, "openai:gpt-5.6", policy) is None


def test_parse_cache_timestamp_requires_timezone() -> None:
    timestamp = datetime(2026, 8, 11, 12, 30, tzinfo=UTC)

    assert parse_cache_timestamp(timestamp.isoformat()) == timestamp
    assert parse_cache_timestamp("2026-08-11T12:30:00") is None
    assert parse_cache_timestamp("not-a-time") is None
    assert parse_cache_timestamp(None) is None


def test_parse_cache_timestamp_normalizes_non_utc_offsets() -> None:
    """A `+05:00` checkpoint must not read as five hours of extra idle time."""
    offset = timezone(timedelta(hours=5))
    local = datetime(2026, 8, 11, 17, 30, tzinfo=offset)

    parsed = parse_cache_timestamp(local.isoformat())

    assert parsed == datetime(2026, 8, 11, 12, 30, tzinfo=UTC)
    assert parsed is not None
    assert parsed.tzinfo is UTC


def test_explicit_official_endpoints_still_resolve() -> None:
    assert resolve_prompt_cache_policy(
        "openai:gpt-5.6", base_url="https://api.openai.com/v1"
    ) == _policy("OpenAI", 1800, "may_be_cold", 1024, "generic")
    anthropic = resolve_prompt_cache_policy(
        "anthropic:claude-sonnet-4-6", base_url="https://api.anthropic.com"
    )
    assert anthropic == _policy("Anthropic", 300, "expired", 1024, "5m")
    # Non-HTTP schemes are not the official API however they parse.
    assert (
        resolve_prompt_cache_policy("openai:gpt-5.6", base_url="ftp://api.openai.com")
        is None
    )


def test_non_dict_cache_control_falls_back_to_default_window() -> None:
    assert resolve_prompt_cache_policy(
        "anthropic:claude-sonnet-4-6", {"cache_control": "ephemeral"}
    ) == _policy("Anthropic", 300, "expired", 1024, "5m")


def test_cache_time_formatting() -> None:
    assert format_cache_age(11_520) == "3h 12m"
    assert format_cache_age(300) == "5m"
    assert format_cache_age(0) == "0m"
    assert format_cache_age(-30) == "0m"
    assert format_cache_window(1800) == "30m"
    assert format_cache_window(3600) == "1h"
