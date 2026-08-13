"""Tests for cold prompt-cache policy and pricing helpers."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from deepagents_code.cold_cache import (
    CacheWriteBucket,
    PromptCachePolicy,
    estimate_rewarm_cost,
    format_cache_age,
    format_cache_window,
    parse_cache_timestamp,
    resolve_prompt_cache_policy,
)


def test_resolves_anthropic_default_and_one_hour_policies() -> None:
    default = resolve_prompt_cache_policy("anthropic:claude-sonnet-4-6")
    extended = resolve_prompt_cache_policy(
        "anthropic:claude-sonnet-4-6",
        {"cache_control": {"type": "ephemeral", "ttl": "1h"}},
    )

    assert default == PromptCachePolicy("Anthropic", 300, "expired", 1024, "5m")
    assert extended == PromptCachePolicy("Anthropic", 3600, "expired", 1024, "1h")


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

    # 30 minutes is the documented guaranteed minimum, so past the window the
    # prefix is treated as expired.
    assert policy == PromptCachePolicy("OpenAI", 1800, "expired", 1024, "generic")


def test_resolves_explicit_older_openai_retention() -> None:
    in_memory = resolve_prompt_cache_policy(
        "openai:gpt-5.5",
        {"prompt_cache_retention": "in_memory"},
    )
    extended = resolve_prompt_cache_policy(
        "openai:gpt-5.5",
        {"prompt_cache_retention": "24h"},
    )

    # Both windows are documented maximums ("up to"), so past the window the
    # cache may still be warm.
    assert in_memory == PromptCachePolicy(
        "OpenAI", 3600, "may_be_cold", 1024, "generic"
    )
    assert extended == PromptCachePolicy(
        "OpenAI", 86400, "may_be_cold", 1024, "generic"
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
    ("bucket", "detail_key"),
    [
        ("generic", "cache_write"),
        ("5m", "ephemeral_5m_input_tokens"),
        ("1h", "ephemeral_1h_input_tokens"),
    ],
)
def test_estimate_rewarm_cost_uses_policy_bucket(
    monkeypatch: pytest.MonkeyPatch,
    bucket: CacheWriteBucket,
    detail_key: str,
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
        details = usage["input_token_details"]
        return 0.1 if "cache_read" in details else 1.25

    monkeypatch.setattr("deepagents_code.cost_tracking.estimate_cost", fake_estimate)
    policy = PromptCachePolicy(
        "Anthropic",
        300,
        "expired",
        1024,
        bucket,
    )

    estimate = estimate_rewarm_cost(50_000, "anthropic:model", policy)

    assert estimate is not None
    assert estimate.cold_cost_usd == pytest.approx(1.25)
    assert estimate.incremental_cost_usd == pytest.approx(1.15)
    assert calls[0]["input_tokens"] == 50_000
    assert calls[0]["input_token_details"] == {"cache_read": 50_000}
    assert calls[1]["input_token_details"] == {detail_key: 50_000}


def test_estimate_rewarm_cost_requires_cacheable_and_priceable_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = PromptCachePolicy("OpenAI", 1800, "may_be_cold", 1024, "generic")
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


def test_cache_time_formatting() -> None:
    assert format_cache_age(11_520) == "3h 12m"
    assert format_cache_age(300) == "5m"
    assert format_cache_window(1800) == "30m"
    assert format_cache_window(3600) == "1h"
