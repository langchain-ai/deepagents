"""Tests for the cold prompt-cache warning modal."""

from textual.app import App

from deepagents_code.cold_cache import (
    CacheConfidence,
    CacheWriteBucket,
    PromptCachePolicy,
    RewarmEstimate,
)
from deepagents_code.tui.modals.cold_cache import (
    ColdCacheChoice,
    ColdCacheWarningScreen,
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


class _Host(App[None]):
    """Minimal host for exercising modal key paths."""


def _screen() -> ColdCacheWarningScreen:
    """Build a representative warning modal."""
    return ColdCacheWarningScreen(
        policy=_policy("OpenAI", 1800, "may_be_cold", 1024, "generic"),
        estimate=RewarmEstimate(0.42, 0.35),
        context_tokens=84_000,
        age_seconds=11_520,
    )


def test_openai_copy_preserves_retention_uncertainty() -> None:
    screen = ColdCacheWarningScreen(
        policy=_policy("OpenAI", 1800, "may_be_cold", 1024, "generic"),
        estimate=RewarmEstimate(0.42, 0.35),
        context_tokens=84_000,
        age_seconds=11_520,
    )

    body = screen._body()

    assert "idle for 3h 12m" in body
    assert "30m minimum cache-retention window" in body
    assert "may still have retained" in body
    # The `may_be_cold` branch keeps the cost sentence conditional, and both
    # figures are rounded estimates framed as an upper bound.
    assert "If the cache has expired" in body
    assert "may cost up to ~$0.42" in body
    assert "~$0.35 more" in body


def test_anthropic_copy_calls_guaranteed_ttl_expired() -> None:
    screen = ColdCacheWarningScreen(
        policy=_policy("Anthropic", 300, "expired", 1024, "5m"),
        estimate=RewarmEstimate(1.25, 1.15),
        context_tokens=50_000,
        age_seconds=600,
    )

    body = screen._body()

    assert "Anthropic's 5m prompt-cache lifetime" in body
    assert "has likely expired" in body
    # Past a documented maximum the prefix is gone, so the cost sentence is
    # unconditional but still rounds and frames the figures as upper bounds.
    assert "If the cache has expired" not in body
    assert "may cost up to ~$1.3" in body
    assert "~$1.2 more" in body


def test_identity_change_uses_model_specific_copy() -> None:
    screen = ColdCacheWarningScreen(
        policy=_policy("OpenAI", 1800, "may_be_cold", 1024, "generic"),
        estimate=RewarmEstimate(0.42, 0.35),
        context_tokens=84_000,
        age_seconds=60,
        identity_changed=True,
    )

    body = screen._body()

    assert "active model or prompt-cache settings differ" in body
    assert "previous cached prefix cannot be reused" in body
    # An identity change guarantees the prefix is unusable, so the cost
    # sentence is unconditional even though the policy is `may_be_cold`.
    assert "If the cache has expired" not in body
    assert "may cost up to ~$0.42" in body


async def test_enter_authorizes_send() -> None:
    app = _Host()
    results: list[ColdCacheChoice | None] = []

    async with app.run_test() as pilot:
        await app.push_screen(_screen(), callback=results.append)
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

    assert results == [ColdCacheChoice.SEND]


async def test_arrow_down_then_enter_suppresses_for_session() -> None:
    app = _Host()
    results: list[ColdCacheChoice | None] = []

    async with app.run_test() as pilot:
        await app.push_screen(_screen(), callback=results.append)
        await pilot.pause()
        await pilot.press("down", "enter")
        await pilot.pause()

    assert results == [ColdCacheChoice.SEND_SUPPRESS_SESSION]


async def test_navigation_wraps_to_suppress_always() -> None:
    app = _Host()
    results: list[ColdCacheChoice | None] = []

    async with app.run_test() as pilot:
        await app.push_screen(_screen(), callback=results.append)
        await pilot.pause()
        await pilot.press("down", "down", "enter")
        await pilot.pause()

    assert results == [ColdCacheChoice.SEND_SUPPRESS_ALWAYS]


async def test_navigation_up_from_top_wraps_to_keep_draft() -> None:
    app = _Host()
    results: list[ColdCacheChoice | None] = []

    async with app.run_test() as pilot:
        await app.push_screen(_screen(), callback=results.append)
        await pilot.pause()
        await pilot.press("up", "enter")
        await pilot.pause()

    assert results == [ColdCacheChoice.CANCEL]


async def test_escape_cancels_to_keep_draft() -> None:
    app = _Host()
    results: list[ColdCacheChoice | None] = []

    async with app.run_test() as pilot:
        await app.push_screen(_screen(), callback=results.append)
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()

    assert results == [ColdCacheChoice.CANCEL]


async def test_clicking_a_choice_does_not_resolve_the_modal() -> None:
    """Clicks are swallowed so a stray press cannot authorize spend.

    The rows look clickable, so this pins the deliberate keyboard-only
    activation described on `_ChoiceOption.on_click`.
    """
    app = _Host()
    results: list[ColdCacheChoice | None] = []

    async with app.run_test() as pilot:
        await app.push_screen(_screen(), callback=results.append)
        await pilot.pause()
        await pilot.click(".cold-cache-choice")
        await pilot.pause()

        assert results == []

        await pilot.press("enter")
        await pilot.pause()

    assert results == [ColdCacheChoice.SEND]
