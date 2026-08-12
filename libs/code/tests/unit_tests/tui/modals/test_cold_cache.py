"""Tests for the cold prompt-cache warning modal."""

from textual.app import App

from deepagents_code.cold_cache import PromptCachePolicy, RewarmEstimate
from deepagents_code.tui.modals.cold_cache import ColdCacheWarningScreen


class _Host(App[None]):
    """Minimal host for exercising modal key paths."""


def _screen() -> ColdCacheWarningScreen:
    """Build a representative warning modal."""
    return ColdCacheWarningScreen(
        policy=PromptCachePolicy("OpenAI", 1800, "may_be_cold", 1024, "generic"),
        estimate=RewarmEstimate(0.42, 0.35),
        context_tokens=84_000,
        age_seconds=11_520,
    )


def test_openai_copy_preserves_retention_uncertainty() -> None:
    screen = ColdCacheWarningScreen(
        policy=PromptCachePolicy(
            "OpenAI",
            1800,
            "may_be_cold",
            1024,
            "generic",
        ),
        estimate=RewarmEstimate(0.42, 0.35),
        context_tokens=84_000,
        age_seconds=11_520,
    )

    body = screen._body()

    assert "idle for 3h 12m" in body
    assert "30m minimum cache-retention window" in body
    assert "may still have retained" in body
    assert "$0.42" in body
    assert "$0.35 more" in body


def test_anthropic_copy_calls_guaranteed_ttl_expired() -> None:
    screen = ColdCacheWarningScreen(
        policy=PromptCachePolicy("Anthropic", 300, "expired", 1024, "5m"),
        estimate=RewarmEstimate(1.25, 1.15),
        context_tokens=50_000,
        age_seconds=600,
    )

    body = screen._body()

    assert "Anthropic's 5m prompt-cache lifetime" in body
    assert "has likely expired" in body


def test_identity_change_uses_model_specific_copy() -> None:
    screen = ColdCacheWarningScreen(
        policy=PromptCachePolicy("OpenAI", 1800, "may_be_cold", 1024, "generic"),
        estimate=RewarmEstimate(0.42, 0.35),
        context_tokens=84_000,
        age_seconds=60,
        identity_changed=True,
    )

    body = screen._body()

    assert "active model or prompt-cache settings differ" in body
    assert "previous cached prefix cannot be reused" in body


async def test_enter_uses_safe_default_cancel() -> None:
    app = _Host()
    results: list[bool | None] = []

    async with app.run_test() as pilot:
        await app.push_screen(_screen(), callback=results.append)
        await pilot.pause()

        assert app.focused is not None
        assert app.focused.id == "cold-cache-cancel"
        await pilot.press("enter")
        await pilot.pause()

    assert results == [False]


async def test_send_shortcut_authorizes_once() -> None:
    app = _Host()
    results: list[bool | None] = []

    async with app.run_test() as pilot:
        await app.push_screen(_screen(), callback=results.append)
        await pilot.pause()
        await pilot.press("s")
        await pilot.pause()

    assert results == [True]
