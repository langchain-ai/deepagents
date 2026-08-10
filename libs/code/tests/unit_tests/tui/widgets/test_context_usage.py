"""Tests for the context-usage report."""

from deepagents_code.tui.widgets.context_usage import build_context_usage_markdown


def test_reports_capacity_and_estimated_breakdown() -> None:
    content = build_context_usage_markdown(
        context_tokens=20_000,
        conversation_tokens=15_000,
        context_limit=100_000,
        model_spec="anthropic:claude-sonnet",
        approximate=False,
    )

    assert "20.0K / 100.0K tokens (20.0%)" in content
    assert "**Remaining:** 80.0K tokens" in content
    assert "**System prompt + tools:** ~5.0K tokens (5.0%)" in content
    assert "**Conversation:** ~15.0K tokens (15.0%)" in content
    assert "**Automatic offload:** enabled" in content
