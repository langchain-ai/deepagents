"""Tests for the Auto first-enable notice body."""

from __future__ import annotations

from deepagents_code.tui.widgets.auto_mode_notice import build_auto_mode_notice_body


def test_escapes_markdown_in_the_model_label() -> None:
    """The label comes from user config, so it cannot inject markdown.

    Round-trips the characters the escape table covers plus a newline, which
    would otherwise break the surrounding Markdown block.
    """
    body = build_auto_mode_notice_body(
        "openai:a_b*c[d]<e>|f~g&h`i\nj", distinct_from_main_model=True
    )

    assert (
        "**classifier model** (openai:a\\_b\\*c\\[d\\]\\<e\\>\\|f\\~g\\&h\\`i j)"
    ) in body
    # Normalized to one line: a raw newline would end the Markdown paragraph.
    assert "i\nj" not in body
