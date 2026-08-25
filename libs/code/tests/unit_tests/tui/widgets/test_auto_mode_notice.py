"""Tests for the Auto first-enable notice body."""

from __future__ import annotations

import pytest

from deepagents_code.tui.widgets.auto_mode_notice import build_auto_mode_notice_body


def test_distinct_classifier_says_it_is_not_the_coding_model() -> None:
    """A separate reviewer must be named *and* distinguished from the main model.

    "active model" reads as the model writing the code, so using it to label the
    classifier stated something false. Naming the classifier alone is not enough
    either — the copy has to say which of the two roles the model fills.
    """
    body = build_auto_mode_notice_body(
        "anthropic:claude-haiku-4-5", distinct_from_main_model=True
    )

    assert "a separate **classifier model** (anthropic:claude-haiku-4-5)" in body
    assert "not the model writing your code" in body
    assert "active model" not in body


def test_inherited_classifier_says_it_is_the_coding_model() -> None:
    """Without a separate classifier, the copy must say the models are the same.

    Otherwise "classifier model (anthropic:claude-opus-5)" implies a second model
    exists when reviews simply reuse the main one.
    """
    body = build_auto_mode_notice_body(
        "anthropic:claude-opus-5", distinct_from_main_model=False
    )

    assert "the **classifier model**, which is the same model writing your code" in body
    assert "anthropic:claude-opus-5" in body
    assert "a separate" not in body


def test_omits_the_parenthetical_when_the_spec_is_unknown() -> None:
    """An unresolved main-model spec must not render an empty `()`."""
    body = build_auto_mode_notice_body(None, distinct_from_main_model=False)

    assert "the same model writing your code." in body
    assert "()" not in body


def test_interpolates_once() -> None:
    """Only the review sentence gains the label, not every anchor occurrence."""
    body = build_auto_mode_notice_body(
        "openai:gpt-5.5-mini", distinct_from_main_model=True
    )

    assert body.count("openai:gpt-5.5-mini") == 1


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


def test_rejects_a_body_missing_the_anchor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rewording the sentence must fail loudly, not drop the disclosure.

    `str.replace` is a silent no-op when the anchor is gone, which would ship a
    notice that never names the reviewing model.
    """
    monkeypatch.setattr(
        "deepagents_code.tui.widgets.auto_mode_notice.AUTO_MODE_NOTICE_BODY",
        "Reviewed by some other phrasing entirely.",
    )

    with pytest.raises(ValueError, match="would not be disclosed"):
        build_auto_mode_notice_body(
            "anthropic:claude-haiku-4-5", distinct_from_main_model=True
        )
