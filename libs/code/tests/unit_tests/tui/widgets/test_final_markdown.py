"""Tests for the one-widget renderer of finished assistant replies."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.geometry import Offset
from textual.selection import Selection

from deepagents_code.tui.widgets.final_markdown import FinalMarkdown
from deepagents_code.tui.widgets.messages import AssistantMessage

if TYPE_CHECKING:
    from textual.widget import Widget

_CORPUS = {
    "paragraphs": (
        "Hello **bold**, *em*, ~~gone~~, `code`, and a [link](https://e.com/x)"
        " that wraps across a long line.\n\nSecond paragraph\nwith a softbreak  \n"
        "and a hard break."
    ),
    "headings": (
        "# A title long enough to wrap at narrow widths\n\n## Sub\n\n### Three\n\n"
        "text\n\n#### Four\n\n##### Five\n\n###### Six\n\nend"
    ),
    "nested_lists": (
        "- a\n- b with a long line of text that needs to wrap at narrow widths\n"
        "  - nested\n    - deeper\n      1. ordered\n      2. again\n\n"
        "1. one\n2. two\n\n   loose paragraph\n\n10. ten\n11. eleven\n\n"
        "-\n- after empty"
    ),
    "fences": (
        "Run:\n\n```python\ndef f(x):\n    return x  # a long trailing comment that"
        " overflows narrow widths for certain\n\n\tindented = 1\n```\n\n"
        "    indented code block\n\nafter"
    ),
    "tables": (
        "| Option | Cost | Notes |\n| --- | --- | --- |\n"
        "| Markdown | 38 nodes | a much longer note that will need to wrap |\n"
        "| Static | 1 node | needs `Content` and [a link](https://x.y) |\n\nafter"
    ),
    "quotes": "> one\n> still one\n>\n> two\n>\n> - item\n>\n> > nested\n\nafter",
    "rules": "above\n\n---\n\nbelow\n\n***",
    "html": "<div>\nhtml block\n</div>\n\ninline <b>bold</b> html\n\nend",
    "reference_links": "See [docs][ref] and ![img](http://i.png).\n\n[ref]: https://e.com",
    "mixed": (
        "- item with fence:\n\n  ```\n  code in list\n  ```\n\n- item with quote\n\n"
        "  > q\n\n# Heading after list\n\n| a |\n|---|\n\n- last"
    ),
    "long_word": "averyveryveryveryveryveryveryveryveryveryverylongword and more",
    "ends_with_fence": "para\n\n```\ncode\n```",
    "ends_with_heading": "para\n\n## End",
    "wide_characters": "中文字符测试 一些宽字符的文本",
}
"""Markdown covering every block type `FinalMarkdown` lays out."""


class _ReplyApp(App[None]):
    def compose(self) -> ComposeResult:
        with VerticalScroll():
            for name in _CORPUS:
                yield AssistantMessage(id=f"reply-{name}")


def _screen_lines(app: App[None], widget: Widget) -> list[str]:
    region = widget.region
    strips = app.screen._compositor.render_strips()
    return [
        "".join(segment.text for segment in strip)[region.x : region.right].rstrip()
        for strip in strips[region.y : region.bottom]
    ]


@pytest.mark.parametrize("width", [32, 57, 100])
@pytest.mark.parametrize("theme", ["textual-dark", "ansi-dark"])
async def test_final_reply_matches_streamed_markdown(width: int, theme: str) -> None:
    """A finished reply shows the same rows as Textual's `Markdown` did.

    Every reply streams into Textual's `Markdown`, then swaps to
    `FinalMarkdown`. Equal rows mean equal heights, so the transcript does
    not jump when a reply finishes. Setting the theme after mount also
    checks that an ANSI theme switch re-lays out fences.
    """
    app = _ReplyApp()
    async with app.run_test(size=(width, 1000)) as pilot:
        app.theme = theme
        replies = {
            name: app.query_one(f"#reply-{name}", AssistantMessage) for name in _CORPUS
        }
        for name, reply in replies.items():
            await reply.append_content(_CORPUS[name])
        await pilot.pause()
        streamed = {name: _screen_lines(app, reply) for name, reply in replies.items()}

        for reply in replies.values():
            await reply.stop_stream()
        await pilot.pause()

        for name, reply in replies.items():
            assert reply.query_one(FinalMarkdown)
            assert _screen_lines(app, reply) == streamed[name], name


class _FinalApp(App[None]):
    def __init__(self, source: str) -> None:
        super().__init__()
        self.source = source

    def compose(self) -> ComposeResult:
        yield FinalMarkdown(self.source)


async def test_encoded_link_opens_original_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Clicking a finished link preserves encoded URL delimiters."""
    url = "https://example.com/a%2Fb?q=a%26b%23c&signature=x%3Dy"
    opener = AsyncMock(return_value=True)
    monkeypatch.setattr(
        "deepagents_code.tui.widgets.final_markdown._widget.open_checked_url_async",
        opener,
    )
    app = _FinalApp(f"[Open]({url})")
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.click(FinalMarkdown, offset=(1, 0))
        await pilot.pause()
        opener.assert_awaited_once_with(url, app=app, notify_on_success=True)


@pytest.mark.parametrize("width", [6, 12, 17])
@pytest.mark.parametrize("theme", ["textual-dark", "ansi-dark"])
async def test_wide_code_preserves_rendered_and_copied_characters(
    width: int, theme: str
) -> None:
    """Code folds at cell boundaries without losing wide characters."""
    code = "中文字符测试一些宽字符的文本用于测试换行\nprint('中文e\u0301')"
    app = _FinalApp(f"```python\n{code}\n```")
    async with app.run_test(size=(width, 100)) as pilot:
        app.theme = theme
        await pilot.pause()
        widget = app.query_one(FinalMarkdown)
        rows = widget.rows_for(width)
        assert all(row.cell_length <= width for row in rows)
        assert "".join(row.text for row in rows) == code.replace("\n", "")
        assert "".join(line.strip() for line in _screen_lines(app, widget)) == (
            code.replace("\n", "")
        )
        assert widget.get_selection(Selection(None, None)) == (code, "\n")


async def test_copy_rejoins_wrapped_lines_and_drops_padding() -> None:
    """Copied text reads like the reply, not like its wrapped layout."""
    source = (
        "A paragraph that wraps across several rows of this narrow widget.\n\n"
        "- item that also wraps onto a second row\n  - nested\n\n"
        "```\nshort\nabcdefghijklmnopqrstuvwxyz0123456789\n```\n\n## Done"
    )
    app = _FinalApp(source)
    async with app.run_test(size=(24, 40)) as pilot:
        await pilot.pause()
        widget = app.query_one(FinalMarkdown)

        copied = widget.get_selection(Selection(None, None))

    expected = (
        "A paragraph that wraps across several rows of this narrow widget.\n\n"
        "• item that also wraps onto a second row\n▪ nested\n\n"
        "short\nabcdefghijklmnopqrstuvwxyz0123456789\n\nDone"
    )
    assert copied == (expected, "\n")


async def test_partial_selection_copies_only_selected_text() -> None:
    """A drag inside one wrapped paragraph copies just the selected span."""
    app = _FinalApp("alpha beta gamma delta")
    async with app.run_test(size=(12, 10)) as pilot:
        await pilot.pause()
        widget = app.query_one(FinalMarkdown)
        assert widget.selectable_text.splitlines() == ["alpha beta", "gamma delta"]

        copied = widget.get_selection(
            Selection.from_offsets(Offset(6, 0), Offset(5, 1))
        )

    assert copied == ("beta gamma", "\n")


class _OneReplyApp(App[None]):
    def compose(self) -> ComposeResult:
        yield AssistantMessage()


def _backgrounds(app: App[None], widget: Widget) -> list[list[str]]:
    region = widget.content_region
    strips = app.screen._compositor.render_strips()
    rows = []
    for strip in strips[region.y : region.bottom]:
        styles = [
            next(iter(strip.crop(x, x + 1))).style
            for x in range(region.x, region.right)
        ]
        rows.append([str(style.bgcolor if style else None) for style in styles])
    return rows


async def test_light_theme_backgrounds_match_streamed_markdown() -> None:
    """Fences and tables keep Textual's light `white 30%` background."""
    app = _OneReplyApp()
    async with app.run_test(size=(30, 40)) as pilot:
        app.theme = "textual-light"
        reply = app.query_one(AssistantMessage)
        await reply.append_content("```\ncode\n```\n\n| a |\n|---|\n| 1 |\n\nend")
        await pilot.pause()
        streamed = _backgrounds(app, reply)

        await reply.stop_stream()
        await pilot.pause()

        assert _backgrounds(app, reply) == streamed
