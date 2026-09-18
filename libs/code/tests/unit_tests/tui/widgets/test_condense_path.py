"""Unit tests for width-aware path condensation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from rich.cells import cell_len

from deepagents_code.config import reset_glyphs_cache
from deepagents_code.tui.widgets._condense_path import condense_path

if TYPE_CHECKING:
    from collections.abc import Iterator

LONG = "/home/user/projects/deepagents/libs/code/deepagents_code/tui/widgets"


@pytest.fixture(autouse=True)
def unicode_glyphs(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Pin the ellipsis glyph so widths are independent of the environment."""
    monkeypatch.setenv("UI_CHARSET_MODE", "unicode")
    reset_glyphs_cache()
    yield
    reset_glyphs_cache()


class TestFitting:
    """Paths that already fit are returned verbatim."""

    def test_short_path_unchanged(self) -> None:
        """A path narrower than the budget is untouched."""
        assert condense_path("/usr/local/bin", 40) == "/usr/local/bin"

    def test_exact_width_fit_unchanged(self) -> None:
        """A path exactly as wide as the budget is untouched."""
        path = "/usr/local/bin"
        assert condense_path(path, len(path)) == path

    def test_trailing_slash_preserved_when_it_fits(self) -> None:
        """Nothing is normalized away while the path fits."""
        assert condense_path("/usr/local/bin/", 40) == "/usr/local/bin/"

    def test_filesystem_root(self) -> None:
        """The root has no components to condense."""
        assert condense_path("/", 10) == "/"

    def test_prefix_counts_against_width(self) -> None:
        """The prefix is returned with the path and shares the budget."""
        assert (
            condense_path("/usr/local/bin", 20, prefix="cwd: ") == "cwd: /usr/local/bin"
        )
        assert cell_len(condense_path(LONG, 20, prefix="cwd: ")) <= 20


class TestCondensation:
    """Directory components collapse from the middle outward."""

    def test_middle_component_goes_first(self) -> None:
        """The first component dropped is the middle one, not the tail."""
        assert condense_path("/usr/local/share/foo/bar", 22) == "/usr/…/share/foo/bar"

    def test_condenses_progressively_toward_the_tail(self) -> None:
        """Narrower budgets keep dropping components nearer the front."""
        widths = [50, 40, 30, 20, 12]
        results = [condense_path(LONG, width) for width in widths]
        assert results == [
            "/home/user/…/code/deepagents_code/tui/widgets",
            "/home/user/…/deepagents_code/tui/widgets",
            "/home/…/tui/widgets",
            "/home/…/tui/widgets",
            "/…/widgets",
        ]
        for width, result in zip(widths, results, strict=True):
            assert cell_len(result) <= width

    def test_trailing_slash_condenses(self) -> None:
        """A trailing slash does not produce an empty component."""
        assert condense_path("/usr/local/share/foo/bar/", 22) == "/usr/…/share/foo/bar"

    def test_home_relative_path(self) -> None:
        """`~` is an ordinary leading component."""
        assert condense_path("~/projects/deepagents/libs/code", 20) == "~/…/libs/code"

    def test_relative_path_keeps_no_root(self) -> None:
        """A relative path never grows a leading slash."""
        assert condense_path("projects/deepagents/libs/code", 16) == "…/libs/code"


class TestNarrowWidths:
    """The last component survives longest, then degrades gracefully."""

    def test_last_component_survives_alone(self) -> None:
        """Once every directory is dropped, the bare tail is shown."""
        assert condense_path(LONG, 8) == "widgets"

    def test_last_component_truncates_only_at_the_end(self) -> None:
        """Below the tail's own width the tail itself is end-truncated."""
        assert condense_path(LONG, 5) == "widg…"

    def test_very_narrow_width(self) -> None:
        """A single cell leaves room for the ellipsis alone."""
        assert condense_path(LONG, 1) == "…"

    def test_zero_width_is_empty(self) -> None:
        """Nothing is rendered when there is no room at all."""
        assert condense_path(LONG, 0) == ""
        assert condense_path(LONG, -3) == ""

    def test_single_component_path(self) -> None:
        """A one-component path drops its root before truncating."""
        assert condense_path("/deepagents_code", 12) == "deepagents_…"

    def test_ascii_charset_uses_ascii_ellipsis(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """In ASCII mode the wider `"..."` marker is used and still fits."""
        monkeypatch.setenv("UI_CHARSET_MODE", "ascii")
        reset_glyphs_cache()
        result = condense_path(LONG, 20)
        assert "\u2026" not in result
        assert "..." in result
        assert cell_len(result) <= 20
