"""Tests for `deepagents_code.diff_utils`."""

from __future__ import annotations

from deepagents_code.diff_utils import file_header_indexes


def _diff(*lines: str) -> str:
    return "\n".join(lines)


class TestSplitDiffLines:
    """The renderer has to split a diff exactly where its producer joined it."""


class TestFileHeaderIndexes:
    """Header detection across the diff shapes the renderers actually see."""

    def test_content_lines_that_look_like_headers_are_not_headers(self) -> None:
        """A diff of a file containing `---`/`+++` must keep them as content.

        This is the regression the hunk line-budget tracking exists for; a
        naive `startswith("+++")` check miscounts here.
        """
        lines = _diff(
            "--- a/x.md",
            "+++ b/x.md",
            "@@ -1,2 +1,2 @@",
            "---- old rule",
            "-+++ old marker",
            "++++ new marker",
            "+---- new rule",
        ).split("\n")
        assert file_header_indexes(lines) == {0, 1}


class TestCountDiffChanges:
    """Change counting, which feeds both the header stats and the tracker."""


class TestIsTruncationMarker:
    """One predicate, so the renderer and the recount cannot disagree."""
