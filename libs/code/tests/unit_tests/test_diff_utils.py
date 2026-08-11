"""Tests for `deepagents_code.diff_utils`."""

from __future__ import annotations

from typing import Any

import pytest

from deepagents_code.diff_utils import (
    DIFF_TRUNCATION_MARKER,
    DiffStats,
    count_diff_change_lines,
    file_header_indexes,
    is_truncation_marker,
    split_diff_lines,
)


def _diff(*lines: str) -> str:
    return "\n".join(lines)


def _count(diff: str) -> DiffStats:
    return count_diff_change_lines(split_diff_lines(diff))


class TestSplitDiffLines:
    """The renderer has to split a diff exactly where its producer joined it."""

    @pytest.mark.parametrize(
        "separator",
        ["\u2028", "\u2029", "\v", "\f", "\x85", "\r"],
        ids=["line-sep", "para-sep", "vtab", "formfeed", "nel", "cr"],
    )
    def test_other_line_separators_stay_inside_their_line(self, separator: str) -> None:
        r"""Only `"\n"` separates diff lines; everything else is content.

        `splitlines()` breaks on all of these, which strands the tail of a line as
        a fragment with no `+`/`-` marker — content that then renders as neutral
        metadata.
        """
        line = f"+value = {separator}payload"
        assert split_diff_lines(_diff("@@ -1 +1 @@", line)) == [
            "@@ -1 +1 @@",
            line,
        ]

    def test_a_terminating_newline_adds_no_empty_line(self) -> None:
        """A trailing newline terminates the last line rather than starting one."""
        assert split_diff_lines("@@ -1 +1 @@\n+a\n") == ["@@ -1 +1 @@", "+a"]

    def test_a_blank_line_inside_the_diff_is_kept(self) -> None:
        """Only the trailing empty entry is dropped, never an interior one."""
        assert split_diff_lines("@@ -1 +1 @@\n\n+a") == ["@@ -1 +1 @@", "", "+a"]

    def test_counts_follow_the_same_boundary(self) -> None:
        """A separator inside a line must not be counted as a second line."""
        assert _count("@@ -1 +1 @@\n+a b") == DiffStats(additions=1, deletions=0)


class TestFileHeaderIndexes:
    """Header detection across the diff shapes the renderers actually see."""

    def test_single_file_header_pair_is_found(self) -> None:
        lines = _diff("--- a/x.py", "+++ b/x.py", "@@ -1,1 +1,1 @@", "-a", "+b").split(
            "\n"
        )
        assert file_header_indexes(lines) == {0, 1}

    def test_multi_file_diff_finds_every_header_pair(self) -> None:
        """Headers recur between files; each pair must be recognized."""
        lines = _diff(
            "--- a/x.py",
            "+++ b/x.py",
            "@@ -1,1 +1,1 @@",
            "-a",
            "+b",
            "--- a/y.py",
            "+++ b/y.py",
            "@@ -1,1 +1,1 @@",
            "-c",
            "+d",
        ).split("\n")
        assert file_header_indexes(lines) == {0, 1, 5, 6}

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

    @pytest.mark.parametrize(
        ("header", "expected"),
        [
            ("@@ -1 +1 @@", {0, 1}),  # counts omitted, default to 1
            ("@@ -1,1 +1,1 @@", {0, 1}),
            ("@@ -0,0 +1,3 @@", {0, 1}),  # new file
            ("@@ -1,3 +0,0 @@", {0, 1}),  # deleted file
        ],
    )
    def test_hunk_header_count_forms(self, header: str, expected: set[int]) -> None:
        lines = _diff("--- a/x.py", "+++ b/x.py", header, "-a", "+b").split("\n")
        assert file_header_indexes(lines) == expected

    def test_no_newline_marker_is_not_a_header(self) -> None:
        lines = _diff(
            "--- a/x.py",
            "+++ b/x.py",
            "@@ -1,1 +1,1 @@",
            "-a",
            "\\ No newline at end of file",
            "+b",
        ).split("\n")
        assert file_header_indexes(lines) == {0, 1}

    def test_an_over_declared_budget_does_not_swallow_the_next_file(self) -> None:
        """A hunk header may claim more lines than its body actually has.

        `difflib` always declares exact counts, so neither in-repo producer
        reaches this — but the module documents multi-file support and invites
        other producers. With the budget still open, `--- a/y.py` was consumed
        as a deletion: the second file's headers went unrecognized, rendered as
        source rows, and counted as changes.
        """
        lines = _diff(
            "--- a/x.py",
            "+++ b/x.py",
            "@@ -1,5 +1,5 @@",  # claims five, supplies one of each
            "-a",
            "+b",
            "--- a/y.py",
            "+++ b/y.py",
            "@@ -1,1 +1,1 @@",
            "-c",
            "+d",
        ).split("\n")

        assert file_header_indexes(lines) == {0, 1, 5, 6}
        assert _count("\n".join(lines)) == DiffStats(additions=2, deletions=2)

    def test_header_pair_not_followed_by_a_hunk_is_ignored(self) -> None:
        """Only a pair immediately preceding a hunk is metadata."""
        assert file_header_indexes(["--- a/x.py", "+++ b/x.py"]) == set()

    def test_empty_input(self) -> None:
        assert file_header_indexes([]) == set()


class TestCountDiffChanges:
    """Change counting, which feeds both the header stats and the tracker."""

    def test_counts_exclude_file_headers(self) -> None:
        diff = _diff("--- a/x.py", "+++ b/x.py", "@@ -1,2 +1,2 @@", " keep", "-a", "+b")
        assert _count(diff) == DiffStats(additions=1, deletions=1)

    def test_counts_span_multiple_files(self) -> None:
        diff = _diff(
            "--- a/x.py",
            "+++ b/x.py",
            "@@ -1,1 +1,1 @@",
            "-a",
            "+b",
            "--- a/y.py",
            "+++ b/y.py",
            "@@ -1,1 +1,1 @@",
            "-c",
            "+d",
        )
        assert _count(diff) == DiffStats(additions=2, deletions=2)

    def test_content_resembling_headers_is_counted(self) -> None:
        diff = _diff(
            "--- a/x.md",
            "+++ b/x.md",
            "@@ -1,1 +1,1 @@",
            "-+++ old marker",
            "++++ new marker",
        )
        assert _count(diff) == DiffStats(additions=1, deletions=1)

    def test_empty_diff_counts_nothing(self) -> None:
        assert _count("") == DiffStats(additions=0, deletions=0)

    def test_result_is_a_named_pair(self) -> None:
        """Fields are named so callers can't silently swap the order."""
        stats = count_diff_change_lines(
            split_diff_lines(
                _diff("--- a/x.py", "+++ b/x.py", "@@ -1,3 +1,2 @@", "-a", "-b", "+c")
            )
        )
        assert stats.additions == 1
        assert stats.deletions == 2

    def test_counts_cannot_be_built_positionally(self) -> None:
        """The type's whole claim is that the pair cannot be transposed.

        A `NamedTuple` would accept `DiffStats(1, 2)` and let a producer emit the
        counts backwards — which reads as a plausible `+2 -1` and is the one
        error nothing downstream can detect.
        """
        # Routed through `Any` so the call reaches the runtime check; the point
        # is that it raises rather than that a checker rejects it, because the
        # producers this guards are not all type-checked at their call site.
        positional: Any = DiffStats
        with pytest.raises(TypeError):
            positional(1, 2)

    @pytest.mark.parametrize(("additions", "deletions"), [(-1, 0), (0, -1), (-1, -1)])
    def test_negative_counts_are_rejected(self, additions: int, deletions: int) -> None:
        """The type is public and its numbers gate destroying a file."""
        with pytest.raises(ValueError, match="cannot be negative"):
            DiffStats(additions=additions, deletions=deletions)


class TestIsTruncationMarker:
    """One predicate, so the renderer and the recount cannot disagree."""

    def test_the_bare_marker_matches(self) -> None:
        assert is_truncation_marker(DIFF_TRUNCATION_MARKER)

    def test_a_context_line_whose_text_is_the_marker_does_not(self) -> None:
        """A source line reading `...` arrives here with its space prefix.

        Stripping would call that a clipped body and suppress the counts for a
        diff that is complete.
        """
        assert not is_truncation_marker(f" {DIFF_TRUNCATION_MARKER}")

    @pytest.mark.parametrize("line", ["+...", "-...", "....", "", "..."[:2]])
    def test_near_misses_do_not_match(self, line: str) -> None:
        assert not is_truncation_marker(line)
