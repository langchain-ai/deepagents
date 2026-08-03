"""Tests for `deepagents_code.diff_utils`."""

from __future__ import annotations

import pytest

from deepagents_code.diff_utils import (
    DiffStats,
    count_diff_changes,
    file_header_indexes,
)


def _diff(*lines: str) -> str:
    return "\n".join(lines)


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

    def test_header_pair_not_followed_by_a_hunk_is_ignored(self) -> None:
        """Only a pair immediately preceding a hunk is metadata."""
        assert file_header_indexes(["--- a/x.py", "+++ b/x.py"]) == set()

    def test_empty_input(self) -> None:
        assert file_header_indexes([]) == set()


class TestCountDiffChanges:
    """Change counting, which feeds both the header stats and the tracker."""

    def test_counts_exclude_file_headers(self) -> None:
        diff = _diff("--- a/x.py", "+++ b/x.py", "@@ -1,2 +1,2 @@", " keep", "-a", "+b")
        assert count_diff_changes(diff) == DiffStats(additions=1, deletions=1)

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
        assert count_diff_changes(diff) == DiffStats(additions=2, deletions=2)

    def test_content_resembling_headers_is_counted(self) -> None:
        diff = _diff(
            "--- a/x.md",
            "+++ b/x.md",
            "@@ -1,1 +1,1 @@",
            "-+++ old marker",
            "++++ new marker",
        )
        assert count_diff_changes(diff) == DiffStats(additions=1, deletions=1)

    def test_empty_diff_counts_nothing(self) -> None:
        assert count_diff_changes("") == DiffStats(additions=0, deletions=0)

    def test_result_is_a_named_pair(self) -> None:
        """Fields are named so callers can't silently swap the order."""
        stats = count_diff_changes(
            _diff("--- a/x.py", "+++ b/x.py", "@@ -1,3 +1,2 @@", "-a", "-b", "+c")
        )
        assert stats.additions == 1
        assert stats.deletions == 2
        assert tuple(stats) == (1, 2)
