"""Tests for the `context_lines` parameter on grep (#3109)."""

from pathlib import Path

import pytest

from deepagents.backends import filesystem as filesystem_module
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends.utils import (
    MAX_GREP_CONTEXT_LINES,
    clamp_context_lines,
    context_window_from_lines,
    format_grep_matches,
    grep_matches_from_files,
)

FILE_CONTENT = "line one\nline two\ntarget line\nline four\nline five"


def _files() -> dict[str, dict[str, str]]:
    return {"/notes.txt": {"content": FILE_CONTENT, "encoding": "utf-8"}}


def test_grep_matches_from_files_default_has_no_context_keys() -> None:
    matches = grep_matches_from_files(_files(), "target").matches
    assert matches is not None and len(matches) == 1
    assert "context_before" not in matches[0]
    assert "context_after" not in matches[0]


def test_grep_matches_from_files_with_context() -> None:
    matches = grep_matches_from_files(_files(), "target", context_lines=2).matches
    assert matches is not None and len(matches) == 1
    match = matches[0]
    assert match["line"] == 3
    assert match["context_before"] == ["line one", "line two"]
    assert match["context_after"] == ["line four", "line five"]


def test_grep_context_clamped_at_file_boundaries() -> None:
    files = {"/short.txt": {"content": "only\ntwo", "encoding": "utf-8"}}
    matches = grep_matches_from_files(files, "only", context_lines=5).matches
    assert matches is not None and len(matches) == 1
    assert matches[0]["context_before"] == []
    assert matches[0]["context_after"] == ["two"]


def test_clamp_context_lines_bounds() -> None:
    assert clamp_context_lines(-3) == 0
    assert clamp_context_lines(0) == 0
    assert clamp_context_lines(MAX_GREP_CONTEXT_LINES + 50) == MAX_GREP_CONTEXT_LINES


def test_context_window_from_lines_first_and_last_line() -> None:
    lines = ["a", "b", "c"]
    assert context_window_from_lines(lines, 1, 2) == ([], ["b", "c"])
    assert context_window_from_lines(lines, 3, 2) == (["a", "b"], [])


@pytest.mark.parametrize("force_python_fallback", [False, True])
def test_filesystem_backend_grep_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, force_python_fallback: bool) -> None:
    """Context attaches identically on the ripgrep path and the Python fallback."""
    if force_python_fallback:
        monkeypatch.setattr(filesystem_module, "_resolve_ripgrep_path", lambda: None)

    (tmp_path / "code.py").write_text(FILE_CONTENT)
    be = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)

    matches = be.grep("target", path="/", context_lines=1).matches
    assert matches is not None and len(matches) == 1
    assert matches[0]["context_before"] == ["line two"]
    assert matches[0]["context_after"] == ["line four"]

    # default stays context-free
    plain = be.grep("target", path="/").matches
    assert plain is not None and "context_before" not in plain[0]


def test_format_grep_matches_content_mode_with_context() -> None:
    matches = grep_matches_from_files(_files(), "target", context_lines=1).matches
    assert matches is not None
    formatted = format_grep_matches(matches, "content")
    assert formatted.splitlines() == [
        "/notes.txt:",
        "  2- line two",
        "  3: target line",
        "  4- line four",
    ]


def test_format_grep_matches_without_context_unchanged() -> None:
    matches = grep_matches_from_files(_files(), "target").matches
    assert matches is not None
    assert format_grep_matches(matches, "content") == "/notes.txt:\n  3: target line"
    assert format_grep_matches(matches, "files_with_matches") == "/notes.txt"
