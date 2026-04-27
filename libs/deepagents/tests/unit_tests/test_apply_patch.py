"""Unit tests for the V4A diff parser and applier."""

from __future__ import annotations

import pytest

from deepagents.utils._apply_patch import (
    PatchError,
    _apply_chunks,
    _Chunk,
    apply_patch,
    list_referenced_files,
)


class TestAddFile:
    """Tests for *** Add File operations."""

    def test_single_line(self) -> None:
        patch = "*** Begin Patch\n*** Add File: /app/hello.py\n+print('hello')\n*** End Patch"
        result = apply_patch(patch, file_reader=lambda _p: None)
        assert result.changes == {"/app/hello.py": "print('hello')"}

    def test_multiple_lines(self) -> None:
        patch = "*** Begin Patch\n*** Add File: /app/main.py\n+def main():\n+    print('hello')\n+\n+main()\n*** End Patch"
        result = apply_patch(patch, file_reader=lambda _p: None)
        assert result.changes == {"/app/main.py": "def main():\n    print('hello')\n\nmain()"}

    def test_duplicate_path_raises(self) -> None:
        patch = "*** Begin Patch\n*** Add File: /app/x.py\n+a\n*** Add File: /app/x.py\n+b\n*** End Patch"
        with pytest.raises(PatchError, match="Duplicate Path"):
            apply_patch(patch, file_reader=lambda _p: None)


class TestDeleteFile:
    """Tests for *** Delete File operations."""

    def test_delete(self) -> None:
        patch = "*** Begin Patch\n*** Delete File: /app/old.py\n*** End Patch"
        result = apply_patch(
            patch,
            file_reader=lambda p: "content" if p == "/app/old.py" else None,
        )
        assert result.changes == {"/app/old.py": None}

    def test_delete_missing_file_raises(self) -> None:
        patch = "*** Begin Patch\n*** Delete File: /app/missing.py\n*** End Patch"
        with pytest.raises(PatchError, match="Missing File"):
            apply_patch(patch, file_reader=lambda _p: None)


class TestUpdateFile:
    """Tests for *** Update File operations."""

    def test_simple_insertion(self) -> None:
        original = "line1\nline2\nline3"
        patch = "*** Begin Patch\n*** Update File: /app/test.py\n@@\n line1\n line2\n+inserted\n line3\n*** End Patch"
        result = apply_patch(patch, file_reader=lambda p: original if p == "/app/test.py" else None)
        assert result.changes == {"/app/test.py": "line1\nline2\ninserted\nline3"}

    def test_simple_deletion(self) -> None:
        original = "line1\nline2\nline3"
        patch = "*** Begin Patch\n*** Update File: /app/test.py\n@@\n line1\n-line2\n line3\n*** End Patch"
        result = apply_patch(patch, file_reader=lambda p: original if p == "/app/test.py" else None)
        assert result.changes == {"/app/test.py": "line1\nline3"}

    def test_replacement(self) -> None:
        original = "line1\nline2\nline3"
        patch = "*** Begin Patch\n*** Update File: /app/test.py\n@@\n line1\n-line2\n+replaced\n line3\n*** End Patch"
        result = apply_patch(patch, file_reader=lambda p: original if p == "/app/test.py" else None)
        assert result.changes == {"/app/test.py": "line1\nreplaced\nline3"}

    def test_anchor_jump(self) -> None:
        original = "a\nb\nc\nd\ne\nf"
        patch = "*** Begin Patch\n*** Update File: /app/test.py\n@@ d\n d\n e\n+new\n f\n*** End Patch"
        result = apply_patch(patch, file_reader=lambda p: original if p == "/app/test.py" else None)
        assert result.changes == {"/app/test.py": "a\nb\nc\nd\ne\nnew\nf"}

    def test_multiple_hunks(self) -> None:
        original = "a\nb\nc\nd\ne"
        patch = "*** Begin Patch\n*** Update File: /app/test.py\n@@\n a\n+x\n b\n@@ d\n d\n+y\n e\n*** End Patch"
        result = apply_patch(patch, file_reader=lambda p: original if p == "/app/test.py" else None)
        assert result.changes == {"/app/test.py": "a\nx\nb\nc\nd\ny\ne"}

    def test_update_missing_file_raises(self) -> None:
        patch = "*** Begin Patch\n*** Update File: /app/missing.py\n@@\n line1\n*** End Patch"
        with pytest.raises(PatchError, match="Missing File"):
            apply_patch(patch, file_reader=lambda _p: None)


class TestMoveTo:
    """Tests for the `*** Move to:` directive on `*** Update File` sections."""

    def test_move_path_captured(self) -> None:
        original = "line1\nline2"
        patch = "*** Begin Patch\n*** Update File: /app/old.py\n*** Move to: /app/new.py\n@@\n line1\n-line2\n+line2b\n*** End Patch"
        result = apply_patch(patch, file_reader=lambda p: original if p == "/app/old.py" else None)
        assert result.changes == {"/app/old.py": "line1\nline2b"}
        assert result.moves == {"/app/old.py": "/app/new.py"}

    def test_update_without_move_leaves_moves_empty(self) -> None:
        original = "line1\nline2"
        patch = "*** Begin Patch\n*** Update File: /app/old.py\n@@\n line1\n-line2\n+line2b\n*** End Patch"
        result = apply_patch(patch, file_reader=lambda p: original if p == "/app/old.py" else None)
        assert result.moves == {}

    def test_move_with_multiple_updates(self) -> None:
        files = {"/a": "a", "/b": "b"}
        patch = "*** Begin Patch\n*** Update File: /a\n*** Move to: /a2\n@@\n-a\n+aa\n*** Update File: /b\n@@\n-b\n+bb\n*** End Patch"
        result = apply_patch(patch, file_reader=files.get)
        assert result.changes == {"/a": "aa", "/b": "bb"}
        assert result.moves == {"/a": "/a2"}


class TestMultiFile:
    """Tests for patches touching multiple files."""

    def test_add_and_update(self) -> None:
        original = "old content"
        patch = (
            "*** Begin Patch\n"
            "*** Add File: /app/new.py\n"
            "+new content\n"
            "*** Update File: /app/existing.py\n"
            "@@\n"
            "-old content\n"
            "+updated content\n"
            "*** End Patch"
        )
        result = apply_patch(
            patch,
            file_reader=lambda p: original if p == "/app/existing.py" else None,
        )
        assert result.changes == {
            "/app/new.py": "new content",
            "/app/existing.py": "updated content",
        }

    def test_add_delete_update(self) -> None:
        files = {"/app/keep.py": "keep\nthis", "/app/remove.py": "bye"}
        patch = (
            "*** Begin Patch\n"
            "*** Add File: /app/brand_new.py\n"
            "+hello\n"
            "*** Delete File: /app/remove.py\n"
            "*** Update File: /app/keep.py\n"
            "@@\n"
            " keep\n"
            "-this\n"
            "+that\n"
            "*** End Patch"
        )
        result = apply_patch(patch, file_reader=files.get)
        assert result.changes == {
            "/app/brand_new.py": "hello",
            "/app/remove.py": None,
            "/app/keep.py": "keep\nthat",
        }


class TestFuzzyMatching:
    """Tests for whitespace-tolerant context matching."""

    def test_exact_match_has_zero_fuzz(self) -> None:
        original = "line1\nline2\nline3"
        patch = "*** Begin Patch\n*** Update File: /app/test.py\n@@\n line1\n+inserted\n line2\n line3\n*** End Patch"
        result = apply_patch(patch, file_reader=lambda p: original if p == "/app/test.py" else None)
        assert result.fuzz == 0

    def test_trailing_whitespace_mismatch(self) -> None:
        original = "line1  \nline2\nline3"
        patch = "*** Begin Patch\n*** Update File: /app/test.py\n@@\n line1\n line2\n+inserted\n line3\n*** End Patch"
        result = apply_patch(patch, file_reader=lambda p: original if p == "/app/test.py" else None)
        assert result.changes == {"/app/test.py": "line1  \nline2\ninserted\nline3"}
        assert result.fuzz == 1, "rstrip fallback should score fuzz=1 per hunk"

    def test_leading_whitespace_mismatch(self) -> None:
        original = "  line1\nline2\nline3"
        patch = "*** Begin Patch\n*** Update File: /app/test.py\n@@\n line1\n line2\n+inserted\n line3\n*** End Patch"
        result = apply_patch(patch, file_reader=lambda p: original if p == "/app/test.py" else None)
        assert result.changes == {"/app/test.py": "  line1\nline2\ninserted\nline3"}
        assert result.fuzz == 100, "strip fallback should score fuzz=100 per hunk"

    def test_multi_hunk_fuzz_accumulates(self) -> None:
        """Fuzz from each hunk must sum into the aggregate score."""
        original = "  a\nb\nc\n  d\ne"
        patch = "*** Begin Patch\n*** Update File: /app/test.py\n@@\n a\n+x\n b\n@@ d\n d\n+y\n e\n*** End Patch"
        result = apply_patch(patch, file_reader=lambda p: original if p == "/app/test.py" else None)
        assert result.fuzz >= 100, "leading-space hunks must accumulate fuzz"


class TestErrorHandling:
    """Tests for invalid patch formats."""

    def test_missing_begin_patch(self) -> None:
        with pytest.raises(PatchError, match="Begin Patch"):
            apply_patch("*** End Patch", file_reader=lambda _p: None)

    def test_missing_end_patch(self) -> None:
        with pytest.raises(PatchError, match="End Patch"):
            apply_patch(
                "*** Begin Patch\n*** Add File: /x\n+y\n",
                file_reader=lambda _p: None,
            )

    def test_invalid_add_file_line(self) -> None:
        patch = "*** Begin Patch\n*** Add File: /app/bad.py\nnot prefixed with plus\n*** End Patch"
        with pytest.raises(PatchError, match="Invalid Add File Line"):
            apply_patch(patch, file_reader=lambda _p: None)

    def test_empty_patch(self) -> None:
        patch = "*** Begin Patch\n*** End Patch"
        result = apply_patch(patch, file_reader=lambda _p: None)
        assert result.changes == {}


class TestLineEndingNormalization:
    """CRLF and lone-CR line endings must be normalized at the entry point."""

    def test_crlf_patch_text_add_file(self) -> None:
        patch = "*** Begin Patch\r\n*** Add File: /app/hello.py\r\n+print('hello')\r\n*** End Patch\r\n"
        result = apply_patch(patch, file_reader=lambda _p: None)
        assert result.changes == {"/app/hello.py": "print('hello')"}

    def test_crlf_patch_text_update_file(self) -> None:
        """CRLF must not leak into extracted paths or section markers."""
        original = "line1\nline2\nline3"
        patch = "*** Begin Patch\r\n*** Update File: /app/test.py\r\n@@\r\n line1\r\n line2\r\n+inserted\r\n line3\r\n*** End Patch\r\n"
        seen: list[str] = []

        def reader(path: str) -> str | None:
            seen.append(path)
            return original if path == "/app/test.py" else None

        result = apply_patch(patch, file_reader=reader)
        assert seen == ["/app/test.py"], "path must not carry a trailing \\r"
        assert result.changes == {"/app/test.py": "line1\nline2\ninserted\nline3"}

    def test_crlf_file_content_update(self) -> None:
        """File content with CRLF must be matched by patches authored with LF."""
        original_crlf = "line1\r\nline2\r\nline3"
        patch = "*** Begin Patch\n*** Update File: /app/test.py\n@@\n line1\n line2\n+inserted\n line3\n*** End Patch"
        result = apply_patch(
            patch,
            file_reader=lambda p: original_crlf if p == "/app/test.py" else None,
        )
        assert result.changes == {"/app/test.py": "line1\nline2\ninserted\nline3"}

    def test_lone_cr_line_endings(self) -> None:
        """Classic-Mac CR line endings must also normalize."""
        patch = "*** Begin Patch\r*** Add File: /app/hello.py\r+print('hello')\r*** End Patch"
        result = apply_patch(patch, file_reader=lambda _p: None)
        assert result.changes == {"/app/hello.py": "print('hello')"}


class TestListReferencedFiles:
    """list_referenced_files shares prefix constants with apply_patch."""

    def test_extracts_update_and_delete(self) -> None:
        patch = "*** Begin Patch\n*** Update File: /app/a.py\n@@\n line\n*** Delete File: /app/b.py\n*** Add File: /app/c.py\n+new\n*** End Patch"
        assert list_referenced_files(patch) == ["/app/a.py", "/app/b.py"]

    def test_ignores_add_file(self) -> None:
        patch = "*** Begin Patch\n*** Add File: /app/only_add.py\n+hello\n*** End Patch"
        assert list_referenced_files(patch) == []

    def test_empty_patch(self) -> None:
        assert list_referenced_files("*** Begin Patch\n*** End Patch") == []

    def test_handles_crlf(self) -> None:
        r"""Extraction must not return paths with trailing \r."""
        patch = "*** Begin Patch\r\n*** Update File: /app/a.py\r\n*** Delete File: /app/b.py\r\n*** End Patch\r\n"
        assert list_referenced_files(patch) == ["/app/a.py", "/app/b.py"]

    def test_matches_parser_for_update(self) -> None:
        """Every path the parser reads must appear in list_referenced_files."""
        files = {"/app/keep.py": "keep\nthis", "/app/remove.py": "bye"}
        patch = (
            "*** Begin Patch\n"
            "*** Add File: /app/brand_new.py\n"
            "+hello\n"
            "*** Delete File: /app/remove.py\n"
            "*** Update File: /app/keep.py\n"
            "@@\n"
            " keep\n"
            "-this\n"
            "+that\n"
            "*** End Patch"
        )

        prescan = set(list_referenced_files(patch))
        observed: set[str] = set()

        def reader(path: str) -> str | None:
            observed.add(path)
            return files.get(path)

        apply_patch(patch, file_reader=reader)
        assert observed.issubset(prescan), "parser requested a path the prescan did not surface"


class TestEndOfFileMarker:
    """The ``*** End of File`` marker anchors hunks to the tail of a file."""

    def test_matches_at_end_of_file(self) -> None:
        original = "alpha\nbeta\ngamma"
        patch = "*** Begin Patch\n*** Update File: /app/test.py\n@@\n beta\n gamma\n+delta\n*** End of File\n*** End Patch"
        result = apply_patch(patch, file_reader=lambda p: original if p == "/app/test.py" else None)
        assert result.changes == {"/app/test.py": "alpha\nbeta\ngamma\ndelta"}

    def test_stale_eof_raises(self) -> None:
        """If the EOF context matches earlier but not at EOF, the patch is stale."""
        # Patch claims the context ``alpha\nbeta`` is at EOF, but the real
        # file has ``gamma`` after beta — so the model was working from a
        # stale view. The parser used to silently apply this with a
        # ``+10000`` fuzz sentinel; now it should reject it outright.
        original = "alpha\nbeta\ngamma"
        patch = "*** Begin Patch\n*** Update File: /app/test.py\n@@\n alpha\n beta\n+inserted\n*** End of File\n*** End Patch"
        with pytest.raises(PatchError, match="stale"):
            apply_patch(
                patch,
                file_reader=lambda p: original if p == "/app/test.py" else None,
            )


class TestBareAnchor:
    """The bare ``@@`` separator (no anchor text) is a legal section start."""

    def test_bare_anchor_between_hunks(self) -> None:
        original = "a\nb\nc\nd\ne"
        patch = "*** Begin Patch\n*** Update File: /app/test.py\n@@\n a\n+x\n b\n@@\n d\n+y\n e\n*** End Patch"
        result = apply_patch(patch, file_reader=lambda p: original if p == "/app/test.py" else None)
        assert result.changes == {"/app/test.py": "a\nx\nb\nc\nd\ny\ne"}


class TestOverlappingChunks:
    """Positioned chunks must not overlap; the applier guards against that."""

    def test_overlap_raises(self) -> None:
        """Two chunks whose spans collide must be rejected.

        The first chunk deletes two lines starting at index 0, so the
        applier's cursor advances to 2. The second chunk is positioned
        at index 1 — already inside the first chunk's span — which must
        trigger the overlap guard rather than silently corrupting the
        output.
        """
        overlapping = [
            _Chunk(orig_index=0, del_lines=["a", "b"], ins_lines=["x"]),
            _Chunk(orig_index=1, del_lines=["b"], ins_lines=["y"]),
        ]
        with pytest.raises(PatchError, match="Overlapping"):
            _apply_chunks("a\nb\nc", overlapping)

    def test_chunk_past_eof_raises(self) -> None:
        with pytest.raises(PatchError, match="exceeds file length"):
            _apply_chunks("a\nb", [_Chunk(orig_index=99, del_lines=[], ins_lines=["x"])])


class TestFuzzSignaling:
    """EOF mismatch is a hard error; ordinary fuzz surfaces in the result."""

    def test_no_silent_eof_fallback(self) -> None:
        """The removed ``+10000`` sentinel must not reappear in ``result.fuzz``."""
        original = "a\nb\nc"
        patch = "*** Begin Patch\n*** Update File: /app/test.py\n@@\n a\n+x\n b\n c\n*** End Patch"
        result = apply_patch(patch, file_reader=lambda p: original if p == "/app/test.py" else None)
        assert result.fuzz < 10000
