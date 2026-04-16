"""Unit tests for the V4A diff parser and applier."""

from __future__ import annotations

import pytest

from deepagents.utils._apply_patch import PatchError, apply_patch


class TestAddFile:
    """Tests for *** Add File operations."""

    def test_single_line(self) -> None:
        patch = (
            "*** Begin Patch\n"
            "*** Add File: /app/hello.py\n"
            "+print('hello')\n"
            "*** End Patch"
        )
        result = apply_patch(patch, file_reader=lambda _p: None)
        assert result == {"/app/hello.py": "print('hello')"}

    def test_multiple_lines(self) -> None:
        patch = (
            "*** Begin Patch\n"
            "*** Add File: /app/main.py\n"
            "+def main():\n"
            "+    print('hello')\n"
            "+\n"
            "+main()\n"
            "*** End Patch"
        )
        result = apply_patch(patch, file_reader=lambda _p: None)
        assert result == {"/app/main.py": "def main():\n    print('hello')\n\nmain()"}

    def test_duplicate_path_raises(self) -> None:
        patch = (
            "*** Begin Patch\n"
            "*** Add File: /app/x.py\n"
            "+a\n"
            "*** Add File: /app/x.py\n"
            "+b\n"
            "*** End Patch"
        )
        with pytest.raises(PatchError, match="Duplicate Path"):
            apply_patch(patch, file_reader=lambda _p: None)


class TestDeleteFile:
    """Tests for *** Delete File operations."""

    def test_delete(self) -> None:
        patch = (
            "*** Begin Patch\n"
            "*** Delete File: /app/old.py\n"
            "*** End Patch"
        )
        result = apply_patch(
            patch,
            file_reader=lambda p: "content" if p == "/app/old.py" else None,
        )
        assert result == {"/app/old.py": None}

    def test_delete_missing_file_raises(self) -> None:
        patch = (
            "*** Begin Patch\n"
            "*** Delete File: /app/missing.py\n"
            "*** End Patch"
        )
        with pytest.raises(PatchError, match="Missing File"):
            apply_patch(patch, file_reader=lambda _p: None)


class TestUpdateFile:
    """Tests for *** Update File operations."""

    def test_simple_insertion(self) -> None:
        original = "line1\nline2\nline3"
        patch = (
            "*** Begin Patch\n"
            "*** Update File: /app/test.py\n"
            "@@\n"
            " line1\n"
            " line2\n"
            "+inserted\n"
            " line3\n"
            "*** End Patch"
        )
        result = apply_patch(
            patch, file_reader=lambda p: original if p == "/app/test.py" else None
        )
        assert result == {"/app/test.py": "line1\nline2\ninserted\nline3"}

    def test_simple_deletion(self) -> None:
        original = "line1\nline2\nline3"
        patch = (
            "*** Begin Patch\n"
            "*** Update File: /app/test.py\n"
            "@@\n"
            " line1\n"
            "-line2\n"
            " line3\n"
            "*** End Patch"
        )
        result = apply_patch(
            patch, file_reader=lambda p: original if p == "/app/test.py" else None
        )
        assert result == {"/app/test.py": "line1\nline3"}

    def test_replacement(self) -> None:
        original = "line1\nline2\nline3"
        patch = (
            "*** Begin Patch\n"
            "*** Update File: /app/test.py\n"
            "@@\n"
            " line1\n"
            "-line2\n"
            "+replaced\n"
            " line3\n"
            "*** End Patch"
        )
        result = apply_patch(
            patch, file_reader=lambda p: original if p == "/app/test.py" else None
        )
        assert result == {"/app/test.py": "line1\nreplaced\nline3"}

    def test_anchor_jump(self) -> None:
        original = "a\nb\nc\nd\ne\nf"
        patch = (
            "*** Begin Patch\n"
            "*** Update File: /app/test.py\n"
            "@@ d\n"
            " d\n"
            " e\n"
            "+new\n"
            " f\n"
            "*** End Patch"
        )
        result = apply_patch(
            patch, file_reader=lambda p: original if p == "/app/test.py" else None
        )
        assert result == {"/app/test.py": "a\nb\nc\nd\ne\nnew\nf"}

    def test_multiple_hunks(self) -> None:
        original = "a\nb\nc\nd\ne"
        patch = (
            "*** Begin Patch\n"
            "*** Update File: /app/test.py\n"
            "@@\n"
            " a\n"
            "+x\n"
            " b\n"
            "@@ d\n"
            " d\n"
            "+y\n"
            " e\n"
            "*** End Patch"
        )
        result = apply_patch(
            patch, file_reader=lambda p: original if p == "/app/test.py" else None
        )
        assert result == {"/app/test.py": "a\nx\nb\nc\nd\ny\ne"}

    def test_update_missing_file_raises(self) -> None:
        patch = (
            "*** Begin Patch\n"
            "*** Update File: /app/missing.py\n"
            "@@\n"
            " line1\n"
            "*** End Patch"
        )
        with pytest.raises(PatchError, match="Missing File"):
            apply_patch(patch, file_reader=lambda _p: None)


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
        assert result == {
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
        result = apply_patch(patch, file_reader=lambda p: files.get(p))
        assert result == {
            "/app/brand_new.py": "hello",
            "/app/remove.py": None,
            "/app/keep.py": "keep\nthat",
        }


class TestFuzzyMatching:
    """Tests for whitespace-tolerant context matching."""

    def test_trailing_whitespace_mismatch(self) -> None:
        original = "line1  \nline2\nline3"
        patch = (
            "*** Begin Patch\n"
            "*** Update File: /app/test.py\n"
            "@@\n"
            " line1\n"
            " line2\n"
            "+inserted\n"
            " line3\n"
            "*** End Patch"
        )
        result = apply_patch(
            patch, file_reader=lambda p: original if p == "/app/test.py" else None
        )
        assert result == {"/app/test.py": "line1  \nline2\ninserted\nline3"}

    def test_leading_whitespace_mismatch(self) -> None:
        original = "  line1\nline2\nline3"
        patch = (
            "*** Begin Patch\n"
            "*** Update File: /app/test.py\n"
            "@@\n"
            " line1\n"
            " line2\n"
            "+inserted\n"
            " line3\n"
            "*** End Patch"
        )
        result = apply_patch(
            patch, file_reader=lambda p: original if p == "/app/test.py" else None
        )
        assert result == {"/app/test.py": "  line1\nline2\ninserted\nline3"}


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
        patch = (
            "*** Begin Patch\n"
            "*** Add File: /app/bad.py\n"
            "not prefixed with plus\n"
            "*** End Patch"
        )
        with pytest.raises(PatchError, match="Invalid Add File Line"):
            apply_patch(patch, file_reader=lambda _p: None)

    def test_empty_patch(self) -> None:
        patch = "*** Begin Patch\n*** End Patch"
        result = apply_patch(patch, file_reader=lambda _p: None)
        assert result == {}
