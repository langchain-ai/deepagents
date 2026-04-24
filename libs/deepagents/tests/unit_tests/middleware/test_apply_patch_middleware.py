"""Unit tests for `_ApplyPatchMiddleware` initialization and apply semantics.

Construction-time behavior (tool wiring, description, default backend)
lives in `TestApplyPatchMiddlewareInit`. End-to-end apply semantics for
`Add`, `Update`, `Delete`, and `Move to` operations run against a real
`FilesystemBackend` rooted at a `tmp_path` so parser + backend integration
is exercised without a LangGraph context. Path-validation edge cases live
in `test_apply_patch_path_validation.py`; parser edge cases live in
`test_apply_patch.py`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain.tools import ToolRuntime

from deepagents.backends import StateBackend
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends.protocol import (
    BackendProtocol,
    EditResult,
    FileData,
    ReadResult,
    WriteResult,
)
from deepagents.middleware._apply_patch import (
    APPLY_PATCH_TOOL_DESCRIPTION,
    _aapply_file_changes,
    _apply_file_changes,
    _ApplyPatchMiddleware,
)
from deepagents.middleware.filesystem import FilesystemState

if TYPE_CHECKING:
    from pathlib import Path


def _runtime() -> ToolRuntime:
    """Minimal `ToolRuntime` for invoking the `apply_patch` tool directly."""
    return ToolRuntime(
        state={},
        context=None,
        tool_call_id="",
        store=None,
        stream_writer=lambda _: None,
        config={},
    )


class TestApplyPatchMiddlewareInit:
    """Tests covering the middleware's public construction surface."""

    def test_registers_single_apply_patch_tool(self) -> None:
        """The middleware exposes exactly one tool, named `apply_patch`."""
        mw = _ApplyPatchMiddleware(backend=StateBackend())
        assert [t.name for t in mw.tools] == ["apply_patch"]

    def test_default_tool_description(self) -> None:
        """Without an override, the tool uses `APPLY_PATCH_TOOL_DESCRIPTION`."""
        mw = _ApplyPatchMiddleware(backend=StateBackend())
        tool = mw.tools[0]
        assert tool.description == APPLY_PATCH_TOOL_DESCRIPTION.rstrip()

    def test_custom_description_overrides_default(self) -> None:
        """`custom_description` replaces the built-in description verbatim."""
        mw = _ApplyPatchMiddleware(
            backend=StateBackend(),
            custom_description="Custom desc",
        )
        tool = mw.tools[0]
        assert tool.description == "Custom desc"

    def test_reuses_filesystem_state_schema(self) -> None:
        """The state schema must be `FilesystemState` so cooperating middleware share a view."""
        mw = _ApplyPatchMiddleware(backend=StateBackend())
        assert mw.state_schema is FilesystemState

    def test_default_backend_is_state_backend(self) -> None:
        """Omitting `backend` installs a fresh `StateBackend()` — parity with `FilesystemMiddleware`."""
        mw = _ApplyPatchMiddleware()
        assert isinstance(mw.backend, StateBackend)

    def test_provided_backend_is_retained(self) -> None:
        """An explicit backend instance is stored on the middleware without wrapping."""
        backend = StateBackend()
        mw = _ApplyPatchMiddleware(backend=backend)
        assert mw.backend is backend


def _fs_backend(tmp_path: Path) -> FilesystemBackend:
    """Build a real filesystem backend rooted at `tmp_path`.

    `virtual_mode=False` is sufficient here — tests only use paths under
    `tmp_path`, and the applier normalizes to absolute paths via
    `validate_path`. Keeping it off makes the test output easier to read.
    """
    return FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False, max_file_size_mb=1)


class TestApplyFileChangesDelete:
    """`*** Delete File:` must remove the file from the backend."""

    def test_delete_removes_file_sync(self, tmp_path: Path) -> None:
        backend = _fs_backend(tmp_path)
        target = tmp_path / "doomed.txt"
        target.write_text("bye")

        output = _apply_file_changes(backend, {str(target): None})

        assert not target.exists()
        assert f"Deleted '{target}'" in output

    async def test_delete_removes_file_async(self, tmp_path: Path) -> None:
        backend = _fs_backend(tmp_path)
        target = tmp_path / "doomed.txt"
        target.write_text("bye")

        output = await _aapply_file_changes(backend, {str(target): None})

        assert not target.exists()
        assert f"Deleted '{target}'" in output

    def test_delete_missing_file_reports_error(self, tmp_path: Path) -> None:
        backend = _fs_backend(tmp_path)
        missing = tmp_path / "never.txt"

        output = _apply_file_changes(backend, {str(missing): None})

        assert "Error deleting" in output
        assert "not found" in output

    def test_delete_unsupported_backend_surfaces_model_friendly_error(self, tmp_path: Path) -> None:
        """Backends without `delete` must surface a readable error, not crash."""
        del tmp_path  # fixture required by pytest-tmp_path but unused here

        class _NoDelete(BackendProtocol):
            def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
                return ReadResult(error="not found")

        output = _apply_file_changes(_NoDelete(), {"/x.txt": None})
        assert "does not support file deletion" in output


class TestApplyFileChangesMove:
    """`*** Move to:` must rename the source file while applying the patch."""

    def test_move_creates_dest_and_removes_source(self, tmp_path: Path) -> None:
        backend = _fs_backend(tmp_path)
        src = tmp_path / "old.txt"
        dest = tmp_path / "new.txt"
        src.write_text("hello")

        output = _apply_file_changes(
            backend,
            {str(src): "hello!"},
            {str(src): str(dest)},
        )

        assert not src.exists()
        assert dest.read_text() == "hello!"
        assert f"Moved '{src}' -> '{dest}'" in output

    async def test_move_creates_dest_and_removes_source_async(self, tmp_path: Path) -> None:
        backend = _fs_backend(tmp_path)
        src = tmp_path / "old.txt"
        dest = tmp_path / "new.txt"
        src.write_text("hello")

        output = await _aapply_file_changes(
            backend,
            {str(src): "hello!"},
            {str(src): str(dest)},
        )

        assert not src.exists()
        assert dest.read_text() == "hello!"
        assert f"Moved '{src}' -> '{dest}'" in output

    def test_move_to_existing_dest_errors_and_preserves_source(self, tmp_path: Path) -> None:
        backend = _fs_backend(tmp_path)
        src = tmp_path / "old.txt"
        dest = tmp_path / "new.txt"
        src.write_text("src-content")
        dest.write_text("occupied")

        output = _apply_file_changes(
            backend,
            {str(src): "patched"},
            {str(src): str(dest)},
        )

        assert src.exists(), "source must be preserved when move is rejected"
        assert src.read_text() == "src-content"
        assert dest.read_text() == "occupied"
        assert "destination already exists" in output

    def test_move_to_same_path_applies_in_place(self, tmp_path: Path) -> None:
        """Degenerate `Move to:` equal to the source path should behave like a plain update."""
        backend = _fs_backend(tmp_path)
        src = tmp_path / "same.txt"
        src.write_text("v1")

        output = _apply_file_changes(
            backend,
            {str(src): "v2"},
            {str(src): str(src)},
        )

        assert src.read_text() == "v2"
        assert f"Updated '{src}'" in output
        assert "Moved" not in output


class _AccessCountingBackend(BackendProtocol):
    """In-memory backend that records how many `read`/`edit`/`write` calls occur.

    Used to verify that the `pre_existing` shortcut in
    `_apply_file_changes` skips the probe + full re-read path inside
    `_update_or_create`.
    """

    def __init__(self, files: dict[str, str] | None = None) -> None:
        self._files: dict[str, str] = dict(files or {})
        self.reads: list[str] = []
        self.writes: list[str] = []
        self.edits: list[str] = []

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        self.reads.append(file_path)
        content = self._files.get(file_path)
        if content is None:
            return ReadResult(error=f"File not found: {file_path}")
        return ReadResult(file_data=FileData(content=content, encoding="utf-8"))

    async def aread(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        return self.read(file_path, offset, limit)

    def write(self, file_path: str, content: str) -> WriteResult:
        self.writes.append(file_path)
        if file_path in self._files:
            return WriteResult(error=f"File exists: {file_path}")
        self._files[file_path] = content
        return WriteResult(path=file_path)

    async def awrite(self, file_path: str, content: str) -> WriteResult:
        return self.write(file_path, content)

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,  # noqa: FBT001, FBT002
    ) -> EditResult:
        self.edits.append(file_path)
        current = self._files.get(file_path)
        if current is None:
            return EditResult(error=f"File not found: {file_path}")
        if old_string not in current:
            return EditResult(error=f"old_string not found in {file_path}")
        self._files[file_path] = current.replace(old_string, new_string)
        return EditResult(path=file_path, occurrences=1)

    async def aedit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,  # noqa: FBT001, FBT002
    ) -> EditResult:
        return self.edit(file_path, old_string, new_string, replace_all)


class TestPreExistingCacheShortcut:
    """`pre_existing` lets `_update_or_create` skip the probe + full re-read."""

    def test_update_with_cache_does_not_reread_sync(self) -> None:
        backend = _AccessCountingBackend({"/foo.txt": "hello"})

        output = _apply_file_changes(
            backend,
            {"/foo.txt": "world"},
            pre_existing={"/foo.txt": "hello"},
        )

        assert "Updated '/foo.txt'" in output
        assert backend._files["/foo.txt"] == "world"
        assert backend.reads == [], "probe + full read should be skipped when cache is provided"
        assert backend.edits == ["/foo.txt"]

    async def test_update_with_cache_does_not_reread_async(self) -> None:
        backend = _AccessCountingBackend({"/foo.txt": "hello"})

        output = await _aapply_file_changes(
            backend,
            {"/foo.txt": "world"},
            pre_existing={"/foo.txt": "hello"},
        )

        assert "Updated '/foo.txt'" in output
        assert backend._files["/foo.txt"] == "world"
        assert backend.reads == []
        assert backend.edits == ["/foo.txt"]

    def test_update_without_cache_falls_back_to_probe(self) -> None:
        """When no cache is provided the legacy probe-based path must still run."""
        backend = _AccessCountingBackend({"/foo.txt": "hello"})

        output = _apply_file_changes(backend, {"/foo.txt": "world"})

        assert "Updated '/foo.txt'" in output
        assert backend._files["/foo.txt"] == "world"
        assert len(backend.reads) >= 1, "probe path must still read when cache is absent"

    def test_add_file_not_in_cache_uses_probe_path(self) -> None:
        """`*** Add File:` paths are never read by the parser, so the cache won't hold them."""
        backend = _AccessCountingBackend()

        output = _apply_file_changes(
            backend,
            {"/new.txt": "hello"},
            pre_existing={},
        )

        assert "Created '/new.txt'" in output
        assert backend._files["/new.txt"] == "hello"
        assert backend.writes == ["/new.txt"]


class TestEmptyFileRegression:
    """Operations on existing but empty files must not be rejected as "missing"."""

    def test_delete_empty_file_sync(self, tmp_path: Path) -> None:
        """`*** Delete File:` on an empty existing file must succeed."""
        backend = _fs_backend(tmp_path)
        target = tmp_path / "empty.txt"
        target.write_text("")
        assert target.exists() and target.read_text() == ""

        mw = _ApplyPatchMiddleware(backend=backend)
        patch = f"*** Begin Patch\n*** Delete File: {target}\n*** End Patch"

        output = mw.tools[0].invoke({"patch": patch, "runtime": _runtime()})

        assert not target.exists()
        assert f"Deleted '{target}'" in output

    async def test_delete_empty_file_async(self, tmp_path: Path) -> None:
        backend = _fs_backend(tmp_path)
        target = tmp_path / "empty.txt"
        target.write_text("")

        mw = _ApplyPatchMiddleware(backend=backend)
        patch = f"*** Begin Patch\n*** Delete File: {target}\n*** End Patch"

        output = await mw.tools[0].ainvoke({"patch": patch, "runtime": _runtime()})

        assert not target.exists()
        assert f"Deleted '{target}'" in output
