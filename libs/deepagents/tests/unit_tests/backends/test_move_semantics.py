"""Cross-backend `move` contract, plus the races and collapses it must not have.

Every backend implements `move` against its own storage model, so the contract
is asserted once here and driven per backend, following the shape of
`test_glob_semantics.py`. Backend-specific mechanics (single `batch` call,
`push_agent` payloads, composite routing) live in each backend's own test file.
"""

import asyncio
import errno
import os
import tempfile
from collections.abc import Callable
from pathlib import Path

import pytest
from langgraph.store.memory import InMemoryStore

from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends.protocol import BackendProtocol
from deepagents.backends.store import StoreBackend

# Files every case starts from, as `path -> content`.
_FIXTURE = {
    "/a.txt": "AAA",
    "/keep.txt": "KKK",
    "/work/nested.txt": "NNN",
}


def _read(backend: BackendProtocol, path: str) -> str | None:
    """Return a file's content, or `None` when it is not there."""
    result = backend.read(path)
    if result.error is not None or result.file_data is None:
        return None
    return result.file_data["content"]


def _make_filesystem() -> BackendProtocol:
    backend = FilesystemBackend(root_dir=tempfile.mkdtemp(), virtual_mode=True)
    for path, content in _FIXTURE.items():
        backend.write(path, content)
    return backend


def _make_store() -> BackendProtocol:
    backend = StoreBackend(store=InMemoryStore(), namespace=lambda _rt: ("filesystem",))
    for path, content in _FIXTURE.items():
        backend.write(path, content)
    return backend


# `StateBackend` needs a live graph context, so it is exercised through
# `create_deep_agent` in `test_end_to_end.py` instead of this matrix.
_BACKEND_FACTORIES: dict[str, Callable[[], BackendProtocol]] = {
    "filesystem": _make_filesystem,
    "store": _make_store,
}


@pytest.fixture(params=sorted(_BACKEND_FACTORIES))
def backend(request: pytest.FixtureRequest) -> BackendProtocol:
    """A prepopulated backend, one per implementation."""
    return _BACKEND_FACTORIES[request.param]()


class TestMoveContract:
    """Behavior every backend's `move` must agree on."""

    def test_relocates_file_and_removes_source(self, backend: BackendProtocol) -> None:
        result = backend.move("/a.txt", "/moved.txt")

        assert result.error is None
        assert result.source_path == "/a.txt"
        assert result.destination_path == "/moved.txt"
        assert _read(backend, "/moved.txt") == "AAA"
        assert _read(backend, "/a.txt") is None

    def test_basename_change_is_allowed(self, backend: BackendProtocol) -> None:
        # "Move, not rename" is framing: a basename change is an ordinary move.
        assert backend.move("/a.txt", "/work/renamed.md").error is None
        assert _read(backend, "/work/renamed.md") == "AAA"

    def test_missing_source_is_an_error(self, backend: BackendProtocol) -> None:
        result = backend.move("/nope.txt", "/dest.txt")

        assert result.source_path is None
        assert result.destination_path is None
        assert result.error is not None
        assert "not found" in result.error
        assert _read(backend, "/dest.txt") is None

    def test_directory_source_is_refused(self, backend: BackendProtocol) -> None:
        result = backend.move("/work", "/dest.txt")

        assert result.error is not None
        assert "is a directory" in result.error
        # Nothing under the refused directory was touched.
        assert _read(backend, "/work/nested.txt") == "NNN"

    def test_directory_destination_is_refused(self, backend: BackendProtocol) -> None:
        # Never "move into the directory" -- the written path would not be the
        # path a permission check upstream validated.
        result = backend.move("/a.txt", "/work")

        assert result.error is not None
        assert "is a directory" in result.error
        assert _read(backend, "/a.txt") == "AAA"

    def test_same_path_is_refused_and_the_file_survives(self, backend: BackendProtocol) -> None:
        # Key-value backends express a move as `{dst: value, src: None}`, which
        # collapses to `{src: None}` when the paths match -- a deletion. This is
        # the guard that stops a no-op request from destroying the file.
        result = backend.move("/keep.txt", "/keep.txt")

        assert result.error is not None
        assert "same path" in result.error
        assert _read(backend, "/keep.txt") == "KKK"

    def test_same_path_modulo_trailing_slash_is_refused(self, backend: BackendProtocol) -> None:
        result = backend.move("/keep.txt", "/keep.txt/")

        assert result.error is not None
        assert _read(backend, "/keep.txt") == "KKK"

    def test_existing_destination_is_refused_without_overwrite(self, backend: BackendProtocol) -> None:
        result = backend.move("/a.txt", "/keep.txt")

        assert result.error is not None
        assert "already exists" in result.error
        # Neither endpoint changed.
        assert _read(backend, "/keep.txt") == "KKK"
        assert _read(backend, "/a.txt") == "AAA"

    def test_overwrite_replaces_the_destination(self, backend: BackendProtocol) -> None:
        result = backend.move("/a.txt", "/keep.txt", overwrite=True)

        assert result.error is None
        assert _read(backend, "/keep.txt") == "AAA"
        assert _read(backend, "/a.txt") is None

    def test_missing_destination_parents_are_created(self, backend: BackendProtocol) -> None:
        assert backend.move("/a.txt", "/new/deep/dir/f.txt").error is None
        assert _read(backend, "/new/deep/dir/f.txt") == "AAA"

    def test_siblings_are_untouched(self, backend: BackendProtocol) -> None:
        assert backend.move("/a.txt", "/moved.txt").error is None
        assert _read(backend, "/keep.txt") == "KKK"
        assert _read(backend, "/work/nested.txt") == "NNN"

    def test_timestamps_are_preserved(self, backend: BackendProtocol) -> None:
        # A move is neither a create nor a write, so `mv` semantics apply and
        # any timestamp the backend tracks carries over unchanged. Only the
        # key-value backends store timestamps in `FileData`; `FilesystemBackend`
        # reports what the filesystem itself has, and `os.replace` preserves
        # that by construction, so assert on whichever keys are present.
        before = backend.read("/a.txt").file_data
        assert before is not None
        tracked = [key for key in ("created_at", "modified_at") if key in before]

        assert backend.move("/a.txt", "/moved.txt").error is None

        after = backend.read("/moved.txt").file_data
        assert after is not None
        for key in tracked:
            assert after[key] == before[key], f"{key} was restamped by the move"

    async def test_amove_matches_move(self, backend: BackendProtocol) -> None:
        result = await backend.amove("/a.txt", "/moved.txt")

        assert result.error is None
        assert result.source_path == "/a.txt"
        assert result.destination_path == "/moved.txt"
        assert _read(backend, "/moved.txt") == "AAA"
        assert _read(backend, "/a.txt") is None

    async def test_amove_same_path_is_refused_and_the_file_survives(self, backend: BackendProtocol) -> None:
        result = await backend.amove("/keep.txt", "/keep.txt")

        assert result.error is not None
        assert _read(backend, "/keep.txt") == "KKK"


class TestFlatBackendDescendants:
    """A key that is also a prefix is a directory, even on a flat backend."""

    def test_exact_key_with_nested_keys_is_refused(self) -> None:
        # `/work/a.txt` and `/work/a.txt/child` can coexist on a key-value
        # backend. Treating the exact key as a plain file would strand the
        # nested key, so the prefix case has to win.
        backend = StoreBackend(store=InMemoryStore(), namespace=lambda _rt: ("filesystem",))
        backend.write("/work/a.txt", "A")
        backend.write("/work/a.txt/child", "C")

        result = backend.move("/work/a.txt", "/dest.txt")

        assert result.error is not None
        assert "is a directory" in result.error
        assert _read(backend, "/work/a.txt") == "A"
        assert _read(backend, "/work/a.txt/child") == "C"

    def test_overwrite_still_works_when_destination_also_has_children(self) -> None:
        # The exact-key check runs before the prefix check, so an `overwrite`
        # of a real file is not refused just because children share its prefix.
        backend = StoreBackend(store=InMemoryStore(), namespace=lambda _rt: ("filesystem",))
        backend.write("/src.txt", "S")
        backend.write("/dst.txt", "D")
        backend.write("/dst.txt/child", "C")

        result = backend.move("/src.txt", "/dst.txt", overwrite=True)

        assert result.error is None
        assert _read(backend, "/dst.txt") == "S"
        assert _read(backend, "/dst.txt/child") == "C"


class TestFilesystemSymlinks:
    """A symlink source is refused rather than followed."""

    def test_symlink_source_is_refused(self) -> None:
        # Moving a link would relocate a path that permission rules, which match
        # on the path string, do not cover: a link landing outside a denied
        # prefix would read through to content inside it.
        root = Path(tempfile.mkdtemp())
        (root / "target.txt").write_text("SECRET")
        (root / "link.txt").symlink_to(root / "target.txt")
        backend = FilesystemBackend(root_dir=str(root), virtual_mode=True)

        result = backend.move("/link.txt", "/relocated.txt")

        assert result.error is not None
        assert "symlink" in result.error
        # Both the link and its target are intact, and nothing was created.
        assert (root / "link.txt").is_symlink()
        assert (root / "target.txt").read_text() == "SECRET"
        assert not (root / "relocated.txt").exists()

    def test_symlink_source_is_refused_in_non_virtual_mode(self) -> None:
        root = Path(tempfile.mkdtemp())
        (root / "target.txt").write_text("SECRET")
        (root / "link.txt").symlink_to(root / "target.txt")
        backend = FilesystemBackend(root_dir=str(root), virtual_mode=False)

        result = backend.move(str(root / "link.txt"), str(root / "relocated.txt"))

        assert result.error is not None
        assert "symlink" in result.error
        assert (root / "link.txt").is_symlink()

    def test_symlink_destination_is_refused(self) -> None:
        root = Path(tempfile.mkdtemp())
        (root / "src.txt").write_text("SRC")
        (root / "target.txt").write_text("TARGET")
        (root / "link.txt").symlink_to(root / "target.txt")
        backend = FilesystemBackend(root_dir=str(root), virtual_mode=True)

        result = backend.move("/src.txt", "/link.txt", overwrite=True)

        assert result.error is not None
        assert "symlink" in result.error
        # Writing through the link would have clobbered its target.
        assert (root / "target.txt").read_text() == "TARGET"
        assert (root / "src.txt").read_text() == "SRC"


class TestFilesystemNoClobberIsRaceFree:
    """`overwrite=False` must not lose a destination that appears mid-call."""

    def test_destination_created_after_the_check_is_not_clobbered(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # `os.replace` clobbers unconditionally, so `exists()`-then-`replace`
        # would destroy a file the caller asked us not to touch. `os.link`
        # refuses with EEXIST instead, which closes that window. Simulate a
        # writer winning the race by creating the destination inside the
        # rename primitive itself.
        root = Path(tempfile.mkdtemp())
        (root / "src.txt").write_text("INCOMING")
        backend = FilesystemBackend(root_dir=str(root), virtual_mode=True)

        real_link = os.link
        raced = {"done": False}

        def racing_link(src: str | Path, dst: str | Path, **kwargs: object) -> None:
            if not raced["done"]:
                raced["done"] = True
                Path(dst).write_text("ALREADY-THERE")
            real_link(src, dst, **kwargs)

        monkeypatch.setattr(os, "link", racing_link)

        result = backend.move("/src.txt", "/dst.txt")

        assert raced["done"], "the race was never simulated"
        assert result.error is not None
        assert "already exists" in result.error
        # The pre-existing destination survived and the source was not consumed.
        assert (root / "dst.txt").read_text() == "ALREADY-THERE"
        assert (root / "src.txt").read_text() == "INCOMING"

    def test_falls_back_to_replace_where_hardlinks_are_unavailable(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Some FUSE and FAT mounts cannot hardlink. The move must still work
        # there, accepting the check-then-act window that fallback implies.
        root = Path(tempfile.mkdtemp())
        (root / "src.txt").write_text("SRC")
        backend = FilesystemBackend(root_dir=str(root), virtual_mode=True)

        def no_hardlinks(_src: str | Path, _dst: str | Path, **_kwargs: object) -> None:
            raise OSError(errno.EOPNOTSUPP, "hardlinks not supported")

        monkeypatch.setattr(os, "link", no_hardlinks)

        result = backend.move("/src.txt", "/dst.txt")

        assert result.error is None
        assert (root / "dst.txt").read_text() == "SRC"
        assert not (root / "src.txt").exists()

    def test_cross_device_failure_to_remove_source_reports_both_locations(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # The EXDEV path is copy-then-unlink and therefore not atomic. If the
        # unlink fails the file exists twice, and the error has to say so rather
        # than implying nothing happened.
        root = Path(tempfile.mkdtemp())
        (root / "src.txt").write_text("SRC")
        backend = FilesystemBackend(root_dir=str(root), virtual_mode=True)

        def cross_device(_src: str | Path, _dst: str | Path, **_kwargs: object) -> None:
            raise OSError(errno.EXDEV, "cross-device link")

        monkeypatch.setattr(os, "link", cross_device)

        def refuse_unlink(_self: Path, *_args: object, **_kwargs: object) -> None:
            raise OSError(errno.EACCES, "permission denied")

        monkeypatch.setattr(Path, "unlink", refuse_unlink)

        result = backend.move("/src.txt", "/dst.txt")

        assert result.error is not None
        assert "both places" in result.error
        assert (root / "dst.txt").read_text() == "SRC"
        assert (root / "src.txt").read_text() == "SRC"


def test_asyncio_amove_on_filesystem_uses_the_threaded_default() -> None:
    """`FilesystemBackend` inherits `amove`, so it must still behave."""
    backend = _make_filesystem()

    result = asyncio.run(backend.amove("/a.txt", "/moved.txt"))

    assert result.error is None
    assert _read(backend, "/moved.txt") == "AAA"
