"""Regression coverage for destructive filesystem and store isolation boundaries."""

import os
from pathlib import Path

import pytest
from langgraph.store.memory import InMemoryStore

from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends.store import StoreBackend


def _directory_link(link: Path, target: Path) -> None:
    """Use a junction on Windows when symlink creation requires a privilege."""
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        if os.name != "nt":
            pytest.skip("platform does not support directory symlinks")
        winapi = pytest.importorskip("_winapi")
        winapi.CreateJunction(str(target), str(link))


@pytest.mark.parametrize("mode", ["virtual", "relative", "absolute"])
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_delete_directory_link_preserves_target(tmp_path: Path, mode: str, *, asynchronous: bool) -> None:
    target = tmp_path / "target"
    target.mkdir()
    (target / "keep.txt").write_text("keep", encoding="utf-8")
    link = tmp_path / "link"
    _directory_link(link, target)
    backend = FilesystemBackend(root_dir=tmp_path, virtual_mode=mode == "virtual")
    path = {"virtual": "/link", "relative": "link", "absolute": str(link)}[mode]

    result = await backend.adelete(path) if asynchronous else backend.delete(path)

    assert result.error is None
    with pytest.raises(FileNotFoundError):
        link.lstat()
    assert (target / "keep.txt").read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize("dangling", [False, True])
@pytest.mark.parametrize("virtual_mode", [False, True])
def test_delete_file_symlink_preserves_target(tmp_path: Path, *, dangling: bool, virtual_mode: bool) -> None:
    target = tmp_path / "target.txt"
    if not dangling:
        target.write_text("keep", encoding="utf-8")
    link = tmp_path / "link.txt"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("platform does not support file symlinks")
    backend = FilesystemBackend(root_dir=tmp_path, virtual_mode=virtual_mode)

    result = backend.delete("/link.txt" if virtual_mode else "link.txt")

    assert result.error is None
    assert not link.is_symlink()
    assert target.exists() is not dangling
    if not dangling:
        assert target.read_text(encoding="utf-8") == "keep"


def test_delete_through_directory_link_preserves_parent(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    (target / "drop.txt").write_text("drop", encoding="utf-8")
    (target / "keep.txt").write_text("keep", encoding="utf-8")
    _directory_link(tmp_path / "link", target)
    backend = FilesystemBackend(root_dir=tmp_path)

    assert backend.delete("/link/drop.txt").error is None
    assert not (target / "drop.txt").exists()
    assert (target / "keep.txt").read_text(encoding="utf-8") == "keep"
    assert (tmp_path / "link").is_dir()


def test_delete_rejects_escaping_directory_link(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("keep", encoding="utf-8")
    _directory_link(root / "escape", outside)
    backend = FilesystemBackend(root_dir=root)

    with pytest.raises(ValueError, match="outside root"):
        backend.delete("/escape")
    with pytest.raises(ValueError, match="outside root"):
        backend.delete("/escape/keep.txt")
    assert (outside / "keep.txt").read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_store_search_excludes_child_namespaces(*, asynchronous: bool) -> None:
    store = InMemoryStore()
    parent = StoreBackend(store=store, namespace=lambda _: ("org",))
    child = StoreBackend(store=store, namespace=lambda _: ("org", "private"))
    parent.write("/shared.txt", "public")
    child.write("/shared.txt", "private")
    child.write("/secret.txt", "private")

    listing = await parent.als("/") if asynchronous else parent.ls("/")
    matches = await parent.agrep("private") if asynchronous else parent.grep("private")
    glob = await parent.aglob("*.txt") if asynchronous else parent.glob("*.txt")

    assert [entry["path"] for entry in listing.entries] == ["/shared.txt"]
    assert matches.matches == []
    assert [entry["path"] for entry in glob.matches] == ["/shared.txt"]
    public = await parent.agrep("public") if asynchronous else parent.grep("public")
    assert public.matches == [{"path": "/shared.txt", "line": 1, "text": "public"}]
    deletion = await parent.adelete("/secret.txt") if asynchronous else parent.delete("/secret.txt")
    assert deletion.error is not None
    assert child.read("/secret.txt").file_data["content"] == "private"


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_store_pagination_continues_past_child_namespace_pages(*, asynchronous: bool) -> None:
    store = InMemoryStore()
    child = StoreBackend(store=store, namespace=lambda _: ("org", "private"))
    parent = StoreBackend(store=store, namespace=lambda _: ("org",))
    for index in range(101):
        child.write(f"/secret/{index}.txt", "private")
    for index in range(101):
        parent.write(f"/public/{index}.txt", "public")

    listing = await parent.als("/public") if asynchronous else parent.ls("/public")
    assert len(listing.entries) == 101
    deletion = await parent.adelete("/") if asynchronous else parent.delete("/")
    assert deletion.error is None
    assert parent.ls("/").entries == []
    assert len(child.ls("/secret").entries) == 101
