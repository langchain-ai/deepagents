"""Tests for `.deepagentsignore` parsing and backend filtering."""

from pathlib import Path

from deepagents.backends.filesystem import FilesystemBackend

from deepagents_code.deepagentsignore import (
    IGNORED_READ_ERROR,
    DeepagentsIgnore,
    IgnoringBackend,
)


def test_deepagentsignore_matches_defaults_and_project_rules(tmp_path: Path) -> None:
    (tmp_path / ".deepagentsignore").write_text(
        ".env\nsecrets/\n!secrets/public.txt\n/root-only.txt\n",
        encoding="utf-8",
    )

    ignore = DeepagentsIgnore.from_project(tmp_path)

    assert ignore.is_ignored_relative("node_modules/pkg/index.js")
    assert ignore.is_ignored_relative("nested/node_modules", is_dir=True)
    assert ignore.is_ignored_relative("dist/app.js")
    assert ignore.is_ignored_relative(".env")
    assert ignore.is_ignored_relative("nested/.env")
    assert ignore.is_ignored_relative("secrets/token.txt")
    assert not ignore.is_ignored_relative("secrets/public.txt")
    assert ignore.is_ignored_relative("root-only.txt")
    assert not ignore.is_ignored_relative("nested/root-only.txt")


def test_ignoring_backend_filters_read_list_glob_and_grep(
    tmp_path: Path,
) -> None:
    (tmp_path / ".deepagentsignore").write_text(
        "secret.txt\nnode_modules/\n",
        encoding="utf-8",
    )
    (tmp_path / "visible.txt").write_text("hello visible\n", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("hello secret\n", encoding="utf-8")
    vendor = tmp_path / "node_modules"
    vendor.mkdir()
    (vendor / "pkg.js").write_text("hello package\n", encoding="utf-8")

    backend = IgnoringBackend(
        FilesystemBackend(root_dir=tmp_path, virtual_mode=False),
        DeepagentsIgnore.from_project(tmp_path),
    )

    read_result = backend.read(str(tmp_path / "secret.txt"))
    assert read_result.error is not None
    assert IGNORED_READ_ERROR in read_result.error

    listed = backend.ls(str(tmp_path))
    listed_paths = {Path(entry["path"]).name for entry in listed.entries or []}
    assert "visible.txt" in listed_paths
    assert "secret.txt" not in listed_paths
    assert "node_modules" not in listed_paths

    globbed = backend.glob("**/*", str(tmp_path))
    globbed_names = {Path(entry["path"]).name for entry in globbed.matches or []}
    assert "visible.txt" in globbed_names
    assert "secret.txt" not in globbed_names
    assert "pkg.js" not in globbed_names

    grep = backend.grep("hello", str(tmp_path))
    grep_names = {Path(match["path"]).name for match in grep.matches or []}
    assert grep_names == {"visible.txt"}
