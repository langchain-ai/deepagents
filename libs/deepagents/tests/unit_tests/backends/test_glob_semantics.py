"""Cross-backend matrix for shared `glob()` pattern semantics.

Locks the contract from #4978 so FS / State / Store / sandbox script cannot
drift:

- Patterns without `/` match the basename at any depth under the search root
  (including under leading-dot directories).
- Patterns containing `/` match the search-root-relative path (with `**`).
  Without DOTMATCH, `**` does not traverse leading-dot path segments; reach
  those paths with an explicit leading-dot segment (e.g. `.github/**`).
- A leading `/` anchors to the search root (narrows, does not widen).
"""

from __future__ import annotations

import base64
import json
import subprocess
import sys
from typing import TYPE_CHECKING, Any

import pytest
from langgraph.store.memory import InMemoryStore

from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends.sandbox import _GLOB_COMMAND_TEMPLATE, _build_glob_cmd
from deepagents.backends.state import StateBackend
from deepagents.backends.store import StoreBackend
from deepagents.backends.utils import _glob_search_files

if TYPE_CHECKING:
    from pathlib import Path

_TREE_FILES = {
    "/a.py": "a",
    "/sub/b.py": "b",
    "/sub/nested/c.py": "c",
    "/readme.md": "docs",
}


def _file_data(content: str) -> dict[str, Any]:
    return {
        "content": content,
        "created_at": "2024-01-01T00:00:00",
        "modified_at": "2024-01-01T00:00:00",
    }


def _matrix_expectations() -> list[tuple[str, str, set[str]]]:
    """(pattern, path, expected absolute virtual paths)."""
    all_py = {"/a.py", "/sub/b.py", "/sub/nested/c.py"}
    return [
        ("*.py", "/", all_py),
        ("**/*.py", "/", all_py),
        ("*.md", "/", {"/readme.md"}),
        ("sub/*.py", "/", {"/sub/b.py"}),
        ("sub/**/*.py", "/", {"/sub/b.py", "/sub/nested/c.py"}),
        ("/*.py", "/", {"/a.py"}),
        ("*.py", "/sub", {"/sub/b.py", "/sub/nested/c.py"}),
    ]


def _write_tree_on_disk(root: Path) -> None:
    (root / "a.py").write_text("a")
    (root / "sub" / "nested").mkdir(parents=True)
    (root / "sub" / "b.py").write_text("b")
    (root / "sub" / "nested" / "c.py").write_text("c")
    (root / "readme.md").write_text("docs")


def _write_hidden_tree_on_disk(root: Path) -> None:
    _write_tree_on_disk(root)
    (root / ".hidden.py").write_text("h")
    (root / ".hidden" / "nested").mkdir(parents=True)
    (root / ".hidden" / "x.py").write_text("x")
    (root / ".hidden" / "nested" / "y.py").write_text("y")
    (root / ".github" / "workflows").mkdir(parents=True)
    (root / ".github" / "workflows" / "ci.yml").write_text("name: ci")


def _paths_from_infos(infos: list[dict[str, Any]] | None) -> set[str]:
    if not infos:
        return set()
    return {i["path"] for i in infos}


class TestGlobSearchFilesMatrix:
    """In-memory helper used by StateBackend / StoreBackend."""

    @pytest.mark.parametrize(("pattern", "path", "expected"), _matrix_expectations())
    def test_matrix(self, pattern: str, path: str, expected: set[str]) -> None:
        files = {p: _file_data(c) for p, c in _TREE_FILES.items()}
        result = _glob_search_files(files, pattern, path)
        if expected:
            assert set(result.strip().split("\n")) == expected
        else:
            assert result == "No files found"


class TestStateBackendGlobMatrix:
    @pytest.mark.parametrize(("pattern", "path", "expected"), _matrix_expectations())
    def test_matrix(
        self,
        pattern: str,
        path: str,
        expected: set[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        be = StateBackend()
        files = {p: _file_data(c) for p, c in _TREE_FILES.items()}
        monkeypatch.setattr(be, "_read_files", lambda: files)
        assert _paths_from_infos(be.glob(pattern, path=path).matches) == expected


class TestStoreBackendGlobMatrix:
    @pytest.mark.parametrize(("pattern", "path", "expected"), _matrix_expectations())
    def test_matrix(self, pattern: str, path: str, expected: set[str]) -> None:
        store = InMemoryStore()
        be = StoreBackend(store=store, namespace=lambda _rt: ("filesystem",))
        for p, content in _TREE_FILES.items():
            res = be.write(p, content)
            assert res.error is None
        assert _paths_from_infos(be.glob(pattern, path=path).matches) == expected


class TestFilesystemBackendGlobMatrix:
    @pytest.mark.parametrize(("pattern", "path", "expected"), _matrix_expectations())
    def test_matrix(self, tmp_path: Path, pattern: str, path: str, expected: set[str]) -> None:
        _write_tree_on_disk(tmp_path)
        be = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)
        assert _paths_from_infos(be.glob(pattern, path=path).matches) == expected


def _run_glob_script(path: Path, pattern: str) -> list[dict[str, Any]]:
    """Run the sandbox remote glob script the same way `_build_glob_cmd` does."""
    cmd = _build_glob_cmd(pattern, str(path))
    _, _, tail = cmd.partition('python3 -c "')
    script, _, _ = tail.partition('" 2>&1')
    assert script, "failed to extract remote glob script from template"
    proc = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    rows: list[dict[str, Any]] = []
    for line in proc.stdout.strip().splitlines():
        if not line:
            continue
        data = json.loads(line)
        if "error" in data:
            pytest.fail(f"glob script error: {data}")
        rows.append(data)
    return rows


def test_build_glob_cmd_passes_user_pattern_unchanged() -> None:
    cmd = _build_glob_cmd("*.py", "/workspace")
    assert base64.b64encode(b"*.py").decode("ascii") in cmd
    # Bare basename expansion is handled by the remote walk matcher, not the host.
    assert base64.b64encode(b"**/*.py").decode("ascii") not in cmd
    cmd_anchored = _build_glob_cmd("/*.py", "/workspace")
    assert base64.b64encode(b"/*.py").decode("ascii") in cmd_anchored


class TestSandboxGlobScriptMatrix:
    """Sandbox remote script parity with the shared backend glob contract."""

    @pytest.mark.parametrize(
        ("pattern", "path_suffix", "expected_rel"),
        [
            ("*.py", "", {"a.py", "sub/b.py", "sub/nested/c.py"}),
            ("**/*.py", "", {"a.py", "sub/b.py", "sub/nested/c.py"}),
            ("*.md", "", {"readme.md"}),
            ("sub/*.py", "", {"sub/b.py"}),
            ("sub/**/*.py", "", {"sub/b.py", "sub/nested/c.py"}),
            ("/*.py", "", {"a.py"}),
            ("*.py", "sub", {"b.py", "nested/c.py"}),
        ],
    )
    def test_matrix(
        self,
        tmp_path: Path,
        pattern: str,
        path_suffix: str,
        expected_rel: set[str],
    ) -> None:
        _write_tree_on_disk(tmp_path)
        search = tmp_path / path_suffix if path_suffix else tmp_path
        rows = _run_glob_script(search, pattern)
        # Script returns paths relative to the search cwd (no leading `/`).
        got = {r["path"].replace("\\", "/") for r in rows if not r.get("is_dir")}
        assert got == expected_rel

    @pytest.mark.parametrize(
        ("pattern", "expected_rel"),
        [
            # Basename-at-any-depth includes files under leading-dot dirs, but
            # still excludes leading-dot basenames unless the pattern is explicit.
            (
                "*.py",
                {
                    "a.py",
                    "sub/b.py",
                    "sub/nested/c.py",
                    ".hidden/x.py",
                    ".hidden/nested/y.py",
                },
            ),
            # Path-relative ** without DOTMATCH skips leading-dot segments.
            ("**/*.py", {"a.py", "sub/b.py", "sub/nested/c.py"}),
            ("**/*.yml", set()),
            (".*.py", {".hidden.py"}),
            (".hidden/*.py", {".hidden/x.py"}),
            (".hidden/**/*.py", {".hidden/x.py", ".hidden/nested/y.py"}),
            (".github/**/*.yml", {".github/workflows/ci.yml"}),
        ],
    )
    def test_hidden_dirs(
        self,
        tmp_path: Path,
        pattern: str,
        expected_rel: set[str],
    ) -> None:
        _write_hidden_tree_on_disk(tmp_path)
        rows = _run_glob_script(tmp_path, pattern)
        got = {r["path"].replace("\\", "/") for r in rows if not r.get("is_dir")}
        assert got == expected_rel

    def test_hidden_dirs_match_filesystem_backend(self, tmp_path: Path) -> None:
        _write_hidden_tree_on_disk(tmp_path)
        be = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)
        patterns = [
            "*.py",
            "**/*.py",
            "**/*.yml",
            ".*.py",
            ".hidden/*.py",
            ".hidden/**/*.py",
            ".github/**/*.yml",
        ]
        for pattern in patterns:
            fs = {info["path"].lstrip("/") for info in be.glob(pattern).matches or []}
            rows = _run_glob_script(tmp_path, pattern)
            sb = {r["path"].replace("\\", "/") for r in rows if not r.get("is_dir")}
            assert sb == fs, pattern


def test_glob_command_template_uses_walk_not_stdlib_glob() -> None:
    # Stdlib glob(recursive=True) neither expands bare basename patterns nor
    # walks into leading-dot directories for `**` matches.
    assert "os.walk" in _GLOB_COMMAND_TEMPLATE
    assert "fnmatch" in _GLOB_COMMAND_TEMPLATE
    assert "glob.glob" not in _GLOB_COMMAND_TEMPLATE
