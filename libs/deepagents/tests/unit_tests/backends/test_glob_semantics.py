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
import os
import stat
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
    # Leading-dot entries live in the shared tree so every backend's matrix
    # exercises the dot rules, which are the substance of #4978.
    "/.hidden.py": "h",
    "/.hidden/x.py": "x",
    "/.github/workflows/ci.yml": "name: ci",
}


def _file_data(content: str) -> dict[str, Any]:
    return {
        "content": content,
        "created_at": "2024-01-01T00:00:00",
        "modified_at": "2024-01-01T00:00:00",
    }


def _matrix_expectations() -> list[tuple[str, str, set[str]]]:
    """(pattern, path, expected absolute virtual paths)."""
    # Bare patterns are basename-scoped, so they reach under `.hidden/` but
    # never match the leading-dot basename `.hidden.py` itself.
    bare_py = {"/a.py", "/sub/b.py", "/sub/nested/c.py", "/.hidden/x.py"}
    # `**` is path-relative and will not traverse a leading-dot segment.
    walked_py = {"/a.py", "/sub/b.py", "/sub/nested/c.py"}
    return [
        ("*.py", "/", bare_py),
        ("**/*.py", "/", walked_py),
        ("*.md", "/", {"/readme.md"}),
        ("sub/*.py", "/", {"/sub/b.py"}),
        ("sub/**/*.py", "/", {"/sub/b.py", "/sub/nested/c.py"}),
        ("/*.py", "/", {"/a.py"}),
        ("a.py/**", "/", set()),
        ("*.py", "/sub", {"/sub/b.py", "/sub/nested/c.py"}),
        # Dot handling: bare `*.yml` reaches into `.github/`, `**/` does not,
        # and an explicit dot segment is the escape hatch.
        ("*.yml", "/", {"/.github/workflows/ci.yml"}),
        ("**/*.yml", "/", set()),
        (".github/**/*.yml", "/", {"/.github/workflows/ci.yml"}),
        (".hidden.py", "/", {"/.hidden.py"}),
        (".*.py", "/", {"/.hidden.py"}),
        ("*", "/", {"/a.py", "/sub/b.py", "/sub/nested/c.py", "/readme.md", "/.hidden/x.py", "/.github/workflows/ci.yml"}),
        # Brace, single-char and character-class syntax the contract advertises.
        ("*.{py,md}", "/", bare_py | {"/readme.md"}),
        ("*.{yml,yaml}", "/", {"/.github/workflows/ci.yml"}),
        ("?.py", "/", bare_py),
        ("[ab].py", "/", {"/a.py", "/sub/b.py"}),
        ("[!a].py", "/", {"/sub/b.py", "/sub/nested/c.py", "/.hidden/x.py"}),
        # `[^...]` must negate like bash/ripgrep, not match a literal '^'.
        ("[^a].py", "/", {"/sub/b.py", "/sub/nested/c.py", "/.hidden/x.py"}),
    ]


def _relative_expectations(path: str, expected: set[str]) -> set[str]:
    """Re-express absolute matrix expectations relative to the search root.

    Lets the sandbox script matrix reuse `_matrix_expectations` instead of
    keeping a parallel hardcoded table that could silently drift from it.
    """
    if path == "/":
        return {p.lstrip("/") for p in expected}
    prefix = path.rstrip("/") + "/"
    return {p[len(prefix) :] for p in expected}


def _write_tree_on_disk(root: Path) -> None:
    (root / "a.py").write_text("a")
    (root / "sub" / "nested").mkdir(parents=True)
    (root / "sub" / "b.py").write_text("b")
    (root / "sub" / "nested" / "c.py").write_text("c")
    (root / "readme.md").write_text("docs")
    (root / ".hidden.py").write_text("h")
    (root / ".hidden").mkdir()
    (root / ".hidden" / "x.py").write_text("x")
    (root / ".github" / "workflows").mkdir(parents=True)
    (root / ".github" / "workflows" / "ci.yml").write_text("name: ci")


def _write_hidden_tree_on_disk(root: Path) -> None:
    """Base tree plus a nested file *inside* a leading-dot directory.

    The base tree already carries the top-level dot entries; this adds the extra
    depth needed to tell `.hidden/*.py` from `.hidden/**/*.py`.
    """
    _write_tree_on_disk(root)
    (root / ".hidden" / "nested").mkdir(parents=True, exist_ok=True)
    (root / ".hidden" / "nested" / "y.py").write_text("y")


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


def _run_glob_script(
    path: Path,
    pattern: str,
    *,
    allow_error: bool = False,
    time_budget: float | None = None,
) -> list[dict[str, Any]]:
    """Run the sandbox remote glob script the same way `_build_glob_cmd` does."""
    cmd = _build_glob_cmd(pattern, str(path))
    _, _, tail = cmd.partition('python3 -c "')
    script, _, _ = tail.partition('" 2>&1')
    assert script, "failed to extract remote glob script from template"
    if time_budget is not None:
        script = script.replace("TIME_BUDGET = 5.0", f"TIME_BUDGET = {time_budget!r}", 1)
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
            if allow_error:
                rows.append(data)
                continue
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

    @pytest.mark.parametrize(("pattern", "path", "expected"), _matrix_expectations())
    def test_matrix(
        self,
        tmp_path: Path,
        pattern: str,
        path: str,
        expected: set[str],
    ) -> None:
        """Drives the same table as the in-process backends, so it cannot drift."""
        _write_tree_on_disk(tmp_path)
        search = tmp_path / path.lstrip("/") if path != "/" else tmp_path
        rows = _run_glob_script(search, pattern)
        # Script returns paths relative to the search cwd (no leading `/`).
        got = {r["path"].replace("\\", "/") for r in rows}
        assert got == _relative_expectations(path, expected)
        # Directories are never emitted, matching FilesystemBackend.glob. Asserted
        # rather than filtered out, so a regression here cannot hide.
        assert all(r["is_dir"] is False for r in rows)

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
        got = {r["path"].replace("\\", "/") for r in rows}
        assert got == expected_rel

    @pytest.mark.parametrize(
        "pattern",
        [
            "*.py",
            "**/*.py",
            "**/*.yml",
            "*.yml",
            "*",
            ".*.py",
            ".hidden/*.py",
            ".hidden/**/*.py",
            ".github/**/*.yml",
            "/*.py",
            "*.{py,yml}",
            "{a,b}.py",
            "{a,{b,c}}.py",
            "?.py",
            "[ab].py",
            "[!a].py",
            "[^a].py",
            "sub/",
            "*.py/",
            "a//b.py",
        ],
    )
    def test_matches_filesystem_backend(self, tmp_path: Path, pattern: str) -> None:
        """Differential against the in-process matcher, not a hardcoded list.

        Covers the syntax classes where a hand-rolled `fnmatch` matcher is most
        likely to drift from `wcmatch`: negated classes, nested braces, trailing
        slashes and empty path segments.
        """
        _write_hidden_tree_on_disk(tmp_path)
        be = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)
        fs = {info["path"].lstrip("/") for info in be.glob(pattern).matches or []}
        rows = _run_glob_script(tmp_path, pattern)
        sb = {r["path"].replace("\\", "/") for r in rows}
        assert sb == fs, pattern


def test_glob_command_template_uses_walk_not_stdlib_glob() -> None:
    # Stdlib glob(recursive=True) neither expands bare basename patterns nor
    # walks into leading-dot directories for `**` matches.
    assert "os.walk" in _GLOB_COMMAND_TEMPLATE
    assert "fnmatch" in _GLOB_COMMAND_TEMPLATE
    assert "glob.glob" not in _GLOB_COMMAND_TEMPLATE


def test_glob_directory_only_walk_honors_time_budget(tmp_path: Path) -> None:
    """An expired budget stops a walk even when no directory contains files."""
    (tmp_path / "one" / "two" / "three").mkdir(parents=True)

    rows = _run_glob_script(tmp_path, "*", time_budget=-1.0)

    assert rows == [{"warning": "truncated"}]


def test_glob_script_body_is_shell_safe() -> None:
    """The script is embedded in `sh -c '... python3 -c "..." ...'`.

    A `$`, backtick, double quote or backslash in the body would be eaten by the
    shell at runtime while every test that runs the body directly still passed
    (hence `chr(92)` rather than a literal backslash in the template).
    """
    cmd = _build_glob_cmd("*.py", "/workspace")
    _, _, tail = cmd.partition('python3 -c "')
    script, _, _ = tail.partition('" 2>&1')
    for char in ("$", "`", '"', "\\"):
        assert char not in script, f"{char!r} in remote script would be mangled by the shell"


def test_glob_script_skips_symlinks_outside_root(tmp_path: Path) -> None:
    """Realpath containment keeps symlink matches inside the declared root.

    Mirrors `test_grep_glob_script_skips_symlinks_outside_root`; the glob route
    had no equivalent, so the containment check was unpinned.
    """
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    src = workspace / "src"
    outside.mkdir()
    src.mkdir(parents=True)
    (outside / "secret.py").write_text("secret")
    (src / "link.py").symlink_to(outside / "secret.py")
    (src / "safe.py").write_text("safe")

    rows = _run_glob_script(workspace, "*.py")

    assert {r["path"] for r in rows} == {"src/safe.py"}


@pytest.mark.skipif(
    sys.platform == "win32" or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="chmod 000 does not deny access on Windows or as root",
)
def test_glob_script_reports_unreadable_subtree_instead_of_shrinking(tmp_path: Path) -> None:
    """`os.walk` discards traversal errors by default; the script must not.

    Without `onerror`, an unreadable directory silently yields fewer matches and
    the caller cannot tell a partial result from a complete one.
    """
    (tmp_path / "ok.py").write_text("ok")
    locked = tmp_path / "locked"
    locked.mkdir()
    (locked / "hidden.py").write_text("x")
    locked.chmod(0o000)
    try:
        rows = _run_glob_script(tmp_path, "*.py", allow_error=True)
    finally:
        locked.chmod(stat.S_IRWXU)

    assert {r["path"] for r in rows if "path" in r} == {"ok.py"}
    assert {"warning": "walk_errors", "count": 1} in rows


class TestSandboxGlobBraceExpansionLimit:
    """Unbounded brace expansion must fail fast instead of hanging the sandbox.

    `glob.pattern` is model/user supplied, so the remote script caps expansion
    (mirroring wcmatch's expansion budget in compile_grep_include_glob) and
    reports `pattern_too_broad` past the limit. That code is deliberately
    distinct from the `invalid_pattern` used for `..` traversal: conflating them
    leaves the caller unable to tell an over-broad pattern from a rejected one.
    """

    def test_over_limit_returns_pattern_too_broad(self, tmp_path: Path) -> None:
        # 18 two-part brace groups -> 262,144 expansions, far past the cap.
        rows = _run_glob_script(tmp_path, "{a,}" * 18 + "probe.py", allow_error=True)
        assert rows == [{"error": "pattern_too_broad"}]

    def test_traversal_uses_a_distinct_code(self, tmp_path: Path) -> None:
        rows = _run_glob_script(tmp_path, "../*.py", allow_error=True)
        assert rows == [{"error": "invalid_pattern"}]

    def test_under_limit_expands_normally(self, tmp_path: Path) -> None:
        # 4 two-part brace groups -> 16 expansions, well under the cap.
        (tmp_path / "probe.py").write_text("x")
        rows = _run_glob_script(tmp_path, "{a,}" * 4 + "probe.py")
        assert {r["path"] for r in rows if not r.get("is_dir")} == {"probe.py"}

    def test_non_expanding_braces_do_not_loop(self, tmp_path: Path) -> None:
        (tmp_path / "{probe}.py").write_text("x")
        rows = _run_glob_script(tmp_path, "{probe}.py")
        assert {r["path"] for r in rows if not r.get("is_dir")} == {"{probe}.py"}

    def test_nested_braces_expand_like_wcmatch(self, tmp_path: Path) -> None:
        """`compile_grep_include_glob` expands nested groups, so the script must too."""
        for name in ("a.py", "b.py", "c.py", "d.py"):
            (tmp_path / name).write_text("x")
        rows = _run_glob_script(tmp_path, "{a,{b,c}}.py")
        assert {r["path"] for r in rows} == {"a.py", "b.py", "c.py"}

    def test_single_element_group_still_expands_the_rest(self, tmp_path: Path) -> None:
        """A literal `{x}` group must not stop later groups from expanding."""
        (tmp_path / "{x}a.py").write_text("x")
        (tmp_path / "{x}b.py").write_text("x")
        rows = _run_glob_script(tmp_path, "{x}{a,b}.py")
        assert {r["path"] for r in rows} == {"{x}a.py", "{x}b.py"}
