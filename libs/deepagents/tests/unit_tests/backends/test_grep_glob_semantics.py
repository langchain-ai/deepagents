"""Cross-backend glob semantics for grep (ripgrep-style everywhere)."""

from pathlib import Path
from unittest.mock import patch

import pytest

import deepagents.backends.filesystem as fsmod
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends.utils import compile_grep_include_glob, grep_matches_from_files

FILES = {
    "/src/app/main.py": {"content": "import os", "encoding": "utf-8"},
    "/readme.md": {"content": "import nothing", "encoding": "utf-8"},
}

CASES = [
    ("src/**/*.py", ["/src/app/main.py"]),
    ("**/*.py", ["/src/app/main.py"]),
    ("*.py", ["/src/app/main.py"]),  # slashless: basename at any depth (rg semantics)
    ("*.md", ["/readme.md"]),
    ("app/*.py", []),  # path patterns are root-relative, like rg
]


@pytest.mark.parametrize(("glob", "expected"), CASES)
def test_in_memory_backend_glob(glob: str, expected: list[str]) -> None:
    matches = grep_matches_from_files(FILES, "import", "/", glob=glob).matches
    assert sorted(m["path"] for m in (matches or [])) == expected


@pytest.mark.parametrize(("glob", "expected"), CASES)
def test_python_fallback_glob(tmp_path: Path, glob: str, expected: list[str]) -> None:
    (tmp_path / "src" / "app").mkdir(parents=True)
    (tmp_path / "src" / "app" / "main.py").write_text("import os")
    (tmp_path / "readme.md").write_text("import nothing")
    with patch.object(fsmod, "_resolve_ripgrep_path", lambda: None):
        be = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)
        matches = be.grep("import", path="/", glob=glob).matches
    assert sorted(m["path"] for m in (matches or [])) == expected


def test_in_memory_glob_with_search_root() -> None:
    # path patterns are relative to the search root, matching rg's cwd behavior
    matches = grep_matches_from_files(FILES, "import", "/src", glob="app/*.py").matches
    assert [m["path"] for m in (matches or [])] == ["/src/app/main.py"]


def test_compile_grep_include_glob_basename_vs_path() -> None:
    assert compile_grep_include_glob("*.py")("src/app/main.py")
    assert not compile_grep_include_glob("app/*.py")("src/app/main.py")
    assert compile_grep_include_glob("src/**/*.py")("src/app/main.py")
