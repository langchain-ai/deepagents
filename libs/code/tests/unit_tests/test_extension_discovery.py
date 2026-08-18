"""Tests for deterministic extension source discovery."""

from pathlib import Path

from deepagents_code.extensions.discovery import (
    discover_extension_files,
    scan_extension_dir,
)
from deepagents_code.extensions.models import UnitOrigin, UnitScope


def _write(path: Path, body: str = "") -> Path:
    """Create a test extension file and return its path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def test_scan_is_shallow_sorted_and_package_aware(tmp_path: Path) -> None:
    """A directory scan should recognize only direct files and packages."""
    _write(tmp_path / "zeta.py")
    _write(tmp_path / "alpha.py")
    _write(tmp_path / "package" / "extension.py")
    _write(tmp_path / "preferred" / "__init__.py")
    _write(tmp_path / "preferred" / "extension.py")
    _write(tmp_path / "ignored" / "nested" / "extension.py")
    _write(tmp_path / "notes.txt")

    entries = scan_extension_dir(tmp_path)

    assert entries == [
        (tmp_path / "alpha.py", UnitOrigin.TOP_LEVEL),
        (tmp_path / "package" / "extension.py", UnitOrigin.PACKAGE),
        (tmp_path / "preferred" / "__init__.py", UnitOrigin.PACKAGE),
        (tmp_path / "zeta.py", UnitOrigin.TOP_LEVEL),
    ]


def test_sources_follow_precedence_and_project_scope(tmp_path: Path) -> None:
    """User, explicit, and trusted project sources should retain precedence."""
    user = tmp_path / "user"
    extra = tmp_path / "extra"
    project = tmp_path / "project"
    user_file = _write(user / "user.py")
    extra_file = _write(extra / "extra.py")
    project_file = _write(project / "project.py")

    discovered = discover_extension_files(
        user_dir=user,
        extra_paths=[extra],
        project_dir=project,
    )

    assert [entry.path for entry in discovered] == [
        user_file,
        extra_file,
        project_file,
    ]
    assert [entry.source.scope for entry in discovered] == [
        UnitScope.USER,
        UnitScope.USER,
        UnitScope.PROJECT,
    ]


def test_explicit_duplicate_path_loads_once(tmp_path: Path) -> None:
    """The same canonical file from two configured sources should deduplicate."""
    extension = _write(tmp_path / "extension.py")

    discovered = discover_extension_files(
        user_dir=tmp_path / "missing",
        extra_paths=[extension, tmp_path],
    )

    assert [entry.path for entry in discovered] == [extension]


def test_missing_or_unreadable_sources_are_skipped(tmp_path: Path) -> None:
    """A missing source should not abort discovery of later sources."""
    extension = _write(tmp_path / "valid" / "extension.py")

    discovered = discover_extension_files(
        user_dir=tmp_path / "missing",
        extra_paths=[tmp_path / "also-missing", extension],
    )

    assert [entry.path for entry in discovered] == [extension]
