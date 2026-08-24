"""Tests for extension source resolution and provenance."""

from pathlib import Path

import pytest

from deepagents_code.extensions.discovery import discover_extensions
from deepagents_code.extensions.registry import SourceScope


@pytest.fixture(autouse=True)
def _experimental(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPAGENTS_CODE_EXPERIMENTAL", "1")
    monkeypatch.setattr(
        "deepagents_code.extensions.discovery.importlib.metadata.entry_points",
        lambda **_: (),
    )


def _extension(directory: Path, name: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text("async def extension(d):\n    pass\n", encoding="utf-8")
    return path


def test_experimental_mode_gates_all_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A disabled experiment performs no extension source inspection."""
    monkeypatch.delenv("DEEPAGENTS_CODE_EXPERIMENTAL")

    def fail(*_: object, **__: object) -> None:
        msg = "extension discovery crossed the experimental gate"
        raise AssertionError(msg)

    for name in (
        "user_extensions_dir",
        "_resolve_paths",
        "_entry_point_sources",
        "_plugin_sources",
    ):
        monkeypatch.setattr(f"deepagents_code.extensions.discovery.{name}", fail)

    result = discover_extensions(
        config_files=(Path("configured.py"),),
        cli_paths=(Path("temporary.py"),),
        project_dir=Path("project"),
    )

    assert not result.sources
    assert not result.errors


def test_sources_resolve_in_authority_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """User, config, CLI, and trusted project sources retain stable order."""
    user = _extension(tmp_path / "user", "a.py")
    configured = _extension(tmp_path / "configured", "b.py")
    temporary = _extension(tmp_path / "temporary", "c.py")
    project = _extension(tmp_path / "project", "d.py")
    monkeypatch.setattr(
        "deepagents_code.extensions.discovery.user_extensions_dir",
        lambda: user.parent,
    )

    result = discover_extensions(
        config_files=(configured, user),
        cli_paths=(temporary,),
        project_dir=project.parent,
    )

    assert [source.path for source in result.sources] == [
        user.resolve(),
        configured.resolve(),
        temporary.resolve(),
        project.resolve(),
    ]
    assert [source.scope for source in result.sources] == [
        SourceScope.USER,
        SourceScope.USER,
        SourceScope.TEMPORARY,
        SourceScope.PROJECT,
    ]


def test_invalid_explicit_path_is_isolated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing CLI source reports an error without blocking valid files."""
    valid = _extension(tmp_path, "valid.py")
    monkeypatch.setattr(
        "deepagents_code.extensions.discovery.user_extensions_dir",
        lambda: tmp_path / "absent-user-dir",
    )

    result = discover_extensions(cli_paths=(tmp_path / "missing.py", valid))

    assert [source.path for source in result.sources] == [valid.resolve()]
    assert len(result.errors) == 1
    assert "missing.py" in result.errors[0]


def test_unreadable_explicit_path_is_isolated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A filesystem error inspecting one explicit path remains non-fatal."""
    valid = _extension(tmp_path / "user", "valid.py")
    unreadable = tmp_path / "unreadable.py"
    monkeypatch.setattr(
        "deepagents_code.extensions.discovery.user_extensions_dir",
        lambda: valid.parent,
    )
    is_dir = Path.is_dir

    def guarded(self: Path) -> bool:
        if self == unreadable:
            msg = "unreadable"
            raise OSError(msg)
        return is_dir(self)

    monkeypatch.setattr(Path, "is_dir", guarded)

    result = discover_extensions(cli_paths=(unreadable,))

    assert [source.path for source in result.sources] == [valid.resolve()]
    assert result.errors == ("Could not inspect an extension path",)


def test_unreadable_directory_entry_is_isolated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An entry that fails inspection is reported without aborting the scan."""
    user = tmp_path / "user"
    _extension(user, "broken.py")
    valid = _extension(user, "valid.py")
    monkeypatch.setattr(
        "deepagents_code.extensions.discovery.user_extensions_dir", lambda: user
    )
    is_file = Path.is_file

    def guarded(self: Path) -> bool:
        if self.name == "broken.py":
            msg = "unreadable"
            raise OSError(msg)
        return is_file(self)

    monkeypatch.setattr(Path, "is_file", guarded)

    result = discover_extensions()

    assert [source.path for source in result.sources] == [valid.resolve()]
    assert len(result.errors) == 1
    assert "broken.py" in result.errors[0]


def test_broken_entry_point_metadata_is_isolated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unreadable distribution metadata reports an error without aborting."""
    valid = _extension(tmp_path / "user", "valid.py")
    monkeypatch.setattr(
        "deepagents_code.extensions.discovery.user_extensions_dir",
        lambda: valid.parent,
    )

    def broken(**_: object) -> tuple[()]:
        msg = "malformed entry_points.txt"
        raise ValueError(msg)

    monkeypatch.setattr(
        "deepagents_code.extensions.discovery.importlib.metadata.entry_points", broken
    )

    result = discover_extensions()

    assert [source.path for source in result.sources] == [valid.resolve()]
    assert len(result.errors) == 1
    assert "malformed entry_points.txt" in result.errors[0]
