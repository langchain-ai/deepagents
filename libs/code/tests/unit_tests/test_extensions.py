"""Unit tests for the extension discovery, loading, and trust pipeline."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003  # used at runtime (Path.cwd() in tests)
from typing import TYPE_CHECKING

import pytest

from deepagents_code.extensions import (
    ExtensionSettings,
    TrustPolicy,
    load_extensions,
    project_extensions_trusted,
)
from deepagents_code.extensions.discovery import scan_extension_dir
from deepagents_code.extensions.models import UnitOrigin
from deepagents_code.extensions.trust import trust_project_extensions


@pytest.fixture(autouse=True)
def _isolate_user_extensions_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Point the global extensions dir at a tmp path.

    `load_extensions` always scans `~/.deepagents/extensions/`, so a real user
    extension would otherwise leak into these tests and break assertions that
    expect an empty registry.
    """
    monkeypatch.setattr(
        "deepagents_code.extensions.discovery.user_extensions_dir",
        lambda: tmp_path / "user-extensions",
    )


_TOOL_EXTENSION = '''
def extension(d):
    def echo(text: str) -> str:
        """Echo text."""
        return text

    d.register_tool(echo)
    d.register_command("demo", lambda ctx: f"ran in {ctx.cwd}")
'''

_ASYNC_EXTENSION = """
async def extension(d):
    d.register_command("demo", lambda ctx: "second")
"""

_BROKEN_EXTENSION = """
raise RuntimeError("boom")
"""

_ROUTE_EXTENSION = """
from deepagents.backends import FilesystemBackend


def extension(d):
    d.register_backend_route("/memories/", FilesystemBackend(virtual_mode=False))
"""

_BAD_ROUTE_EXTENSION = """
from deepagents.backends import FilesystemBackend


def extension(d):
    d.register_backend_route("memories", FilesystemBackend(virtual_mode=False))
"""


def test_route_prefixes_must_be_absolute_and_slash_terminated() -> None:
    """A malformed route prefix is rejected with an ExtensionError."""
    from pathlib import Path

    import pytest

    from deepagents_code.extensions import ExtensionAPI, ExtensionRegistry
    from deepagents_code.extensions.models import (
        ExtensionError,
        SourceInfo,
        UnitOrigin,
        UnitScope,
        UnitSource,
    )

    source = SourceInfo(
        path=None,
        source=UnitSource.EXTENSION,
        scope=UnitScope.USER,
        origin=UnitOrigin.TOP_LEVEL,
    )
    api = ExtensionAPI(ExtensionRegistry(), source, cwd=Path.cwd(), mode="interactive")
    from deepagents.backends import FilesystemBackend

    with pytest.raises(ExtensionError):
        api.register_backend_route("memories", FilesystemBackend(virtual_mode=False))
    with pytest.raises(ExtensionError):
        api.register_backend_route("/../etc/", FilesystemBackend(virtual_mode=False))
    with pytest.raises(ExtensionError):
        api.register_backend_route("/memories/", object())


async def test_registers_backend_routes_in_load_order(tmp_path: Path) -> None:
    """A route registers cleanly and is recorded with its prefix as its name."""
    extensions_dir = tmp_path / "extensions"
    _write(extensions_dir, "routes.py", _ROUTE_EXTENSION)

    result = await load_extensions(
        cwd=tmp_path,
        settings=ExtensionSettings(paths=(extensions_dir,)),
    )

    assert not result.errors
    assert [unit.name for unit in result.registry.backend_routes] == ["/memories/"]
    assert not result.registry.is_empty()


def test_duplicate_route_prefix_is_dropped() -> None:
    """The first registration of a prefix wins; the second is dropped."""
    from pathlib import Path

    from deepagents.backends import FilesystemBackend

    from deepagents_code.extensions import ExtensionAPI, ExtensionRegistry
    from deepagents_code.extensions.models import (
        SourceInfo,
        UnitOrigin,
        UnitScope,
        UnitSource,
    )

    source = SourceInfo(
        path=None,
        source=UnitSource.EXTENSION,
        scope=UnitScope.USER,
        origin=UnitOrigin.TOP_LEVEL,
    )
    registry = ExtensionRegistry()
    api = ExtensionAPI(registry, source, cwd=Path.cwd(), mode="interactive")
    first = FilesystemBackend(virtual_mode=False)
    second = FilesystemBackend(virtual_mode=False)
    api.register_backend_route("/memories/", first)
    api.register_backend_route("/memories/", second)

    assert [unit.name for unit in registry.backend_routes] == ["/memories/"]
    assert registry.backend_routes[0].unit is first


def _write(directory: Path, name: str, body: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(body, encoding="utf-8")
    return path


def test_scan_resolves_files_and_package_entry_points(tmp_path: Path) -> None:
    """Direct modules and one-level package entry files are both discovered."""
    _write(tmp_path, "single.py", _TOOL_EXTENSION)
    _write(tmp_path / "pkg", "extension.py", _TOOL_EXTENSION)
    _write(tmp_path / "pkg" / "nested", "extension.py", _TOOL_EXTENSION)
    _write(tmp_path, "notes.md", "ignored")

    resolved = scan_extension_dir(tmp_path)

    assert resolved == [
        (tmp_path / "pkg" / "extension.py", UnitOrigin.PACKAGE),
        (tmp_path / "single.py", UnitOrigin.TOP_LEVEL),
    ]


async def test_loads_units_and_suffixes_colliding_commands(tmp_path: Path) -> None:
    """Both extensions load, and the duplicate command name is suffixed."""
    extensions_dir = tmp_path / "extensions"
    _write(extensions_dir, "a_first.py", _TOOL_EXTENSION)
    _write(extensions_dir, "b_second.py", _ASYNC_EXTENSION)

    result = await load_extensions(
        cwd=tmp_path,
        settings=ExtensionSettings(paths=(extensions_dir,)),
    )

    assert not result.errors
    assert [unit.name for unit in result.registry.tools] == ["echo"]
    assert [unit.name for unit in result.registry.commands] == ["demo", "demo-2"]
    assert result.registry.command_description("demo") == ""


async def test_broken_extension_is_isolated(tmp_path: Path) -> None:
    """A failing extension is reported without preventing the others."""
    extensions_dir = tmp_path / "extensions"
    _write(extensions_dir, "a_broken.py", _BROKEN_EXTENSION)
    _write(extensions_dir, "b_good.py", _TOOL_EXTENSION)

    result = await load_extensions(
        cwd=tmp_path,
        settings=ExtensionSettings(paths=(extensions_dir,)),
    )

    assert len(result.errors) == 1
    assert "a_broken.py" in result.errors[0]
    assert [unit.name for unit in result.registry.tools] == ["echo"]


async def test_project_extensions_load_only_once_trusted(tmp_path: Path) -> None:
    """The project source stays unscanned until a trust decision exists."""
    project_root = tmp_path / "repo"
    _write(project_root / ".deepagents" / "extensions", "ext.py", _TOOL_EXTENSION)
    store = tmp_path / "trust.json"
    settings = ExtensionSettings(paths=())

    untrusted = await load_extensions(
        cwd=project_root,
        project_root=project_root,
        settings=settings,
        trust_store_path=store,
    )
    assert untrusted.registry.is_empty()

    assert trust_project_extensions(project_root, store_path=store)
    trusted = await load_extensions(
        cwd=project_root,
        project_root=project_root,
        settings=settings,
        trust_store_path=store,
    )
    assert [unit.name for unit in trusted.registry.tools] == ["echo"]


def test_never_policy_overrides_persisted_trust(tmp_path: Path) -> None:
    """`never` refuses project extensions even with a stored decision."""
    store = tmp_path / "trust.json"
    assert trust_project_extensions(tmp_path, store_path=store)

    assert not project_extensions_trusted(
        tmp_path, policy=TrustPolicy.NEVER, store_path=store
    )
    assert project_extensions_trusted(
        tmp_path, policy=TrustPolicy.ASK, store_path=store
    )
