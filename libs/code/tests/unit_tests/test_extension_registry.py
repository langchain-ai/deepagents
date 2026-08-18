"""Tests for extension provenance and registration behavior."""

from pathlib import Path

from deepagents.backends import FilesystemBackend
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.tools import tool

from deepagents_code.extensions import (
    ExtensionRegistry,
    SourceInfo,
    UnitOrigin,
    UnitScope,
)


def _source(name: str) -> SourceInfo:
    """Build extension provenance for a top-level user file."""
    return SourceInfo(
        path=Path(f"/extensions/{name}.py"),
        scope=UnitScope.USER,
        origin=UnitOrigin.TOP_LEVEL,
    )


class _Middleware(AgentMiddleware):
    """Minimal middleware used to exercise registry naming."""

    name = "extension-test"


@tool("extension_test")
def _first_tool() -> str:
    """Return the first result."""
    return "first"


@tool("extension_test")
def _second_tool() -> str:
    """Return the second result."""
    return "second"


def test_source_label_uses_package_directory() -> None:
    """Package provenance should identify the extension directory."""
    source = SourceInfo(
        path=Path("/extensions/example/__init__.py"),
        scope=UnitScope.PROJECT,
        origin=UnitOrigin.PACKAGE,
    )

    assert source.label == "example"


def test_duplicate_registrations_keep_first_unit() -> None:
    """Duplicate names and route prefixes must not replace earlier units."""
    registry = ExtensionRegistry()
    first = _source("first")
    second = _source("second")
    first_backend = FilesystemBackend(virtual_mode=False)

    registry.add_middleware(_Middleware(), first)
    registry.add_middleware(_Middleware(), second)
    registry.add_tool(_first_tool, first)
    registry.add_tool(_second_tool, second)
    registry.add_backend_route("/memories/", first_backend, first)
    registry.add_backend_route(
        "/memories/", FilesystemBackend(virtual_mode=False), second
    )

    assert [unit.source for unit in registry.middleware] == [first]
    assert [unit.source for unit in registry.tools] == [first]
    assert [unit.source for unit in registry.backend_routes] == [first]
    assert registry.backend_routes[0].unit is first_backend


def test_rollback_removes_partial_factory_state() -> None:
    """Factory rollback should remove every registration kind and hook."""
    registry = ExtensionRegistry()
    source = _source("broken")
    snapshot = registry._snapshot()

    registry.add_middleware(_Middleware(), source)
    registry.add_tool(_first_tool, source)
    registry.add_backend_route(
        "/memories/", FilesystemBackend(virtual_mode=False), source
    )
    registry.add_shutdown_hook(lambda: None, source)
    registry._rollback(snapshot)

    assert registry.is_empty()
    assert not registry.shutdown_hooks
