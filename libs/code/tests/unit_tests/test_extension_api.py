"""Tests for the public extension registrar."""

from pathlib import Path
from typing import Any, cast

import pytest
from deepagents.backends import FilesystemBackend
from langchain.agents.middleware.types import AgentMiddleware

from deepagents_code.extensions import (
    ExtensionAPI,
    ExtensionError,
    ExtensionRegistry,
    SourceInfo,
    UnitOrigin,
    UnitScope,
)


class _Middleware(AgentMiddleware):
    """Minimal valid middleware registration."""


class _ConfiguredMiddleware(AgentMiddleware):
    """Middleware that must be registered as an instance."""

    def __init__(self, value: str) -> None:
        self.value = value


@pytest.fixture
def api() -> ExtensionAPI:
    """Create a registrar with user-extension provenance."""
    source = SourceInfo(
        path=Path("/extensions/example.py"),
        scope=UnitScope.USER,
        origin=UnitOrigin.TOP_LEVEL,
    )
    return ExtensionAPI(
        ExtensionRegistry(), source, cwd=Path("/workspace"), mode="interactive"
    )


def test_registers_supported_extension_units(api: ExtensionAPI) -> None:
    """The P0 API should register middleware, tools, routes, and teardown."""
    backend = FilesystemBackend(virtual_mode=False)

    def status() -> str:
        """Report extension status."""
        return "ready"

    api.register_middleware(_Middleware)
    api.register_tool(status)
    api.register_backend_route("/memories/", backend)
    api.on_shutdown(lambda: None)

    registry = api._registry
    assert registry.middleware[0].unit.name == "_Middleware"
    assert registry.tools[0].name == "status"
    assert registry.backend_routes[0].unit is backend
    assert len(registry.shutdown_hooks) == 1
    assert api.cwd == Path("/workspace")
    assert api.mode == "interactive"
    assert api.path == Path("/extensions/example.py")


@pytest.mark.parametrize(
    "prefix",
    [
        "memories/",
        "/memories",
        "/../memories/",
        "/./memories/",
        "/memory//notes/",
        "/Memories/",
        "/memory.notes/",
        "/memories\\notes/",
        "/memories/?user=1",
        "/memories/#fragment",
    ],
)
def test_rejects_unsafe_backend_route_prefixes(api: ExtensionAPI, prefix: str) -> None:
    """Routes must be canonical virtual paths without traversal syntax."""
    with pytest.raises(ExtensionError, match="Invalid backend route prefix"):
        api.register_backend_route(prefix, FilesystemBackend(virtual_mode=False))


def test_rejects_invalid_registration_values(api: ExtensionAPI) -> None:
    """Invalid unit types should fail at the extension boundary."""
    with pytest.raises(ExtensionError, match="Could not construct middleware"):
        api.register_middleware(_ConfiguredMiddleware)
    with pytest.raises(ExtensionError, match="must be an AgentMiddleware"):
        # Wrong runtime types are intentional in these boundary checks.
        api.register_middleware(cast("Any", object()))
    with pytest.raises(ExtensionError, match="not a BackendProtocol"):
        api.register_backend_route("/memories/", cast("Any", object()))
    with pytest.raises(ExtensionError, match="not callable"):
        api.on_shutdown(cast("Any", None))


def test_registration_closes_with_factory_scope(api: ExtensionAPI) -> None:
    """Retaining the registrar must not allow post-compilation mutation."""
    api._close()

    with pytest.raises(ExtensionError, match="only while their factory is running"):
        api.register_middleware(_Middleware())
