"""Tests for merging extension units into the dcode agent."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

from deepagents.backends import FilesystemBackend
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.tools import tool

from deepagents_code.agent import (
    _extension_backend_routes,
    _merge_extension_middleware,
    _merge_extension_tools,
)
from deepagents_code.extensions import (
    ExtensionRegistry,
    SourceInfo,
    UnitOrigin,
    UnitScope,
)


def _source() -> SourceInfo:
    """Build provenance for the hosting tests."""
    return SourceInfo(
        path=Path("/extensions/example.py"),
        scope=UnitScope.USER,
        origin=UnitOrigin.TOP_LEVEL,
    )


@tool("existing")
def _existing_tool() -> str:
    """Represent an existing dcode tool."""
    return "existing"


@tool("unique")
def _unique_tool() -> str:
    """Represent a unique extension tool."""
    return "unique"


class _ExistingMiddleware(AgentMiddleware):
    """Represent middleware already installed by dcode."""

    name = "existing"


class _UniqueMiddleware(AgentMiddleware):
    """Represent unique extension middleware."""

    name = "unique"


def test_internal_backend_namespaces_reject_overlapping_routes() -> None:
    """Extensions must not intercept any part of an internal route tree."""
    registry = ExtensionRegistry()
    source = _source()
    reserved_child = FilesystemBackend(virtual_mode=False)
    reserved_parent = FilesystemBackend(virtual_mode=False)
    safe = FilesystemBackend(virtual_mode=False)
    registry.add_backend_route("/artifacts/messages/private/", reserved_child, source)
    registry.add_backend_route("/artifacts/", reserved_parent, source)
    registry.add_backend_route("/memories/", safe, source)

    routes = _extension_backend_routes(
        registry,
        reserved=["/artifacts/messages/"],
    )

    assert routes == {"/memories/": safe}


def test_builtin_tool_and_middleware_names_take_precedence() -> None:
    """Extension units should append only when their names remain available."""
    registry = ExtensionRegistry()
    source = _source()
    registry.add_tool(_existing_tool, source)
    registry.add_tool(_unique_tool, source)
    registry.add_middleware(_ExistingMiddleware(), source)
    registry.add_middleware(_UniqueMiddleware(), source)

    tools = _merge_extension_tools([_existing_tool], registry)
    middleware = _merge_extension_middleware([_ExistingMiddleware()], registry)

    assert tools == [_existing_tool, _unique_tool]
    assert [item.name for item in middleware] == ["existing", "unique"]


async def test_server_lifespan_releases_extensions() -> None:
    """LangGraph shutdown should await server-owned extension teardown."""
    from starlette.applications import Starlette

    from deepagents_code.server_lifespan import _lifespan

    shutdown = AsyncMock()
    with patch(
        "deepagents_code.extensions.runtime.shutdown_server_extensions", shutdown
    ):
        async with _lifespan(Starlette()):
            pass

    shutdown.assert_awaited_once_with()
