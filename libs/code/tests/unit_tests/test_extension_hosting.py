"""Tests for agent-server extension hosting."""

from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest
from deepagents.backends import FilesystemBackend
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from starlette.requests import Request

from deepagents_code.agent import create_cli_agent
from deepagents_code.extensions.registry import (
    ExtensionError,
    ExtensionRegistry,
    SourceInfo,
)
from deepagents_code.offload import _ArtifactsStorage


@tool("existing")
def _existing() -> str:
    """Represent an existing dcode tool."""
    return "existing"


@tool("unique")
def _unique() -> str:
    """Represent an extension tool."""
    return "unique"


@tool("existing")
def _replacement() -> str:
    """Represent an extension override."""
    return "replacement"


class _Middleware(AgentMiddleware):
    """Represent extension middleware."""

    name = "extension-middleware"


def test_extension_units_override_tools_and_add_backend_routes(
    tmp_path: Path,
) -> None:
    """Extension tools win collisions while internal routes remain protected."""
    source = SourceInfo(tmp_path / "extension.py", plugin_id="memory@test")
    registry = ExtensionRegistry()
    registry.add_tool(_replacement, source)
    registry.add_tool(_unique, source)
    registry.add_middleware(_Middleware(), source)
    registry.add_backend_route(
        "/memories/",
        FilesystemBackend(root_dir=tmp_path / "memories", virtual_mode=True),
        source,
    )
    graph = Mock()
    graph.with_config.return_value = graph
    model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))
    model.profile = {"max_input_tokens": 200000}

    with (
        patch("deepagents_code.agent.create_deep_agent", return_value=graph) as create,
        patch(
            "deepagents_code.agent._artifacts_root",
            return_value=_ArtifactsStorage(root="/artifacts"),
        ),
    ):
        _, backend = create_cli_agent(
            model,
            "test",
            tools=[_existing],
            enable_memory=False,
            enable_skills=False,
            enable_shell=False,
            system_prompt="test",
            cwd=tmp_path,
            extension_registry=registry,
        )

    kwargs = create.call_args.kwargs
    assert [item.name for item in kwargs["tools"]].count("existing") == 1
    assert (
        next(item for item in kwargs["tools"] if item.name == "existing")
        is _replacement
    )
    assert any(item.name == "unique" for item in kwargs["tools"])
    assert any(item.name == "extension-middleware" for item in kwargs["middleware"])
    assert "/memories/" in backend.routes
    assert registry.backend_routes[0].source.plugin_id == "memory@test"


def test_internal_backend_route_overlap_fails_agent_startup(tmp_path: Path) -> None:
    """An internal storage collision fails instead of being silently ignored."""
    registry = ExtensionRegistry()
    registry.add_backend_route(
        "/artifacts/",
        FilesystemBackend(root_dir=tmp_path / "blocked", virtual_mode=True),
        SourceInfo(tmp_path / "extension.py"),
    )
    model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))
    model.profile = {"max_input_tokens": 200000}

    with (
        patch("deepagents_code.agent.create_deep_agent", return_value=Mock()),
        patch(
            "deepagents_code.agent._artifacts_root",
            return_value=_ArtifactsStorage(root="/artifacts"),
        ),
        pytest.raises(ExtensionError, match="overlaps an internal route"),
    ):
        create_cli_agent(
            model,
            "test",
            enable_memory=False,
            enable_skills=False,
            enable_shell=False,
            system_prompt="test",
            cwd=tmp_path,
            extension_registry=registry,
        )


async def test_server_lifespan_releases_extensions() -> None:
    """LangGraph shutdown awaits server-owned extension teardown."""
    from starlette.applications import Starlette

    from deepagents_code.server_lifespan import _lifespan

    shutdown = AsyncMock()
    with patch(
        "deepagents_code.extensions.runtime.shutdown_server_extensions", shutdown
    ):
        async with _lifespan(Starlette()):
            pass
    shutdown.assert_awaited_once_with()


def test_provenance_endpoint_is_loopback_only(tmp_path: Path) -> None:
    """The metadata route exposes no extension objects and rejects remote peers."""
    from deepagents_code.extensions.runtime import bind_server_extensions
    from deepagents_code.server_lifespan import _extensions

    registry = ExtensionRegistry()
    registry.add_tool(_unique, SourceInfo(tmp_path / "extension.py"))
    bind_server_extensions(registry, errors=("broken",))

    local = Request({"type": "http", "client": ("127.0.0.1", 1234)})
    response = _extensions(local)
    remote = Request({"type": "http", "client": ("192.0.2.1", 1234)})

    assert response.status_code == 200
    assert b'"name":"unique"' in response.body
    assert b'"errors":["broken"]' in response.body
    assert b'"restart_required":false' in response.body
    assert _extensions(remote).status_code == 404
