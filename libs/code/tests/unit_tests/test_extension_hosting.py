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
from starlette.routing import Route

from deepagents_code.agent import create_cli_agent
from deepagents_code.extensions.registry import (
    ExtensionError,
    ExtensionRegistry,
    SourceInfo,
)
from deepagents_code.offload import _ArtifactsStorage


@pytest.fixture(autouse=True)
def _experimental(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPAGENTS_CODE_EXPERIMENTAL", "1")


@tool("existing")
def _existing() -> str:
    """Represent an existing dcode tool."""
    return "existing"


@tool("existing")
def _replacement() -> str:
    """Represent an extension override."""
    return "replacement"


class _Middleware(AgentMiddleware):
    """Represent extension middleware."""

    name = "extension-middleware"


def test_agent_ignores_registry_outside_experimental_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Direct callers cannot bypass the process-wide experimental gate."""
    monkeypatch.delenv("DEEPAGENTS_CODE_EXPERIMENTAL")
    registry = ExtensionRegistry()
    registry.add_tool(_replacement, SourceInfo(tmp_path / "extension.py"))
    graph = Mock()
    graph.with_config.return_value = graph
    model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))
    model.profile = {"max_input_tokens": 200000}

    with patch("deepagents_code.agent.create_deep_agent", return_value=graph) as create:
        create_cli_agent(
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

    assert create.call_args.kwargs["tools"] == [_existing]
    assert all(
        item.name != "__deepagents_extension_runtime__"
        for item in create.call_args.kwargs["middleware"]
    )


def test_extension_units_override_tools_and_add_backend_routes(
    tmp_path: Path,
) -> None:
    """Extension tools win collisions while internal routes remain protected."""
    source = SourceInfo(tmp_path / "extension.py")
    registry = ExtensionRegistry()
    registry.add_tool(_replacement, source)
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
    assert _existing not in kwargs["tools"]
    assert _replacement in kwargs["tools"]
    assert any(item.name == "extension-middleware" for item in kwargs["middleware"])
    assert "/memories/" in backend.routes


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


def test_provenance_endpoint_enforces_flag_and_loopback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The metadata route is experimental and loopback-only."""
    from deepagents_code.offload_api import _extensions

    local = Request({"type": "http", "client": ("127.0.0.1", 1234)})
    remote = Request({"type": "http", "client": ("192.0.2.1", 1234)})

    assert _extensions(local).status_code == 200
    assert _extensions(remote).status_code == 404
    monkeypatch.delenv("DEEPAGENTS_CODE_EXPERIMENTAL")
    assert _extensions(local).status_code == 404


async def test_builtin_http_app_hosts_extension_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The built-in offload app also owns extension lifecycle and metadata."""
    from deepagents_code.offload_api import app

    shutdown = AsyncMock()
    with patch(
        "deepagents_code.extensions.runtime.shutdown_server_extensions", shutdown
    ):
        async with app.router.lifespan_context(app):
            monkeypatch.delenv("DEEPAGENTS_CODE_EXPERIMENTAL")
    shutdown.assert_awaited_once_with()
    assert any(
        isinstance(route, Route) and route.path == "/extensions" for route in app.routes
    )
