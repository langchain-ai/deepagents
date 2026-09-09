from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, ClassVar, Self

import anyio
import pytest
from langchain_mcp_adapters.interceptors import MCPToolCallRequest
from mcp.client.auth import OAuthFlowError
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken
from mcp.types import CallToolResult
from pydantic import SecretStr

from deepagents_talon.authorization import (
    AuthorizationAttempt,
    AuthorizationBinding,
    AuthorizationCompleted,
    AuthorizationEvent,
    AuthorizationURL,
    CallbackURLRequested,
    DeviceCode,
    current_authorization_attempt,
    reset_authorization_handler,
    set_authorization_handler,
)
from deepagents_talon.config import TalonConfig
from deepagents_talon.mcp import (
    MCPConfigError,
    MCPServerInfo,
    MCPToolInfo,
    MCPToolProvider,
    _argument_normalization_interceptor,
    _authorization_interceptor,
    _connection,
    _normalize_mcp_arguments,
    _run_authorized,
    load_mcp_tools,
    login_mcp_server,
    mcp_config_path,
)
from deepagents_talon.mcp_auth import (
    DeviceAuthorizationCompletedError,
    FileTokenStorage,
    MCPAuthorizationError,
    _DeviceCodeResponse,
    _present_device_code,
    build_oauth_provider,
)

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class DummyTool:
    name: str
    description: str = ""
    args_schema: dict[str, object] | None = None


class FakeMCPClient:
    calls: ClassVar[list[dict[str, object]]] = []

    def __init__(self, connections: dict[str, object], **kwargs: object) -> None:
        self.connections = connections
        self.calls.append({"connections": connections, **kwargs})

    async def get_tools(self, *, server_name: str | None = None) -> list[DummyTool]:
        assert server_name is not None
        return [
            DummyTool(
                f"{server_name}_read",
                "Read files",
                {"type": "object", "properties": {"path": {"type": "string"}}},
            )
        ]


def _oauth_token() -> OAuthToken:
    return OAuthToken(access_token="secret-token")  # noqa: S106


async def _no_tokens() -> None:
    return None


async def _stored_tokens() -> OAuthToken:
    return _oauth_token()


class EmptyOAuthStorage:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass

    async def get_tokens(self) -> None:
        return None


def _write_config(path: Path, servers: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"mcpServers": servers}), encoding="utf-8")


def _config(tmp_path: Path, env: dict[str, str] | None = None) -> TalonConfig:
    return TalonConfig.from_env(
        {
            "AGENT_ASSISTANT_ID": "test",
            "DEEPAGENTS_TALON_WORKSPACE": str(tmp_path / "workspace"),
            **(env or {}),
        },
        base_home=tmp_path,
    )


async def _assert_refresh_scheduled(
    provider: MCPToolProvider,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def load() -> SimpleNamespace:
        return SimpleNamespace(tools=(DummyTool("refreshed"),))

    monkeypatch.setattr(provider, "load", load)
    refreshed = await provider.refresh_if_needed()
    assert refreshed is not None
    assert [tool.name for tool in refreshed] == ["refreshed"]


def test_mcp_metadata_contracts_are_talon_owned() -> None:
    tool = MCPToolInfo(
        name="search",
        description="Search documents",
        input_schema={"type": "object"},
    )
    server = MCPServerInfo(
        name="docs",
        transport="http",
        tools=(tool,),
        uses_oauth=True,
    )

    assert server.tools == (tool,)
    assert server.status == "ok"
    assert server.needs_attention() is False


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"error": "failed"}, "status='ok' cannot carry an error"),
        ({"status": "error"}, "requires an error message"),
        (
            {
                "status": "unauthenticated",
                "error": "login",
                "tools": (MCPToolInfo(name="search", description=""),),
            },
            "cannot carry tools",
        ),
        ({"pending_reconnect": True}, "pending_reconnect requires status='disabled'"),
    ],
)
def test_mcp_server_info_rejects_inconsistent_state(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        MCPServerInfo(name="docs", transport="http", **kwargs)


def test_mcp_server_info_reports_authentication_attention() -> None:
    server = MCPServerInfo(
        name="docs",
        transport="http",
        status="unauthenticated",
        error="login required",
    )

    assert server.needs_attention() is True


def test_mcp_config_path_uses_standard_path_or_env_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    monkeypatch.setattr("deepagents_talon.mcp.Path.home", lambda: home)

    assert mcp_config_path(_config(tmp_path)) == home / ".deepagents" / ".mcp.json"

    custom = tmp_path / "custom.mcp.json"
    assert (
        mcp_config_path(_config(tmp_path, {"DEEPAGENTS_TALON_MCP_CONFIG": str(custom)})) == custom
    )


async def test_load_mcp_tools_uses_standard_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeMCPClient.calls.clear()
    home = tmp_path / "home"
    monkeypatch.setattr("deepagents_talon.mcp.Path.home", lambda: home)
    monkeypatch.setattr("deepagents_talon.mcp.MultiServerMCPClient", FakeMCPClient)
    _write_config(
        home / ".deepagents" / ".mcp.json",
        {"remote": {"type": "http", "url": "https://example.com/mcp"}},
    )
    result = await load_mcp_tools(_config(tmp_path))

    assert [tool.name for tool in result.tools] == ["remote_read"]
    assert [server.name for server in result.servers] == ["remote"]
    assert result.servers[0].tools[0].input_schema == {
        "type": "object",
        "properties": {"path": {"type": "string"}},
    }
    assert FakeMCPClient.calls[0]["connections"] == {
        "remote": {
            "transport": "streamable_http",
            "url": "https://example.com/mcp",
            "timeout": 30.0,
        }
    }


async def test_mcp_interceptor_resumes_same_bound_invocation_without_secret_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("deepagents_talon.mcp_auth.Path.home", lambda: tmp_path)
    storage = FileTokenStorage("notion", server_url="https://mcp.example")
    provider = build_oauth_provider(
        server_name="notion",
        server_url="https://mcp.example",
        storage=storage,
        interactive=False,
    )
    events: list[AuthorizationEvent] = []
    callback = "http://localhost:3000/callback?code=secret-code&state=secret-state"

    async def authorize(event: AuthorizationEvent) -> str | None:
        events.append(event)
        return callback if isinstance(event, CallbackURLRequested) else None

    async def execute(_request: MCPToolCallRequest) -> CallToolResult:
        redirect_handler = provider.context.redirect_handler
        callback_handler = provider.context.callback_handler
        assert redirect_handler is not None
        assert callback_handler is not None
        await redirect_handler("https://auth.example/authorize?state=secret-state")
        assert await callback_handler() == ("secret-code", "secret-state")
        await storage.set_tokens(_oauth_token())
        return CallToolResult(content=[])

    token = set_authorization_handler(authorize)
    try:
        result = await _authorization_interceptor(
            MCPToolCallRequest(
                name="search",
                args={},
                server_name="notion",
                runtime=SimpleNamespace(tool_call_id="tool-call-42"),
            ),
            execute,
        )
    finally:
        reset_authorization_handler(token)

    assert result.content == []
    assert [event.type for event in events] == [
        "authorization_url",
        "callback_url_requested",
        "completed",
    ]
    assert all(event.binding.invocation_id == "tool-call-42" for event in events)
    assert isinstance(events[0], AuthorizationURL)
    assert isinstance(events[-1], AuthorizationCompleted)
    assert "secret-state" not in repr(events[0])
    assert "secret-code" not in repr(events)


async def test_github_device_code_is_bound_outside_model_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("deepagents_talon.mcp_auth.Path.home", lambda: tmp_path)
    storage = FileTokenStorage("github", server_url="https://api.githubcopilot.com/mcp")
    device = _DeviceCodeResponse(
        device_code=SecretStr("device-secret"),
        user_code=SecretStr("ABCD-1234"),
        verification_uri="https://github.com/login/device",
        expires_in=120,
    )
    events: list[AuthorizationEvent] = []

    async def authorize(event: AuthorizationEvent) -> None:
        events.append(event)

    async def execute() -> None:
        await _present_device_code(
            "github",
            device,
            deadline=asyncio.get_running_loop().time() + 120,
            interactive=False,
        )
        await storage.set_tokens_and_client_info(
            _oauth_token(),
            OAuthClientInformationFull(
                redirect_uris=["http://localhost/callback"],
                client_id="client-id",
            ),
        )

    attempt = AuthorizationAttempt(terminal=True)
    token = set_authorization_handler(authorize)
    try:
        await _run_authorized("tool-call-42", execute, attempt=attempt)
    finally:
        reset_authorization_handler(token)

    assert [type(event) for event in events] == [DeviceCode, AuthorizationCompleted]
    assert all(event.binding.invocation_id == "tool-call-42" for event in events)
    assert attempt.completed is True
    assert "ABCD-1234" not in repr(events[0])
    assert "github.com/login/device" not in repr(events[0])


def test_normalize_mcp_arguments_omits_only_optional_empty_strings() -> None:
    schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "integrationId": {"type": "string"},
            "fetchMode": {"type": "object"},
        },
        "required": ["query"],
    }

    arguments = _normalize_mcp_arguments(
        {"query": "", "integrationId": "", "fetchMode": {}}, schema
    )

    assert arguments == {"query": "", "fetchMode": {}}


async def test_argument_normalization_interceptor_overrides_request_arguments() -> None:
    request = MCPToolCallRequest(
        name="listVulnerabilities",
        args={"severity": "CRITICAL", "integrationId": ""},
        server_name="vanta",
    )
    received: list[MCPToolCallRequest] = []
    expected = CallToolResult(content=[])

    async def execute(normalized: MCPToolCallRequest) -> CallToolResult:
        received.append(normalized)
        return expected

    result = await _argument_normalization_interceptor(
        request,
        execute,
        input_schemas={
            "listVulnerabilities": {
                "type": "object",
                "properties": {
                    "severity": {"type": "string"},
                    "integrationId": {"type": "string"},
                },
            }
        },
    )

    assert result is expected
    assert received[0].args == {"severity": "CRITICAL"}
    assert request.args == {"severity": "CRITICAL", "integrationId": ""}


async def test_mcp_tool_provider_exposes_only_configured_server_authentication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "oauth.mcp.json"
    _write_config(
        config_path,
        {"notion": {"url": "https://mcp.example", "auth": "oauth"}},
    )
    monkeypatch.setattr(
        "deepagents_talon.mcp.FileTokenStorage.get_tokens",
        lambda _self: _no_tokens(),
    )
    provider = MCPToolProvider(_config(tmp_path, {"DEEPAGENTS_TALON_MCP_CONFIG": str(config_path)}))

    loaded = await provider.load()

    tools = {tool.name: tool for tool in loaded.tools}
    status_tool = tools["get_mcp_server_status"]
    assert status_tool.description == (
        "Report configured MCP server availability. Current servers: notion (unauthenticated)."
    )
    assert status_tool.invoke({}) == (
        {
            "server_name": "notion",
            "status": "unauthenticated",
            "can_authenticate": True,
        },
    )
    assert loaded.servers[0].status == "unauthenticated"
    assert loaded.servers[0].uses_oauth is True
    schema = tools["authenticate_mcp_server"].tool_call_schema.model_json_schema()
    assert schema["properties"]["reauthenticate"] == {
        "default": False,
        "description": (
            "Set true only when the user explicitly asks to log in again or switch accounts."
        ),
        "title": "Reauthenticate",
        "type": "boolean",
    }
    assert await provider._authenticate("unconfigured", "tool-call") == {"status": "failed"}


async def test_mcp_reload_tool_schedules_refresh_without_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = MCPToolProvider(
        _config(
            tmp_path,
            {"DEEPAGENTS_TALON_MCP_CONFIG": str(tmp_path / "missing.json")},
        )
    )
    reload_tool = (await provider.load()).tools[0]

    result = reload_tool.invoke({})

    assert reload_tool.name == "reload_mcp_configuration"
    assert result == {"status": "scheduled", "available": "after_successful_reload"}

    async def load() -> SimpleNamespace:
        return SimpleNamespace(tools=(DummyTool("refreshed"),))

    monkeypatch.setattr(provider, "load", load)
    refreshed = await provider.refresh_if_needed()
    assert refreshed is not None
    assert [tool.name for tool in refreshed] == ["refreshed"]


async def test_mcp_tool_provider_serializes_concurrent_refreshes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = MCPToolProvider(_config(tmp_path))
    provider.request_refresh()
    load_started = asyncio.Event()
    release_load = asyncio.Event()

    async def load() -> SimpleNamespace:
        load_started.set()
        await release_load.wait()
        return SimpleNamespace(tools=(DummyTool("refreshed"),))

    monkeypatch.setattr(provider, "load", load)
    first = asyncio.create_task(provider.refresh_if_needed())
    await load_started.wait()
    second = asyncio.create_task(provider.refresh_if_needed())
    await asyncio.sleep(0)

    assert not second.done()

    release_load.set()
    first_result, second_result = await asyncio.gather(first, second)

    assert first_result is not None
    assert [tool.name for tool in first_result] == ["refreshed"]
    assert second_result is None


async def test_mcp_tool_provider_preserves_refresh_requested_during_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = MCPToolProvider(_config(tmp_path))
    provider.request_refresh()
    load_started = asyncio.Event()
    release_load = asyncio.Event()
    loads = 0

    async def load() -> SimpleNamespace:
        nonlocal loads
        loads += 1
        if loads == 1:
            load_started.set()
            await release_load.wait()
        return SimpleNamespace(tools=(DummyTool(f"refreshed-{loads}"),))

    monkeypatch.setattr(provider, "load", load)
    first = asyncio.create_task(provider.refresh_if_needed())
    await load_started.wait()
    provider.request_refresh()
    release_load.set()

    first_result = await first
    second_result = await provider.refresh_if_needed()

    assert first_result is not None
    assert second_result is not None
    assert [tool.name for tool in first_result] == ["refreshed-1"]
    assert [tool.name for tool in second_result] == ["refreshed-2"]


async def test_mcp_tool_provider_retries_cancelled_refresh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = MCPToolProvider(_config(tmp_path))
    provider.request_refresh()
    load_started = asyncio.Event()
    release_load = asyncio.Event()
    loads = 0

    async def load() -> SimpleNamespace:
        nonlocal loads
        loads += 1
        if loads == 1:
            load_started.set()
            await release_load.wait()
        return SimpleNamespace(tools=(DummyTool("refreshed"),))

    monkeypatch.setattr(provider, "load", load)
    refresh = asyncio.create_task(provider.refresh_if_needed())
    await load_started.wait()
    refresh.cancel()

    with pytest.raises(asyncio.CancelledError):
        await refresh

    refreshed = await provider.refresh_if_needed()
    assert refreshed is not None
    assert [tool.name for tool in refreshed] == ["refreshed"]


async def test_mcp_tool_provider_does_not_retry_failed_refresh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = MCPToolProvider(_config(tmp_path))
    provider.request_refresh()

    async def load() -> SimpleNamespace:
        message = "invalid MCP configuration"
        raise MCPConfigError(message)

    monkeypatch.setattr(provider, "load", load)

    with pytest.raises(MCPConfigError, match="invalid MCP configuration"):
        await provider.refresh_if_needed()

    assert await provider.refresh_if_needed() is None


async def test_mcp_tool_provider_reports_existing_authorization_without_refresh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "oauth.mcp.json"
    _write_config(
        config_path,
        {"notion": {"url": "https://mcp.example", "auth": "oauth"}},
    )
    provider = MCPToolProvider(_config(tmp_path, {"DEEPAGENTS_TALON_MCP_CONFIG": str(config_path)}))
    provider._oauth_servers = frozenset({"notion"})
    monkeypatch.setattr(
        "deepagents_talon.mcp.FileTokenStorage.get_tokens",
        lambda _self: _stored_tokens(),
    )

    async def open_existing_session(_client: object, _server_name: str) -> None:
        return None

    monkeypatch.setattr("deepagents_talon.mcp._open_mcp_session", open_existing_session)

    result = await provider._authenticate("notion", "tool-call")

    assert result == {"status": "already_authenticated", "server_name": "notion"}
    assert await provider.refresh_if_needed() is None


async def test_mcp_tool_provider_forces_explicit_reauthentication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "oauth.mcp.json"
    _write_config(
        config_path,
        {"notion": {"url": "https://mcp.example", "auth": "oauth"}},
    )
    provider = MCPToolProvider(_config(tmp_path, {"DEEPAGENTS_TALON_MCP_CONFIG": str(config_path)}))
    provider._oauth_servers = frozenset({"notion"})
    forced: list[bool] = []

    async def connection(
        _server_name: str,
        _server: object,
        *,
        channel_authorization: bool,
        force_authorization: bool,
    ) -> tuple[dict[str, object], str]:
        assert channel_authorization is True
        assert current_authorization_attempt() is not None
        forced.append(force_authorization)
        return {}, "streamable_http"

    async def complete_authorization(_client: object, _server_name: str) -> None:
        attempt = current_authorization_attempt()
        assert attempt is not None
        attempt.binding = AuthorizationBinding(
            server_name="notion",
            invocation_id="tool-call",
            expires_at=asyncio.get_running_loop().time() + 30,
        )
        attempt.completed = True

    monkeypatch.setattr("deepagents_talon.mcp._connection", connection)
    monkeypatch.setattr("deepagents_talon.mcp._open_mcp_session", complete_authorization)

    result = await provider._authenticate("notion", "tool-call", reauthenticate=True)

    assert forced == [True]
    assert result == {"status": "completed", "server_name": "notion"}
    await _assert_refresh_scheduled(provider, monkeypatch)


def _provider_with_post_persistence_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exception_type: type[BaseException],
) -> MCPToolProvider:
    config_path = tmp_path / "oauth.mcp.json"
    _write_config(
        config_path,
        {"notion": {"url": "https://mcp.example", "auth": "oauth"}},
    )
    provider = MCPToolProvider(_config(tmp_path, {"DEEPAGENTS_TALON_MCP_CONFIG": str(config_path)}))
    provider._oauth_servers = frozenset({"notion"})
    monkeypatch.setattr(
        "deepagents_talon.mcp.FileTokenStorage.get_tokens",
        lambda _self: _no_tokens(),
    )

    async def complete_then_fail(_client: object, _server_name: str) -> None:
        attempt = current_authorization_attempt()
        assert attempt is not None
        attempt.binding = AuthorizationBinding(
            server_name="notion",
            invocation_id="tool-call",
            expires_at=asyncio.get_running_loop().time() + 30,
        )
        attempt.completed = True
        raise exception_type

    monkeypatch.setattr("deepagents_talon.mcp._open_mcp_session", complete_then_fail)
    return provider


@pytest.mark.parametrize("exception_type", [RuntimeError, KeyError])
async def test_mcp_tool_provider_refreshes_after_credentials_persist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exception_type: type[BaseException],
) -> None:
    provider = _provider_with_post_persistence_error(tmp_path, monkeypatch, exception_type)
    events: list[AuthorizationEvent] = []

    async def authorize(event: AuthorizationEvent) -> str | None:
        events.append(event)
        return None

    token = set_authorization_handler(authorize)
    try:
        result = await provider._authenticate("notion", "tool-call")
    finally:
        reset_authorization_handler(token)

    assert result == {"status": "completed", "server_name": "notion"}
    await _assert_refresh_scheduled(provider, monkeypatch)
    assert [type(event) for event in events] == [AuthorizationCompleted]
    assert isinstance(events[0], AuthorizationCompleted)
    assert events[0].terminal is True


async def test_mcp_tool_provider_propagates_cancellation_after_credentials_persist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _provider_with_post_persistence_error(
        tmp_path,
        monkeypatch,
        asyncio.CancelledError,
    )
    events: list[AuthorizationEvent] = []

    async def authorize(event: AuthorizationEvent) -> str | None:
        events.append(event)
        return None

    token = set_authorization_handler(authorize)
    try:
        with pytest.raises(asyncio.CancelledError):
            await provider._authenticate("notion", "tool-call")
    finally:
        reset_authorization_handler(token)

    await _assert_refresh_scheduled(provider, monkeypatch)
    assert [type(event) for event in events] == [AuthorizationCompleted]


async def test_explicit_config_interpolates_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeMCPClient.calls.clear()
    monkeypatch.setattr("deepagents_talon.mcp.MultiServerMCPClient", FakeMCPClient)
    config_path = tmp_path / "custom.mcp.json"
    _write_config(
        config_path,
        {
            "remote": {
                "type": "sse",
                "url": "${MCP_URL}",
                "headers": {"Authorization": "Bearer ${MCP_TOKEN}"},
            }
        },
    )

    monkeypatch.setenv("MCP_URL", "https://example.com/sse")
    monkeypatch.setenv("MCP_TOKEN", "secret")

    await load_mcp_tools(_config(tmp_path, {"DEEPAGENTS_TALON_MCP_CONFIG": str(config_path)}))

    assert FakeMCPClient.calls[0]["connections"] == {
        "remote": {
            "transport": "sse",
            "url": "https://example.com/sse",
            "timeout": 30.0,
            "headers": {"Authorization": "Bearer secret"},
        }
    }


@pytest.mark.parametrize(
    ("document", "match"),
    [
        ([], "must contain a JSON object"),
        ({}, "must contain an mcpServers object"),
        ({"mcpServers": {"bad/name": {"command": "x"}}}, "server name"),
    ],
)
async def test_invalid_config_fails_before_connecting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    document: object,
    match: str,
) -> None:
    FakeMCPClient.calls.clear()
    monkeypatch.setattr("deepagents_talon.mcp.MultiServerMCPClient", FakeMCPClient)
    config_path = tmp_path / "invalid.mcp.json"
    config_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(MCPConfigError, match=match):
        await load_mcp_tools(_config(tmp_path, {"DEEPAGENTS_TALON_MCP_CONFIG": str(config_path)}))

    assert FakeMCPClient.calls == []


async def test_server_connection_error_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FailingMCPClient(FakeMCPClient):
        async def get_tools(self, *, server_name: str | None = None) -> list[DummyTool]:
            assert server_name is not None
            msg = "connection failed"
            raise RuntimeError(msg)

    config_path = tmp_path / "custom.mcp.json"
    _write_config(config_path, {"remote": {"url": "https://example.com/mcp"}})
    monkeypatch.setattr("deepagents_talon.mcp.MultiServerMCPClient", FailingMCPClient)

    result = await load_mcp_tools(
        _config(tmp_path, {"DEEPAGENTS_TALON_MCP_CONFIG": str(config_path)})
    )

    assert result.tools == ()
    assert result.servers[0].status == "error"
    assert result.servers[0].error == "connection failed"

    provider = MCPToolProvider(_config(tmp_path, {"DEEPAGENTS_TALON_MCP_CONFIG": str(config_path)}))
    loaded = await provider.load()
    status_tool = loaded.tools[0]
    assert status_tool.description == (
        "Report configured MCP server availability. Current servers: remote (error)."
    )
    assert status_tool.invoke({}) == (
        {"server_name": "remote", "status": "error", "can_authenticate": False},
    )
    assert "connection failed" not in status_tool.description
    assert "connection failed" not in str(status_tool.invoke({}))


@pytest.mark.parametrize(
    "wrapped",
    [
        MCPAuthorizationError("authorization detail must not escape"),
        ExceptionGroup(
            "nested task group detail must not escape",
            [MCPAuthorizationError("nested authorization detail must not escape")],
        ),
    ],
)
async def test_wrapped_channel_authorization_error_does_not_abort_startup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    wrapped: Exception,
) -> None:
    class AuthorizationRequiredClient(FakeMCPClient):
        async def get_tools(self, *, server_name: str | None = None) -> list[DummyTool]:
            assert server_name == "notion"
            raise wrapped

    config_path = tmp_path / "oauth.mcp.json"
    _write_config(
        config_path,
        {"notion": {"url": "https://mcp.example", "auth": "oauth"}},
    )
    monkeypatch.setattr("deepagents_talon.mcp.MultiServerMCPClient", AuthorizationRequiredClient)
    monkeypatch.setattr(
        "deepagents_talon.mcp.FileTokenStorage.get_tokens",
        lambda _self: _stored_tokens(),
    )

    result = await load_mcp_tools(
        _config(tmp_path, {"DEEPAGENTS_TALON_MCP_CONFIG": str(config_path)})
    )

    assert result.tools == ()
    assert result.servers[0].status == "unauthenticated"
    assert result.servers[0].error == "MCP server 'notion' needs authentication"
    assert result.servers[0].uses_oauth is True
    assert "detail must not escape" not in caplog.text


async def test_unrelated_exception_group_remains_server_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class GroupedFailureClient(FakeMCPClient):
        async def get_tools(self, *, server_name: str | None = None) -> list[DummyTool]:
            assert server_name == "remote"
            msg = "nested detail"
            group_msg = "internal detail"
            raise ExceptionGroup(group_msg, [RuntimeError(msg)])

    config_path = tmp_path / "custom.mcp.json"
    _write_config(config_path, {"remote": {"url": "https://example.com/mcp"}})
    monkeypatch.setattr("deepagents_talon.mcp.MultiServerMCPClient", GroupedFailureClient)

    result = await load_mcp_tools(
        _config(tmp_path, {"DEEPAGENTS_TALON_MCP_CONFIG": str(config_path)})
    )

    assert result.tools == ()
    assert result.servers[0].status == "error"
    assert result.servers[0].error == "ExceptionGroup"


async def test_unexpected_server_error_does_not_block_other_servers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class PartiallyFailingMCPClient(FakeMCPClient):
        async def get_tools(self, *, server_name: str | None = None) -> list[DummyTool]:
            if server_name == "broken":
                msg = "unexpected failure"
                raise OAuthFlowError(msg)
            return await super().get_tools(server_name=server_name)

    config_path = tmp_path / "custom.mcp.json"
    _write_config(
        config_path,
        {
            "broken": {"url": "https://broken.example.com/mcp"},
            "working": {"url": "https://working.example.com/mcp"},
        },
    )
    monkeypatch.setattr("deepagents_talon.mcp.MultiServerMCPClient", PartiallyFailingMCPClient)

    result = await load_mcp_tools(
        _config(tmp_path, {"DEEPAGENTS_TALON_MCP_CONFIG": str(config_path)})
    )

    assert [tool.name for tool in result.tools] == ["working_read"]
    assert [(server.name, server.status) for server in result.servers] == [
        ("broken", "error"),
        ("working", "ok"),
    ]


async def test_tool_allowlist_filters_loaded_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class MultipleToolClient(FakeMCPClient):
        async def get_tools(self, *, server_name: str | None = None) -> list[DummyTool]:
            assert server_name is not None
            return [DummyTool(f"{server_name}_read"), DummyTool(f"{server_name}_write")]

    config_path = tmp_path / "custom.mcp.json"
    _write_config(
        config_path,
        {
            "remote": {
                "url": "https://example.com/mcp",
                "allowedTools": ["read"],
            }
        },
    )
    monkeypatch.setattr("deepagents_talon.mcp.MultiServerMCPClient", MultipleToolClient)

    result = await load_mcp_tools(
        _config(tmp_path, {"DEEPAGENTS_TALON_MCP_CONFIG": str(config_path)})
    )

    assert [tool.name for tool in result.tools] == ["remote_read"]
    assert [tool.name for tool in result.servers[0].tools] == ["remote_read"]


async def test_oauth_connection_uses_stored_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = object()

    class FakeStorage:
        def __init__(self, server_name: str, *, server_url: str) -> None:
            assert (server_name, server_url) == ("remote", "https://example.com/mcp")

        async def get_tokens(self) -> object:
            return object()

    monkeypatch.setattr("deepagents_talon.mcp.FileTokenStorage", FakeStorage)
    monkeypatch.setattr("deepagents_talon.mcp.build_oauth_provider", lambda **_kwargs: provider)
    monkeypatch.setattr("deepagents_talon.mcp.MultiServerMCPClient", FakeMCPClient)
    config_path = tmp_path / "custom.mcp.json"
    _write_config(
        config_path,
        {"remote": {"url": "https://example.com/mcp", "auth": "oauth"}},
    )

    result = await load_mcp_tools(
        _config(tmp_path, {"DEEPAGENTS_TALON_MCP_CONFIG": str(config_path)})
    )

    connection = FakeMCPClient.calls[-1]["connections"]
    assert isinstance(connection, dict)
    assert connection["remote"]["auth"] is provider
    assert result.servers[0].uses_oauth is True


async def test_oauth_connection_prepares_oauth_login(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = object()
    prepared: list[tuple[str, object]] = []

    class EmptyStorage:
        def __init__(self, server_name: str, *, server_url: str) -> None:
            assert (server_name, server_url) == (
                "github",
                "https://api.githubcopilot.com/mcp",
            )

    async def prepare(*, server_url: str, storage: object) -> None:
        prepared.append((server_url, storage))

    monkeypatch.setattr("deepagents_talon.mcp.FileTokenStorage", EmptyStorage)
    monkeypatch.setattr("deepagents_talon.mcp.prepare_oauth_login", prepare)
    monkeypatch.setattr("deepagents_talon.mcp.build_oauth_provider", lambda **_kwargs: provider)

    connection, transport = await _connection(
        "github",
        {"url": "https://api.githubcopilot.com/mcp", "auth": "oauth"},
        interactive=True,
    )

    assert transport == "streamable_http"
    assert connection["auth"] is provider
    assert len(prepared) == 1
    assert prepared[0][0] == "https://api.githubcopilot.com/mcp"


async def test_oauth_connection_reuses_stored_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = object()

    class StoredStorage:
        def __init__(self, _server_name: str, *, server_url: str) -> None:
            assert server_url == "https://example.com/mcp"

        async def get_tokens(self) -> OAuthToken:
            return _oauth_token()

    monkeypatch.setattr("deepagents_talon.mcp.FileTokenStorage", StoredStorage)
    monkeypatch.setattr("deepagents_talon.mcp.build_oauth_provider", lambda **_kwargs: provider)

    connection, _ = await _connection(
        "remote",
        {"url": "https://example.com/mcp", "auth": "oauth"},
    )

    assert connection["auth"] is provider


async def test_forced_oauth_connection_bypasses_stored_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = object()

    class ForcedStorage:
        def __init__(
            self,
            server_name: str,
            *,
            server_url: str,
            force_authorization: bool,
        ) -> None:
            assert (server_name, server_url) == ("remote", "https://example.com/mcp")
            assert force_authorization is True

    monkeypatch.setattr("deepagents_talon.mcp.FileTokenStorage", ForcedStorage)
    monkeypatch.setattr("deepagents_talon.mcp.build_oauth_provider", lambda **_kwargs: provider)

    connection, transport = await _connection(
        "remote",
        {"url": "https://example.com/mcp", "auth": "oauth"},
        channel_authorization=True,
        force_authorization=True,
    )

    assert transport == "streamable_http"
    assert connection["auth"] is provider


async def test_oauth_without_stored_credentials_requires_login(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class EmptyStorage:
        def __init__(self, server_name: str, *, server_url: str) -> None:
            assert (server_name, server_url) == ("remote", "https://example.com/mcp")

        async def get_tokens(self) -> None:
            return None

    monkeypatch.setattr("deepagents_talon.mcp.FileTokenStorage", EmptyStorage)
    config_path = tmp_path / "custom.mcp.json"
    _write_config(
        config_path,
        {"remote": {"url": "https://example.com/mcp", "auth": "oauth"}},
    )

    result = await load_mcp_tools(
        _config(tmp_path, {"DEEPAGENTS_TALON_MCP_CONFIG": str(config_path)})
    )

    assert result.tools == ()
    assert result.servers[0].status == "unauthenticated"
    assert result.servers[0].needs_attention() is True


async def test_login_uses_talon_config_and_interactive_oauth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "custom.mcp.json"
    _write_config(
        config_path,
        {"remote": {"url": "https://example.com/mcp", "auth": "oauth"}},
    )
    provider = object()
    calls: list[dict[str, object]] = []

    class LoginClient:
        def __init__(self, connections: dict[str, object]) -> None:
            calls.append(connections)

        def session(self, server_name: str):
            assert server_name == "remote"

            class Session:
                async def __aenter__(self) -> Self:
                    return self

                async def __aexit__(self, *_args: object) -> None:
                    return None

            return Session()

    monkeypatch.setattr("deepagents_talon.mcp.MultiServerMCPClient", LoginClient)
    monkeypatch.setattr("deepagents_talon.mcp._MCP_LOAD_TIMEOUT_SECONDS", 1)
    forced: list[bool] = []

    class LoginStorage:
        def __init__(
            self,
            _server_name: str,
            *,
            server_url: str,
            force_authorization: bool,
        ) -> None:
            assert server_url == "https://example.com/mcp"
            forced.append(force_authorization)

    monkeypatch.setattr("deepagents_talon.mcp.FileTokenStorage", LoginStorage)
    monkeypatch.setattr(
        "deepagents_talon.mcp.build_oauth_provider",
        lambda **kwargs: provider if kwargs["interactive"] else None,
    )

    result = await login_mcp_server(_config(tmp_path), "remote", str(config_path))

    assert result == 0
    assert forced == [True]
    assert calls == [
        {
            "remote": {
                "transport": "streamable_http",
                "url": "https://example.com/mcp",
                "timeout": 30.0,
                "auth": provider,
            }
        }
    ]


async def test_login_reports_oauth_failure_without_details(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = tmp_path / "custom.mcp.json"
    _write_config(config_path, {"remote": {"url": "https://example.com/mcp"}})

    failure = OAuthFlowError("secret token exchange response")

    async def fail_login(*_args: object) -> None:
        raise failure

    monkeypatch.setattr("deepagents_talon.mcp._open_mcp_session", fail_login)
    monkeypatch.setattr("deepagents_talon.mcp.FileTokenStorage", EmptyOAuthStorage)
    monkeypatch.setattr("deepagents_talon.mcp.build_oauth_provider", lambda **_kwargs: object())

    result = await login_mcp_server(_config(tmp_path), "remote", str(config_path))

    assert result == 1
    assert capsys.readouterr().err == "MCP login failed: OAuthFlowError\n"


async def test_login_does_not_timeout_interactive_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "custom.mcp.json"
    _write_config(config_path, {"remote": {"url": "https://example.com/mcp"}})

    async def slow_login(*_args: object) -> None:
        await asyncio.sleep(0.02)

    monkeypatch.setattr("deepagents_talon.mcp._open_mcp_session", slow_login)
    monkeypatch.setattr("deepagents_talon.mcp._MCP_LOAD_TIMEOUT_SECONDS", 0.001)
    monkeypatch.setattr("deepagents_talon.mcp.FileTokenStorage", EmptyOAuthStorage)
    monkeypatch.setattr("deepagents_talon.mcp.build_oauth_provider", lambda **_kwargs: object())

    assert await login_mcp_server(_config(tmp_path), "remote", str(config_path)) == 0


async def test_login_reports_missing_server_without_deepagents_code(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = tmp_path / "custom.mcp.json"
    _write_config(config_path, {})

    result = await login_mcp_server(_config(tmp_path), "missing", str(config_path))

    assert result == 1
    assert "was not found" in capsys.readouterr().err


async def test_invalid_stdio_environment_does_not_block_valid_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "custom.mcp.json"
    _write_config(
        config_path,
        {
            "unsafe": {"command": "x", "env": {"LD_PRELOAD": "x"}},
            "valid": {"url": "https://example.com/mcp"},
        },
    )
    monkeypatch.setattr("deepagents_talon.mcp.MultiServerMCPClient", FakeMCPClient)

    result = await load_mcp_tools(
        _config(tmp_path, {"DEEPAGENTS_TALON_MCP_CONFIG": str(config_path)})
    )

    assert [tool.name for tool in result.tools] == ["valid_read"]
    assert result.servers[0].error == "MCP stdio server 'unsafe' cannot set LD_PRELOAD"


async def test_invalid_server_does_not_block_valid_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "custom.mcp.json"
    _write_config(
        config_path,
        {
            "invalid": {"transport": "websocket", "url": "https://example.com"},
            "valid": {"url": "https://example.com/mcp"},
        },
    )
    monkeypatch.setattr("deepagents_talon.mcp.MultiServerMCPClient", FakeMCPClient)

    result = await load_mcp_tools(
        _config(tmp_path, {"DEEPAGENTS_TALON_MCP_CONFIG": str(config_path)})
    )

    assert [tool.name for tool in result.tools] == ["valid_read"]
    assert result.servers[0].status == "error"
    assert result.servers[1].status == "ok"


async def _raise_device_completion() -> None:
    raise DeviceAuthorizationCompletedError


async def _raise_oauth_failure() -> None:
    msg = "secret token exchange response"
    raise OAuthFlowError(msg)


async def _raise_connection_failure() -> None:
    msg = "connection reset"
    raise ConnectionError(msg)


async def test_login_succeeds_when_the_device_flow_completes_inside_a_task_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A completed device login exits 0: the session task group wraps the marker."""
    config_path = tmp_path / "custom.mcp.json"
    _write_config(config_path, {"remote": {"url": "https://example.com/mcp", "auth": "oauth"}})

    async def complete_device_login(*_args: object) -> None:
        async with anyio.create_task_group() as group:
            group.start_soon(_raise_device_completion)

    monkeypatch.setattr("deepagents_talon.mcp._open_mcp_session", complete_device_login)
    monkeypatch.setattr("deepagents_talon.mcp.FileTokenStorage", EmptyOAuthStorage)
    monkeypatch.setattr("deepagents_talon.mcp.build_oauth_provider", lambda **_kwargs: object())

    result = await login_mcp_server(_config(tmp_path), "remote", str(config_path))

    captured = capsys.readouterr()
    assert result == 0
    assert captured.err == ""
    assert captured.out == "Logged in to MCP server 'remote'.\n"


async def test_login_reports_a_grouped_failure_without_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = tmp_path / "custom.mcp.json"
    _write_config(config_path, {"remote": {"url": "https://example.com/mcp"}})

    async def fail_inside_task_group(*_args: object) -> None:
        async with anyio.create_task_group() as group:
            group.start_soon(_raise_oauth_failure)

    monkeypatch.setattr("deepagents_talon.mcp._open_mcp_session", fail_inside_task_group)
    monkeypatch.setattr("deepagents_talon.mcp.FileTokenStorage", EmptyOAuthStorage)
    monkeypatch.setattr("deepagents_talon.mcp.build_oauth_provider", lambda **_kwargs: object())

    result = await login_mcp_server(_config(tmp_path), "remote", str(config_path))

    error = capsys.readouterr().err
    assert result == 1
    assert error.startswith("MCP login failed: ")
    assert "secret token exchange response" not in error


async def test_authenticate_reports_failure_for_a_grouped_session_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A network failure with no binding must not raise the group into the agent."""
    config_path = tmp_path / "oauth.mcp.json"
    _write_config(config_path, {"notion": {"url": "https://mcp.example", "auth": "oauth"}})
    provider = MCPToolProvider(_config(tmp_path, {"DEEPAGENTS_TALON_MCP_CONFIG": str(config_path)}))
    provider._oauth_servers = frozenset({"notion"})
    monkeypatch.setattr(
        "deepagents_talon.mcp.FileTokenStorage.get_tokens",
        lambda _self: _stored_tokens(),
    )

    async def fail_inside_task_group(_client: object, _server_name: str) -> None:
        async with anyio.create_task_group() as group:
            group.start_soon(_raise_connection_failure)

    monkeypatch.setattr("deepagents_talon.mcp._open_mcp_session", fail_inside_task_group)

    result = await provider._authenticate("notion", "tool-call")

    assert result == {"status": "failed", "server_name": "notion"}
    assert await provider.refresh_if_needed() is None


async def test_login_succeeds_and_logs_a_failure_alongside_the_completion_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The marker is only raised after credentials persist, so the login did succeed."""
    config_path = tmp_path / "custom.mcp.json"
    _write_config(config_path, {"remote": {"url": "https://example.com/mcp", "auth": "oauth"}})

    async def complete_then_fail(*_args: object) -> None:
        async with anyio.create_task_group() as group:
            group.start_soon(_raise_device_completion)
            group.start_soon(_raise_connection_failure)

    monkeypatch.setattr("deepagents_talon.mcp._open_mcp_session", complete_then_fail)
    monkeypatch.setattr("deepagents_talon.mcp.FileTokenStorage", EmptyOAuthStorage)
    monkeypatch.setattr("deepagents_talon.mcp.build_oauth_provider", lambda **_kwargs: object())

    with caplog.at_level(logging.WARNING, logger="deepagents_talon.mcp"):
        result = await login_mcp_server(_config(tmp_path), "remote", str(config_path))

    captured = capsys.readouterr()
    assert result == 0
    assert captured.out == "Logged in to MCP server 'remote'.\n"
    assert captured.err == ""
    assert "after credentials were saved" in caplog.text
    assert "connection reset" in caplog.text
