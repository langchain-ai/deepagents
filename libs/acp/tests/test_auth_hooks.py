"""Sign-in and readiness hooks: what a client sees before it has credentials."""

from __future__ import annotations

from typing import Any, TypeVar

import pytest
from acp import PROTOCOL_VERSION
from acp.exceptions import RequestError
from acp.schema import AuthCapabilities, ClientCapabilities, TerminalAuthMethod

from deepagents_acp.server import AgentServerACP, AgentSessionContext
from tests.conformance.harness import RecordingClient, deep_agent, scripted_model

_ServerT = TypeVar("_ServerT", bound=AgentServerACP)

_LOGIN = TerminalAuthMethod(id="anthropic", name="Anthropic API key", args=["auth", "login"])


class _SignInServer(AgentServerACP):
    ready = False

    def auth_methods(self, client_capabilities: ClientCapabilities) -> list[TerminalAuthMethod]:
        terminal = client_capabilities.auth is not None and client_capabilities.auth.terminal
        return [_LOGIN] if terminal else []

    async def check_ready(self, context: AgentSessionContext) -> None:
        if not self.ready:
            raise RequestError.auth_required({"message": f"no key for {context.model}"})


def _connected(server: _ServerT) -> _ServerT:
    server.on_connect(RecordingClient())
    return server


def _server() -> _SignInServer:
    return _SignInServer(
        agent=lambda _context: deep_agent(scripted_model()),
        models=[{"value": "anthropic:claude", "name": "Claude"}],
    )


async def test_initialize_answers_with_the_servers_protocol_version() -> None:
    response = await _server().initialize(protocol_version=PROTOCOL_VERSION + 1)

    assert response.protocol_version == PROTOCOL_VERSION
    assert response.agent_info is not None
    assert response.agent_info.name == "deepagents-acp"


async def test_terminal_methods_need_the_client_capability() -> None:
    terminal = ClientCapabilities(auth=AuthCapabilities(terminal=True))

    without = await _server().initialize(protocol_version=1)
    with_terminal = await _server().initialize(protocol_version=1, client_capabilities=terminal)

    assert without.auth_methods == []
    assert with_terminal.auth_methods == [_LOGIN]


async def test_new_session_requires_auth_until_ready() -> None:
    server = _server()

    with pytest.raises(RequestError) as excinfo:
        await server.new_session(cwd="/tmp", mcp_servers=[])
    assert excinfo.value.code == -32000
    assert excinfo.value.data == {"message": "no key for anthropic:claude"}

    server.ready = True
    assert (await server.new_session(cwd="/tmp", mcp_servers=[])).session_id


async def test_authenticate_rejects_terminal_methods() -> None:
    server = _server()
    terminal = ClientCapabilities(auth=AuthCapabilities(terminal=True))
    await server.initialize(protocol_version=1, client_capabilities=terminal)

    with pytest.raises(RequestError) as excinfo:
        await server.authenticate(method_id="anthropic")
    assert excinfo.value.code == -32600

    with pytest.raises(RequestError) as excinfo:
        await server.authenticate(method_id="missing")
    assert excinfo.value.code == -32602


async def test_load_checks_readiness_for_the_restored_model() -> None:
    from langgraph.checkpoint.memory import MemorySaver

    checkpointer = MemorySaver()
    checked: list[str | None] = []

    class Server(AgentServerACP):
        async def check_ready(self, context: AgentSessionContext) -> None:
            checked.append(context.model)
            if context.model == "openai:gpt":
                raise RequestError.auth_required({"message": "no key for openai"})

    models = [
        {"value": "anthropic:claude", "name": "Claude"},
        {"value": "openai:gpt", "name": "GPT"},
    ]

    def build(_context: AgentSessionContext) -> Any:
        return deep_agent(scripted_model(), checkpointer=checkpointer)

    first = _connected(Server(agent=build, models=models, load_sessions=True))
    session = await first.new_session(cwd="/tmp", mcp_servers=[])
    await first.set_config_option(
        config_id="model", session_id=session.session_id, value="openai:gpt"
    )

    restarted = _connected(Server(agent=build, models=models, load_sessions=True))
    with pytest.raises(RequestError) as excinfo:
        await restarted.load_session(cwd="/tmp", session_id=session.session_id, mcp_servers=[])

    assert excinfo.value.code == -32000
    assert checked[-1] == "openai:gpt"
    assert session.session_id not in restarted._sessions


async def test_load_never_builds_the_unusable_launch_model() -> None:
    from langgraph.checkpoint.memory import MemorySaver

    checkpointer = MemorySaver()
    usable = {"value": True}

    class Server(AgentServerACP):
        async def check_ready(self, context: AgentSessionContext) -> None:
            if context.model == "anthropic:claude" and not usable["value"]:
                raise RequestError.auth_required({"message": "no key for anthropic"})

    models = [
        {"value": "anthropic:claude", "name": "Claude"},
        {"value": "openai:gpt", "name": "GPT"},
    ]

    def build(context: AgentSessionContext) -> Any:
        if context.model == "anthropic:claude" and not usable["value"]:
            msg = "building the launch model needs an Anthropic key"
            raise AssertionError(msg)
        return deep_agent(scripted_model(), checkpointer=checkpointer)

    first = _connected(Server(agent=build, models=models, load_sessions=True))
    session = await first.new_session(cwd="/tmp", mcp_servers=[])
    await first.set_config_option(
        config_id="model", session_id=session.session_id, value="openai:gpt"
    )

    usable["value"] = False
    restarted = _connected(
        Server(agent=build, models=models, load_sessions=True, checkpointer=checkpointer)
    )
    loaded = await restarted.load_session(cwd="/tmp", session_id=session.session_id, mcp_servers=[])

    model = next(option for option in loaded.config_options if option.id == "model")
    assert model.current_value == "openai:gpt"


async def test_load_restores_a_saved_model_when_there_is_no_default_yet() -> None:
    # dcode's startup path when the launch model is unusable: a catalog, no default.
    from langgraph.checkpoint.memory import MemorySaver

    checkpointer = MemorySaver()
    built: list[str | None] = []
    checked: list[str | None] = []
    models = [
        {"value": "anthropic:claude", "name": "Claude"},
        {"value": "openai:gpt", "name": "GPT"},
    ]

    def build(context: AgentSessionContext) -> Any:
        built.append(context.model)
        return deep_agent(scripted_model(), checkpointer=checkpointer)

    first = _connected(AgentServerACP(agent=build, models=models, load_sessions=True))
    session = await first.new_session(cwd="/tmp", mcp_servers=[])
    await first.set_config_option(
        config_id="model", session_id=session.session_id, value="openai:gpt"
    )

    class NoDefault(AgentServerACP):
        def default_model(self) -> str | None:
            return None

        async def check_ready(self, context: AgentSessionContext) -> None:
            checked.append(context.model)
            if context.model is None:
                raise RequestError.auth_required({"message": "no key for anthropic"})

    built.clear()
    restarted = _connected(
        NoDefault(agent=build, models=models, load_sessions=True, checkpointer=checkpointer)
    )
    await restarted.load_session(cwd="/tmp", session_id=session.session_id, mcp_servers=[])

    assert checked == ["openai:gpt"]
    assert built == ["openai:gpt"]
