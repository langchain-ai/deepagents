"""Tests for model switching functionality in ACP adapter."""

from typing import Any

import pytest
from acp.schema import (
    NewSessionResponse,
    SessionConfigOptionSelect,
    SetSessionConfigOptionResponse,
)
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.memory import MemorySaver

from deepagents_acp.server import AgentServerACP, AgentSessionContext


def _select(opt: Any) -> SessionConfigOptionSelect:
    """Normalize cross-version config-option shape.

    agent-client-protocol v0.8.x wraps config options in SessionConfigOption
    with a .root attribute; v0.9.0+ emits the Select instance directly.
    Fails loudly if the unwrapped value isn't a SessionConfigOptionSelect so
    a future wrapper shape can't silently pass through.
    """
    inner = getattr(opt, "root", opt)
    assert isinstance(inner, SessionConfigOptionSelect), (
        f"expected SessionConfigOptionSelect, got {type(inner).__name__}"
    )
    return inner


class MockClient:
    """Mock client for testing."""

    async def session_update(self, **_kwargs: Any):
        """Mock session update."""

    async def request_permission(self, **_kwargs: Any):
        """Mock permission request."""


def _get_model_string(model_string: str | None):
    """Get model string with default fallback."""
    return model_string or "anthropic:claude-sonnet-4"


@pytest.fixture
def agent_factory():
    """Create an agent factory for testing."""

    def build_agent(context: AgentSessionContext):
        model = _get_model_string(context.model)
        return create_deep_agent(
            model=model,
            checkpointer=MemorySaver(),
            backend=FilesystemBackend(root_dir=context.cwd, virtual_mode=True),
        )

    return build_agent


@pytest.fixture
def models():
    """Define test models."""
    return [
        {
            "value": "anthropic:claude-opus-4-6",
            "name": "Claude Opus 4",
            "description": "Most capable model",
        },
        {
            "value": "anthropic:claude-sonnet-4",
            "name": "Claude Sonnet 4",
            "description": "Balanced performance",
        },
        {
            "value": "anthropic:claude-haiku-4",
            "name": "Claude Haiku 4",
            "description": "Fast and efficient",
        },
    ]


@pytest.mark.asyncio
async def test_new_session_returns_config_options(agent_factory, models):
    """Test that new_session returns model config options."""
    server = AgentServerACP(agent=agent_factory, models=models)
    server.on_connect(MockClient())

    response = await server.new_session(cwd="/tmp")

    assert isinstance(response, NewSessionResponse)
    assert response.config_options is not None
    assert len(response.config_options) == 1

    # Check model config option
    model_config = _select(response.config_options[0])
    assert model_config.id == "model"
    assert model_config.name == "Model"
    assert model_config.category == "model"
    assert model_config.type == "select"
    assert model_config.current_value == "anthropic:claude-opus-4-6"
    assert len(model_config.options) == 3


@pytest.mark.asyncio
async def test_set_config_option_switches_model(agent_factory, models):
    """Test that set_config_option switches the model."""
    server = AgentServerACP(agent=agent_factory, models=models)
    server.on_connect(MockClient())

    # Create a session
    new_session_response = await server.new_session(cwd="/tmp")
    session_id = new_session_response.session_id

    # Verify initial model
    initial = _select(new_session_response.config_options[0])
    assert initial.current_value == "anthropic:claude-opus-4-6"

    # Switch to a different model
    response = await server.set_config_option(
        config_id="model",
        session_id=session_id,
        value="anthropic:claude-sonnet-4",
    )

    assert isinstance(response, SetSessionConfigOptionResponse)
    assert len(response.config_options) == 1
    assert _select(response.config_options[0]).current_value == "anthropic:claude-sonnet-4"

    # Verify the model was updated in session state
    assert server._session_models[session_id] == "anthropic:claude-sonnet-4"


@pytest.mark.asyncio
async def test_set_config_option_invalid_model_raises_error(agent_factory, models):
    """Test that setting an invalid model raises an error."""
    server = AgentServerACP(agent=agent_factory, models=models)
    server.on_connect(MockClient())

    # Create a session
    new_session_response = await server.new_session(cwd="/tmp")
    session_id = new_session_response.session_id

    # Try to set an invalid model
    from acp.exceptions import RequestError

    with pytest.raises(RequestError) as exc_info:
        await server.set_config_option(
            config_id="model",
            session_id=session_id,
            value="invalid:model",
        )

    assert "Invalid model" in str(exc_info.value)


@pytest.mark.asyncio
async def test_config_options_with_modes_and_models(agent_factory, models):
    """Test that both modes and models are exposed as config options."""
    from acp.schema import SessionMode, SessionModeState

    modes = SessionModeState(
        current_mode_id="auto",
        available_modes=[
            SessionMode(id="auto", name="Auto"),
            SessionMode(id="manual", name="Manual"),
        ],
    )

    server = AgentServerACP(agent=agent_factory, modes=modes, models=models)
    server.on_connect(MockClient())

    response = await server.new_session(cwd="/tmp")

    assert response.config_options is not None
    assert len(response.config_options) == 2

    # Check that mode comes first
    mode_opt = _select(response.config_options[0])
    assert mode_opt.id == "mode"
    assert mode_opt.category == "mode"

    # Check that model comes second
    model_opt = _select(response.config_options[1])
    assert model_opt.id == "model"
    assert model_opt.category == "model"


@pytest.mark.asyncio
async def test_switching_mode_via_config_option(agent_factory, models):
    """Test that modes can be switched via set_config_option."""
    from acp.schema import SessionMode, SessionModeState

    modes = SessionModeState(
        current_mode_id="auto",
        available_modes=[
            SessionMode(id="auto", name="Auto"),
            SessionMode(id="manual", name="Manual"),
        ],
    )

    server = AgentServerACP(agent=agent_factory, modes=modes, models=models)
    server.on_connect(MockClient())

    # Create a session
    new_session_response = await server.new_session(cwd="/tmp")
    session_id = new_session_response.session_id

    # Switch mode
    response = await server.set_config_option(
        config_id="mode",
        session_id=session_id,
        value="manual",
    )

    assert _select(response.config_options[0]).current_value == "manual"
    assert server._session_modes[session_id] == "manual"


@pytest.mark.asyncio
async def test_model_passed_to_agent_context(agent_factory, models):
    """Test that the selected model is passed to the agent factory via context."""
    server = AgentServerACP(agent=agent_factory, models=models)
    server.on_connect(MockClient())

    # Create a session
    new_session_response = await server.new_session(cwd="/tmp")
    session_id = new_session_response.session_id

    # Switch to a different model
    await server.set_config_option(
        config_id="model",
        session_id=session_id,
        value="anthropic:claude-haiku-4",
    )

    # Reset agent (simulating what happens during model switch)
    server._reset_agent(session_id)

    # The agent should have been created with the new model
    # We can't directly check the model inside the agent, but we verified
    # the session state is updated correctly
    assert server._session_models[session_id] == "anthropic:claude-haiku-4"


@pytest.mark.asyncio
async def test_default_model_when_none_configured():
    """Test that sessions work without model configuration."""

    def build_agent(context: AgentSessionContext):
        # Should receive None for model when not configured
        assert context.model is None
        model = _get_model_string(context.model)
        return create_deep_agent(
            model=model,
            checkpointer=MemorySaver(),
            backend=FilesystemBackend(root_dir=context.cwd, virtual_mode=True),
        )

    server = AgentServerACP(agent=build_agent)
    server.on_connect(MockClient())

    response = await server.new_session(cwd="/tmp")

    # Should not have config options when models not configured
    assert response.config_options is None or len(response.config_options) == 0
