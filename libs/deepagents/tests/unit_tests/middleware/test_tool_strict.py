import pytest
from unittest.mock import MagicMock
from deepagents.middleware.patch_tool_calls import ToolStrictMiddleware
from langchain.agents.middleware.types import ModelRequest

def test_tool_strict_middleware_openai():
    """Verify that ToolStrictMiddleware enables strict=True for OpenAI models."""
    middleware = ToolStrictMiddleware()
    
    # Mock OpenAI model
    mock_model = MagicMock()
    mock_model.model_dump.return_value = {"model_name": "gpt-4o"}
    
    request = ModelRequest(
        model=mock_model,
        messages=[],
        system_message=None,
        tools=[],
        tool_choice=None,
        response_format=None,
        state={},
        runtime=MagicMock(),
        model_settings={}
    )
    
    modified_request = middleware.modify_request(request)
    assert modified_request.model_settings.get("strict") is True

def test_tool_strict_middleware_google():
    """Verify that ToolStrictMiddleware enables strict=True for Google models."""
    middleware = ToolStrictMiddleware()
    
    # Mock Google model
    mock_model = MagicMock()
    mock_model.model_dump.return_value = {"model": "gemini-1.5-flash"}
    
    request = ModelRequest(
        model=mock_model,
        messages=[],
        system_message=None,
        tools=[],
        tool_choice=None,
        response_format=None,
        state={},
        runtime=MagicMock(),
        model_settings={}
    )
    
    modified_request = middleware.modify_request(request)
    assert modified_request.model_settings.get("strict") is True

def test_tool_strict_middleware_anthropic_ignored():
    """Verify that ToolStrictMiddleware ignores Anthropic models."""
    middleware = ToolStrictMiddleware()
    
    # Mock Anthropic model
    mock_model = MagicMock()
    mock_model.model_dump.return_value = {"model": "claude-3-sonnet"}
    
    request = ModelRequest(
        model=mock_model,
        messages=[],
        system_message=None,
        tools=[],
        tool_choice=None,
        response_format=None,
        state={},
        runtime=MagicMock(),
        model_settings={}
    )
    
    modified_request = middleware.modify_request(request)
    assert "strict" not in modified_request.model_settings
