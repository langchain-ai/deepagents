import pytest
from langchain_core.messages import AIMessage
from deepagents.middleware.gemini import GeminiEmptyResponseMiddleware
from langgraph.types import Command

def test_gemini_ghosting_detection():
    """Verify that GeminiEmptyResponseMiddleware detects empty responses and returns a Command."""
    middleware = GeminiEmptyResponseMiddleware()
    
    # 1. Non-empty response
    msg = AIMessage(content="Hello")
    assert middleware.after_agent(msg, {}, None) == msg
    
    # 2. Empty response (string content)
    empty_msg = AIMessage(content="")
    result = middleware.after_agent(empty_msg, {}, None)
    assert isinstance(result, Command)
    assert "System: The last response was empty" in result.update["messages"][1].content
    
    # 3. Empty response (list content)
    empty_list_msg = AIMessage(content=[])
    result = middleware.after_agent(empty_list_msg, {}, None)
    assert isinstance(result, Command)
    
    # 4. Tool call (not empty)
    tool_msg = AIMessage(content="", tool_calls=[{"name": "test", "args": {}, "id": "1"}])
    assert middleware.after_agent(tool_msg, {}, None) == tool_msg
