"""Unit tests for ToolSafetyMiddleware.

This module tests the tool safety middleware, focusing on how it detects empty
responses (tool calls with no text content) and injects warning messages.
"""

from unittest.mock import Mock

from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from deepagents.middleware.tool_safety import ToolSafetyMiddleware


class TestToolSafetyMiddleware:
    """Tests for ToolSafetyMiddleware behavior."""

    def test_empty_response_detection(self) -> None:
        """Test that middleware detects AIMessage with tool_calls but no content."""
        middleware = ToolSafetyMiddleware()

        mock_model = Mock()
        request = ModelRequest(model=mock_model, messages=[HumanMessage(content="Test question")])

        # Create empty response (tool_calls but no content)
        empty_response = ModelResponse(
            result=[
                HumanMessage(content="Test question"),
                AIMessage(content="", tool_calls=[{"name": "mock_tool", "args": {"query": "test"}, "id": "call_1"}]),
            ]
        )

        def handler(_request: ModelRequest) -> ModelResponse:
            return empty_response

        # Process response through middleware
        middleware.on_model_response(request, handler)

        # Check if warning flag was set
        assert middleware._empty_response_detected is True

    def test_warning_injection(self) -> None:
        """Test that middleware injects warning message on next request."""
        middleware = ToolSafetyMiddleware()

        # Manually set the warning flag (as if empty response was detected)
        middleware._empty_response_detected = True

        # Create request that should trigger warning injection
        request = ModelRequest(
            model=Mock(),
            messages=[
                HumanMessage(content="Test question"),
                AIMessage(content="", tool_calls=[{"name": "mock_tool", "args": {}, "id": "call_1"}]),
                ToolMessage(content="Tool result", tool_call_id="call_1"),
            ],
        )

        def handler(req: ModelRequest) -> ModelResponse:
            return ModelResponse(result=req.messages)

        # Process request through middleware
        result = middleware.on_model_request(request, handler)

        # Check if warning was injected
        warning_found = False
        for msg in result.result:
            if isinstance(msg, ToolMessage) and "SYSTEM WARNING" in msg.content:
                warning_found = True
                assert "last response" in msg.content.lower()
                assert "tool calls" in msg.content.lower()
                break

        assert warning_found is True
        assert middleware._empty_response_detected is False  # Flag should be reset

    def test_normal_response_no_warning(self) -> None:
        """Test that middleware doesn't flag responses with content."""
        middleware = ToolSafetyMiddleware()

        request = ModelRequest(model=Mock(), messages=[HumanMessage(content="Test")])

        # Test multiple scenarios in one test
        test_cases = [
            # Normal response with tool calls and text
            ModelResponse(
                result=[
                    HumanMessage(content="Test"),
                    AIMessage(
                        content="Here's a normal response with text",
                        tool_calls=[{"name": "mock_tool", "args": {}, "id": "call_1"}],
                    ),
                ]
            ),
            # Pure text response without tool calls
            ModelResponse(result=[HumanMessage(content="Test"), AIMessage(content="Here's my answer")]),
        ]

        for response in test_cases:
            middleware._empty_response_detected = False
            middleware.on_model_response(request, lambda _request, r=response: r)
            assert middleware._empty_response_detected is False

    def test_warning_message_content(self) -> None:
        """Test the actual warning message content."""
        middleware = ToolSafetyMiddleware()

        warning = middleware._WARNING_MESSAGE

        # Check for key elements
        assert "SYSTEM WARNING" in warning
        assert "last response" in warning.lower()
        assert "tool calls" in warning.lower()
        assert "no text" in warning.lower()
        assert "include text" in warning.lower()
        assert "reasoning" in warning.lower()
        assert len(warning) < 200
        assert not any(ord(c) > 127 for c in warning)  # No emojis
        assert "\n" not in warning.strip()  # Single line

    def test_edge_cases_no_false_positives(self) -> None:
        """Test edge cases that should NOT trigger warnings."""
        middleware = ToolSafetyMiddleware()

        request = ModelRequest(model=Mock(), messages=[HumanMessage(content="Test")])

        edge_cases = [
            # Empty result array
            ModelResponse(result=[]),
            # Non-AIMessage as last message
            ModelResponse(
                result=[
                    HumanMessage(content="Test"),
                    AIMessage(content="", tool_calls=[{"name": "mock_tool", "args": {}, "id": "call_1"}]),
                    ToolMessage(content="Tool result", tool_call_id="call_1"),
                ]
            ),
            # AIMessage without tool_calls attribute
            ModelResponse(result=[HumanMessage(content="Test"), AIMessage(content="")]),
            # Empty tool_calls list
            ModelResponse(result=[HumanMessage(content="Test"), AIMessage(content="", tool_calls=[])]),
        ]

        for response in edge_cases:
            middleware._empty_response_detected = False
            middleware.on_model_response(request, lambda _request, r=response: r)
            assert middleware._empty_response_detected is False, f"False positive for {response}"

    def test_whitespace_and_multimodal_detection(self) -> None:
        """Test whitespace-only and multimodal content detection."""
        middleware = ToolSafetyMiddleware()

        request = ModelRequest(model=Mock(), messages=[HumanMessage(content="Test")])

        # Cases that SHOULD trigger detection
        should_detect = [
            # Whitespace-only
            AIMessage(content="   ", tool_calls=[{"name": "mock_tool", "args": {}, "id": "call_1"}]),
            AIMessage(content="\t\t", tool_calls=[{"name": "mock_tool", "args": {}, "id": "call_1"}]),
            AIMessage(content="\n\n", tool_calls=[{"name": "mock_tool", "args": {}, "id": "call_1"}]),
            # Multimodal with empty text
            AIMessage(
                content=[{"type": "text", "text": ""}],
                tool_calls=[{"name": "mock_tool", "args": {}, "id": "call_1"}],
            ),
        ]

        for ai_msg in should_detect:
            middleware._empty_response_detected = False
            response = ModelResponse(result=[HumanMessage(content="Test"), ai_msg])
            middleware.on_model_response(request, lambda _request, r=response: r)
            assert middleware._empty_response_detected is True, f"Failed to detect: {ai_msg.content}"

        # Cases that should NOT trigger detection
        should_not_detect = [
            # Multimodal with actual text
            AIMessage(
                content=[{"type": "text", "text": "Let me help"}],
                tool_calls=[{"name": "mock_tool", "args": {}, "id": "call_1"}],
            ),
        ]

        for ai_msg in should_not_detect:
            middleware._empty_response_detected = False
            response = ModelResponse(result=[HumanMessage(content="Test"), ai_msg])
            middleware.on_model_response(request, lambda _request, r=response: r)
            assert middleware._empty_response_detected is False, f"False positive for: {ai_msg.content}"

    def test_consecutive_empty_responses(self) -> None:
        """Test handling of multiple consecutive empty responses."""
        middleware = ToolSafetyMiddleware()

        request = ModelRequest(model=Mock(), messages=[HumanMessage(content="Test")])
        empty_response = ModelResponse(
            result=[
                HumanMessage(content="Test"),
                AIMessage(content="", tool_calls=[{"name": "mock_tool", "args": {}, "id": "call_1"}]),
            ]
        )

        # First empty response
        middleware.on_model_response(request, lambda _request: empty_response)
        assert middleware._empty_response_detected is True

        # Warning injection should reset flag
        middleware.on_model_request(request, lambda req: ModelResponse(result=req.messages))
        assert middleware._empty_response_detected is False

        # Second empty response should trigger again
        middleware.on_model_response(request, lambda _request: empty_response)
        assert middleware._empty_response_detected is True

    async def test_async_behavior(self) -> None:
        """Test both async detection and injection."""
        middleware = ToolSafetyMiddleware()

        request = ModelRequest(model=Mock(), messages=[HumanMessage(content="Test")])
        empty_response = ModelResponse(
            result=[
                HumanMessage(content="Test"),
                AIMessage(content="", tool_calls=[{"name": "mock_tool", "args": {}, "id": "call_1"}]),
            ]
        )

        # Test async detection
        async def detection_handler(_request: ModelRequest) -> ModelResponse:
            return empty_response

        await middleware.on_model_response_async(request, detection_handler)
        assert middleware._empty_response_detected is True

        # Test async injection
        async def injection_handler(req: ModelRequest) -> ModelResponse:
            return ModelResponse(result=req.messages)

        result = await middleware.on_model_request_async(request, injection_handler)

        warning_found = False
        for msg in result.result:
            if isinstance(msg, ToolMessage) and "SYSTEM WARNING" in msg.content:
                warning_found = True
                break

        assert warning_found is True
        assert middleware._empty_response_detected is False
