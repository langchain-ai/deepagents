"""Regression tests for `PatchToolCallsMiddleware` synthetic-tool-message visibility.

The synthetic `ToolMessage` injected for cancelled or invalid tool calls
previously carried no `status` field, while the rest of the SDK
(``FilesystemMiddleware._tool_error``) sets `status="error"` on tool
results that did not execute as intended. These tests pin the contract.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime

from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware


def _runtime() -> Runtime:
    return Runtime()


def _patched_tool_messages(call: AIMessage) -> list:
    # `AgentState` is a TypedDict; the middleware only reads `state["messages"]`.
    state = {"messages": [call]}
    update = PatchToolCallsMiddleware().before_agent(state, _runtime())  # type: ignore[arg-type]
    assert update is not None
    msgs = update["messages"]
    assert isinstance(msgs[0], RemoveMessage)
    assert msgs[0].id == REMOVE_ALL_MESSAGES
    return [m for m in msgs[1:] if m.type == "tool"]


class TestPatchedToolMessageStatus:
    def test_synthetic_tool_message_for_invalid_call_has_error_status(self) -> None:
        ai = AIMessage(
            content="",
            id="ai-1",
            invalid_tool_calls=[
                {
                    "id": "call-1",
                    "name": "read_file",
                    "args": "{not json",
                    "type": "invalid_tool_call",
                }
            ],
        )
        patched = _patched_tool_messages(ai)
        assert len(patched) == 1
        assert patched[0].tool_call_id == "call-1"
        assert patched[0].status == "error"

    def test_synthetic_tool_message_for_cancelled_call_has_error_status(self) -> None:
        ai = AIMessage(
            content="",
            id="ai-2",
            tool_calls=[{"id": "call-2", "name": "write_file", "args": {"file_path": "/x"}}],
        )
        patched = _patched_tool_messages(ai)
        assert len(patched) == 1
        assert patched[0].tool_call_id == "call-2"
        assert patched[0].status == "error"

    def test_synthetic_content_preserved(self) -> None:
        ai_invalid = AIMessage(
            content="",
            id="ai-3",
            invalid_tool_calls=[
                {
                    "id": "call-3",
                    "name": "read_file",
                    "args": "{not json",
                    "type": "invalid_tool_call",
                }
            ],
        )
        ai_cancelled = AIMessage(
            content="",
            id="ai-4",
            tool_calls=[{"id": "call-4", "name": "write_file", "args": {}}],
        )

        invalid_msg = _patched_tool_messages(ai_invalid)[0]
        cancelled_msg = _patched_tool_messages(ai_cancelled)[0]

        assert "could not be executed" in invalid_msg.text
        assert "was cancelled" in cancelled_msg.text
