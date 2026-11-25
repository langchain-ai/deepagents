"""Conformance: what a client sees during a turn, including the tool-call lifecycle.

Every tool call should follow `tool_call` (pending) -> optional `in_progress` ->
exactly one final `tool_call_update` with `completed` or `failed`.
"""

from __future__ import annotations

import pytest
from acp.schema import ToolCallStart
from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool, ToolException, tool

from tests.conformance.harness import (
    Harness,
    LifecycleError,
    ToolCallNotFinishedError,
    ToolCallNotStartedError,
    WrongFinalStatusError,
    call,
    check_tool_call,
    deep_agent,
    echo,
    reply,
    scripted_model,
)


@tool
def now() -> str:
    """Return the current time."""
    return "noon"


def _explode(reason: str) -> str:
    """Always fail."""
    raise ToolException(reason)


explode = StructuredTool.from_function(_explode, name="explode", handle_tool_error=True)


def _replies(harness: Harness, session_id: str) -> list[str]:
    return [entry[1] for entry in harness.trace(session_id) if entry[0] == "agent_message_chunk"]


async def test_text_turn_streams_the_reply() -> None:
    harness = Harness.start(deep_agent(scripted_model(reply("Hello there"))))
    session_id = await harness.new_session()

    response = await harness.prompt(session_id, "Hi")

    assert response.stop_reason == "end_turn"
    assert _replies(harness, session_id) == ["Hello there"]


async def _replies_with_a_nested_model_call(source: object) -> list[str]:
    """Run a turn whose tool calls another model tagged with `lc_source=source`."""
    nested = scripted_model("Nested reply")

    @tool
    async def review() -> str:
        """Ask another model."""
        await nested.ainvoke("Summarize", config={"metadata": {"lc_source": source}})
        return "reviewed"

    model = scripted_model(reply("", call("review")), "Done")
    harness = Harness.start(deep_agent(model, tools=[review]))
    session_id = await harness.new_session()

    await harness.prompt(session_id, "Go")

    return _replies(harness, session_id)


async def test_internal_model_calls_are_not_shown_as_replies() -> None:
    assert await _replies_with_a_nested_model_call("summarization") == ["Done"]


@pytest.mark.parametrize("source", ["a_custom_source", ["summarization"], {"summarization": True}])
async def test_model_calls_from_other_sources_are_shown(source: object) -> None:
    assert await _replies_with_a_nested_model_call(source) == ["Nested reply", "Done"]


async def test_truncated_reply_stops_with_max_tokens() -> None:
    truncated = AIMessage(content="Partial", response_metadata={"stop_reason": "max_tokens"})
    harness = Harness.start(deep_agent(scripted_model(truncated)))
    session_id = await harness.new_session()

    response = await harness.prompt(session_id, "Write a lot")

    assert response.stop_reason == "max_tokens"


async def test_refusal_block_is_shown_and_stops_with_refusal() -> None:
    refusal = AIMessage(content=[{"type": "refusal", "refusal": "I can't help with that."}])
    harness = Harness.start(deep_agent(scripted_model(refusal)))
    session_id = await harness.new_session()

    response = await harness.prompt(session_id, "Do something bad")

    assert response.stop_reason == "refusal"
    assert _replies(harness, session_id) == ["I can't help with that."]


async def test_model_error_fails_the_prompt() -> None:
    harness = Harness.start(deep_agent(scripted_model(RuntimeError("provider unavailable"))))
    session_id = await harness.new_session()

    with pytest.raises(RuntimeError, match="provider unavailable"):
        await harness.prompt(session_id, "Hi")


async def test_successful_tool_call_completes() -> None:
    model = scripted_model(reply("", call("echo", {"text": "hi"})), reply("Done"))
    harness = Harness.start(deep_agent(model, tools=[echo]))
    session_id = await harness.new_session()

    response = await harness.prompt(session_id, "Echo hi")

    assert response.stop_reason == "end_turn"
    assert _replies(harness, session_id) == ["Done"]
    start, final = check_tool_call(harness.client.tool_call(session_id, "call-echo"), "completed")
    assert start.raw_input == {"text": "hi"}
    assert final.content[0].content.text == "hi"


async def test_write_todos_sends_a_plan() -> None:
    todos = [
        {"content": "Read the code", "status": "in_progress"},
        {"content": "Fix the bug", "status": "pending"},
    ]
    model = scripted_model(reply("", call("write_todos", {"todos": todos})), reply("Planned"))
    harness = Harness.start(deep_agent(model))
    session_id = await harness.new_session()

    await harness.prompt(session_id, "Plan it")

    assert [entry for entry in harness.trace(session_id) if entry[0] == "plan"] == [
        ("plan", (("Read the code", "in_progress"), ("Fix the bug", "pending"))),
    ]


async def test_failed_tool_call_reports_failed() -> None:
    model = scripted_model(reply("", call("explode", {"reason": "boom"})), reply("It failed"))
    harness = Harness.start(deep_agent(model, tools=[explode]))
    session_id = await harness.new_session()

    await harness.prompt(session_id, "Try it")

    assert _replies(harness, session_id) == ["It failed"]
    check_tool_call(harness.client.tool_call(session_id, "call-explode"), "failed")


async def test_tool_call_with_empty_args_is_surfaced() -> None:
    model = scripted_model(reply("", call("now")), reply("It is noon"))
    harness = Harness.start(deep_agent(model, tools=[now]))
    session_id = await harness.new_session()

    await harness.prompt(session_id, "What time is it?")

    assert _replies(harness, session_id) == ["It is noon"]
    check_tool_call(harness.client.tool_call(session_id, "call-now"), "completed")


async def test_edit_file_gets_a_final_update() -> None:
    model = scripted_model(
        reply("", call("write_file", {"file_path": "/notes.txt", "content": "hello"})),
        reply(
            "",
            call(
                "edit_file",
                {"file_path": "/notes.txt", "old_string": "hello", "new_string": "goodbye"},
            ),
        ),
        reply("Edited"),
    )
    harness = Harness.start(deep_agent(model))
    session_id = await harness.new_session()

    await harness.prompt(session_id, "Write then edit")

    assert _replies(harness, session_id) == ["Edited"]
    # A plain assert, so a write_file regression can't pass as the tracked edit_file bug.
    assert harness.client.tool_call(session_id, "call-write_file")[-1].status == "completed"
    check_tool_call(harness.client.tool_call(session_id, "call-edit_file"), "completed")


def test_check_tool_call_rejects_a_second_start_as_the_final_update() -> None:
    # Guards the harness: a repeated `tool_call` must not count as completion.
    start = ToolCallStart(tool_call_id="c1", title="echo", status="pending")
    restart = ToolCallStart(tool_call_id="c1", title="echo", status="completed")
    with pytest.raises(LifecycleError):
        check_tool_call([start, restart], "completed")
