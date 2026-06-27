"""Tests for `SecretsRedactionMiddleware`."""

from __future__ import annotations

from typing import Any

from langchain.tools import ToolRuntime
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import HumanMessage, RemoveMessage, SystemMessage, ToolCall, ToolMessage

from deepagents.middleware._secrets import (
    SecretsRedactionMiddleware,
    _scrub_text,
)

FAKE_PYLON_KEY = "pylon_api_" + "a" * 40
FAKE_LANGSMITH_KEY = "lsv2_pt_" + "A" * 40 + "_abc123"
FAKE_ANTHROPIC_KEY = "sk-ant-" + "A" * 40
FAKE_OPENAI_PROJECT_KEY = "sk-proj-" + "B" * 40
FAKE_OPENAI_KEY = "sk-" + "C" * 40


def _runtime(tool_call_id: str = "tc1") -> ToolRuntime:
    return ToolRuntime(
        state={},
        context=None,
        tool_call_id=tool_call_id,
        store=None,
        stream_writer=lambda _: None,
        config={},
    )


class TestScrubText:
    def test_pylon_key(self) -> None:
        text = f"My key is {FAKE_PYLON_KEY} use it."
        scrubbed, matched = _scrub_text(text)
        assert FAKE_PYLON_KEY not in scrubbed
        assert "<<PYLON_API_KEY>>" in scrubbed
        assert matched == {"PYLON_API_KEY"}

    def test_langsmith_key(self) -> None:
        scrubbed, matched = _scrub_text(FAKE_LANGSMITH_KEY)
        assert FAKE_LANGSMITH_KEY not in scrubbed
        assert "<<LANGSMITH_API_KEY>>" in scrubbed
        assert "LANGSMITH_API_KEY" in matched

    def test_anthropic_key(self) -> None:
        scrubbed, _matched = _scrub_text(FAKE_ANTHROPIC_KEY)
        assert FAKE_ANTHROPIC_KEY not in scrubbed
        assert "<<ANTHROPIC_API_KEY>>" in scrubbed

    def test_openai_project_key(self) -> None:
        scrubbed, matched = _scrub_text(FAKE_OPENAI_PROJECT_KEY)
        assert FAKE_OPENAI_PROJECT_KEY not in scrubbed
        assert "<<OPENAI_PROJECT_KEY>>" in scrubbed
        assert "OPENAI_PROJECT_KEY" in matched

    def test_openai_key(self) -> None:
        scrubbed, _matched = _scrub_text(FAKE_OPENAI_KEY)
        assert FAKE_OPENAI_KEY not in scrubbed
        assert "<<OPENAI_API_KEY>>" in scrubbed

    def test_generic_assignment(self) -> None:
        text = "FOO_API_KEY=AKIA1234567890abcdef1234"
        scrubbed, matched = _scrub_text(text)
        assert "AKIA1234567890abcdef1234" not in scrubbed
        assert "<<REDACTED_SECRET>>" in scrubbed
        assert "REDACTED_SECRET" in matched

    def test_no_match_returns_input(self) -> None:
        text = "no secrets here, just a normal sentence."
        scrubbed, matched = _scrub_text(text)
        assert scrubbed == text
        assert matched == set()


class TestBeforeModel:
    def test_scrubs_human_message_string_content(self) -> None:
        mw = SecretsRedactionMiddleware()
        state: dict[str, Any] = {
            "messages": [
                HumanMessage(content=f"here is my key: {FAKE_PYLON_KEY}"),
            ]
        }
        update = mw.before_model(state, runtime=None)  # type: ignore[arg-type]
        assert update is not None
        msgs = update["messages"]
        assert isinstance(msgs[0], RemoveMessage)
        scrubbed_human = msgs[1]
        assert isinstance(scrubbed_human, HumanMessage)
        assert FAKE_PYLON_KEY not in scrubbed_human.content
        assert "<<PYLON_API_KEY>>" in scrubbed_human.content
        note = msgs[-1]
        assert isinstance(note, SystemMessage)
        assert "PYLON_API_KEY" in note.content

    def test_noop_when_no_secrets(self) -> None:
        mw = SecretsRedactionMiddleware()
        state: dict[str, Any] = {"messages": [HumanMessage(content="hello world")]}
        assert mw.before_model(state, runtime=None) is None  # type: ignore[arg-type]

    def test_scrubs_human_message_block_content(self) -> None:
        mw = SecretsRedactionMiddleware()
        state: dict[str, Any] = {
            "messages": [
                HumanMessage(content=[{"type": "text", "text": f"key={FAKE_ANTHROPIC_KEY}"}]),
            ]
        }
        update = mw.before_model(state, runtime=None)  # type: ignore[arg-type]
        assert update is not None
        scrubbed_human = update["messages"][1]
        assert FAKE_ANTHROPIC_KEY not in str(scrubbed_human.content)
        assert "<<ANTHROPIC_API_KEY>>" in str(scrubbed_human.content)

    def test_idempotent_when_note_already_present(self) -> None:
        """Re-running on already-scrubbed state with note doesn't add a second note."""
        mw = SecretsRedactionMiddleware()
        state: dict[str, Any] = {
            "messages": [
                HumanMessage(content="key: <<PYLON_API_KEY>>"),
                SystemMessage(content="A secret matching PYLON_API_KEY was detected..."),
            ]
        }
        assert mw.before_model(state, runtime=None) is None  # type: ignore[arg-type]


class TestWrapToolCall:
    def _make_request(self, command: str) -> ToolCallRequest:
        tool_call: ToolCall = {
            "name": "execute",
            "args": {"command": command},
            "id": "tc1",
            "type": "tool_call",
        }
        return ToolCallRequest(tool_call=tool_call, tool=None, state={}, runtime=_runtime())

    def test_scrubs_execute_command_args(self) -> None:
        mw = SecretsRedactionMiddleware()
        captured: list[ToolCallRequest] = []

        def handler(req: ToolCallRequest) -> ToolMessage:
            captured.append(req)
            return ToolMessage(content="ok", name="execute", tool_call_id="tc1")

        original_command = f'curl -H "Authorization: Bearer {FAKE_PYLON_KEY}" https://api.example.com'
        request = self._make_request(original_command)
        mw.wrap_tool_call(request, handler)

        assert len(captured) == 1
        passed_command = captured[0].tool_call["args"]["command"]
        assert FAKE_PYLON_KEY not in passed_command
        assert "<<PYLON_API_KEY>>" in passed_command

    def test_scrubs_execute_tool_result(self) -> None:
        mw = SecretsRedactionMiddleware()

        def handler(_req: ToolCallRequest) -> ToolMessage:
            return ToolMessage(
                content=f"output contained {FAKE_LANGSMITH_KEY} verbatim",
                name="execute",
                tool_call_id="tc1",
            )

        request = self._make_request("echo hello")
        result = mw.wrap_tool_call(request, handler)
        assert isinstance(result, ToolMessage)
        assert FAKE_LANGSMITH_KEY not in result.content
        assert "<<LANGSMITH_API_KEY>>" in result.content

    def test_ignores_non_execute_tools(self) -> None:
        mw = SecretsRedactionMiddleware()
        captured: list[ToolCallRequest] = []

        def handler(req: ToolCallRequest) -> ToolMessage:
            captured.append(req)
            return ToolMessage(content="ok", name="write_file", tool_call_id="tc1")

        tool_call: ToolCall = {
            "name": "write_file",
            "args": {"path": "/x", "content": f"key={FAKE_PYLON_KEY}"},
            "id": "tc1",
            "type": "tool_call",
        }
        request = ToolCallRequest(tool_call=tool_call, tool=None, state={}, runtime=_runtime())
        mw.wrap_tool_call(request, handler)
        # write_file tool args are left untouched; this middleware only
        # scrubs the `execute` boundary per spec.
        assert captured[0].tool_call["args"]["content"] == f"key={FAKE_PYLON_KEY}"


class TestRegressionEndToEnd:
    """Regression for issue: secrets must not appear verbatim in prompt, tool args, or tool results."""

    def test_pylon_key_not_in_prompt_tool_args_or_results(self) -> None:
        mw = SecretsRedactionMiddleware()
        state: dict[str, Any] = {
            "messages": [
                HumanMessage(content=f"please call the API with {FAKE_PYLON_KEY}"),
            ]
        }
        update = mw.before_model(state, runtime=None)  # type: ignore[arg-type]
        assert update is not None
        # (a) Prompt history sent to the model.
        assert all(FAKE_PYLON_KEY not in str(m.content) for m in update["messages"] if hasattr(m, "content"))

        # (b) Recorded tool-call args.
        tool_call: ToolCall = {
            "name": "execute",
            "args": {"command": f'curl -H "Authorization: Bearer {FAKE_PYLON_KEY}"'},
            "id": "tc1",
            "type": "tool_call",
        }
        captured: list[ToolCallRequest] = []

        def handler(req: ToolCallRequest) -> ToolMessage:
            captured.append(req)
            # (c) Tool-result payload echoes the key.
            return ToolMessage(content=f"hit with {FAKE_PYLON_KEY}", name="execute", tool_call_id="tc1")

        request = ToolCallRequest(tool_call=tool_call, tool=None, state={}, runtime=_runtime())
        result = mw.wrap_tool_call(request, handler)
        assert FAKE_PYLON_KEY not in captured[0].tool_call["args"]["command"]
        assert isinstance(result, ToolMessage)
        assert FAKE_PYLON_KEY not in result.content
