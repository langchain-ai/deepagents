"""Tests for AgentTrajectory.pretty() serialization.

Ensures tool-call steps are always visible in the serialised output so that
downstream consumers (e.g. the LLM judge) see the full trajectory.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from tests.evals.utils import AgentStep, AgentTrajectory


def _traj(*steps: AgentStep) -> AgentTrajectory:
    return AgentTrajectory(steps=list(steps), files={})


class TestPrettyTextOnly:
    """Steps with only text content."""

    def test_single_text_step(self) -> None:
        t = _traj(AgentStep(index=1, action=AIMessage(content="hello"), observations=[]))
        assert "text: hello" in t.pretty()

    def test_multiline_text_escaped(self) -> None:
        t = _traj(AgentStep(index=1, action=AIMessage(content="line1\nline2"), observations=[]))
        out = t.pretty()
        assert r"line1\nline2" in out


class TestPrettyToolCallsOnly:
    """Steps with only tool calls (no text)."""

    def test_tool_call_visible(self) -> None:
        msg = AIMessage(
            content="",
            tool_calls=[{"name": "edit_file", "args": {"file_path": "/a.md"}, "id": "1"}],
        )
        t = _traj(AgentStep(index=1, action=msg, observations=[]))
        out = t.pretty()
        assert "edit_file" in out
        assert "/a.md" in out

    def test_empty_text_not_shown(self) -> None:
        msg = AIMessage(
            content="",
            tool_calls=[{"name": "read_file", "args": {}, "id": "1"}],
        )
        t = _traj(AgentStep(index=1, action=msg, observations=[]))
        out = t.pretty()
        assert "text:" not in out


class TestPrettyMixed:
    """Steps with both tool calls and text."""

    def test_both_tool_call_and_text_present(self) -> None:
        msg = AIMessage(
            content="I'll update the file now.",
            tool_calls=[{"name": "edit_file", "args": {"file_path": "/a.md"}, "id": "1"}],
        )
        t = _traj(AgentStep(index=1, action=msg, observations=[]))
        out = t.pretty()
        assert "edit_file" in out
        assert "I'll update the file now." in out


class TestPrettyMultiStep:
    """Multi-step trajectories matching the memory_multiturn pattern."""

    def test_read_edit_text_all_visible(self) -> None:
        steps = [
            AgentStep(
                index=1,
                action=AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "read_file", "args": {"file_path": "/AGENTS.md"}, "id": "1"}
                    ],
                ),
                observations=[],
            ),
            AgentStep(
                index=2,
                action=AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "edit_file",
                            "args": {
                                "file_path": "/AGENTS.md",
                                "new_string": "## Preferences\n- formal language",
                            },
                            "id": "2",
                        }
                    ],
                ),
                observations=[],
            ),
            AgentStep(
                index=3,
                action=AIMessage(content="Done, preference saved."),
                observations=[],
            ),
        ]
        out = _traj(*steps).pretty()
        assert "read_file" in out
        assert "edit_file" in out
        assert "formal language" in out
        assert "Done, preference saved." in out
