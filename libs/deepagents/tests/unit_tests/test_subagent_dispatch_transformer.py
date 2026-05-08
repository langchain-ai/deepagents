"""Unit tests for `SubagentDispatchTransformer`.

Pushes synthetic `tools` protocol events through the transformer and
asserts the structured payloads emitted on the `subagent_dispatch`
channel. The transformer is non-native (`_native = False`), so on the
wire its channel name is auto-prefixed to `custom:subagent_dispatch`
by the mux — this file tests the projection pre-prefix; the wire
prefix is langgraph's responsibility.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from deepagents import SubagentDispatchTransformer

if TYPE_CHECKING:
    from langgraph.stream._types import ProtocolEvent

TS = int(time.time() * 1000)


def _tool_event(
    *,
    event: str,
    tool_name: str,
    tool_call_id: str,
    namespace: list[str] | None = None,
    tool_input: str | dict[str, Any] | None = None,
    tool_output: dict[str, Any] | None = None,
) -> ProtocolEvent:
    data: dict[str, Any] = {
        "event": event,
        "tool_call_id": tool_call_id,
        "tool_name": tool_name,
    }
    if tool_input is not None:
        data["input"] = tool_input
    if tool_output is not None:
        data["output"] = tool_output
    return {
        "type": "event",
        "method": "tools",
        "params": {
            "namespace": namespace or [],
            "timestamp": TS,
            "data": data,
        },
    }


def _capture(transformer: SubagentDispatchTransformer) -> list[dict[str, Any]]:
    """Wire the transformer's channel to a list and return it for inspection."""
    captured: list[dict[str, Any]] = []
    chan = transformer.init()["subagent_dispatch"]
    chan._wire(captured.append)
    return captured


class TestSubagentDispatchTransformerProcess:
    def test_task_tool_started_with_string_input_emits_record(self) -> None:
        """`task` tool-started with a JSON-stringified input emits one record."""
        t = SubagentDispatchTransformer(())
        captured = _capture(t)
        t.process(
            _tool_event(
                event="tool-started",
                tool_name="task",
                tool_call_id="toolu_abc",
                tool_input='{"description": "do research", "subagent_type": "researcher"}',
            )
        )
        assert captured == [
            {
                "tool_call_id": "toolu_abc",
                "subagent_type": "researcher",
                "namespace": [],
                "description": "do research",
            }
        ]

    def test_task_tool_started_with_dict_input_emits_record(self) -> None:
        """`task` tool-started with an already-parsed dict input emits one record."""
        t = SubagentDispatchTransformer(())
        captured = _capture(t)
        t.process(
            _tool_event(
                event="tool-started",
                tool_name="task",
                tool_call_id="toolu_abc",
                tool_input={"description": "find facts", "subagent_type": "researcher"},
            )
        )
        assert captured == [
            {
                "tool_call_id": "toolu_abc",
                "subagent_type": "researcher",
                "namespace": [],
                "description": "find facts",
            }
        ]

    def test_namespace_is_propagated(self) -> None:
        """The dispatch namespace is included on the payload (joins to lifecycle ns)."""
        t = SubagentDispatchTransformer(())
        captured = _capture(t)
        t.process(
            _tool_event(
                event="tool-started",
                tool_name="task",
                tool_call_id="toolu_abc",
                namespace=["agent:1"],
                tool_input={"subagent_type": "researcher"},
            )
        )
        assert captured[0]["namespace"] == ["agent:1"]

    def test_description_omitted_when_absent(self) -> None:
        """Missing description leaves the field off the payload."""
        t = SubagentDispatchTransformer(())
        captured = _capture(t)
        t.process(
            _tool_event(
                event="tool-started",
                tool_name="task",
                tool_call_id="toolu_abc",
                tool_input={"subagent_type": "researcher"},
            )
        )
        assert "description" not in captured[0]

    def test_non_task_tool_is_ignored(self) -> None:
        t = SubagentDispatchTransformer(())
        captured = _capture(t)
        t.process(
            _tool_event(
                event="tool-started",
                tool_name="calculator",
                tool_call_id="toolu_abc",
                tool_input='{"expression": "1+1"}',
            )
        )
        assert captured == []

    def test_tool_finished_is_ignored(self) -> None:
        """Only `tool-started` triggers a dispatch record; finishes/errors don't."""
        t = SubagentDispatchTransformer(())
        captured = _capture(t)
        t.process(
            _tool_event(
                event="tool-finished",
                tool_name="task",
                tool_call_id="toolu_abc",
                tool_output={"messages": []},
            )
        )
        t.process(
            _tool_event(
                event="tool-error",
                tool_name="task",
                tool_call_id="toolu_abc",
            )
        )
        assert captured == []

    def test_missing_tool_call_id_is_ignored(self) -> None:
        """No tool_call_id means no join key — drop the event silently."""
        t = SubagentDispatchTransformer(())
        captured = _capture(t)
        event = _tool_event(
            event="tool-started",
            tool_name="task",
            tool_call_id="",
            tool_input={"subagent_type": "researcher"},
        )
        t.process(event)
        assert captured == []

    def test_missing_subagent_type_is_ignored(self) -> None:
        """No subagent_type means there's nothing useful to surface."""
        t = SubagentDispatchTransformer(())
        captured = _capture(t)
        t.process(
            _tool_event(
                event="tool-started",
                tool_name="task",
                tool_call_id="toolu_abc",
                tool_input={"description": "do something"},
            )
        )
        assert captured == []

    def test_malformed_json_input_is_ignored(self) -> None:
        """A non-JSON string input doesn't crash the transformer."""
        t = SubagentDispatchTransformer(())
        captured = _capture(t)
        t.process(
            _tool_event(
                event="tool-started",
                tool_name="task",
                tool_call_id="toolu_abc",
                tool_input="not-json{",
            )
        )
        assert captured == []

    def test_non_tools_event_is_ignored(self) -> None:
        """Lifecycle, messages, etc. don't trigger emission."""
        t = SubagentDispatchTransformer(())
        captured = _capture(t)
        t.process(
            {
                "type": "event",
                "method": "lifecycle",
                "params": {
                    "namespace": [],
                    "timestamp": TS,
                    "data": {"event": "started"},
                },
            }
        )
        assert captured == []

    def test_process_returns_true_to_keep_event_in_main_log(self) -> None:
        """Transformer should never suppress the underlying tools event."""
        t = SubagentDispatchTransformer(())
        _capture(t)
        keep = t.process(
            _tool_event(
                event="tool-started",
                tool_name="task",
                tool_call_id="toolu_abc",
                tool_input={"subagent_type": "researcher", "description": "x"},
            )
        )
        assert keep is True
