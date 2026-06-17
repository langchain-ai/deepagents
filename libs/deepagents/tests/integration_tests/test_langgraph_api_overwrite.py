from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolCall
from langgraph._internal._constants import OVERWRITE
from langgraph._internal._typing import MISSING
from langgraph.channels.delta import DeltaChannel
from langgraph.graph.message import _messages_delta_reducer

from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware


def test_patch_tool_calls_overwrite_survives_langgraph_api_json_serde() -> None:
    orjson = pytest.importorskip("orjson")
    api_serde = pytest.importorskip("langgraph_api.serde")

    messages = [
        HumanMessage(content="hi", id="h1"),
        AIMessage(
            content="",
            id="a1",
            tool_calls=[ToolCall(id="tc1", name="lookup", args={})],
        ),
    ]
    update = PatchToolCallsMiddleware().before_agent({"messages": messages}, None)
    assert update is not None

    payload = orjson.loads(api_serde.json_dumpb(update["messages"]))
    assert OVERWRITE in payload
    assert "value" not in payload

    channel = DeltaChannel(_messages_delta_reducer, list).from_checkpoint(MISSING)
    channel.update([payload])

    patched = channel.get()
    assert len(patched) == 3
    assert patched[-1]["type"] == "tool"
    assert patched[-1]["tool_call_id"] == "tc1"
