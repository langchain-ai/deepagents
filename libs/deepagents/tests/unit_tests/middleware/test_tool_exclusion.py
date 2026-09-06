"""Tests that a profile's `excluded_tools` are refused at the tool-call boundary.

`_ToolExclusionMiddleware.wrap_model_call` keeps excluded tools out of the
request, but the executor still has them registered and dispatches on the name
the model emits. These tests drive a compiled agent with a scripted model that
emits an excluded name anyway, and assert the call is answered with an error
instead of running, while a non-excluded call in the same turn still runs.
"""

from collections.abc import Iterator
from typing import Any
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph.state import CompiledStateGraph

from deepagents import create_deep_agent
from deepagents.profiles.harness.harness_profiles import (
    _HARNESS_PROFILES,
    HarnessProfile,
    register_harness_profile,
)
from tests.unit_tests.chat_model import GenericFakeChatModel

_EXCLUDED = "write_file"
_ALLOWED = "ls"
_TARGET = "/excluded.txt"

_Agent = CompiledStateGraph[Any, Any, Any, Any]


@pytest.fixture
def agent() -> Iterator[_Agent]:
    """A compiled agent whose profile excludes `write_file`.

    The scripted model calls the excluded tool and an allowed tool in one turn,
    then finishes.
    """
    original = dict(_HARNESS_PROFILES)
    register_harness_profile("exclprov", HarnessProfile(excluded_tools=frozenset({_EXCLUDED})))
    model = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": _EXCLUDED,
                            "args": {"file_path": _TARGET, "content": "written anyway"},
                            "id": "call_excluded",
                            "type": "tool_call",
                        },
                        {"name": _ALLOWED, "args": {"path": "/"}, "id": "call_allowed", "type": "tool_call"},
                    ],
                ),
                AIMessage(content="done"),
            ]
        )
    )
    try:
        with patch("deepagents.graph.resolve_model", return_value=model):
            yield create_deep_agent(model="exclprov:some-model")
    finally:
        _HARNESS_PROFILES.clear()
        _HARNESS_PROFILES.update(original)


def _results(result: dict[str, Any]) -> dict[str, ToolMessage]:
    return {m.name: m for m in result["messages"] if isinstance(m, ToolMessage)}


def _assert_refused(result: dict[str, Any]) -> None:
    tool_messages = _results(result)

    refused = tool_messages[_EXCLUDED]
    assert refused.status == "error"
    assert _EXCLUDED in str(refused.content)
    # The tool never reached the backend, so no file was created.
    assert _TARGET not in (result.get("files") or {})

    # A non-excluded call in the same turn is unaffected.
    assert tool_messages[_ALLOWED].status != "error"


def test_excluded_tool_call_is_refused(agent: _Agent) -> None:
    """A call naming an excluded tool errors instead of executing."""
    _assert_refused(agent.invoke({"messages": [{"role": "user", "content": "go"}]}))


async def test_excluded_tool_call_is_refused_async(agent: _Agent) -> None:
    """The async path refuses the same call."""
    _assert_refused(await agent.ainvoke({"messages": [{"role": "user", "content": "go"}]}))
