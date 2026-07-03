"""Unit tests for Deep Agents HITL interrupt configuration normalization."""

from typing import Any

import pytest
from langchain.agents.middleware import ToolCallRequest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from deepagents.graph import _merge_fs_interrupt_on, create_deep_agent
from deepagents.middleware._fs_interrupt import _build_interrupt_on_from_permissions
from deepagents.middleware.filesystem import FilesystemPermission
from tests.unit_tests.chat_model import GenericFakeChatModel


def _always_interrupt(request: ToolCallRequest) -> bool:
    return bool(request.tool_call)


def test_enabled_true_dict_defaults_to_true_config() -> None:
    interrupt_on = _merge_fs_interrupt_on({}, {"sample_tool": {"enabled": True}})

    assert interrupt_on == {"sample_tool": {"allowed_decisions": ["approve", "edit", "reject", "respond"]}}


def test_enabled_true_dict_preserves_custom_config() -> None:
    interrupt_on = _merge_fs_interrupt_on(
        {},
        {
            "sample_tool": {
                "enabled": True,
                "allowed_decisions": ["approve", "reject"],
                "when": _always_interrupt,
            }
        },
    )

    assert interrupt_on is not None
    config = interrupt_on["sample_tool"]
    assert config["allowed_decisions"] == ["approve", "reject"]
    assert config["when"] is _always_interrupt
    assert "enabled" not in config


def test_enabled_false_dict_overrides_filesystem_interrupt() -> None:
    rules = [FilesystemPermission(operations=["write"], paths=["/secrets/**"], mode="interrupt")]
    interrupt_on = _merge_fs_interrupt_on(
        _build_interrupt_on_from_permissions(rules),
        {"write_file": {"enabled": False}},
    )

    assert interrupt_on is not None
    assert interrupt_on["write_file"] is False
    assert "edit_file" in interrupt_on
    assert "delete" in interrupt_on


def test_enabled_requires_bool() -> None:
    with pytest.raises(TypeError, match="enabled"):
        _merge_fs_interrupt_on({}, {"sample_tool": {"enabled": "yes"}})


def test_enabled_true_dict_interrupts_with_default_decisions() -> None:
    interrupt_payloads: list[Any] = []

    @tool
    def requires_approval() -> str:
        """A tool that should trigger HITL."""
        return "approved"

    model = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "requires_approval",
                            "args": {},
                            "id": "call_requires_approval",
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        )
    )

    agent = create_deep_agent(
        model=model,
        tools=[requires_approval],
        checkpointer=InMemorySaver(),
        interrupt_on={"requires_approval": {"enabled": True}},
    )

    for chunk in agent.stream(
        {"messages": [HumanMessage(content="Use the approval-gated tool.")]},
        config={"configurable": {"thread_id": "enabled-true-dict"}},
        stream_mode="updates",
    ):
        if "__interrupt__" in chunk:
            interrupt_payloads.extend(chunk["__interrupt__"])

    assert len(interrupt_payloads) == 1
    interrupt_value = interrupt_payloads[0].value
    assert interrupt_value["action_requests"][0]["name"] == "requires_approval"
    assert interrupt_value["review_configs"][0]["allowed_decisions"] == ["approve", "edit", "reject", "respond"]


def test_enabled_false_dict_auto_approves_tool_call() -> None:
    called = False

    @tool
    def requires_approval() -> str:
        """A tool that should run without HITL."""
        nonlocal called
        called = True
        return "approved"

    model = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "requires_approval",
                            "args": {},
                            "id": "call_requires_approval",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="done"),
            ]
        )
    )

    agent = create_deep_agent(
        model=model,
        tools=[requires_approval],
        checkpointer=InMemorySaver(),
        interrupt_on={"requires_approval": {"enabled": False}},
    )

    result = agent.invoke(
        {"messages": [HumanMessage(content="Use the approval-gated tool.")]},
        config={"configurable": {"thread_id": "enabled-false-dict"}},
    )

    assert called
    assert "__interrupt__" not in result
