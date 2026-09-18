"""Batch approvals preserve the scope of parallel protected actions."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Interrupt, interrupt

from deepagents_talon.host import _format_tool_approval_prompt, _parse_tool_approval_reply
from deepagents_talon.interfaces import AgentRequest, ToolApprovalRequest
from deepagents_talon.runtime import DeepAgentRuntime


@pytest.mark.parametrize("decision", ["approve", "reject"])
async def test_parallel_interrupts_share_one_decision(decision):
    effects, approvals = [], []

    def first(_state):
        result = interrupt({"action_requests": [{"name": "first", "args": {"item": 1}}]})
        if result["decisions"][0]["type"] == "approve":
            effects.append("first")
        return {}

    def second(_state):
        result = interrupt(
            {
                "action_requests": [
                    {"name": "second", "args": {"item": 2}},
                    {"name": "second", "args": {"item": 3}},
                ]
            }
        )
        assert len(result["decisions"]) == 2
        if all(item["type"] == "approve" for item in result["decisions"]):
            effects.append("second")
        return {}

    builder = StateGraph(MessagesState)
    for name, node in (("first", first), ("second", second)):
        builder.add_node(name, node)
        builder.add_edge(START, name)
        builder.add_edge(name, END)
    graph = builder.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "batch"}}
    state = await graph.ainvoke({}, config)
    assert len(state["__interrupt__"]) == 2

    async def decide(request):
        assert effects == []
        approvals.append(request)
        return decision

    runtime = DeepAgentRuntime(model="test:batch")
    resume = await runtime._build_approval_resume(
        AgentRequest("batch", "work", approval_handler=decide), state["__interrupt__"]
    )
    assert len(approvals) == 1
    assert [action["args"]["item"] for action in approvals[0].action_requests] == [1, 2, 3]
    await graph.ainvoke(resume, config)
    assert sorted(effects) == (["first", "second"] if decision == "approve" else [])


@pytest.mark.parametrize("metadata", [{"trigger": "cron"}, {"background_delivery": True}, {}])
async def test_unattended_batch_is_denied(metadata):
    async def unexpected(_request):
        pytest.fail("Unattended requests must not prompt")

    request = AgentRequest(
        "batch", "work", metadata=metadata, approval_handler=unexpected if metadata else None
    )
    interrupts = [
        Interrupt(value={"action_requests": [{"name": name}]}, id=name)
        for name in ("first", "second")
    ]
    resume = await DeepAgentRuntime(model="test:batch")._build_approval_resume(request, interrupts)
    assert set(resume.resume) == {"first", "second"}
    assert all(value["decisions"][0]["type"] == "reject" for value in resume.resume.values())


@pytest.mark.parametrize(
    "value",
    [None, {}, {"action_requests": []}, {"action_requests": [{"name": "visible"}, "hidden"]}],
)
async def test_malformed_batch_never_prompts(value):
    async def unexpected(_request):
        pytest.fail("Malformed batches must not prompt")

    with pytest.raises(ValueError, match="malformed"):
        await DeepAgentRuntime(model="test:batch")._build_approval_resume(
            AgentRequest("batch", "work", approval_handler=unexpected),
            [
                Interrupt(value={"action_requests": [{"name": "valid"}]}, id="valid"),
                Interrupt(value=value, id="invalid"),
            ],
        )


@pytest.mark.parametrize("ids", [("same", "same"), ("valid", None)])
async def test_invalid_interrupt_ids_never_prompt(ids):
    async def unexpected(_request):
        pytest.fail("Invalid interrupt identities must not prompt")

    with pytest.raises(RuntimeError, match="unique resumable ids"):
        await DeepAgentRuntime(model="test:batch")._build_approval_resume(
            AgentRequest("batch", "work", approval_handler=unexpected),
            [
                SimpleNamespace(id=name, value={"action_requests": [{"name": "tool"}]})
                for name in ids
            ],
        )


async def test_cancelled_batch_never_returns_resume():
    entered = asyncio.Event()

    async def wait(_request):
        entered.set()
        await asyncio.Future()

    task = asyncio.create_task(
        DeepAgentRuntime(model="test:batch")._build_approval_resume(
            AgentRequest("batch", "work", approval_handler=wait),
            [
                Interrupt(value={"action_requests": [{"name": name}]}, id=name)
                for name in ("first", "second")
            ],
        )
    )
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.parametrize("text", ["approve 1", "approve except 2", "👍 1", "deny 2"])
def test_partial_batch_reply_is_not_a_decision(text):
    assert _parse_tool_approval_reply(text, batch=True) is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [("approve", "approve"), ("👍", "approve"), ("deny", "reject"), ("👎", "reject")],
)
def test_batch_reply_applies_to_all(text, expected):
    assert _parse_tool_approval_reply(text, batch=True) == expected


def test_batch_prompt_displays_every_action_and_scope():
    prompt = _format_tool_approval_prompt(
        ToolApprovalRequest(
            "batch",
            "first",
            (
                {"name": "first", "args": {"item": 1}},
                {"name": "second", "args": {"item": 2}},
            ),
        )
    )
    assert '1. `first`\nArgs: `{"item": 1}`' in prompt
    assert '2. `second`\nArgs: `{"item": 2}`' in prompt
    assert "run ALL actions" in prompt
    assert "skip ALL actions" in prompt
