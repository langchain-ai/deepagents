"""Batch approvals preserve the scope of parallel protected actions."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Interrupt, interrupt

from deepagents_talon.interfaces import AgentRequest, ToolApprovalDecision, ToolApprovalRequest
from deepagents_talon.runtime import DeepAgentRuntime


@pytest.mark.parametrize("decision", ["approve", "reject"])
async def test_mixed_batch_cancels_elicitation(decision: ToolApprovalDecision) -> None:
    approvals: list[ToolApprovalRequest] = []

    async def decide(request: ToolApprovalRequest) -> ToolApprovalDecision:
        approvals.append(request)
        return decision

    elicitation = Interrupt(
        value={"type": "mcp_elicitation", "requests": [{"key": "question"}]}, id="input"
    )
    actions = [{"name": name} for name in ("first", "second")]
    resume = await DeepAgentRuntime(model="test:batch")._build_approval_resume(
        AgentRequest("batch", "work", approval_handler=decide),
        [elicitation, *(Interrupt(value={"action_requests": [a]}, id=a["name"]) for a in actions)],
    )
    assert len(approvals) == 1
    assert approvals[0].interrupt_id == "first"
    assert approvals[0].action_requests == tuple(actions)
    expected = {"type": decision}
    if decision == "reject":
        expected["message"] = "Denied by operator."
    assert resume.resume == {
        "input": {"responses": {"question": {"action": "cancel"}}},
        "first": {"decisions": [expected]},
        "second": {"decisions": [expected]},
    }


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
