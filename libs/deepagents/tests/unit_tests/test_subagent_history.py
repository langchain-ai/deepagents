"""A `task` subagent's own state must be readable back from checkpoint history.

`Pregel.get_subgraphs()` — which `get_state_history` uses to resolve a namespace
— reads `node.subgraphs`, auto-detected by scanning the node's bound runnable. A
subagent lives in the `task` tool's *closure*, so that scan cannot see it: the
tools node ends up with no subgraphs and reading a subagent's namespace raises
`Subgraph tools not found`, even though its checkpoints were written.

`create_deep_agent` therefore declares the compiled subagent graphs on the tools
node. Without that, a history/inspection UI can see that a subagent ran and what
it returned (both are in the parent's transcript) but never what it did.
"""

from types import SimpleNamespace
from typing import Annotated

import pytest
from langchain_core.messages import AIMessage
from langgraph.channels.delta import DeltaChannel
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import _messages_delta_reducer
from typing_extensions import TypedDict

from deepagents.graph import _declare_subagent_subgraphs, create_deep_agent
from tests.unit_tests.chat_model import GenericFakeChatModel


class _SubState(TypedDict, total=False):
    messages: Annotated[list, DeltaChannel(_messages_delta_reducer)]


def _subagent():
    graph = StateGraph(_SubState)
    graph.add_node("work", lambda _state: {"messages": [AIMessage("sub worked")]})
    graph.add_edge(START, "work")
    graph.add_edge("work", END)
    return graph.compile()


def _agent(checkpointer):
    model = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "c1",
                            "name": "task",
                            "args": {
                                "description": "do it",
                                "subagent_type": "helper",
                            },
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="all done"),
            ]
        )
    )
    return create_deep_agent(
        model=model,
        subagents=[
            {
                "name": "helper",
                "description": "Does things.",
                "runnable": _subagent(),
            }
        ],
        checkpointer=checkpointer,
    )


def test_tools_node_declares_subagent_graphs() -> None:
    """The declaration itself — what makes the namespace resolvable."""
    agent = _agent(InMemorySaver())
    assert agent.nodes["tools"].subgraphs, "tools node has no subgraphs, so `get_state_history` cannot resolve a `tools:<id>` namespace"


def test_subagent_state_is_readable_from_history() -> None:
    saver = InMemorySaver()
    agent = _agent(saver)
    config = {"configurable": {"thread_id": "1"}}
    agent.invoke({"messages": [{"role": "user", "content": "go"}]}, config)

    namespaces = sorted({tup.config["configurable"]["checkpoint_ns"] for tup in saver.list({"configurable": {"thread_id": "1"}})} - {""})
    assert namespaces, "the subagent did not checkpoint under its own namespace"

    for namespace in namespaces:
        snapshots = list(agent.get_state_history({"configurable": {"thread_id": "1", "checkpoint_ns": namespace}}))
        assert snapshots, f"no snapshots for {namespace}"


@pytest.mark.parametrize("subagents", [None, []])
def test_builtin_general_purpose_subagent_is_declared_too(subagents) -> None:
    """Even with no user subagents, deepagents registers `general-purpose`.

    Its checkpoints are just as unreadable without the declaration, so it must
    be covered by the same mechanism.
    """
    agent = create_deep_agent(
        model=GenericFakeChatModel(messages=iter([AIMessage(content="done")])),
        subagents=subagents,
    )
    assert agent.nodes["tools"].subgraphs


def test_declaration_does_not_clobber_existing_subgraphs() -> None:
    """Only fills the gap; never overwrites what auto-detection already found."""
    sentinel = object()

    node = SimpleNamespace(subgraphs=[sentinel])
    agent = SimpleNamespace(nodes={"tools": node})
    middleware = SimpleNamespace(subagent_graphs={"helper": _subagent()})

    _declare_subagent_subgraphs(agent, [middleware])
    assert node.subgraphs == [sentinel]
