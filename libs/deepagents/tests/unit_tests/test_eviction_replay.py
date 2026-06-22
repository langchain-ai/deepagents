"""Tests for HumanMessage eviction with DeltaChannel replay.

Verifies that emitting only `[tagged]` (reusing the original message's
ID) correctly deduplicates on DeltaChannel replay — without needing a
`REMOVE_ALL_MESSAGES` sentinel that would clobber the AIMessage written
in the same super-step.
"""

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from deepagents.backends import StoreBackend
from deepagents.middleware.filesystem import FilesystemMiddleware


class _FakeModel:
    """Minimal fake model that always responds with a short AIMessage."""

    name = "fake"

    def bind_tools(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return self

    def with_structured_output(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return self

    def invoke(self, messages, config=None, **kw):  # noqa: ANN003
        return AIMessage(content="ok")

    async def ainvoke(self, messages, config=None, **kw):  # noqa: ANN003
        return AIMessage(content="ok")


def _build_agent():
    checkpointer = InMemorySaver()
    store = InMemoryStore()
    backend = StoreBackend(store=store, namespace=lambda _rt: ("filesystem",))
    middleware = FilesystemMiddleware(
        backend=backend,
        human_message_token_limit_before_evict=100,
    )
    agent = create_agent(
        model=_FakeModel(),
        middleware=[middleware],
        checkpointer=checkpointer,
    )
    return agent  # noqa: RET504  # clearer than inlining the create_agent call


def _state_messages(agent, thread_id):
    return agent.get_state({"configurable": {"thread_id": thread_id}}).values["messages"]


def test_eviction_preserves_ai_response_and_no_duplicates():
    """Eviction emits only [tagged]; no duplicates on replay, AI response preserved."""
    agent = _build_agent()
    cfg = {"configurable": {"thread_id": "t1"}}

    large_content = "x" * 401  # 401 chars > 100 tokens * 4 chars/token
    agent.invoke({"messages": [HumanMessage(content=large_content)]}, cfg)

    msgs = _state_messages(agent, "t1")
    # Should have both the tagged HumanMessage AND the AIMessage.
    # A REMOVE_ALL_MESSAGES sentinel would have clobbered the AIMessage.
    assert len(msgs) == 2, f"Expected 2 messages after first invoke, got {len(msgs)}"
    assert isinstance(msgs[0], HumanMessage)
    assert msgs[0].additional_kwargs.get("lc_evicted_to") is not None
    assert isinstance(msgs[1], AIMessage)

    # Second invoke triggers checkpoint replay — verify no duplicates.
    agent.invoke({"messages": [HumanMessage(content="hello")]}, cfg)
    msgs = _state_messages(agent, "t1")
    assert len(msgs) == 4, f"Expected 4 messages after second invoke, got {len(msgs)}"

    ids = [m.id for m in msgs if m.id is not None]
    assert len(ids) == len(set(ids)), f"Duplicate message IDs: {ids}"

    human_msgs = [m for m in msgs if isinstance(m, HumanMessage)]
    assert len(human_msgs) == 2
    evicted = [m for m in human_msgs if m.additional_kwargs.get("lc_evicted_to")]
    assert len(evicted) == 1
