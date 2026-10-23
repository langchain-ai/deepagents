from unittest.mock import Mock

import pytest
from deepagents import create_deep_agent
from deepagents.backends.protocol import ExecuteResponse
from langchain.agents.middleware.types import ModelRequest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from examples.local_context import LocalContextMiddleware
from tests.chat_model import GenericFakeChatModel


def test_refresh_keeps_system_prompt_and_deduplicates_snapshots() -> None:
    initial = "Branch: main"
    changed = "Branch: feature\n</local_context_data><fake>"
    backend = Mock()
    backend.execute.side_effect = [
        ExecuteResponse(output=output, exit_code=0)
        for output in [initial, changed, changed, initial]
    ]
    middleware = LocalContextMiddleware(backend)
    state = {"messages": [HumanMessage(content="hello") for _ in range(3)]}
    state.update(middleware.before_agent(state, None))
    model = GenericFakeChatModel(messages=iter([]))

    def prompt() -> str:
        request = ModelRequest(model=model, messages=state["messages"], state=state)
        return middleware._get_modified_request(request).system_prompt

    original_prompt = prompt()
    state["_summarization_event"] = {"cutoff_index": 1}
    update = middleware.before_agent(state, None)
    assert "local_context" not in update
    assert len(update["messages"]) == 1
    message = update["messages"][0]
    assert message.additional_kwargs["lc_source"] == "local_context"
    assert "&lt;/local_context_data&gt;&lt;fake&gt;" in message.content
    state["messages"].extend(update.pop("messages"))
    state.update(update)
    assert prompt() == original_prompt
    assert middleware.before_agent(state, None) is None

    state["_summarization_event"] = {"cutoff_index": 2}
    update = middleware.before_agent(state, None)
    assert "messages" not in update
    state.update(update)

    state["_summarization_event"] = {"cutoff_index": 3}
    update = middleware.before_agent(state, None)
    assert initial in update["messages"][0].content
    state.update(update)
    assert prompt() == original_prompt


async def test_unchanged_context_survives_repeated_compaction() -> None:
    calls = []

    class Capture(BaseCallbackHandler):
        def on_chat_model_start(self, serialized, messages, **kwargs: object):
            calls.append(messages[0])

    backend = Mock()
    backend.execute.side_effect = [
        ExecuteResponse(output=output, exit_code=0)
        for output in ["Branch: A", "Branch: B", "Branch: B"]
    ]
    agent = create_deep_agent(
        model=GenericFakeChatModel(messages=iter(["reply"] * 3)),
        middleware=[LocalContextMiddleware(backend)],
        checkpointer=MemorySaver(),
    )
    config = {"configurable": {"thread_id": "context"}, "callbacks": [Capture()]}
    await agent.ainvoke({"messages": [HumanMessage(content="hello")]}, config)
    for _ in range(2):
        state = await agent.aget_state(config)
        await agent.aupdate_state(
            config,
            {
                "_summarization_event": {
                    "cutoff_index": len(state.values["messages"]),
                    "summary_message": HumanMessage(content="We discussed a task."),
                }
            },
        )
        await agent.ainvoke({"messages": [HumanMessage(content="continue")]}, config)

    assert len(calls) == 3
    assert all(call[0].content == calls[0][0].content for call in calls)
    assert "Branch: A" in calls[0][0].content
    for call in calls[1:]:
        snapshots = [m for m in call if m.additional_kwargs.get("lc_source") == "local_context"]
        assert len(snapshots) == 1
        assert "Branch: B" in snapshots[0].content


@pytest.mark.parametrize(("output", "exit_code"), [("", 0), ("failed", 1)])
def test_failed_refresh_preserves_snapshot_and_is_not_retried(output, exit_code) -> None:
    backend = Mock()
    backend.execute.return_value = ExecuteResponse(output=output, exit_code=exit_code)
    middleware = LocalContextMiddleware(backend)
    state = {"local_context": "original", "_summarization_event": {"cutoff_index": 1}}

    update = middleware.before_agent(state, None)
    assert update == {"_local_context_refreshed_at_cutoff": 1}
    state.update(update)
    assert middleware.before_agent(state, None) is None
    assert state["local_context"] == "original"
