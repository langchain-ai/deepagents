"""Opaque context survives native compaction and checkpoint continuation."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from openai.types.responses import Response
from openai.types.responses.compacted_response import CompactedResponse
from pydantic import Field

from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from deepagents.middleware.native_compaction import OpenAICompactionMiddleware
from deepagents.middleware.summarization import SummarizationToolMiddleware, create_summarization_middleware


class NativeModel(BaseChatModel):
    model_name: str = "native-test"
    use_responses_api: bool = True
    profile: dict[str, Any] = Field(default_factory=lambda: {"max_input_tokens": 100, "structured_output": True})
    calls: list = Field(default_factory=list)
    counts: list = Field(default_factory=list)
    failure: bool = False

    @property
    def _llm_type(self) -> str:
        return "native-test"

    def bind_tools(self, tools, **kwargs: Any):
        return self.bind(tools=tools, **kwargs)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs: Any):
        self.calls.append((messages, kwargs))
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="continued"))])

    def compact(self, messages, **kwargs: Any):
        if self.failure:
            msg = "compact failed"
            raise RuntimeError(msg)
        self.calls.append((messages, kwargs))
        return AIMessage(content=[{"type": "compaction", "encrypted_content": "opaque"}])

    async def acompact(self, messages, **kwargs: Any):
        return self.compact(messages, **kwargs)

    def count_input_tokens(self, messages, **kwargs: Any):
        self.counts.append((messages, kwargs))
        return 95 if not isinstance(messages[-1], AIMessage) else 5

    async def acount_input_tokens(self, messages, **kwargs: Any):
        return self.count_input_tokens(messages, **kwargs)


def test_checkpoint_continuation_preserves_opaque_context_and_model():
    model = NativeModel()
    agent = create_deep_agent(model, compaction_strategy="openai", checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "source"}}
    history = [
        HumanMessage(content="work"),
        AIMessage(content="", tool_calls=[{"id": "old", "name": "side_effect", "args": {}}]),
        ToolMessage(content="done", tool_call_id="old"),
    ]
    agent.invoke({"messages": history}, config)
    snapshot = agent.get_state(config)
    event = snapshot.values["_summarization_event"]
    assert isinstance(event["summary_message"], AIMessage)
    assert event["summary_message"].content == [{"type": "compaction", "encrypted_content": "opaque"}]
    assert event["cutoff_index"] == len(history)
    assert all(not message.tool_calls for message in model.calls[-1][0] if isinstance(message, AIMessage))
    assert model.calls[-1][1]["store"] is False
    fork = {"configurable": {"thread_id": "fork"}}
    agent.update_state(fork, snapshot.values)
    agent.invoke({"messages": [HumanMessage(content="continue")]}, fork)
    assert agent.get_state(config).values == snapshot.values
    assert any(isinstance(message, AIMessage) and message.content == event["summary_message"].content for message in model.calls[-1][0])
    assert all(not isinstance(message, ToolMessage) for message in model.calls[-1][0])


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_request_count_includes_instructions_tools_and_schema(asynchronous):
    model = NativeModel()
    middleware = OpenAICompactionMiddleware(model, backend=StateBackend())
    request = ModelRequest(
        model=model,
        messages=[HumanMessage(content="work")],
        system_message=SystemMessage(content="instructions"),
        tools=[{"type": "web_search"}],
        response_format={"title": "Answer", "type": "object", "properties": {"answer": {"type": "string"}}},
    )
    if asynchronous:
        await middleware.acount_request_tokens(request)
    else:
        middleware.count_request_tokens(request)
    inputs, kwargs = model.counts[-1]
    assert inputs[0] == request.system_message
    assert kwargs["tools"] == request.tools
    assert "response_format" in kwargs
    assert kwargs["store"] is False


async def test_failed_compaction_leaves_history_and_state_unchanged():
    model = NativeModel(failure=True)
    middleware = OpenAICompactionMiddleware(model, backend=StateBackend())
    messages = [HumanMessage(content="work")]
    state = {"messages": messages.copy()}
    request = ModelRequest(model=model, messages=messages, state=state)
    with pytest.raises(RuntimeError, match="compact failed"):
        await middleware.awrap_model_call(request, lambda _: None)
    assert state == {"messages": messages}
    assert model.calls == []


def test_incomplete_tool_exchange_is_not_compacted():
    middleware = OpenAICompactionMiddleware(NativeModel(), backend=StateBackend())
    history = [
        HumanMessage(content="work"),
        AIMessage(content="", tool_calls=[{"id": "one", "name": "tool", "args": {}}, {"id": "two", "name": "tool", "args": {}}]),
        ToolMessage(content="done", tool_call_id="one"),
    ]
    assert middleware._determine_cutoff_index(history) == 0
    with pytest.raises(ValueError, match="all tool results"):
        middleware._create_summary(history)


def test_incompatible_model_and_text_strategy_reject_native_checkpoint():
    model = NativeModel()
    middleware = OpenAICompactionMiddleware(model, backend=StateBackend())
    opaque = middleware._build_new_messages_with_path(middleware._create_summary([HumanMessage(content="work")]), None)[0]
    state = {"messages": [HumanMessage(content="work")], "_summarization_event": {"cutoff_index": 1, "summary_message": opaque, "file_path": None}}
    request = ModelRequest(model=NativeModel(model_name="different"), messages=state["messages"], state=state)
    with pytest.raises(ValueError, match="Cannot switch"):
        middleware.wrap_model_call(request, lambda _: ModelResponse(result=[]))
    text = create_summarization_middleware(model, StateBackend())
    with pytest.raises(ValueError, match="original model"):
        text.wrap_model_call(request, lambda _: ModelResponse(result=[]))


async def test_manual_compaction_keeps_pending_tool_exchange(monkeypatch):
    engine = OpenAICompactionMiddleware(NativeModel(), backend=StateBackend())
    tool = SummarizationToolMiddleware(engine)
    monkeypatch.setattr(tool, "_is_eligible_for_compaction", lambda _: True)
    monkeypatch.setattr(engine, "_aoffload_to_backend", lambda *_: _no_archive())
    messages = [HumanMessage(content="work"), AIMessage(content="", tool_calls=[{"id": "compact", "name": "compact_conversation", "args": {}}])]
    runtime = SimpleNamespace(state={"messages": messages}, tool_call_id="compact", config={"configurable": {"thread_id": "manual"}})
    command = await tool._arun_compact(runtime)
    event = command.update["_summarization_event"]
    assert isinstance(event["summary_message"], AIMessage)
    assert event["cutoff_index"] == 1
    assert command.update["messages"][0].tool_call_id == "compact"
    assert runtime.state == {"messages": messages}


async def _no_archive():
    return None


def test_native_count_sees_final_prompt_and_tool_transformations():
    class FinalPrompt(AgentMiddleware):
        def wrap_model_call(self, request, handler):
            prompt = SystemMessage(content="final instructions")
            return handler(request.override(system_message=prompt, tools=[*request.tools, {"type": "web_search"}]))

    model = NativeModel()
    agent = create_deep_agent(model, middleware=[FinalPrompt()], compaction_strategy="openai")
    agent.invoke({"messages": [HumanMessage(content="work")]})
    inputs, kwargs = model.counts[0]
    assert inputs[0].content == "final instructions"
    assert {"type": "web_search"} in kwargs["tools"]


def test_real_adapter_compaction_serializes_count_and_stateless_continuation():
    if not hasattr(ChatOpenAI, "compact"):
        pytest.skip("native compaction adapter has not been released")
    model = ChatOpenAI(model="native-test", api_key="test", use_responses_api=True, profile={"max_input_tokens": 100, "structured_output": True})
    counted = MagicMock(return_value=SimpleNamespace(input_tokens=95))
    model.root_client.responses.input_tokens.count = counted
    output = Response.model_validate(
        {
            "id": "resp_test",
            "created_at": 0,
            "model": "native-test",
            "object": "response",
            "parallel_tool_calls": True,
            "tools": [],
            "tool_choice": "auto",
            "output": [{"type": "compaction", "id": "cmp_item", "encrypted_content": "opaque"}],
        }
    ).output
    compact = MagicMock(return_value=CompactedResponse.model_construct(id="cmp_test", output=output))
    model.root_client.responses.compact = compact
    engine = OpenAICompactionMiddleware(model, backend=StateBackend())
    history = [HumanMessage(content="work")]
    request = ModelRequest(
        model=model,
        messages=history,
        system_message=SystemMessage(content="instructions"),
        tools=[{"type": "web_search"}],
        response_format={"title": "Answer", "type": "object", "properties": {"answer": {"type": "string"}}},
    )
    generation = []

    def continue_model(prepared):
        generation.append(model._get_request_payload([*prepared.messages, HumanMessage(content="continue")], **prepared.model_settings))
        return ModelResponse(result=[AIMessage(content="continued")])

    engine.wrap_model_call(request, continue_model)
    counted_payload = counted.call_args.kwargs
    assert counted_payload["tools"] == [{"type": "web_search"}]
    assert counted_payload["text"]["format"]["schema"]["properties"] == {"answer": {"type": "string"}}
    assert counted_payload["input"][0]["role"] == "system"
    assert compact.call_args.kwargs["input"] == counted_payload["input"]
    assert "tools" not in compact.call_args.kwargs
    assert generation[0]["store"] is False
    assert generation[0]["input"][0] == {"type": "compaction", "id": "cmp_item", "encrypted_content": "opaque"}
    assert "previous_response_id" not in generation[0] or generation[0]["previous_response_id"] is None


def test_unsupported_adapter_fails_explicitly():
    with pytest.raises(ValueError, match="installed adapter is unsupported"):
        create_summarization_middleware(FakeListChatModel(responses=["unused"]), StateBackend(), compaction_strategy="openai")
