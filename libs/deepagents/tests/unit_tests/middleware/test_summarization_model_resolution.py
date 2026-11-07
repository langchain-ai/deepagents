"""The summarizer sizes, compacts, and writes against the model each call actually goes to."""

import asyncio
import uuid
from collections.abc import Awaitable, Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.model_selection import MODEL_SELECTION_KEY, model_spec_from_model
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.exceptions import ContextOverflowError
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, BaseMessage, HumanMessage, ToolCall
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from pydantic import Field, SecretStr

from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from deepagents.middleware.summarization import (
    SUMMARIZATION_EVENT_KEY,
    create_summarization_middleware,
)

SMALL_WINDOW = 32_000
LARGE_WINDOW = 128_000


class ChatWindow(BaseChatModel):
    """Fake chat model with a context window that records the calls and summaries it serves."""

    model: str
    temperature: float | None = None
    api_key: SecretStr = SecretStr("sk-never-recorded")
    script: list[AIMessage] = Field(default_factory=list)
    fail_first_answer: bool = False
    calls: list[list[BaseMessage]] = Field(default_factory=list)
    summaries: list[str] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "window-fake"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "ChatWindow":
        return self

    def _generate(self, messages: list[BaseMessage], stop: list[str] | None = None, run_manager: Any = None, **kwargs: Any) -> ChatResult:  # noqa: ANN401
        text = messages[-1].text if messages else ""
        if "<messages>" in text:
            self.summaries.append(text)
            reply = AIMessage(content=f"summary by {self.model}")
        else:
            if self.fail_first_answer:
                self.fail_first_answer = False
                msg = "transient"
                raise RuntimeError(msg)
            self.calls.append(list(messages))
            reply = self.script.pop(0) if self.script else AIMessage(content=f"answer from {self.model}")
        return ChatResult(generations=[ChatGeneration(message=reply)])


def window_model(name: str, window: int, **kwargs: Any) -> ChatWindow:
    return ChatWindow(model=name, profile={"max_input_tokens": window}, **kwargs)


class SwapModel(AgentMiddleware):
    """Custom middleware that routes each call to a model of its choosing."""

    def __init__(self, pick: Callable[[ModelRequest], BaseChatModel], name: str = "SwapModel") -> None:
        self._pick = pick
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
        return handler(request.override(model=self._pick(request)))

    async def awrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]]) -> ModelResponse:
        return await handler(request.override(model=self._pick(request)))


def long_history(turns: int = 10, tokens_per_turn: int = 4_000, *, prefix: str = "") -> list[AnyMessage]:
    """A conversation of roughly `turns * tokens_per_turn` tokens that ends on a user turn."""
    messages: list[AnyMessage] = []
    for i in range(turns):
        messages.append(HumanMessage(content=f"{prefix}turn {i} " + "x" * (tokens_per_turn * 4), id=f"h{i}"))
        messages.append(AIMessage(content=f"reply {i}", id=f"a{i}"))
    messages.append(HumanMessage(content=f"{prefix}latest question", id="latest"))
    return messages


def tokens(messages: list[BaseMessage]) -> int:
    return count_tokens_approximately(messages)


async def run(agent: Any, state: dict[str, Any], mode: str, thread_id: str | None = None) -> dict[str, Any]:  # noqa: ANN401
    """Invoke `agent` on a thread (fresh by default) and return its full state, private keys included."""
    config = {"configurable": {"thread_id": thread_id or str(uuid.uuid4())}}
    if mode == "async":
        await agent.ainvoke(state, config)
    else:
        agent.invoke(state, config)
    return agent.get_state(config).values


@pytest.fixture(params=["sync", "async"])
def mode(request: pytest.FixtureRequest) -> str:
    return request.param


async def test_swap_to_larger_window_accepts_input_that_fits(mode: str) -> None:
    """A 32K build model swapped for a 128K one no longer rejects input that fits the 128K window."""
    small, large = window_model("small", SMALL_WINDOW), window_model("large", LARGE_WINDOW)
    agent = create_deep_agent(model=small, middleware=[SwapModel(lambda _: large)], checkpointer=InMemorySaver())
    question = HumanMessage(content="y" * 50_000 * 4, id="big")

    result = await run(agent, {"messages": [question]}, mode)

    assert result["messages"][-1].content == "answer from large"
    assert small.calls == []
    assert large.calls[-1][-1].content == question.content
    assert result.get(SUMMARIZATION_EVENT_KEY) is None


async def test_swap_to_smaller_window_compacts_for_it(mode: str) -> None:
    """A 128K build model swapped for a 32K one compacts with the 32K model before the call."""
    small, large = window_model("small", SMALL_WINDOW), window_model("large", LARGE_WINDOW)
    agent = create_deep_agent(model=large, middleware=[SwapModel(lambda _: small)], checkpointer=InMemorySaver())

    result = await run(agent, {"messages": long_history()}, mode)

    assert result[SUMMARIZATION_EVENT_KEY]["summary_message"].text.endswith("summary by small\n</summary>")
    assert len(small.summaries) == 1
    assert large.calls == []
    assert large.summaries == []
    assert tokens(small.calls[-1]) < SMALL_WINDOW
    assert small.calls[-1][-1].content == "latest question"


async def test_same_history_is_left_alone_for_a_larger_window(mode: str) -> None:
    """The history that compacts for a 32K model is sent whole to a 128K model."""
    small, large = window_model("small", SMALL_WINDOW), window_model("large", LARGE_WINDOW)
    agent = create_deep_agent(model=small, middleware=[SwapModel(lambda _: large)], checkpointer=InMemorySaver())
    history = long_history()

    result = await run(agent, {"messages": history}, mode)

    assert result.get(SUMMARIZATION_EVENT_KEY) is None
    assert small.summaries == []
    assert [m.content for m in large.calls[-1][-len(history) :]] == [m.content for m in history]


async def test_summarizer_override_writes_while_thresholds_follow_the_call(mode: str) -> None:
    """`summarizer=` still writes the summary; the swapped-in model still sizes the thresholds."""
    small, large = window_model("small", SMALL_WINDOW), window_model("large", LARGE_WINDOW)
    writer = window_model("writer", LARGE_WINDOW)
    summarization = create_summarization_middleware(large, StateBackend(), summarizer=writer)
    agent = create_deep_agent(model=large, middleware=[SwapModel(lambda _: small), summarization], checkpointer=InMemorySaver())

    result = await run(agent, {"messages": long_history()}, mode)

    assert result[SUMMARIZATION_EVENT_KEY] is not None
    assert len(writer.summaries) == 1
    assert small.summaries == []
    assert large.summaries == []
    assert tokens(small.calls[-1]) < SMALL_WINDOW


async def test_subagent_follows_its_own_calls_model(mode: str) -> None:
    """A subagent sizes against the model its own middleware routes to, not the parent's or its build model."""
    description = "z" * 50_000 * 4
    task_call: ToolCall = {"name": "task", "args": {"description": description, "subagent_type": "worker"}, "id": "t1"}
    parent = window_model("parent", 200_000, script=[AIMessage(content="", tool_calls=[task_call])])
    worker_small, worker_large = window_model("worker-small", SMALL_WINDOW), window_model("worker-large", LARGE_WINDOW)
    worker = {
        "name": "worker",
        "description": "Does work.",
        "system_prompt": "Work.",
        "model": worker_small,
        "middleware": [SwapModel(lambda _: worker_large)],
    }
    agent = create_deep_agent(model=parent, subagents=[worker], checkpointer=InMemorySaver())

    result = await run(agent, {"messages": [HumanMessage(content="delegate")]}, mode)

    assert result["messages"][-1].content == "answer from parent"
    assert worker_small.calls == []
    assert worker_large.calls[-1][-1].content == description


async def test_subagent_selection_stays_out_of_parent(mode: str) -> None:
    """The #6644 shape: the parent stays on "strong" and its `model_selection` record survives `task`."""
    task_call: ToolCall = {"name": "task", "args": {"description": "Summarize the page", "subagent_type": "summarizer"}, "id": "t1"}
    strong = window_model("strong", LARGE_WINDOW, temperature=0.2, script=[AIMessage(content="", tool_calls=[task_call])])
    cheap = window_model("cheap", SMALL_WINDOW)
    summarizer_spec = {"name": "summarizer", "description": "Summarizes pages.", "system_prompt": "Summarize.", "model": cheap}
    agent = create_deep_agent(model=strong, subagents=[summarizer_spec], checkpointer=InMemorySaver())
    config: Any = {"configurable": {"thread_id": "selection"}}
    state = {"messages": [HumanMessage(content="Look this up")]}

    if mode == "async":
        await agent.ainvoke(state, config)
    else:
        agent.invoke(state, config)

    assert len(cheap.calls) == 1
    assert len(strong.calls) == 2
    selections = [snapshot.values.get(MODEL_SELECTION_KEY) for snapshot in agent.get_state_history(config)]
    assert {selection["current"]["model"] for selection in selections if selection} == {"strong"}
    assert agent.get_state(config).values[MODEL_SELECTION_KEY] == {
        "current": {"provider": "window", "model": "strong", "max_input_tokens": LARGE_WINDOW, "init_kwargs": {"temperature": 0.2}},
    }


def test_selection_round_trips_through_strict_serializer() -> None:
    """The recorded spec holds plain values only, and never credentials."""
    spec = model_spec_from_model(window_model("strong", LARGE_WINDOW, temperature=0.2))
    serde = JsonPlusSerializer(allowed_msgpack_modules=None)

    assert serde.loads_typed(serde.dumps_typed({"current": spec})) == {"current": spec}
    assert "sk-never-recorded" not in repr(spec)


async def test_concurrent_runs_on_different_models_stay_separate(mode: str) -> None:
    """Runs routed to different models at once each compact (or not) for their own model only."""
    small, large = window_model("small", SMALL_WINDOW), window_model("large", LARGE_WINDOW)

    def pick(request: ModelRequest) -> BaseChatModel:
        return small if request.state["messages"][0].text.startswith("[small]") else large

    agent = create_deep_agent(model=large, middleware=[SwapModel(pick)], checkpointer=InMemorySaver())
    inputs = [{"messages": long_history(prefix=f"[{size}]")} for size in ["small", "large"] * 4]

    if mode == "async":
        results = await asyncio.gather(*(run(agent, state, mode) for state in inputs))
    else:
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda state: asyncio.run(run(agent, state, mode)), inputs))

    for state, result in zip(inputs, results, strict=True):
        is_small = state["messages"][0].text.startswith("[small]")
        assert result["messages"][-1].content == f"answer from {'small' if is_small else 'large'}"
        assert (result.get(SUMMARIZATION_EVENT_KEY) is not None) is is_small
        assert result[MODEL_SELECTION_KEY]["current"]["max_input_tokens"] == (SMALL_WINDOW if is_small else LARGE_WINDOW)
    assert len(small.summaries) == 4
    assert large.summaries == []
    assert all(tokens(call) < SMALL_WINDOW for call in small.calls)


class SummarizationRuns(BaseCallbackHandler):
    """Collects the metadata of each `summarization` run."""

    def __init__(self) -> None:
        self.metadata: list[dict[str, Any]] = []

    def on_chain_start(self, serialized: Any, inputs: Any, *, metadata: dict[str, Any] | None = None, **kwargs: Any) -> None:  # noqa: ANN401
        if kwargs.get("name") == "summarization":
            self.metadata.append(metadata or {})


@pytest.mark.parametrize("override", [False, True])
async def test_summary_trace_records_compaction_model_and_window(mode: str, override: bool) -> None:  # noqa: FBT001
    """The `summarization` run names the model whose window was compacted, that window, and the writer."""
    small, large = window_model("small", SMALL_WINDOW), window_model("large", LARGE_WINDOW)
    writer = window_model("writer", LARGE_WINDOW)
    middleware: list[AgentMiddleware] = [SwapModel(lambda _: small)]
    if override:
        middleware.append(create_summarization_middleware(large, StateBackend(), summarizer=writer))
    agent = create_deep_agent(model=large, middleware=middleware)
    runs = SummarizationRuns()
    config: Any = {"callbacks": [runs]}

    if mode == "async":
        await agent.ainvoke({"messages": long_history()}, config)
    else:
        agent.invoke({"messages": long_history()}, config)

    [metadata] = runs.metadata
    assert metadata["summarization_context_model"] == "window:small"
    assert metadata["summarization_context_window"] == SMALL_WINDOW
    assert metadata["summarization_model"] == ("window:writer" if override else "window:small")


def test_build_model_overflow_error_is_gone_only_because_of_the_swap() -> None:
    """Without a swap, the 32K model still rejects input it cannot hold."""
    small = window_model("small", SMALL_WINDOW)
    agent = create_deep_agent(model=small)

    with pytest.raises(ContextOverflowError):
        agent.invoke({"messages": [HumanMessage(content="y" * 50_000 * 4)]})


class PrependReminder(AgentMiddleware):
    """Custom middleware that puts a message ahead of the conversation."""

    def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
        return handler(request.override(messages=[HumanMessage(content="reminder"), *request.messages]))

    async def awrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]]) -> ModelResponse:
        return await handler(request.override(messages=[HumanMessage(content="reminder"), *request.messages]))


async def test_stored_summary_is_applied_once_when_middleware_prepends(mode: str) -> None:
    """A message prepended by custom middleware does not make the stored summary apply a second time."""
    small = window_model("small", SMALL_WINDOW)
    agent = create_deep_agent(model=small, middleware=[PrependReminder()], checkpointer=InMemorySaver())
    thread = str(uuid.uuid4())
    await run(agent, {"messages": long_history()}, mode, thread)
    small.calls.clear()

    await run(agent, {"messages": [HumanMessage(content="follow-up", id="f1")]}, mode, thread)

    sent = small.calls[-1]
    summaries = [m for m in sent if isinstance(m, HumanMessage) and m.additional_kwargs.get("lc_source") == "summarization"]
    assert len(summaries) == 1
    assert [m.text for m in sent if isinstance(m, HumanMessage)][-2:] == ["latest question", "follow-up"]
    assert sum(m.text == "latest question" for m in sent) == 1


class RetryOnce(AgentMiddleware):
    """Custom retry middleware: calls again once when the model call fails."""

    def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
        try:
            return handler(request)
        except RuntimeError:
            return handler(request)

    async def awrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]]) -> ModelResponse:
        try:
            return await handler(request)
        except RuntimeError:
            return await handler(request)


async def test_retry_around_summarizer_reuses_summary_and_offload(mode: str) -> None:
    """Retry middleware around the summarizer neither writes a second summary nor offloads history twice."""
    small = window_model("small", SMALL_WINDOW, fail_first_answer=True)
    agent = create_deep_agent(model=small, middleware=[RetryOnce()], checkpointer=InMemorySaver())

    result = await run(agent, {"messages": long_history()}, mode)

    assert result["messages"][-1].content == "answer from small"
    assert len(small.summaries) == 1
    [history] = [f for path, f in result["files"].items() if path.startswith("/conversation_history/")]
    content = history["content"] if isinstance(history["content"], str) else "\n".join(history["content"])
    assert content.count("## Summarized at") == 1
