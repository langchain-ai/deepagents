"""The summarizer sizes, compacts, and writes against the model each call actually goes to."""

import asyncio
import uuid
from collections.abc import Awaitable, Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, cast

import pytest
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ExtendedModelResponse, ModelRequest, ModelResponse
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.exceptions import ContextOverflowError
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, BaseMessage, HumanMessage, ToolCall
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import Field, SecretStr

from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from deepagents.middleware._state import private_state_field_names
from deepagents.middleware.summarization import (
    LAST_CALL_MODEL_KEY,
    SUMMARIZATION_EVENT_KEY,
    ModelResolution,
    SummarizationMiddleware,
    SummarizationState,
    create_summarization_middleware,
)
from tests.unit_tests.middleware.test_summarization_middleware import MockBackend, make_mock_runtime

SMALL_WINDOW = 32_000
LARGE_WINDOW = 128_000


class ChatWindow(BaseChatModel):
    """Fake chat model with a context window that records the calls and summaries it serves."""

    model: str
    temperature: float | None = None
    api_key: SecretStr = SecretStr("sk-never-recorded")
    script: list[AIMessage] = Field(default_factory=list)
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


async def run(agent: Any, state: dict[str, Any], mode: str) -> dict[str, Any]:  # noqa: ANN401
    """Invoke `agent` on a fresh thread and return its full state, private keys included."""
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
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


async def test_subagent_model_record_stays_out_of_parent(mode: str) -> None:
    """The #6644 shape: the parent stays on "strong" and its private record survives `task`."""
    task_call: ToolCall = {"name": "task", "args": {"description": "Summarize the page", "subagent_type": "summarizer"}, "id": "t1"}
    strong = window_model("strong", LARGE_WINDOW, temperature=0.2, script=[AIMessage(content="", tool_calls=[task_call])])
    cheap = window_model("cheap", SMALL_WINDOW)
    summarizer_spec = {"name": "summarizer", "description": "Summarizes pages.", "system_prompt": "Summarize.", "model": cheap}
    agent = create_deep_agent(model=strong, subagents=[summarizer_spec], checkpointer=InMemorySaver())
    config: Any = {"configurable": {"thread_id": "record"}}
    state = {"messages": [HumanMessage(content="Look this up")]}

    if mode == "async":
        await agent.ainvoke(state, config)
    else:
        agent.invoke(state, config)
    result = agent.get_state(config).values

    assert result["messages"][-1].content == "answer from strong"
    assert len(cheap.calls) == 1
    assert len(strong.calls) == 2
    records = [snapshot.values.get(LAST_CALL_MODEL_KEY) for snapshot in agent.get_state_history(config)]
    assert {record["model"] for record in records if record is not None} == {"strong"}
    assert result[LAST_CALL_MODEL_KEY] == {
        "provider": "window",
        "model": "strong",
        "params": {"temperature": 0.2},
        "max_input_tokens": LARGE_WINDOW,
    }


def test_model_record_is_private_and_holds_no_secrets() -> None:
    """The record is stripped by `task` and never carries credentials."""
    assert LAST_CALL_MODEL_KEY in private_state_field_names(SummarizationState)
    middleware = SummarizationMiddleware(backend=MockBackend(), trigger=("messages", 100))
    model = window_model("strong", LARGE_WINDOW)
    request = ModelRequest(model=model, messages=[HumanMessage(content="hi")], state=cast("Any", {"messages": []}), runtime=make_mock_runtime())

    first = middleware.wrap_model_call(request, lambda _: ModelResponse(result=[AIMessage(content="ok")]))

    assert isinstance(first, ExtendedModelResponse)
    record = first.command.update[LAST_CALL_MODEL_KEY]
    assert record["model"] == "strong"
    assert "sk-never-recorded" not in repr(record)
    unchanged = request.override(state=cast("Any", {"messages": [], LAST_CALL_MODEL_KEY: record}))
    second = middleware.wrap_model_call(unchanged, lambda _: ModelResponse(result=[AIMessage(content="ok")]))
    assert isinstance(second, ModelResponse)


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
        assert result[LAST_CALL_MODEL_KEY]["max_input_tokens"] == (SMALL_WINDOW if is_small else LARGE_WINDOW)
    assert len(small.summaries) == 4
    assert large.summaries == []
    assert all(tokens(call) < SMALL_WINDOW for call in small.calls)


def test_callable_model_resolves_per_call() -> None:
    """A callable `model` picks the managed model for each call."""
    small, large = window_model("small", SMALL_WINDOW), window_model("large", LARGE_WINDOW)
    resolutions: list[ModelResolution] = []

    def resolve(resolution: ModelResolution) -> BaseChatModel:
        resolutions.append(resolution)
        return small

    middleware = SummarizationMiddleware(
        resolve, backend=MockBackend(), trigger=("fraction", 0.5), keep=("messages", 1), trim_tokens_to_summarize=None
    )
    history = long_history(turns=5)
    request = ModelRequest(model=large, messages=history, state=cast("Any", {"messages": history}), runtime=make_mock_runtime())

    result = middleware.wrap_model_call(request, lambda _: ModelResponse(result=[AIMessage(content="ok")]))

    assert isinstance(result, ExtendedModelResponse)
    assert SUMMARIZATION_EVENT_KEY in result.command.update
    assert len(small.summaries) == 1
    assert large.summaries == []
    assert resolutions[0].request is request
    assert resolutions[0].state is request.state


def test_callable_model_must_not_return_a_name() -> None:
    """A resolver that returns a model name is rejected rather than building a client."""
    middleware = SummarizationMiddleware(lambda _: "openai:gpt-5.5", backend=MockBackend(), trigger=("messages", 100))
    model = window_model("large", LARGE_WINDOW)
    request = ModelRequest(model=model, messages=[HumanMessage(content="hi")], state=cast("Any", {"messages": []}), runtime=make_mock_runtime())

    with pytest.raises(TypeError, match="not a model name"):
        middleware.wrap_model_call(request, lambda _: ModelResponse(result=[AIMessage(content="ok")]))


def test_followed_model_without_profile_skips_fractional_limits() -> None:
    """A followed model without a profile skips fractional limits instead of failing the call."""
    middleware = SummarizationMiddleware(backend=MockBackend(), trigger=[("fraction", 0.5), ("messages", 4)], keep=("fraction", 0.1))
    model = ChatWindow(model="no-profile")
    history = long_history(turns=12, tokens_per_turn=10)
    request = ModelRequest(model=model, messages=history, state=cast("Any", {"messages": history}), runtime=make_mock_runtime())

    result = middleware.wrap_model_call(request, lambda _: ModelResponse(result=[AIMessage(content="ok")]))

    assert isinstance(result, ExtendedModelResponse)
    assert SUMMARIZATION_EVENT_KEY in result.command.update
    assert len(model.summaries) == 1


def test_fixed_model_without_profile_still_rejects_fractional_settings() -> None:
    """A fixed model is validated when the middleware is built, as before."""
    with pytest.raises(ValueError, match="Model profile information is required"):
        SummarizationMiddleware(ChatWindow(model="no-profile"), backend=MockBackend(), trigger=("fraction", 0.5))


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
