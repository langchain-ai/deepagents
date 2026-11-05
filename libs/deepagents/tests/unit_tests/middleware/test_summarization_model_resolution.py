"""The summarizer sizes, compacts, and writes against the agent's current model from `model_selection`."""

import asyncio
import uuid
from collections.abc import Awaitable, Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, cast

import pytest
from langchain.agents.middleware import AgentMiddleware
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
from deepagents.middleware import summarization as summarization_module
from deepagents.middleware._model_selection import MODEL_SELECTION_KEY, model_spec_from_model
from deepagents.middleware.summarization import SUMMARIZATION_EVENT_KEY, SummarizationMiddleware, create_summarization_middleware
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


def window_model(name: str, window: int | None, **kwargs: Any) -> ChatWindow:
    return ChatWindow(model=name, profile={"max_input_tokens": window} if window else None, **kwargs)


class ChooseModel(AgentMiddleware):
    """Chooses each call's model: records it in `model_selection` and routes the call to it."""

    def __init__(self, pick: Callable[[dict[str, Any]], BaseChatModel], *, record: bool = True) -> None:
        self._pick = pick
        self._record = record

    def before_model(self, state: dict[str, Any], runtime: Any) -> dict[str, Any] | None:  # noqa: ANN401
        if not self._record:
            return None
        return {MODEL_SELECTION_KEY: {"current": model_spec_from_model(self._pick(state))}}

    def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
        return handler(request.override(model=self._pick(request.state)))

    async def awrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]]) -> ModelResponse:
        return await handler(request.override(model=self._pick(request.state)))


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
    """Invoke `agent` on a thread and return its full state, private keys included."""
    config = {"configurable": {"thread_id": thread_id or str(uuid.uuid4())}}
    if mode == "async":
        await agent.ainvoke(state, config)
    else:
        agent.invoke(state, config)
    return agent.get_state(config).values


@pytest.fixture(params=["sync", "async"])
def mode(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture
def rebuild(monkeypatch: pytest.MonkeyPatch) -> dict[str, BaseChatModel]:
    """Rebuild recorded fake models by name, as `init_chat_model` does for real providers."""
    models: dict[str, BaseChatModel] = {}
    monkeypatch.setattr(summarization_module, "init_model_from_spec", lambda spec: models[spec["model"]])
    return models


async def test_choosing_a_larger_window_accepts_input_that_fits(mode: str) -> None:
    """A 32K build model switched to a 128K one no longer rejects input that fits the 128K window."""
    small, large = window_model("small", SMALL_WINDOW), window_model("large", LARGE_WINDOW)
    agent = create_deep_agent(model=small, middleware=[ChooseModel(lambda _: large)], checkpointer=InMemorySaver())
    question = HumanMessage(content="y" * 50_000 * 4, id="big")

    result = await run(agent, {"messages": [question]}, mode)

    assert result["messages"][-1].content == "answer from large"
    assert small.calls == []
    assert result.get(SUMMARIZATION_EVENT_KEY) is None


async def test_choosing_a_smaller_window_compacts_and_writes_with_it(mode: str, rebuild: dict[str, BaseChatModel]) -> None:
    """A 128K build model switched to a 32K one compacts for the 32K window, and the 32K model writes the summary."""
    small, large = window_model("small", SMALL_WINDOW), window_model("large", LARGE_WINDOW)
    rebuild["small"] = small
    agent = create_deep_agent(model=large, middleware=[ChooseModel(lambda _: small)], checkpointer=InMemorySaver())

    result = await run(agent, {"messages": long_history()}, mode)

    assert result[SUMMARIZATION_EVENT_KEY] is not None
    assert len(small.summaries) == 1
    assert large.summaries == []
    assert large.calls == []
    assert tokens(small.calls[-1]) < SMALL_WINDOW


async def test_unrebuildable_model_falls_back_to_the_calls_model_for_writing(mode: str) -> None:
    """When the recorded model cannot be rebuilt, the call's own model writes; thresholds still follow the recorded one."""
    small, large = window_model("small", SMALL_WINDOW), window_model("large", LARGE_WINDOW)
    agent = create_deep_agent(model=large, middleware=[ChooseModel(lambda _: small)], checkpointer=InMemorySaver())

    result = await run(agent, {"messages": long_history()}, mode)

    assert result[SUMMARIZATION_EVENT_KEY] is not None
    assert len(large.summaries) == 1
    assert tokens(small.calls[-1]) < SMALL_WINDOW


async def test_recorded_model_is_followed_on_the_next_call(mode: str, rebuild: dict[str, BaseChatModel]) -> None:
    """A swap that only routes the call is recorded after it, and the next call compacts for that model."""
    small, large = window_model("small", SMALL_WINDOW), window_model("large", LARGE_WINDOW)
    rebuild["small"] = small
    agent = create_deep_agent(model=large, middleware=[ChooseModel(lambda _: small, record=False)], checkpointer=InMemorySaver())
    thread = str(uuid.uuid4())

    first = await run(agent, {"messages": [HumanMessage(content="hi", id="hi")]}, mode, thread)
    second = await run(agent, {"messages": long_history()}, mode, thread)

    assert first[MODEL_SELECTION_KEY]["current"] == model_spec_from_model(small)
    assert first.get(SUMMARIZATION_EVENT_KEY) is None
    assert second[SUMMARIZATION_EVENT_KEY] is not None
    assert len(small.summaries) == 1


async def test_summarizer_override_writes_while_thresholds_follow_the_selection(mode: str) -> None:
    """`summarizer=` still writes the summary; the selected model still sizes the thresholds."""
    small, large = window_model("small", SMALL_WINDOW), window_model("large", LARGE_WINDOW)
    writer = window_model("writer", LARGE_WINDOW)
    summarization = create_summarization_middleware(large, StateBackend(), summarizer=writer)
    agent = create_deep_agent(model=large, middleware=[ChooseModel(lambda _: small), summarization], checkpointer=InMemorySaver())

    result = await run(agent, {"messages": long_history()}, mode)

    assert result[SUMMARIZATION_EVENT_KEY] is not None
    assert len(writer.summaries) == 1
    assert small.summaries == []
    assert large.summaries == []


async def test_pinned_model_ignores_the_selection(mode: str) -> None:
    """A pinned `model=` keeps sizing against itself whatever the agent selects."""
    small, large = window_model("small", SMALL_WINDOW), window_model("large", LARGE_WINDOW)
    pinned = SummarizationMiddleware(large, backend=StateBackend(), trigger=("fraction", 0.85), keep=("fraction", 0.1))
    agent = create_deep_agent(model=large, middleware=[ChooseModel(lambda _: small), pinned], checkpointer=InMemorySaver())

    result = await run(agent, {"messages": long_history(turns=6)}, mode)

    assert result.get(SUMMARIZATION_EVENT_KEY) is None
    assert small.summaries == []


async def test_subagent_follows_its_own_selection(mode: str) -> None:
    """A subagent sizes against the model its own middleware selects, not the parent's or its build model."""
    description = "z" * 50_000 * 4
    task_call: ToolCall = {"name": "task", "args": {"description": description, "subagent_type": "worker"}, "id": "t1"}
    parent = window_model("parent", 200_000, script=[AIMessage(content="", tool_calls=[task_call])])
    worker_small, worker_large = window_model("worker-small", SMALL_WINDOW), window_model("worker-large", LARGE_WINDOW)
    worker = {
        "name": "worker",
        "description": "Does work.",
        "system_prompt": "Work.",
        "model": worker_small,
        "middleware": [ChooseModel(lambda _: worker_large)],
    }
    agent = create_deep_agent(model=parent, subagents=[worker], checkpointer=InMemorySaver())

    result = await run(agent, {"messages": [HumanMessage(content="delegate")]}, mode)

    assert result["messages"][-1].content == "answer from parent"
    assert worker_small.calls == []
    assert worker_large.calls[-1][-1].content == description


async def test_subagent_selection_stays_out_of_parent(mode: str) -> None:
    """The #6644 shape: the parent stays on "strong" and its selection survives `task`."""
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


async def test_concurrent_runs_on_different_models_stay_separate(mode: str, rebuild: dict[str, BaseChatModel]) -> None:
    """Runs on different models at once each compact (or not) for their own model only."""
    small, large = window_model("small", SMALL_WINDOW), window_model("large", LARGE_WINDOW)
    rebuild["small"] = small

    def pick(state: dict[str, Any]) -> BaseChatModel:
        return small if state["messages"][0].text.startswith("[small]") else large

    agent = create_deep_agent(model=large, middleware=[ChooseModel(pick)], checkpointer=InMemorySaver())
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
    assert len(small.summaries) == 4
    assert large.summaries == []
    assert all(tokens(call) < SMALL_WINDOW for call in small.calls)


def test_recorded_model_without_window_skips_fractional_limits() -> None:
    """A recorded model with no context window skips fractional limits instead of failing the call."""
    middleware = SummarizationMiddleware(backend=MockBackend(), trigger=[("fraction", 0.5), ("messages", 4)], keep=("fraction", 0.1))
    model = window_model("no-window", None)
    history = long_history(turns=12, tokens_per_turn=10)
    state = {"messages": history, MODEL_SELECTION_KEY: {"current": model_spec_from_model(model)}}
    request = ModelRequest(model=model, messages=history, state=cast("Any", state), runtime=make_mock_runtime())

    result = middleware.wrap_model_call(request, lambda _: ModelResponse(result=[AIMessage(content="ok")]))

    assert SUMMARIZATION_EVENT_KEY in result.command.update  # ty: ignore[unresolved-attribute]
    assert len(model.summaries) == 1


def test_pinned_model_without_profile_still_rejects_fractional_settings() -> None:
    """A pinned model is validated when the middleware is built, as before."""
    with pytest.raises(ValueError, match="Model profile information is required"):
        SummarizationMiddleware(window_model("no-window", None), backend=MockBackend(), trigger=("fraction", 0.5))


class SummarizationRuns(BaseCallbackHandler):
    """Collects the metadata of each `summarization` run."""

    def __init__(self) -> None:
        self.metadata: list[dict[str, Any]] = []

    def on_chain_start(self, serialized: Any, inputs: Any, *, metadata: dict[str, Any] | None = None, **kwargs: Any) -> None:  # noqa: ANN401
        if kwargs.get("name") == "summarization":
            self.metadata.append(metadata or {})


@pytest.mark.parametrize("override", [False, True])
async def test_summary_trace_records_compaction_model_and_window(mode: str, override: bool, rebuild: dict[str, BaseChatModel]) -> None:  # noqa: FBT001
    """The `summarization` run names the model whose window was compacted, that window, and the writer."""
    small, large = window_model("small", SMALL_WINDOW), window_model("large", LARGE_WINDOW)
    rebuild["small"] = small
    writer = window_model("writer", LARGE_WINDOW)
    middleware: list[AgentMiddleware] = [ChooseModel(lambda _: small)]
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


def test_build_model_still_rejects_input_it_cannot_hold() -> None:
    """Without a switch, the 32K model still rejects input it cannot hold."""
    agent = create_deep_agent(model=window_model("small", SMALL_WINDOW))

    with pytest.raises(ContextOverflowError):
        agent.invoke({"messages": [HumanMessage(content="y" * 50_000 * 4)]})
