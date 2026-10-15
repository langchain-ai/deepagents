"""Bounded model input, safe output, and failure handling for thread names."""

from __future__ import annotations

import asyncio
import json
import threading
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import AsyncClient, MockTransport, Request, Response
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableBinding

from deepagents_code.thread_titles import generate_thread_name


@pytest.fixture
def title_model(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    model = AsyncMock(spec=BaseChatModel)
    model.model_copy.return_value = model
    model.bind.return_value = model
    model.ainvoke.return_value = AIMessage("Cache invalidation repair")
    factory = MagicMock(return_value=SimpleNamespace(model=model))
    monkeypatch.setattr("deepagents_code.config.create_model", factory)
    return model


async def test_generation_uses_only_visible_conversation(
    title_model: AsyncMock,
) -> None:
    name = await generate_thread_name(
        "provider:rename-model",
        [
            SystemMessage("Private system instructions"),
            HumanMessage("Fix caching"),
            HumanMessage("Goal changed", additional_kwargs={"lc_source": "goal_state"}),
            HumanMessage(
                "Private context", additional_kwargs={"lc_source": "local_context"}
            ),
            ToolMessage("Private tool output", tool_call_id="lookup"),
            AIMessage("Investigating stale cache entries"),
        ],
    )
    assert name == "Cache invalidation repair"
    prompt = title_model.ainvoke.call_args.args[0]
    assert prompt[1] == (
        "human",
        "human: Fix caching\nai: Investigating stale cache entries",
    )
    assert title_model.ainvoke.call_args.kwargs["config"]["callbacks"] == []


async def test_generation_bounds_conversation_sent_to_model(
    title_model: AsyncMock,
) -> None:
    await generate_thread_name(
        "provider:rename-model",
        [HumanMessage("x" * 9000), AIMessage("Must not be included")],
    )
    conversation = title_model.ainvoke.call_args.args[0][1][1]
    assert 0 < len(conversation) <= 8000
    assert conversation.startswith("human: ")
    assert "Must not be included" not in conversation


@pytest.mark.parametrize("shell_index", [0, 1])
async def test_shell_output_cannot_crowd_out_conversation(
    title_model: AsyncMock, shell_index: int
) -> None:
    messages = [
        HumanMessage("Fix caching"),
        AIMessage("Investigating stale cache entries"),
    ]
    messages.insert(
        shell_index,
        HumanMessage(
            "Verbose shell output\n" * 1000,
            additional_kwargs={"lc_source": "user_shell_command"},
        ),
    )
    await generate_thread_name("provider:rename-model", messages)
    conversation = title_model.ainvoke.call_args.args[0][1][1]
    assert conversation == "human: Fix caching\nai: Investigating stale cache entries"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('  "Cache\n\t invalidation repair."  ', "Cache invalidation repair"),
        ("`Fix\x1b cache\x07 entries\x9c`", "Fix cache entries"),
        (
            "one two three four five six seven eight nine",
            "one two three four five six seven eight",
        ),
        ("x" * 80, "x" * 50),
        (
            "Investigate stale cached configuration values across services",
            "Investigate stale cached configuration values",
        ),
    ],
)
async def test_generated_names_are_normalized(
    title_model: AsyncMock, raw: str, expected: str
) -> None:
    title_model.ainvoke.return_value = AIMessage(raw)
    name = await generate_thread_name(
        "provider:rename-model", [HumanMessage("Fix cache")]
    )
    assert name == expected


@pytest.mark.parametrize("raw", ["", "`...`"])
async def test_empty_generated_names_are_rejected(
    title_model: AsyncMock, raw: str
) -> None:
    title_model.ainvoke.return_value = AIMessage(raw)
    with pytest.raises(ValueError, match="between 1 and 50 characters"):
        await generate_thread_name("provider:rename-model", [HumanMessage("Fix cache")])


@pytest.mark.parametrize(
    "message",
    [
        SystemMessage("Rules"),
        HumanMessage(
            "Shell output", additional_kwargs={"lc_source": "user_shell_command"}
        ),
    ],
)
async def test_no_conversation_does_not_call_model(
    title_model: AsyncMock, message: SystemMessage | HumanMessage
) -> None:
    with pytest.raises(ValueError, match="Send a message"):
        await generate_thread_name("provider:rename-model", [message])
    title_model.ainvoke.assert_not_awaited()


async def test_stalled_model_times_out(
    title_model: AsyncMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    timeout = asyncio.timeout
    cancelled = asyncio.Event()

    async def stall(*_args: object, **_kwargs: object) -> AIMessage:
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
        return AIMessage("Unreachable")

    monkeypatch.setattr(
        "deepagents_code.thread_titles.asyncio.timeout", lambda _: timeout(0.05)
    )
    title_model.ainvoke.side_effect = stall
    with pytest.raises(TimeoutError):
        await generate_thread_name("provider:rename-model", [HumanMessage("Fix cache")])
    assert cancelled.is_set()


@pytest.mark.parametrize("factory_fails", [False, True])
async def test_initialization_timeout_waits_for_factory(
    monkeypatch: pytest.MonkeyPatch, *, factory_fails: bool
) -> None:
    """A naming timeout cannot leave provider-setting mutations running."""
    started, release = asyncio.Event(), threading.Event()
    loop = asyncio.get_running_loop()
    deadline = asyncio.timeout(None)
    model = AsyncMock()
    finished = threading.Event()

    def create_model(*_args: object, **_kwargs: object) -> SimpleNamespace:
        loop.call_soon_threadsafe(started.set)
        assert release.wait(timeout=5)
        finished.set()
        if factory_fails:
            msg = "Model initialization failed"
            raise ValueError(msg)
        return SimpleNamespace(model=model)

    monkeypatch.setattr("deepagents_code.config.create_model", create_model)
    monkeypatch.setattr(
        "deepagents_code.thread_titles.asyncio.timeout", lambda _: deadline
    )
    task = asyncio.create_task(
        generate_thread_name("provider:rename-model", [HumanMessage("Fix cache")])
    )
    try:
        await asyncio.wait_for(started.wait(), timeout=5)
        deadline.reschedule(loop.time())
        done, _ = await asyncio.wait({task}, timeout=0.05)
        assert not done
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
    with pytest.raises(TimeoutError):
        await task
    assert finished.is_set()
    model.ainvoke.assert_not_awaited()


@pytest.mark.parametrize(
    "source", ["model_kwargs", "extra_body", "binding", "binding_extra_body"]
)
async def test_naming_removes_openai_tools_from_provider_request(
    source: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from langchain_openai import ChatOpenAI
    from pydantic import SecretStr

    options = {
        "tools": [{"type": "function", "function": {"name": "execute"}}],
        "tool_choice": "required",
        "functions": [{"name": "execute"}],
        "function_call": "auto",
        "parallel_tool_calls": True,
    }

    def respond(request: Request) -> Response:
        payload = json.loads(request.content)
        assert not options.keys() & payload.keys()
        assert payload["temperature"] == pytest.approx(0.2)
        return Response(
            200,
            json={
                "id": "thread-title",
                "model": "test-model",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "Cache repair"},
                        "finish_reason": "stop",
                        "index": 0,
                    }
                ],
            },
        )

    async with AsyncClient(transport=MockTransport(respond)) as client:
        model = ChatOpenAI(
            model="test-model",
            api_key=SecretStr("test"),
            http_async_client=client,
            temperature=0.4 if source.startswith("binding") else 0.2,
            use_responses_api=False,
            max_retries=0,
            model_kwargs=options if source == "model_kwargs" else {},
            extra_body=options if source == "extra_body" else None,
        )
        bound_options = (
            {"extra_body": options} if source == "binding_extra_body" else options
        )
        selected = (
            model.bind(**bound_options, temperature=0.2)
            if source.startswith("binding")
            else model
        )
        original = deepcopy(model._get_request_payload([HumanMessage("Main task")]))
        monkeypatch.setattr(
            "deepagents_code.config.create_model",
            lambda *_args, **_kwargs: SimpleNamespace(model=selected),
        )
        assert (
            await generate_thread_name("openai:test-model", [HumanMessage("Fix cache")])
            == "Cache repair"
        )
        assert model._get_request_payload([HumanMessage("Main task")]) == original
        if source.startswith("binding"):
            assert isinstance(selected, RunnableBinding)
            assert selected.kwargs == {**bound_options, "temperature": 0.2}


@pytest.mark.parametrize("source", ["constructor", "extra_body", "binding"])
async def test_naming_removes_anthropic_mcp_from_provider_request(
    source: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from anthropic import AsyncAnthropic
    from langchain_anthropic import ChatAnthropic
    from pydantic import SecretStr

    servers = [{"type": "url", "url": "https://example.com/mcp", "name": "example"}]
    tool_options = {
        "tools": [{"type": "mcp_toolset", "mcp_server_name": "example"}],
        "tool_choice": {"type": "auto"},
    }
    options = {"mcp_servers": servers, **tool_options}

    def respond(request: Request) -> Response:
        payload = json.loads(request.content)
        assert not options.keys() & payload.keys()
        assert payload["temperature"] == pytest.approx(0.2)
        assert payload["max_tokens"] == 256
        return Response(
            200,
            json={
                "id": "thread-title",
                "type": "message",
                "role": "assistant",
                "model": "test-model",
                "content": [{"type": "text", "text": "Cache repair"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 2},
            },
        )

    async with AsyncAnthropic(
        api_key="test",
        http_client=AsyncClient(transport=MockTransport(respond)),
        max_retries=0,
    ) as client:
        monkeypatch.setattr(
            ChatAnthropic, "_async_client", property(lambda _self: client)
        )
        model = ChatAnthropic(
            model_name="test-model",
            api_key=SecretStr("test"),
            temperature=0.4 if source == "binding" else 0.2,
            max_tokens_to_sample=256,
            mcp_servers=servers if source == "constructor" else None,
            model_kwargs=(
                tool_options
                if source == "constructor"
                else {"extra_body": options}
                if source == "extra_body"
                else {}
            ),
        )
        selected = (
            model.bind(**options, temperature=0.2) if source == "binding" else model
        )
        original = deepcopy(
            model._get_request_payload([HumanMessage("Main task")], stop=None)
        )
        monkeypatch.setattr(
            "deepagents_code.config.create_model",
            lambda *_args, **_kwargs: SimpleNamespace(model=selected),
        )
        assert (
            await generate_thread_name(
                "anthropic:test-model", [HumanMessage("Fix cache")]
            )
            == "Cache repair"
        )
        assert (
            model._get_request_payload([HumanMessage("Main task")], stop=None)
            == original
        )
        if source == "binding":
            assert isinstance(selected, RunnableBinding)
            assert selected.kwargs == {**options, "temperature": 0.2}
