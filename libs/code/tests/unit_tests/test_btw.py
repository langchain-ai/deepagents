"""Side answers must never run tools or mutate the main conversation."""

from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest
from httpx import (
    ASGITransport,
    AsyncClient,
    MockTransport,
    ReadError,
    Request,
    Response,
)
from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.runtime import ExecutionInfo, Runtime
from textual import events
from textual.containers import VerticalScroll
from textual.widgets import Markdown, Static, TextArea

from deepagents_code.btw import BtwOperation
from deepagents_code.client.remote_client import RemoteAgent
from deepagents_code.tui.modals.btw import BtwScreen, BtwTextArea
from deepagents_code.tui.widgets.messages import AssistantMessage, UserMessage

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable
    from pathlib import Path

    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import BaseMessage
    from langchain_core.runnables import RunnableConfig
    from langgraph.pregel import Pregel

    from deepagents_code.app import DeepAgentsApp


@pytest.fixture
def invoke(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    mock = AsyncMock(return_value=AIMessage(content="The answer"))
    monkeypatch.setattr(FakeMessagesListChatModel, "ainvoke", mock)
    return mock


type BtwServer = tuple[AsyncClient, BtwOperation, MagicMock]


@pytest.fixture
async def btw_server(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[BtwServer]:
    from deepagents_code import offload_api

    operation = BtwOperation(FakeMessagesListChatModel(responses=[]), "system", None)
    threads = MagicMock(get_state=AsyncMock(return_value={"values": {}}))
    monkeypatch.setattr("deepagents_code.btw_api.require_thread_workspace", AsyncMock())
    monkeypatch.setattr(
        offload_api,
        "get_server_runtime",
        AsyncMock(
            return_value=SimpleNamespace(backend=SimpleNamespace(_dcode_btw=operation))
        ),
    )
    monkeypatch.setattr(
        offload_api, "_thread_client", lambda: SimpleNamespace(threads=threads)
    )
    async with AsyncClient(
        transport=ASGITransport(app=offload_api.app), base_url="http://test"
    ) as client:
        yield client, operation, threads


async def test_tool_free_snapshot_keeps_state_and_uses_compaction(
    invoke: AsyncMock,
) -> None:
    model = FakeMessagesListChatModel(responses=[AIMessage(content="The answer")])
    operation = BtwOperation(model, "Main instructions", None)
    state = {
        "messages": [
            HumanMessage(content="Archived secret"),
            HumanMessage(content="Current request"),
            AIMessage(
                content="Looking",
                tool_calls=[
                    {"name": "execute", "args": {"command": "git status"}, "id": "1"}
                ],
            ),
            ToolMessage(content="Tool context", tool_call_id="1", name="execute"),
            AIMessage(
                content="", tool_calls=[{"name": "execute", "args": {}, "id": "2"}]
            ),
        ],
        "_summarization_event": {
            "cutoff_index": 1,
            "summary_message": HumanMessage(content="Earlier summary"),
        },
    }
    before = deepcopy(state)
    assert await operation.answer("thread", state, "Why?") == "The answer"
    messages = invoke.call_args.args[0]
    assert [m.text for m in messages[1:-1]] == [
        "Earlier summary",
        "Current request",
        'Looking\n\n[tool call 1: execute]\n{"command": "git status"}',
        "[tool result 1: execute]\nTool context",
        "[tool call 2: execute]\n{}",
    ]
    assert all(not isinstance(m, ToolMessage) for m in messages)
    assert all(not m.tool_calls for m in messages if isinstance(m, AIMessage))
    assert "Do not call any tools" in messages[0].text
    assert invoke.call_args.kwargs == {
        "config": {"callbacks": [], "metadata": {"thread_id": "thread"}}
    }
    assert state == before


@pytest.mark.parametrize(
    "attachment",
    [
        {"type": "image_url", "image_url": {"url": "https://example.com/design.png"}},
        {"type": "image", "base64": "aW1hZ2U=", "mime_type": "image/png"},
        {"type": "video", "base64": "dmlkZW8=", "mime_type": "video/mp4"},
    ],
)
@pytest.mark.parametrize("include_text", [False, True])
async def test_side_context_preserves_human_attachments(
    attachment: dict[str, object], invoke: AsyncMock, *, include_text: bool
) -> None:
    model = FakeMessagesListChatModel(responses=[], profile={"max_input_tokens": 8000})
    operation = BtwOperation(model, "Main instructions", None)
    human = HumanMessage(
        content=[
            *([{"type": "text", "text": "Review this design"}] if include_text else []),
            attachment,
        ]
    )
    state = {
        "messages": [
            human.model_dump(),
            AIMessage(
                content=[
                    {"type": "thinking", "thinking": "Private", "signature": "sig"},
                    {"type": "text", "text": "I see the design"},
                ],
                additional_kwargs={"provider_state": "opaque"},
            ).model_dump(),
        ]
    }
    before = deepcopy(state)
    await operation.answer("thread", state, "What color is the background?")
    messages = invoke.call_args.args[0]
    assert messages[1] == human
    assert messages[2] == AIMessage(content="I see the design")
    assert messages[-1].text.endswith("What color is the background?")
    messages[1].content[-1]["mime_type"] = "changed"
    assert state == before


@pytest.mark.parametrize(
    "attachment",
    [
        {"type": "image_url", "image_url": {"url": "https://example.com/design.png"}},
        {"type": "image", "base64": "aW1hZ2U=", "mime_type": "image/png"},
        {"type": "file", "base64": "cGRm", "mime_type": "application/pdf"},
        {
            "type": "document",
            "source": {
                "type": "base64",
                "data": "cGRm",
                "media_type": "application/pdf",
            },
        },
    ],
    ids=["image-url", "image", "file", "document"],
)
@pytest.mark.parametrize(("include_text", "serialized"), [(False, False), (True, True)])
async def test_side_context_preserves_tool_attachments(
    attachment: dict[str, object],
    invoke: AsyncMock,
    *,
    include_text: bool,
    serialized: bool,
) -> None:
    model = FakeMessagesListChatModel(
        responses=[],
        profile={"max_input_tokens": 8000, "image_inputs": True, "pdf_inputs": True},
    )
    operation = BtwOperation(model, "Main instructions", None)
    result = ToolMessage(
        content=[
            *([{"type": "text", "text": "The design:"}] if include_text else []),
            attachment,
            *(["End of design"] if include_text else []),
        ],
        tool_call_id="read-design",
        name="read_file",
        artifact={"internal": "Do not send to the model"},
    )
    state = {"messages": [result.model_dump() if serialized else result]}
    before = deepcopy(state)
    await operation.answer("thread", state, "What does the design show?")
    messages = invoke.call_args.args[0]
    assert messages[1] == HumanMessage(
        content=[
            {"type": "text", "text": "[tool result read-design: read_file]\n"},
            *result.content,
        ]
    )
    assert all(not isinstance(message, ToolMessage) for message in messages)
    messages[1].content[2 if include_text else 1]["mime_type"] = "changed"
    assert state == before


@pytest.mark.parametrize(
    ("attachment", "capability"),
    [
        (
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/design.png"},
            },
            "image_inputs",
        ),
        (
            {"type": "image", "base64": "aW1hZ2U=", "mime_type": "image/png"},
            "image_inputs",
        ),
        (
            {"type": "video", "base64": "dmlkZW8=", "mime_type": "video/mp4"},
            "video_inputs",
        ),
        (
            {"type": "audio", "base64": "YXVkaW8=", "mime_type": "audio/wav"},
            "audio_inputs",
        ),
        (
            {"type": "file", "base64": "cGRm", "mime_type": "application/pdf"},
            "pdf_inputs",
        ),
    ],
)
@pytest.mark.parametrize("tool_result", [False, True])
async def test_side_context_filters_unsupported_attachments_without_changing_state(
    attachment: dict[str, object],
    capability: str,
    invoke: AsyncMock,
    *,
    tool_result: bool,
) -> None:
    model = FakeMessagesListChatModel(responses=[], profile={capability: False})
    operation = BtwOperation(model, "Main instructions", None)
    content: list[str | dict[str, object]] = [
        {"type": "text", "text": "Review this design"},
        attachment,
    ]
    message = (
        ToolMessage(content=content, tool_call_id="read-design", name="read_file")
        if tool_result
        else HumanMessage(content=content)
    )
    state = {"messages": [message.model_dump()]}
    before = deepcopy(state)
    await operation.answer("thread", state, "What does the design show?")
    context = invoke.call_args.args[0][1]
    assert all(block["type"] == "text" for block in context.content_blocks)
    assert "Review this design" in context.text
    assert "was not attached because this model does not support" in context.text
    assert state == before


@pytest.mark.parametrize("source", ["bootstrap", "snapshot", "checkpoint"])
async def test_side_context_filters_for_selected_model_before_budgeting(
    source: str, invoke: AsyncMock
) -> None:
    old = FakeMessagesListChatModel(responses=[], profile={"image_inputs": True})
    active = FakeMessagesListChatModel(
        responses=[], profile={"max_input_tokens": 1600, "image_inputs": False}
    )
    operation = BtwOperation(active if source == "bootstrap" else old, "system", None)
    state: dict[str, object] = {
        "messages": [
            HumanMessage(
                content=[
                    {"type": "text", "text": "Keep this context"},
                    *[
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.com/design.png"},
                        }
                        for _ in range(20)
                    ],
                ]
            )
        ]
    }
    if source == "snapshot":
        operation._snapshots["thread"] = (active, SystemMessage(content="system"), {})
    elif source == "checkpoint":
        state["_model_spec"] = "test:text-only"
    before = deepcopy(state)
    with patch(
        "deepagents_code.config.create_model",
        return_value=SimpleNamespace(model=active),
    ):
        await operation.answer("thread", state, "Why?")
    messages = invoke.call_args.args[0]
    assert "Keep this context" in messages[1].text
    assert all(block["type"] == "text" for block in messages[1].content_blocks)
    assert messages[-1].text.endswith("Why?")
    assert state == before


@pytest.mark.parametrize(
    ("tool", "summarized", "context_limit"),
    [
        ("write_file", False, 8000),
        ("write_file", True, None),
        ("edit_file", False, None),
        ("edit_file", True, 8000),
    ],
)
async def test_side_context_truncates_old_file_arguments(
    tool: str, context_limit: int | None, invoke: AsyncMock, *, summarized: bool
) -> None:
    model = FakeMessagesListChatModel(
        responses=[],
        profile={"max_input_tokens": context_limit} if context_limit else None,
    )
    operation = BtwOperation(model, "Main instructions", None)
    old_content = "old file contents\n" * 4000
    recent_content = "recent file contents\n" * 101
    argument = "content" if tool == "write_file" else "new_string"
    state: dict[str, object] = {
        "messages": [
            HumanMessage(content="Archived request"),
            HumanMessage(content="Update the file"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": tool,
                        "args": {"file_path": "/example.py", argument: old_content},
                        "id": "old",
                    }
                ],
            ),
            ToolMessage(content="Saved", tool_call_id="old"),
            *(HumanMessage(content="More context") for _ in range(20)),
            HumanMessage(content="Make one more change"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": tool,
                        "args": {"file_path": "/example.py", argument: recent_content},
                        "id": "recent",
                    }
                ],
            ),
            ToolMessage(content="Saved", tool_call_id="recent"),
        ],
    }
    if summarized:
        state["_summarization_event"] = {
            "cutoff_index": 1,
            "summary_message": HumanMessage(content="Earlier summary"),
        }
    before = deepcopy(state)
    assert await operation.answer("thread", state, "Why?") == "The answer"
    messages = invoke.call_args.args[0]
    transcript = "\n".join(message.text for message in messages)
    assert "...(argument truncated)" in transcript
    assert json.dumps(old_content) not in transcript
    assert json.dumps(recent_content) in transcript
    assert ("Earlier summary" if summarized else "Archived request") in transcript
    assert all(not isinstance(message, ToolMessage) for message in messages)
    assert all(
        not message.tool_calls for message in messages if isinstance(message, AIMessage)
    )
    assert state == before


@pytest.mark.parametrize("max_tokens", [0, 400])
async def test_side_context_budget_includes_instructions_history_and_output(
    max_tokens: int,
    invoke: AsyncMock,
) -> None:
    from langchain_core.messages.utils import count_tokens_approximately

    model = FakeMessagesListChatModel(responses=[], profile={"max_input_tokens": 2000})
    operation = BtwOperation(
        cast("BaseChatModel", model.bind(max_tokens=max_tokens)),
        "Main instructions " * 40,
        None,
    )
    state = {"messages": [HumanMessage(content="Old context " * 1000)]}
    history = [
        ("Old side question", "Old side answer " * 340),
        ("Which file?", "example.py"),
    ]
    before = deepcopy(state)
    history_before = deepcopy(history)
    await operation.answer("thread", state, "Why?", history=history)
    messages = invoke.call_args.args[0]
    assert count_tokens_approximately(messages) <= 1900 - max_tokens
    assert messages[0].text.startswith("Main instructions " * 40)
    retained_history = history[-1:] if max_tokens else history
    assert [message.text for message in messages[1:-1]] == [
        text for pair in retained_history for text in pair
    ]
    assert messages[-1].text.endswith("Why?")
    assert history == history_before
    assert state == before


@pytest.mark.parametrize(
    ("model_spec", "snapshot"),
    [
        ("test:bootstrap", "resumed"),
        ("test:switched", "resumed"),
        ("test:switched", "live"),
    ],
)
async def test_side_context_keeps_session_profile_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model_spec: str, snapshot: str
) -> None:
    from langchain_core.messages.utils import count_tokens_approximately

    from deepagents_code import server_graph
    from deepagents_code._server_config import ServerConfig
    from deepagents_code._testing_models import DeterministicIntegrationChatModel
    from deepagents_code.btw import BTW_OPERATION_ATTR

    monkeypatch.setattr(
        "deepagents_code.config._create_model_via_init",
        lambda *_args, **_kwargs: DeterministicIntegrationChatModel(
            profile={"max_input_tokens": 8000, "tool_calling": True}
        ),
    )
    server = await server_graph._make_graphs(
        config_override=ServerConfig(
            model="test:bootstrap",
            profile_overrides={"max_input_tokens": 2000},
            cwd=str(tmp_path),
            system_prompt="Main instructions",
            no_mcp=True,
            enable_shell=False,
            enable_memory=False,
            enable_skills=False,
        )
    )
    operation: BtwOperation = getattr(server.backend, BTW_OPERATION_ATTR)
    if snapshot == "live":
        operation._snapshots["thread"] = (
            cast("BaseChatModel", operation._model),
            SystemMessage(content="Main instructions"),
            {},
        )
    state = {
        "_model_spec": model_spec,
        "messages": [
            HumanMessage(content="Old context " * 1000),
            HumanMessage(content="Recent context"),
        ],
    }
    before = deepcopy(state)
    with patch.object(
        DeterministicIntegrationChatModel,
        "ainvoke",
        new=AsyncMock(return_value=AIMessage(content="The answer")),
    ) as invoke:
        assert await operation.answer("thread", state, "Why?") == "The answer"
    messages = invoke.call_args.args[0]
    assert count_tokens_approximately(messages) <= 1900
    assert messages[0].text.startswith("Main instructions")
    assert [message.text for message in messages[1:-1]] == ["Recent context"]
    assert messages[-1].text.endswith("Why?")
    assert state == before


@pytest.mark.parametrize("oversized", ["system", "question"])
async def test_side_context_rejects_required_input_over_budget(
    oversized: str, invoke: AsyncMock
) -> None:
    from langchain_core.exceptions import ContextOverflowError

    model = FakeMessagesListChatModel(responses=[], profile={"max_input_tokens": 1000})
    operation = BtwOperation(
        model, "Instructions " * (1000 if oversized == "system" else 1), None
    )
    question = "Why? " * (1000 if oversized == "question" else 1)
    with pytest.raises(ContextOverflowError, match="input budget"):
        await operation.answer("thread", {}, question)
    invoke.assert_not_awaited()


async def test_live_model_selected_before_main_response_and_no_wait() -> None:
    old = FakeMessagesListChatModel(responses=[AIMessage(content="old")])
    active = FakeMessagesListChatModel(responses=[AIMessage(content="active")])
    operation = BtwOperation(old, "old system", None)
    entered = asyncio.Event()
    release = asyncio.Event()

    async def main_handler(_request: ModelRequest) -> ModelResponse:
        entered.set()
        await release.wait()
        return ModelResponse(result=[AIMessage(content="main finished")])

    request = ModelRequest(
        model=active,
        messages=[HumanMessage(content="main")],
        system_message=SystemMessage(content="Current system"),
        tools=[],
        runtime=Runtime(
            execution_info=ExecutionInfo(
                thread_id="thread",
                checkpoint_id="checkpoint",
                checkpoint_ns="",
                task_id="task",
            )
        ),
    )
    main = asyncio.create_task(operation.awrap_model_call(request, main_handler))
    try:
        await entered.wait()
        assert (
            await operation.answer("thread", {"messages": request.messages}, "aside")
            == "active"
        )
        assert not main.done()
        assert await operation.answer("other-thread", {}, "aside") == "old"
    finally:
        release.set()
        await main


async def test_snapshot_preserves_settings_without_tools_or_shared_mutation() -> None:
    model = FakeMessagesListChatModel(responses=[AIMessage(content="answer")])
    operation = BtwOperation(model, "system", None)
    request: ModelRequest = ModelRequest(
        model=cast(
            "BaseChatModel",
            model.bind(temperature=0.9, max_tokens=512, tools=[{"name": "execute"}]),
        ),
        messages=[],
        tools=[],
        model_settings={
            "temperature": 0.2,
            "reasoning": {"effort": "high"},
            "tool_choice": "required",
            "parallel_tool_calls": True,
        },
        runtime=Runtime(
            execution_info=ExecutionInfo(
                thread_id="thread",
                checkpoint_id="c",
                checkpoint_ns="model:t",
                task_id="t",
            )
        ),
    )
    response = ModelResponse(result=[AIMessage(content="main answer")])
    operation.wrap_model_call(request, MagicMock(return_value=response))
    fork_model = FakeMessagesListChatModel(responses=[AIMessage(content="fork")])
    fork_request = request.override(
        model=fork_model,
        system_message=SystemMessage(content="Fork instructions"),
        model_settings={"temperature": 0.8},
        runtime=Runtime(
            execution_info=ExecutionInfo(
                thread_id="thread",
                checkpoint_id="fork-checkpoint",
                checkpoint_ns="tools:task|model:fork-task",
                task_id="fork-task",
            )
        ),
    )
    operation.wrap_model_call(fork_request, MagicMock(return_value=response))
    assert await operation.answer("thread", {}, "why") == "answer"
    request.model_settings["reasoning"]["effort"] = "low"

    def invoke(messages: list[SystemMessage], **kwargs: object) -> AIMessage:
        assert messages[0].text.startswith("system\n\n")
        assert kwargs == {
            "config": {"callbacks": [], "metadata": {"thread_id": "thread"}},
            "temperature": 0.2,
            "max_tokens": 512,
            "reasoning": {"effort": "high"},
        }
        reasoning = kwargs["reasoning"]
        assert isinstance(reasoning, dict)
        cast("dict[str, object]", reasoning)["effort"] = "low"
        return AIMessage(content="answer")

    with patch.object(
        FakeMessagesListChatModel, "ainvoke", new=AsyncMock(side_effect=invoke)
    ):
        assert await operation.answer("thread", {}, "why") == "answer"
        assert await operation.answer("thread", {}, "again") == "answer"
    with patch.object(
        FakeMessagesListChatModel,
        "ainvoke",
        new=AsyncMock(return_value=AIMessage(content="other")),
    ) as other:
        assert await operation.answer("other-thread", {}, "why") == "other"
    assert other.call_args.kwargs == {
        "config": {"callbacks": [], "metadata": {"thread_id": "other-thread"}}
    }
    assert request.model_settings["tool_choice"] == "required"


@pytest.mark.parametrize("synchronous", [False, True])
async def test_effective_context_survives_restart_and_eviction(
    *,
    synchronous: bool,
) -> None:
    from langchain.agents import create_agent
    from langchain.agents.middleware import dynamic_prompt
    from langgraph.checkpoint.memory import InMemorySaver

    @dynamic_prompt
    def extension_prompt(_request: ModelRequest) -> str:
        return "Current model identity. Extension instruction: answer in Spanish."

    class GenerationSettings(AgentMiddleware):
        def wrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], ModelResponse],
        ) -> ModelResponse:
            return handler(self._override(request))

        async def awrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            return await handler(self._override(request))

        def _override(self, request: ModelRequest) -> ModelRequest:
            return request.override(
                model=cast(
                    "BaseChatModel", request.model.bind(temperature=0.9, max_tokens=512)
                ),
                model_settings={"temperature": 0.2, "reasoning": {"effort": "high"}},
            )

    model = FakeMessagesListChatModel(responses=[AIMessage(content="main")])
    warm = BtwOperation(model, "Bootstrap instructions", None)
    graph = create_agent(
        model,
        middleware=[extension_prompt, GenerationSettings(), warm],
        checkpointer=InMemorySaver(),
    )
    config: RunnableConfig = {"configurable": {"thread_id": "instruction-parity"}}
    inputs = {"messages": [HumanMessage("Hello")]}
    if synchronous:
        await asyncio.to_thread(graph.invoke, inputs, config)
    else:
        await graph.ainvoke(inputs, config)
    checkpoint = await graph.aget_state(config)
    cold = BtwOperation(model, "Bootstrap instructions", None)
    prompts: list[str] = []

    def answer(messages: list[BaseMessage], **kwargs: object) -> AIMessage:
        prompts.append(messages[0].text)
        assert kwargs["temperature"] == pytest.approx(0.2)
        assert kwargs["max_tokens"] == 512
        assert kwargs["reasoning"] == {"effort": "high"}
        cast("dict[str, object]", kwargs["reasoning"])["effort"] = "low"
        return AIMessage(content="Respuesta")

    with patch.object(
        FakeMessagesListChatModel, "ainvoke", new=AsyncMock(side_effect=answer)
    ):
        for operation in (warm, cold):
            await operation.answer("instruction-parity", checkpoint.values, "Why?")
        warm._snapshots.clear()
        await warm.answer("instruction-parity", checkpoint.values, "Why?")
    assert len(set(prompts)) == 1
    assert "Extension instruction: answer in Spanish." in prompts[0]
    assert (await graph.aget_state(config)) == checkpoint


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("retries", [0, 1])
@pytest.mark.parametrize("restart", [False, True])
async def test_side_answer_honors_retry_budget(
    streaming: bool, retries: int, restart: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    from deepagents_code.config import MODEL_RETRIES_ATTR

    model = FakeMessagesListChatModel(responses=[])
    setattr(model, MODEL_RETRIES_ATTR, retries)
    operation = BtwOperation(model, "system", None)
    state: dict[str, object] = {}
    if restart:
        captured = operation.wrap_model_call(
            ModelRequest(
                model=model,
                messages=[],
                tools=[],
                runtime=Runtime(
                    execution_info=ExecutionInfo(
                        thread_id="thread",
                        checkpoint_id="c",
                        checkpoint_ns="",
                        task_id="t",
                    )
                ),
            ),
            lambda _request: ModelResponse(result=[AIMessage(content="main")]),
        )
        assert captured.command is not None
        assert isinstance(captured.command.update, dict)
        state = captured.command.update
        replacement = FakeMessagesListChatModel(responses=[])
        setattr(replacement, MODEL_RETRIES_ATTR, 1 - retries)
        operation = BtwOperation(replacement, "system", None)
    monkeypatch.setattr(
        "deepagents_code.model_retry._compute_backoff_delay", lambda _: 0
    )
    attempts = 0
    closed: list[int] = []
    fragments: list[str] = []

    def invoke(_messages: object, **_kwargs: object) -> AIMessage:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            msg = "Connection dropped"
            raise ReadError(msg)
        return AIMessage(content="Recovered answer")

    invocation = AsyncMock(side_effect=invoke)

    async def stream(
        _self: object, _messages: object, **_kwargs: object
    ) -> AsyncIterator[AIMessageChunk]:
        try:
            # Non-text chunks must not prevent a retry before visible output.
            yield AIMessageChunk(content=[{"type": "reasoning", "reasoning": "plan"}])
            result = await invocation(_messages, **_kwargs)
            yield AIMessageChunk(content=result.content)
        finally:
            closed.append(attempts)

    on_text = AsyncMock(side_effect=fragments.append)
    monkeypatch.setattr(FakeMessagesListChatModel, "ainvoke", invocation)
    monkeypatch.setattr(FakeMessagesListChatModel, "astream", stream)
    answer = operation.answer(
        "thread", state, "why", on_text=on_text if streaming else None
    )
    if retries:
        assert await answer == "Recovered answer"
        assert fragments == (["Recovered answer"] if streaming else [])
    else:
        with pytest.raises(ReadError, match="Connection dropped"):
            await answer
        assert not fragments
    assert attempts == retries + 1
    assert closed == (list(range(1, attempts + 1)) if streaming else [])


@pytest.mark.parametrize("failure", ["provider", "receiver"])
async def test_stream_never_retries_after_visible_text(
    failure: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from deepagents_code.config import MODEL_RETRIES_ATTR

    model = FakeMessagesListChatModel(responses=[])
    setattr(model, MODEL_RETRIES_ATTR, 2)
    operation = BtwOperation(model, "system", None)
    attempts = 0
    closed = asyncio.Event()
    fragments: list[str] = []

    async def stream(
        _self: object, _messages: object, **_kwargs: object
    ) -> AsyncIterator[AIMessageChunk]:
        nonlocal attempts
        attempts += 1
        try:
            await asyncio.sleep(0)
            yield AIMessageChunk(content="Partial answer")
            msg = "Provider connection dropped"
            raise ReadError(msg)
        finally:
            closed.set()

    def on_text(text: str) -> None:
        fragments.append(text)
        if failure == "receiver":
            msg = "Receiver connection dropped"
            raise ReadError(msg)

    monkeypatch.setattr(FakeMessagesListChatModel, "astream", stream)
    with pytest.raises(ReadError, match="connection dropped"):
        await operation.answer(
            "thread", {}, "why", on_text=AsyncMock(side_effect=on_text)
        )
    assert fragments == ["Partial answer"]
    assert attempts == 1
    assert closed.is_set()


@pytest.mark.parametrize("phase", ["startup", "backoff"])
async def test_cancel_before_visible_text_stops_retries_and_closes_stream(
    phase: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from deepagents_code.config import MODEL_RETRIES_ATTR

    model = FakeMessagesListChatModel(responses=[])
    setattr(model, MODEL_RETRIES_ATTR, 2)
    operation = BtwOperation(model, "system", None)
    ready = asyncio.Event()
    closed = asyncio.Event()
    attempts = 0

    def delay(_attempt: int) -> float:
        ready.set()
        return 10

    async def stream(
        _self: object, _messages: object, **_kwargs: object
    ) -> AsyncIterator[AIMessageChunk]:
        nonlocal attempts
        attempts += 1
        try:
            if phase == "backoff":
                msg = "Connection dropped"
                raise ReadError(msg)
            ready.set()
            await asyncio.Event().wait()
            yield AIMessageChunk(content="unused")
        finally:
            closed.set()

    monkeypatch.setattr("deepagents_code.model_retry._compute_backoff_delay", delay)
    monkeypatch.setattr(FakeMessagesListChatModel, "astream", stream)
    on_text = AsyncMock()
    task = asyncio.create_task(operation.answer("thread", {}, "why", on_text=on_text))
    try:
        await asyncio.wait_for(ready.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert attempts == 1
        assert closed.is_set()
        on_text.assert_not_awaited()
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.fixture
def instruction_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from deepagents_code import agent

    user = tmp_path / "user.md"
    project = tmp_path / "AGENTS.md"
    user.write_text("Always answer in Spanish.\n<!-- private marker -->\n")
    project.write_text("Reuse the existing HTTP client.\n")
    skills = tmp_path / "skills"
    skill = skills / "review"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: review\ndescription: Review database migrations.\n---\nCheck SQL.\n"
    )
    monkeypatch.setattr(agent, "get_user_agent_md_path", lambda _name: user)
    monkeypatch.setattr(agent, "get_project_agent_md_path", lambda _root: [project])
    monkeypatch.setattr(
        agent, "get_skill_sources", lambda **_kwargs: [(str(skills), "Project")]
    )
    return tmp_path


@pytest.mark.parametrize("resumed", [False, True])
async def test_side_instructions_survive_a_fresh_runtime(
    instruction_workspace: Path, monkeypatch: pytest.MonkeyPatch, *, resumed: bool
) -> None:
    from langgraph.checkpoint.memory import InMemorySaver

    from deepagents_code._testing_models import DeterministicIntegrationChatModel
    from deepagents_code.agent import create_cli_agent
    from deepagents_code.btw import BTW_OPERATION_ATTR

    model = DeterministicIntegrationChatModel()
    monkeypatch.setattr(
        "deepagents_code.config.create_model",
        lambda *_args, **_kwargs: SimpleNamespace(model=model),
    )
    saver = InMemorySaver()
    config: RunnableConfig = {"configurable": {"thread_id": "instructions"}}

    def build() -> tuple[Pregel, BtwOperation]:
        graph, backend = create_cli_agent(
            model=model,
            assistant_id="test-btw-instructions",
            cwd=instruction_workspace,
            enable_shell=False,
            system_prompt="Explain the project.",
            checkpointer=saver,
        )
        return graph, getattr(backend, BTW_OPERATION_ATTR)

    graph, warm = build()
    if resumed:
        await graph.ainvoke({"messages": [HumanMessage("Hello")]}, config)
    before = await graph.aget_state(config)
    _new_graph, cold = build()
    captured: list[str] = []

    def answer(messages: list[BaseMessage], **_kwargs: object) -> AIMessage:
        captured.append(messages[0].text)
        return AIMessage(content="Una respuesta.")

    with patch.object(
        DeterministicIntegrationChatModel, "ainvoke", new=AsyncMock(side_effect=answer)
    ):
        # Calling the operation directly must neither populate the checkpoint
        # nor need a regular agent turn in the newly constructed runtime.
        for operation in (warm, cold):
            assert (
                await operation.answer("instructions", before.values, "Why?")
                == "Una respuesta."
            )
        warm._snapshots.clear()  # Simulate eviction within the same runtime.
        assert (
            await warm.answer("instructions", before.values, "Why?") == "Una respuesta."
        )
    for prompt in captured:
        assert "Always answer in Spanish." in prompt
        assert "Reuse the existing HTTP client." in prompt
        assert "Review database migrations." in prompt
        assert "private marker" not in prompt
    assert (await graph.aget_state(config)) == before


@pytest.mark.parametrize("source", ["bootstrap", "snapshot", "checkpoint"])
@pytest.mark.parametrize("container", ["model_kwargs", "extra_body"])
async def test_constructor_tools_are_absent_from_provider_request(
    source: str, container: str
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
    params = {container: options}
    original = deepcopy(params)

    def respond(request: Request) -> Response:
        payload = json.loads(request.content)
        assert not options.keys() & payload.keys()
        assert payload["temperature"] == pytest.approx(0.2)
        return Response(
            200,
            json={
                "id": "side-answer",
                "model": "test-model",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "answer"},
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
            temperature=0.2,
            use_responses_api=False,
            max_retries=0,
            model_kwargs=options if container == "model_kwargs" else {},
            extra_body=options if container == "extra_body" else None,
        )
        operation = BtwOperation(model, "system", None)
        state: dict[str, object] = {}
        if source == "snapshot":
            operation._snapshots["thread"] = (
                model,
                SystemMessage(content="system"),
                {},
            )
        elif source == "checkpoint":
            state = {"_model_spec": "openai:test-model", "_model_params": params}
        with patch(
            "deepagents_code.config.create_model",
            return_value=SimpleNamespace(model=model),
        ):
            assert await operation.answer("thread", state, "why") == "answer"
        assert params == original
        defaults = model._get_request_payload([HumanMessage(content="main")])
        assert (defaults.get("extra_body") or defaults)["tools"] == options["tools"]


@pytest.mark.parametrize(
    "source", ["constructor", "binding", "snapshot", "checkpoint", "extra_body"]
)
async def test_native_anthropic_mcp_is_absent_from_provider_request(
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
                "id": "side-answer",
                "type": "message",
                "role": "assistant",
                "model": "test-model",
                "content": [{"type": "text", "text": "answer"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 1},
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
        constructor = source in {"constructor", "checkpoint"}
        model = ChatAnthropic(
            model_name="test-model",
            api_key=SecretStr("test"),
            temperature=0.2,
            max_tokens_to_sample=256,
            mcp_servers=servers if constructor else None,
            model_kwargs=(
                tool_options
                if constructor
                else {"extra_body": options}
                if source == "extra_body"
                else {}
            ),
        )
        settings = options if source in {"binding", "snapshot"} else {}
        original = deepcopy(
            model._get_request_payload([HumanMessage("main")], stop=None, **settings)
        )
        operation = BtwOperation(
            cast("BaseChatModel", model.bind(**settings))
            if source == "binding"
            else model,
            "system",
            None,
        )
        state: dict[str, object] = {}
        if source == "snapshot":
            operation._snapshots["thread"] = (model, SystemMessage("system"), settings)
        elif source == "checkpoint":
            state = {"_model_spec": "anthropic:test-model", "_model_params": options}
            monkeypatch.setattr(
                "deepagents_code.config.create_model",
                lambda *_args, **_kwargs: SimpleNamespace(model=model),
            )
        assert await operation.answer("thread", state, "why") == "answer"
        assert (
            model._get_request_payload([HumanMessage("main")], stop=None, **settings)
            == original
        )


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {},
        {"question": " ", "workspace": {}},
        {"question": "x" * 16001, "workspace": {}},
    ],
)
async def test_route_rejects_invalid_question(payload: object) -> None:
    from deepagents_code import offload_api

    with patch(
        "deepagents_code.btw_api.require_thread_workspace", new=AsyncMock()
    ) as workspace:
        async with AsyncClient(
            transport=ASGITransport(app=offload_api.app), base_url="http://test"
        ) as client:
            result = await client.post("/dcode/threads/thread/btw", json=payload)
    assert result.status_code == 422
    workspace.assert_not_awaited()


@pytest.mark.parametrize("live_model", [False, True])
async def test_route_reads_busy_thread_without_writes(
    btw_server: BtwServer,
    *,
    live_model: bool,
) -> None:
    client, operation, threads = btw_server
    if live_model:
        operation._snapshots["thread"] = (
            FakeMessagesListChatModel(responses=[AIMessage(content="live answer")]),
            SystemMessage(content="system"),
            {"temperature": 0.2},
        )
    saved = FakeMessagesListChatModel(responses=[AIMessage(content="saved answer")])
    threads.get_state.return_value = {
        "values": {
            "messages": [HumanMessage(content="Working")],
            "_model_spec": "provider:previous",
            "_model_params": {"output_config": {"effort": "high"}},
        },
        "next": ["tools"],
    }
    with patch(
        "deepagents_code.config.create_model",
        return_value=SimpleNamespace(model=saved),
    ) as create:
        result = await client.post(
            "/dcode/threads/thread/btw",
            json={"question": "why", "workspace": {"workspace_id": "1"}},
        )
    assert result.status_code == 200
    assert result.json() == {"text": "live answer" if live_model else "saved answer"}
    if live_model:
        create.assert_not_called()
    else:
        assert create.call_args.args == ("provider:previous",)
        assert create.call_args.kwargs["extra_kwargs"] == {
            "output_config": {"effort": "high"}
        }
    assert [call[0] for call in threads.mock_calls] == ["get_state"]


async def test_route_rejects_wrong_workspace_before_reading() -> None:
    from deepagents_code import offload_api
    from deepagents_code.workspace import WorkspaceConflictError

    with (
        patch(
            "deepagents_code.btw_api.require_thread_workspace",
            new=AsyncMock(side_effect=WorkspaceConflictError("wrong workspace")),
        ),
        patch.object(offload_api, "_thread_client") as client_factory,
    ):
        async with AsyncClient(
            transport=ASGITransport(app=offload_api.app), base_url="http://test"
        ) as client:
            result = await client.post(
                "/dcode/threads/thread/btw", json={"question": "why", "workspace": {}}
            )
    assert result.status_code == 409
    client_factory.assert_not_called()


async def test_follow_up_reaches_model_with_side_history_and_leaves_state_unchanged(
    btw_server: BtwServer,
    invoke: AsyncMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _operation, threads = btw_server
    state = {"messages": [HumanMessage(content="Main task")]}
    before = deepcopy(state)
    threads.get_state.return_value = {"values": state}
    remote = RemoteAgent("http://test")

    async def post(path: str, *, json: dict[str, object]) -> object:
        response = await client.post(path, json=json)
        response.raise_for_status()
        return response.json()

    graph = MagicMock()
    graph.client.http.post = post
    graph.client.http.get = AsyncMock(return_value={"cost": None})
    monkeypatch.setattr(remote, "_get_graph", lambda: graph)
    monkeypatch.setattr(remote, "_workspace_for_thread", AsyncMock(return_value={}))
    monkeypatch.setattr(remote, "aensure_thread", AsyncMock())
    assert (
        await remote.abtw(
            "Why that option?",
            config={
                "configurable": {
                    "thread_id": "thread",
                    "model": "provider:selected",
                    "model_params": {"temperature": 0.2},
                }
            },
            history=[("Which option?", "Use the cache.")],
        )
        == "The answer"
    )
    messages = invoke.call_args.args[0]
    assert [(message.type, message.text) for message in messages[1:-1]] == [
        ("human", "Main task"),
        ("human", "Which option?"),
        ("ai", "Use the cache."),
    ]
    assert messages[-1].text.endswith("Why that option?")
    assert state == before
    assert [call[0] for call in threads.mock_calls] == ["get_state"]


@pytest.mark.parametrize("main_total", [1.0, 2.0])
async def test_checkpoint_reconciles_cost_when_accounting_fails(
    main_total: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    from deepagents_code.app import DeepAgentsApp
    from deepagents_code.cost_tracking import _empty_cost_breakdown

    remote = RemoteAgent("http://test")
    graph = MagicMock()
    remote._graph = graph
    side = _empty_cost_breakdown()
    side.update(total_cost_usd=0.5, request_count=1)
    graph.client.http.get = AsyncMock(return_value={"cost": side})
    app = DeepAgentsApp(agent=MagicMock(), thread_id="reconcile")
    monkeypatch.setattr(app, "_remote_agent", lambda: remote)
    main = _empty_cost_breakdown()
    main.update(total_cost_usd=main_total, request_count=2)
    state = {"_session_cost_usd": main_total, "_session_cost_breakdown": main}
    monkeypatch.setattr(app, "_get_thread_state_values", AsyncMock(return_value=state))
    config = {"configurable": {"thread_id": app._lc_thread_id}}
    await remote.aget_session_cost(config, checkpoint={"_session_cost_usd": 1.0})
    app._set_session_cost(1.5)
    app._add_provisional_cost(1.0, request_id="missed-final-event")
    graph.client.http.get.side_effect = RuntimeError("unavailable")

    await app._sync_session_cost_from_checkpoint()

    assert app._displayed_cost_usd == pytest.approx(main_total + 0.5)
    assert app._session_cost_breakdown is not None
    assert app._session_cost_breakdown["total_cost_usd"] == pytest.approx(
        main_total + 0.5
    )
    assert app._session_cost_breakdown["request_count"] == 3
    # A later side refresh must retain the graph total recovered from state.
    graph.client.http.get.side_effect = None
    refreshed = await remote.aget_session_cost(config, checkpoint=state)
    assert refreshed is not None
    assert refreshed["total"] == pytest.approx(main_total + 0.5)


async def test_cached_cost_without_checkpoint_preserves_provisional_spend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from deepagents_code.app import DeepAgentsApp

    remote = RemoteAgent("http://test")
    graph = MagicMock()
    remote._graph = graph
    graph.client.http.get = AsyncMock(return_value={"cost": None})
    app = DeepAgentsApp(agent=MagicMock(), thread_id="reconcile")
    monkeypatch.setattr(app, "_remote_agent", lambda: remote)
    monkeypatch.setattr(app, "_get_thread_state_values", AsyncMock(return_value={}))
    await remote.aget_session_cost(
        {"configurable": {"thread_id": app._lc_thread_id}},
        checkpoint={"_session_cost_usd": 1.5},
    )
    app._set_session_cost(1.5)
    app._add_provisional_cost(1.0, request_id="unfinished")
    graph.client.http.get.side_effect = TimeoutError()

    await app._sync_session_cost_from_checkpoint()

    assert app._displayed_cost_usd == pytest.approx(2.5)
    app._add_provisional_cost(-1.0, request_id="unfinished", is_correction=True)
    assert app._displayed_cost_usd == pytest.approx(1.5)


async def test_side_cost_survives_main_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from deepagents_code.app import DeepAgentsApp
    from deepagents_code.cost_tracking import _empty_cost_breakdown

    main_total = 2.0
    provisional = 0.2
    remote = RemoteAgent("http://test")
    graph = MagicMock()
    remote._graph = graph
    app = DeepAgentsApp(agent=MagicMock())
    monkeypatch.setattr(app, "_remote_agent", lambda: remote)
    monkeypatch.setattr(app, "_post_paint_init", AsyncMock())
    monkeypatch.setattr(
        app,
        "_get_thread_state_values",
        AsyncMock(return_value={"messages": [HumanMessage(content="main")]}),
    )
    monkeypatch.setattr(app, "_ui_adapter", MagicMock())
    monkeypatch.setattr(app, "_ensure_goal_state_notice", AsyncMock(return_value=True))
    monkeypatch.setattr(remote, "_workspace_for_thread", AsyncMock(return_value={}))
    monkeypatch.setattr(remote, "aensure_thread", AsyncMock())
    started = asyncio.Event()

    async def execute(*_args: object, **_kwargs: object) -> None:
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(
        "deepagents_code.tui.textual_adapter.execute_task_textual", execute
    )
    async with app.run_test() as pilot:
        app._connecting = False
        app._agent_running = True
        app._lc_thread_id = "btw-cancellation"
        config = {"configurable": {"thread_id": app._lc_thread_id}}
        # The last observed main total may be ahead of its durable checkpoint.
        graph.client.http.get = AsyncMock(return_value={"cost": None})
        await remote.aget_session_cost(
            config, checkpoint={"_session_cost_usd": main_total}
        )
        app._set_session_cost(main_total)
        app._add_provisional_cost(provisional, request_id="unfinished")
        side = _empty_cost_breakdown()
        side.update(total_cost_usd=0.5, request_count=1)

        async def stream(*_args: object, **_kwargs: object) -> AsyncIterator[object]:
            await asyncio.sleep(0)
            yield SimpleNamespace(
                event="complete", data={"text": "Side answer", "cost": side}
            )

        graph.client.http.stream = stream
        graph.client.http.get.return_value = {"cost": side}
        main = asyncio.create_task(app._run_agent_task("main"))
        try:
            await asyncio.wait_for(started.wait(), 2)
            await app._handle_command("/btw why")
            await pilot.pause()
            assert app.screen.query_one(Markdown)._markdown == "Side answer"
            assert not main.done()
        finally:
            main.cancel()
            with pytest.raises(asyncio.CancelledError):
                await main
        assert not app._agent_running
        assert app._displayed_cost_usd == pytest.approx(main_total + 0.5 + provisional)
        # Re-reading the same side subtotal must neither double-charge it nor
        # settle the unfinished main request's provisional estimate.
        await pilot.press("escape")
        await app._handle_command("/btw again")
        await pilot.pause()
        assert app._displayed_cost_usd == pytest.approx(main_total + 0.5 + provisional)
        app._add_provisional_cost(
            -provisional, request_id="unfinished", is_correction=True
        )
        assert app._displayed_cost_usd == pytest.approx(main_total + 0.5)


async def test_app_requires_a_message_before_btw(
    btw_app: tuple[DeepAgentsApp, MagicMock],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, remote = btw_app
    monkeypatch.setattr(app, "_get_thread_state_values", AsyncMock(return_value={}))
    notify = MagicMock()
    monkeypatch.setattr(app, "notify", notify)
    async with app.run_test() as pilot:
        await pilot.pause()
        app._connecting = False
        composer = app.query_one("#chat-input", TextArea)
        composer.focus()
        await pilot.press(*"/btw Why this approach?")
        await pilot.press("enter")
        await pilot.pause()

        assert not isinstance(app.screen, BtwScreen)
        remote.abtw.assert_not_called()
        notify.assert_called_once_with("Send a message before asking /btw.")


async def test_app_keyboard_scroll_and_escape_leave_main_worker_running(
    btw_app: tuple[DeepAgentsApp, MagicMock],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, remote = btw_app
    monkeypatch.setattr(app, "_get_thread_state_values", AsyncMock(return_value={}))
    remote.abtw.return_value = "\n\n".join(f"Line {i}" for i in range(80))
    main_started = asyncio.Event()
    release = asyncio.Event()
    main_finished = asyncio.Event()

    async def main_run() -> None:
        main_started.set()
        await release.wait()
        main_finished.set()

    async with app.run_test(size=(110, 36)) as pilot:
        await pilot.pause()
        app._connecting = False
        app._agent_running = True
        # The first main turn has no checkpoint yet, but /btw is already usable.
        app._agent_turn_started = True
        app._active_user_message = UserMessage("main")
        app._model_override = "provider:selected"
        app._model_params_override = {"temperature": 0.2}
        worker = app.run_worker(main_run(), group="agent")
        await asyncio.wait_for(main_started.wait(), 2)
        before = app._message_store.get_all_messages()
        await app._submit_input("/btw why", "command")
        await pilot.pause()
        remote.abtw.assert_awaited_once_with(
            "why",
            config={"configurable": {"thread_id": app._lc_thread_id}},
            history=(),
            on_text=ANY,
        )
        scroll = app.screen.query_one("#btw-scroll", VerticalScroll)
        assert scroll.max_scroll_y > 0
        await pilot.press("tab", "home")
        assert scroll.has_focus
        assert scroll.scroll_y == 0
        await pilot.press("down", "down", "down")
        await pilot.pause()
        position = scroll.scroll_y
        assert position > 0
        await pilot.press("up")
        await pilot.pause()
        assert scroll.scroll_y < position
        await pilot.press("escape")
        await pilot.pause()
        assert not isinstance(app.screen, BtwScreen)
        assert worker.is_running
        assert not worker.is_cancelled
        assert app._agent_running
        assert app._message_store.get_all_messages() == before
        assert not app._pending_messages
        release.set()
        await asyncio.wait_for(main_finished.wait(), 2)
        app._agent_running = False


async def test_app_follow_ups_preserve_exchanges_and_recover_after_error(
    btw_app: tuple[DeepAgentsApp, MagicMock],
) -> None:
    app, remote = btw_app
    remote.abtw.side_effect = [
        "Use a cache.",
        RuntimeError("Try again [/tmp/file]"),
        "It is faster.",
        "Same conversation.",
    ]
    async with app.run_test(size=(110, 36)) as pilot:
        await pilot.pause()
        app._connecting = False
        before = app._message_store.get_all_messages()
        await app._handle_command("/btw Which option?")
        await pilot.pause()
        editor = app.screen.query_one(TextArea)
        assert editor.has_focus
        assert editor.text == ""
        await pilot.press(*"Why?", "enter")
        await pilot.pause()
        assert editor.has_focus
        assert not editor.disabled
        assert (
            app.screen.query(".btw-error").last(Static).content
            == "Try again [/tmp/file]"
        )
        await pilot.press(*"Why?", "enter")
        await pilot.pause()
        assert [widget.raw_text for widget in app.screen.query(UserMessage)] == [
            "Which option?",
            "Why?",
            "Why?",
        ]
        assert [
            widget.query_one(Markdown)._markdown
            for widget in app.screen.query(AssistantMessage)
            if widget.display
        ] == [
            "Use a cache.",
            "It is faster.",
        ]
        assert [call.kwargs["history"] for call in remote.abtw.await_args_list] == [
            (),
            (("Which option?", "Use a cache."),),
            (("Which option?", "Use a cache."),),
        ]
        await pilot.press("escape")
        await pilot.pause()
        assert app.query_one("#chat-input", TextArea).has_focus
        assert app._message_store.get_all_messages() == before
        assert not app._pending_messages
        await pilot.press(*"/btw", "enter")
        await pilot.pause()
        # Reopening restores only completed exchanges without another model call.
        assert remote.abtw.await_count == 3
        assert [widget.raw_text for widget in app.screen.query(UserMessage)] == [
            "Which option?",
            "Why?",
        ]
        assert [
            widget.query_one(Markdown)._markdown
            for widget in app.screen.query(AssistantMessage)
        ] == ["Use a cache.", "It is faster."]
        assert not app.screen.query(".btw-error")
        assert app.screen.query_one(BtwTextArea).has_focus
        await pilot.press(*"Still?", "enter")
        await pilot.pause()
        assert remote.abtw.call_args.kwargs["history"] == (
            ("Which option?", "Use a cache."),
            ("Why?", "It is faster."),
        )
        assert app._message_store.get_all_messages() == before


async def test_side_history_is_separate_per_thread_and_clear_survives_reopening(
    btw_app: tuple[DeepAgentsApp, MagicMock],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from deepagents_code.app import TextualSessionState

    app, remote = btw_app
    app._session_state = TextualSessionState(thread_id="btw-test")
    monkeypatch.setattr(app, "_thread_resume_block", AsyncMock(return_value=None))
    monkeypatch.setattr(
        app, "_offer_thread_cwd_switch", AsyncMock(return_value="continue")
    )
    monkeypatch.setattr(app, "_fetch_thread_history_data", AsyncMock())
    monkeypatch.setattr(app, "_load_thread_history", AsyncMock())
    monkeypatch.setattr(app, "_reload_hooks", AsyncMock())
    monkeypatch.setattr(app, "_run_session_start_hook", AsyncMock(return_value=False))
    monkeypatch.setattr(app, "_remount_pending_goal_rubric_review", AsyncMock())
    async with app.run_test(size=(110, 36)) as pilot:
        await pilot.pause()
        app._connecting = False
        await pilot.press(*"/btw First", "enter")
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()

        await app._resume_thread("other-thread")
        await pilot.press(*"/btw Second", "enter")
        await pilot.pause()
        assert remote.abtw.call_args.kwargs["history"] == ()
        assert remote.abtw.call_args.kwargs["config"] == {
            "configurable": {"thread_id": "other-thread"}
        }
        assert [widget.raw_text for widget in app.screen.query(UserMessage)] == [
            "Second"
        ]
        await pilot.press("escape")
        await pilot.pause()

        await app._resume_thread("btw-test")
        await pilot.press(*"/btw Third", "enter")
        await pilot.pause()
        assert remote.abtw.call_args.kwargs["history"] == (("First", "Side answer"),)
        assert [widget.raw_text for widget in app.screen.query(UserMessage)] == [
            "First",
            "Third",
        ]
        await pilot.press("ctrl+x", "escape")
        await pilot.pause()
        await pilot.press(*"/btw", "enter")
        await pilot.pause()
        assert not app.screen.query(UserMessage)
        assert not app.screen.query(AssistantMessage)
        await pilot.press(*"Fresh", "enter")
        await pilot.pause()
        assert remote.abtw.call_args.kwargs["history"] == ()
        await pilot.press("escape")
        await pilot.pause()

        await app._resume_thread("other-thread")
        await pilot.press(*"/btw Fourth", "enter")
        await pilot.pause()
        assert remote.abtw.call_args.kwargs["history"] == (("Second", "Side answer"),)


async def test_reopening_after_cancellation_keeps_only_completed_exchanges(
    btw_app: tuple[DeepAgentsApp, MagicMock],
) -> None:
    app, remote = btw_app
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def answer(
        question: str, *, on_text: Callable[[str], Awaitable[None]], **_kwargs: object
    ) -> str:
        if question != "Pending":
            return "Completed answer"
        try:
            await on_text("Partial answer")
            started.set()
            await asyncio.Event().wait()
        finally:
            cancelled.set()
        return "unreachable"

    remote.abtw.side_effect = answer
    async with app.run_test(size=(110, 36)) as pilot:
        await pilot.pause()
        app._connecting = False
        await pilot.press(*"/btw First", "enter")
        await pilot.pause()
        await pilot.press(*"Pending", "enter")
        await asyncio.wait_for(started.wait(), 2)
        message = app.screen.query(AssistantMessage).last()
        await pilot.press("escape")
        await asyncio.wait_for(cancelled.wait(), 2)
        await pilot.pause()
        assert message._stream is None
        await pilot.press(*"/btw", "enter")
        await pilot.pause()
        assert [widget.raw_text for widget in app.screen.query(UserMessage)] == [
            "First"
        ]
        assert app.screen.query_one(AssistantMessage).query_one(Markdown)._markdown == (
            "Completed answer"
        )
        await pilot.press(*"Next", "enter")
        await pilot.pause()
        assert remote.abtw.call_args.kwargs["history"] == (
            ("First", "Completed answer"),
        )


@pytest.mark.parametrize("focus_history", [False, True])
async def test_clear_discards_side_history_errors_and_draft(
    btw_app: tuple[DeepAgentsApp, MagicMock], focus_history: bool
) -> None:
    app, remote = btw_app
    remote.abtw.side_effect = [
        "Old answer",
        RuntimeError("Old error"),
        "Fresh answer",
        "Follow-up answer",
    ]
    async with app.run_test(size=(110, 36)) as pilot:
        await pilot.pause()
        app._connecting = False
        app._set_session_cost(1.5)
        before = app._message_store.get_all_messages()
        await pilot.press(*"/btw", "enter")
        await pilot.pause()
        screen = app.screen
        editor = screen.query_one(BtwTextArea)
        placeholder = editor.placeholder
        # Clearing an empty modal is harmless, including repeated resets.
        await pilot.press("ctrl+x", "ctrl+x", *"First", "enter")
        await pilot.pause()
        await pilot.press(*"Failed follow-up", "enter")
        await pilot.pause()
        assert screen.query(".btw-error").last(Static).content == "Old error"
        app.post_message(events.Paste("Draft line\n" * 100))
        await pilot.pause()
        assert editor.submitted_value == "Draft line\n" * 100
        if focus_history:
            await pilot.press("tab")
            assert screen.query_one("#btw-scroll").has_focus

        await pilot.press("ctrl+x")
        await pilot.pause()
        assert app.screen is screen
        assert not screen.query(UserMessage)
        assert not screen.query(AssistantMessage)
        assert not screen.query(".btw-error")
        assert not screen.query_one("#btw-scroll").display
        assert not screen.query_one("#btw-loading").display
        assert not screen.has_class("has-history")
        assert editor.has_focus
        assert editor.placeholder == placeholder
        assert editor.submitted_value == ""
        assert app._session_cost_usd == pytest.approx(1.5)
        # Undo must not resurrect the discarded draft or its paste placeholders.
        await pilot.press("ctrl+z", *"Fresh", "enter")
        await pilot.pause()
        assert remote.abtw.call_args.args == ("Fresh",)
        assert remote.abtw.call_args.kwargs["history"] == ()
        await pilot.press(*"Follow-up", "enter")
        await pilot.pause()
        assert remote.abtw.call_args.kwargs["history"] == (("Fresh", "Fresh answer"),)
        assert app._message_store.get_all_messages() == before
        assert not app._pending_messages


@pytest.mark.parametrize("streaming", [False, True])
async def test_clear_cancels_side_answer_and_keeps_main_worker_running(
    btw_app: tuple[DeepAgentsApp, MagicMock], streaming: bool
) -> None:
    app, remote = btw_app
    started = asyncio.Event()
    cancelled = asyncio.Event()
    release_main = asyncio.Event()

    async def answer(
        question: str, *, on_text: Callable[[str], Awaitable[None]], **_kwargs: object
    ) -> str:
        if question == "Fresh":
            return "Fresh answer"
        try:
            if streaming:
                await on_text("Partial answer")
            started.set()
            await asyncio.Event().wait()
        finally:
            cancelled.set()
        return "unreachable"

    remote.abtw.side_effect = answer
    async with app.run_test(size=(110, 36)) as pilot:
        await pilot.pause()
        app._connecting = False
        app._agent_running = True
        main = app.run_worker(release_main.wait(), group="agent")
        before = app._message_store.get_all_messages()
        try:
            await pilot.press(*"/btw why", "enter")
            await asyncio.wait_for(started.wait(), 2)
            await pilot.pause()
            screen = app.screen
            message = screen.query_one(AssistantMessage)
            assert screen.query_one(BtwTextArea).disabled
            assert screen.query_one("#btw-loading").display is not streaming

            await pilot.press("ctrl+x")
            await asyncio.wait_for(cancelled.wait(), 2)
            await pilot.pause()
            assert app.screen is screen
            assert not screen.query(AssistantMessage)
            assert message._stream is None
            assert not screen.query_one("#btw-loading").display
            assert screen.query_one(BtwTextArea).has_focus
            assert not screen.query_one(BtwTextArea).disabled
            assert main.is_running
            assert not main.is_cancelled
            assert app._agent_running

            await pilot.press(*"Fresh", "enter")
            await pilot.pause()
            assert remote.abtw.call_args.kwargs["history"] == ()
            assert screen.query_one(AssistantMessage).query_one(Markdown)._markdown == (
                "Fresh answer"
            )
            assert app._message_store.get_all_messages() == before
        finally:
            release_main.set()
            await main.wait()
            app._agent_running = False


async def test_thinking_follows_question_and_prevents_duplicate_submits() -> None:
    from textual.app import App

    app = App()
    response: asyncio.Future[str] = asyncio.get_running_loop().create_future()

    async def answer(_question: str) -> str:
        return await response

    callback = AsyncMock(side_effect=answer)
    async with app.run_test(size=(80, 24)) as pilot:
        app.push_screen(BtwScreen(callback, "First question"))
        await pilot.pause()
        for question in ("First question", "Follow-up question"):
            scroll = app.screen.query_one("#btw-scroll", VerticalScroll)
            loading = app.screen.query_one("#btw-loading", Static)
            latest = app.screen.query(".btw-question").last(UserMessage)
            assert latest.raw_text == question
            assert loading.region.y >= latest.region.bottom
            assert loading.region in scroll.content_region
            editor = app.screen.query_one(TextArea)
            assert editor.disabled
            await pilot.press("enter", "enter")
            assert callback.await_count == (1 if question == "First question" else 2)
            response.set_result(
                "A cache reuses earlier results to make repeated requests faster."
            )
            await pilot.pause()
            assert not loading.display
            assert editor.has_focus
            if question == "First question":
                response = asyncio.get_running_loop().create_future()
                await pilot.press(*"Follow-up question", "enter")
                await pilot.pause()


async def test_modal_rejects_oversized_expanded_paste() -> None:
    """The size limit uses the full pasted text and leaves it editable."""
    from textual.app import App

    from deepagents_code.tui.modals.btw import BtwTextArea

    app = App()
    answer = AsyncMock(return_value="Side answer")
    question = "x" * 16_001
    async with app.run_test() as pilot:
        app.push_screen(BtwScreen(answer))
        await pilot.pause()
        editor = app.screen.query_one(BtwTextArea)
        app.post_message(events.Paste(question))
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        answer.assert_not_awaited()
        assert editor.has_focus
        assert editor.submitted_value == question
        accepted = "line\n" * 3199 + "last!"
        editor.clear()
        app.post_message(events.Paste(accepted))
        await pilot.pause()
        answer.assert_not_awaited()
        await pilot.press("enter")
        await pilot.pause()
        answer.assert_awaited_once_with(accepted)
        assert app.screen.query_one(".btw-question", UserMessage).raw_text == accepted


async def test_multiline_editor_keeps_help_visible() -> None:
    """A long draft scrolls inside its editor without hiding modal controls."""
    from textual.app import App

    app = App()
    answer = AsyncMock()
    async with app.run_test(size=(80, 24)) as pilot:
        app.push_screen(BtwScreen(answer))
        await pilot.pause()
        editor = app.screen.query_one(TextArea)
        editor.load_text("Draft line\n" * 30)
        await pilot.pause()
        dialog = app.screen.query_one("#btw-dialog")
        hint = app.screen.query_one("#btw-help", Static)
        assert editor.region in dialog.content_region
        assert hint.region in dialog.content_region
        assert hint.region in app.screen.region
        assert "newline" in str(hint.content)
        assert editor.max_scroll_y > 0
        await pilot.press("escape")
        assert not isinstance(app.screen, BtwScreen)
        answer.assert_not_awaited()


async def test_long_question_keeps_answer_and_dismissal_hint_visible() -> None:
    from textual.app import App

    app = App()
    question = "What does this mean? " * 125
    async with app.run_test(size=(80, 24)) as pilot:
        app.push_screen(BtwScreen(AsyncMock(return_value="A short answer."), question))
        await pilot.pause()
        dialog = app.screen.query_one("#btw-dialog")
        answer = app.screen.query_one(Markdown)
        hint = app.screen.query_one("#btw-help")
        assert answer.region in dialog.content_region
        assert answer.region in app.screen.region
        assert hint.region in dialog.content_region
        scroll = app.screen.query_one("#btw-scroll", VerticalScroll)
        assert scroll.max_scroll_y > 0
        await pilot.press("tab", "home")
        await pilot.pause()
        assert scroll.scroll_y == 0
        question_widget = app.screen.query_one(".btw-question", UserMessage)
        assert question_widget.region.y >= scroll.region.y
        assert question_widget.raw_text == question
        await pilot.press("end")
        await pilot.pause()
        assert answer.region in scroll.content_region
        await pilot.press("escape")
        assert not isinstance(app.screen, BtwScreen)


@pytest.mark.parametrize("cancel", [False, True])
async def test_model_delivers_text_before_completion_and_closes_stream(
    cancel: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = asyncio.Event()
    received = asyncio.Event()
    closed = asyncio.Event()
    fragments: list[str] = []

    async def stream(
        _self: object, _messages: object, **_kwargs: object
    ) -> AsyncIterator[AIMessageChunk]:
        try:
            yield AIMessageChunk(
                content=[{"type": "reasoning", "reasoning": "private"}]
            )
            yield AIMessageChunk(content="First ")
            await release.wait()
            yield AIMessageChunk(content="answer")
        finally:
            closed.set()

    async def on_text(text: str) -> None:
        fragments.append(text)
        received.set()
        await asyncio.sleep(0)

    monkeypatch.setattr(FakeMessagesListChatModel, "astream", stream)
    operation = BtwOperation(FakeMessagesListChatModel(responses=[]), "system", None)
    task = asyncio.create_task(operation.answer("thread", {}, "why", on_text=on_text))
    try:
        await asyncio.wait_for(received.wait(), 2)
        assert fragments == ["First "]
        assert not task.done()
        if cancel:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            release.set()
            assert await task == "First answer"
            assert fragments == ["First ", "answer"]
        assert closed.is_set()
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("outcome", ["complete", "error"])
async def test_modal_renders_streamed_text(
    btw_app: tuple[DeepAgentsApp, MagicMock], outcome: str
) -> None:
    app, remote = btw_app
    release = asyncio.Event()

    async def answer(
        _question: str, *, on_text: Callable[[str], Awaitable[None]], **_kwargs: object
    ) -> str:
        await on_text("First fragment")
        await release.wait()
        if outcome == "error":
            msg = "Stream failed"
            raise RuntimeError(msg)
        await on_text(" and final fragment.")
        return "First fragment and final fragment."

    remote.abtw.side_effect = answer
    async with app.run_test(size=(110, 36)) as pilot:
        await pilot.pause()
        app._connecting = False
        before = app._message_store.get_all_messages()
        await pilot.press(*"/btw why", "enter")
        await pilot.pause()
        message = app.screen.query_one(AssistantMessage)
        assert message.display
        assert message.query_one(Markdown)._markdown == "First fragment"
        assert app.screen.query_one(TextArea).disabled
        assert not app.screen.query_one("#btw-loading").display
        question = app.screen.query_one(UserMessage)
        assert question.raw_text == "why"
        release.set()
        await pilot.pause()
        assert app.screen.query_one(TextArea).has_focus
        assert not app.screen.query_one(TextArea).disabled
        if outcome == "complete":
            assert (
                message.query_one(Markdown)._markdown
                == "First fragment and final fragment."
            )
        else:
            assert message.query_one(Markdown)._markdown == "First fragment"
            assert app.screen.query_one(".btw-error", Static).content == "Stream failed"
        assert app._message_store.get_all_messages() == before


@pytest.mark.parametrize("read_history", [False, True])
@pytest.mark.parametrize("delay_render", [False, True])
async def test_stream_follows_bottom_without_displacing_history_reader(
    monkeypatch: pytest.MonkeyPatch, read_history: bool, delay_render: bool
) -> None:
    from textual.app import App

    release = asyncio.Event()
    finish = asyncio.Event()
    render_started = asyncio.Event()
    render_release = asyncio.Event()
    first_rendered = asyncio.Event()
    final_rendered = asyncio.Event()
    text = "\n\n".join(f"Paragraph {i}" for i in range(40))
    append = Markdown.append
    if not delay_render:
        render_release.set()

    async def delayed_append(markdown: Markdown, content: str) -> None:
        render_started.set()
        await render_release.wait()
        await append(markdown, content)
        (first_rendered if content == text else final_rendered).set()

    monkeypatch.setattr(Markdown, "append", delayed_append)

    async def stream(_question: str, on_text: Callable[[str], Awaitable[None]]) -> str:
        await on_text(text)
        await release.wait()
        await on_text("\n\nFinal paragraph.")
        await finish.wait()
        return text + "\n\nFinal paragraph."

    app = App()
    async with app.run_test(size=(80, 24)) as pilot:
        app.push_screen(BtwScreen(AsyncMock(), "why", stream_answer=stream))
        await asyncio.wait_for(render_started.wait(), 2)
        await pilot.pause()
        render_release.set()
        await asyncio.wait_for(first_rendered.wait(), 2)
        await pilot.pause()
        scroll = app.screen.query_one("#btw-scroll", VerticalScroll)

        async def wait_for_bottom() -> None:
            async with asyncio.timeout(2):
                while not scroll.is_vertical_scroll_end:
                    await pilot.pause()
            assert scroll.is_vertical_scroll_end

        assert scroll.max_scroll_y > 0
        await wait_for_bottom()
        if read_history:
            await pilot.press("home")
            await pilot.pause()
            assert scroll.scroll_y == 0
        previous_height = scroll.virtual_size.height
        release.set()
        await asyncio.wait_for(final_rendered.wait(), 2)
        await pilot.pause()
        assert scroll.virtual_size.height > previous_height
        assert app.screen.query_one(TextArea).disabled
        if read_history:
            assert scroll.scroll_y == 0
        else:
            await wait_for_bottom()
        finish.set()
        await app.workers.wait_for_complete()
        await pilot.pause()
        if read_history:
            assert scroll.scroll_y == 0
        else:
            await wait_for_bottom()
