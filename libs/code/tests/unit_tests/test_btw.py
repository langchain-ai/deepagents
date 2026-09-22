"""Side answers must never run tools or mutate the main conversation."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.runtime import ExecutionInfo, Runtime
from textual.containers import VerticalScroll
from textual.widgets import Input, Markdown, Static

from deepagents_code.btw import BtwOperation
from deepagents_code.client.remote_client import RemoteAgent
from deepagents_code.tui.modals.btw import BtwScreen

if TYPE_CHECKING:
    from pathlib import Path

    from langchain_core.language_models import BaseChatModel
    from langchain_core.runnables import RunnableConfig


async def test_tool_free_snapshot_keeps_state_and_uses_compaction() -> None:
    model = FakeMessagesListChatModel(responses=[AIMessage(content="The answer")])
    operation = BtwOperation(model, "Main instructions", None)
    state = {
        "messages": [
            HumanMessage(content="Archived secret"),
            HumanMessage(content="Current request"),
            AIMessage(
                content="Looking",
                tool_calls=[{"name": "execute", "args": {}, "id": "1"}],
            ),
            ToolMessage(content="Tool context", tool_call_id="1"),
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
    with patch.object(
        FakeMessagesListChatModel,
        "ainvoke",
        new=AsyncMock(return_value=AIMessage(content="The answer")),
    ) as invoke:
        assert await operation.answer("thread", state, "Why?") == "The answer"
    messages = invoke.call_args.args[0]
    assert [m.text for m in messages[1:-1]] == [
        "Earlier summary",
        "Current request",
        "Looking",
        "[tool context]\nTool context",
    ]
    assert all(not isinstance(m, ToolMessage) for m in messages)
    assert all(not m.tool_calls for m in messages if isinstance(m, AIMessage))
    assert "Do not call any tools" in messages[0].text
    assert invoke.call_args.kwargs == {"config": {"callbacks": []}}
    assert state == before


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


@pytest.mark.parametrize("synchronous", [False, True])
async def test_snapshot_preserves_settings_without_tools_or_shared_mutation(
    *, synchronous: bool
) -> None:
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
                thread_id="thread", checkpoint_id="c", checkpoint_ns="", task_id="t"
            )
        ),
    )
    response = ModelResponse(result=[AIMessage(content="main answer")])
    if synchronous:
        assert (
            operation.wrap_model_call(request, MagicMock(return_value=response))
            is response
        )
    else:
        assert (
            await operation.awrap_model_call(request, AsyncMock(return_value=response))
            is response
        )
    request.model_settings["reasoning"]["effort"] = "low"

    def invoke(_messages: object, **kwargs: object) -> AIMessage:
        assert kwargs == {
            "config": {"callbacks": []},
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
    assert other.call_args.kwargs == {"config": {"callbacks": []}}
    assert request.model_settings["tool_choice"] == "required"


@pytest.mark.parametrize("synchronous", [False, True])
async def test_real_agent_wiring_preserves_checkpoint(
    tmp_path: Path, *, synchronous: bool
) -> None:
    from langgraph.checkpoint.memory import InMemorySaver

    from deepagents_code._testing_models import DeterministicIntegrationChatModel
    from deepagents_code.agent import create_cli_agent
    from deepagents_code.btw import BTW_OPERATION_ATTR

    agent, backend = create_cli_agent(
        model=DeterministicIntegrationChatModel(),
        assistant_id="test-btw",
        enable_memory=False,
        enable_skills=False,
        enable_shell=False,
        system_prompt="Answer the main task.",
        cwd=tmp_path,
        checkpointer=InMemorySaver(),
    )
    config: RunnableConfig = {"configurable": {"thread_id": "btw-wiring"}}
    inputs = {"messages": [HumanMessage(content="Hello")]}
    if synchronous:
        agent.invoke(inputs, config=config)
    else:
        await agent.ainvoke(inputs, config=config)
    before = await agent.aget_state(config)
    operation = getattr(backend, BTW_OPERATION_ATTR)
    assert isinstance(operation, BtwOperation)
    assert "btw-wiring" in operation._snapshots
    assert "side question" in await operation.answer(
        "btw-wiring", before.values, "side question"
    )
    after = await agent.aget_state(config)
    assert before == after


async def test_resumed_model_comes_only_from_checkpoint() -> None:
    model = FakeMessagesListChatModel(responses=[AIMessage(content="resumed")])
    operation = BtwOperation(model, "system", None)
    with patch(
        "deepagents_code.config.create_model", return_value=SimpleNamespace(model=model)
    ) as create:
        assert (
            await operation.answer(
                "thread",
                {
                    "_model_spec": "provider:resumed",
                    "_model_params": {"temperature": 0},
                },
                "why",
            )
            == "resumed"
        )
    create.assert_called_once_with(
        "provider:resumed",
        extra_kwargs={"temperature": 0},
        bind_preserved_thinking=False,
    )


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {},
        {"question": " ", "workspace": {}},
        {"question": "x" * 16001, "workspace": {}},
        {
            "question": "why",
            "workspace": {},
            "model_params": {"base_url": "http://attacker"},
        },
    ],
)
async def test_route_rejects_untrusted_model_and_invalid_question(
    payload: object,
) -> None:
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


async def test_route_reads_busy_thread_without_writes() -> None:
    from deepagents_code import offload_api

    operation = BtwOperation(
        FakeMessagesListChatModel(responses=[AIMessage(content="side answer")]),
        "system",
        None,
    )
    threads = MagicMock()
    threads.get_state = AsyncMock(
        return_value={
            "values": {"messages": [HumanMessage(content="Working")]},
            "next": ["tools"],
        }
    )
    with (
        patch("deepagents_code.btw_api.require_thread_workspace", new=AsyncMock()),
        patch.object(
            offload_api,
            "get_server_runtime",
            new=AsyncMock(
                return_value=SimpleNamespace(
                    backend=SimpleNamespace(_dcode_btw=operation)
                )
            ),
        ),
        patch.object(
            offload_api, "_thread_client", return_value=SimpleNamespace(threads=threads)
        ),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=offload_api.app), base_url="http://test"
        ) as client:
            result = await client.post(
                "/dcode/threads/thread/btw",
                json={"question": "why", "workspace": {"workspace_id": "1"}},
            )
    assert result.status_code == 200
    assert result.json() == {"text": "side answer"}
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


async def test_remote_uses_side_route_not_runs() -> None:
    agent = RemoteAgent("http://test")
    graph = MagicMock()
    graph.client.http.post = AsyncMock(return_value={"text": "answer"})
    config = {"configurable": {"thread_id": "thread"}}
    with (
        patch.object(agent, "_get_graph", return_value=graph),
        patch.object(
            agent,
            "_workspace_for_thread",
            new=AsyncMock(return_value={"workspace_id": "1"}),
        ),
        patch.object(agent, "aensure_thread", new=AsyncMock()),
    ):
        assert await agent.abtw("why", config=config) == "answer"
    graph.client.http.post.assert_awaited_once_with(
        "/dcode/threads/thread/btw",
        json={"question": "why", "workspace": {"workspace_id": "1"}},
    )
    graph.client.runs.assert_not_called()


@pytest.mark.parametrize("question", ["", "Why this approach?"])
async def test_app_modal_while_main_run_continues(
    question: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from deepagents_code.app import DeepAgentsApp

    app = DeepAgentsApp(agent=MagicMock())
    monkeypatch.setattr(app, "_post_paint_init", AsyncMock())
    remote = MagicMock(spec=RemoteAgent)
    started = asyncio.Event()
    release = asyncio.Event()

    async def answer(question: str, *, config: object) -> str:
        assert question == "Why this approach?"
        assert isinstance(config, dict)
        started.set()
        await release.wait()
        return "**Side answer**"

    remote.abtw = AsyncMock(side_effect=answer)
    monkeypatch.setattr(app, "_remote_agent", lambda: remote)
    async with app.run_test(size=(110, 36)) as pilot:
        await pilot.pause()
        app._agent_running = True
        app._connecting = False
        before = app._message_store.get_all_messages()
        assert app._can_bypass_queue("/btw why")
        await app._submit_input(f"/btw {question}".strip(), "command")
        await pilot.pause()
        assert isinstance(app.screen, BtwScreen)
        if not question:
            app.screen.query_one(Input).value = "Why this approach?"
            await pilot.press("enter")
        await asyncio.wait_for(started.wait(), 2)
        assert app._agent_running
        release.set()
        await pilot.pause()
        assert app.screen.query_one(Markdown)._markdown == "**Side answer**"
        await pilot.press("escape")
        await pilot.pause()
        assert not isinstance(app.screen, BtwScreen)
        assert app._agent_running
        assert app._message_store.get_all_messages() == before
        assert not app._pending_messages
        app._agent_running = False


async def test_app_keyboard_scroll_and_escape_leave_main_worker_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from deepagents_code.app import DeepAgentsApp

    app = DeepAgentsApp(agent=MagicMock())
    monkeypatch.setattr(app, "_post_paint_init", AsyncMock())
    remote = MagicMock(spec=RemoteAgent)
    remote.abtw = AsyncMock(return_value="\n\n".join(f"Line {i}" for i in range(80)))
    monkeypatch.setattr(app, "_remote_agent", lambda: remote)
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
        worker = app.run_worker(main_run(), group="agent")
        await asyncio.wait_for(main_started.wait(), 2)
        before = app._message_store.get_all_messages()
        await app._submit_input("/btw why", "command")
        await pilot.pause()
        scroll = app.screen.query_one("#btw-scroll", VerticalScroll)
        assert scroll.max_scroll_y > 0
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
        release.set()
        await asyncio.wait_for(main_finished.wait(), 2)
        app._agent_running = False


async def test_modal_dismiss_cancels_only_side_question() -> None:
    from textual.app import App

    app = App()
    cancelled = asyncio.Event()

    async def answer(_question: str) -> str:
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
        return "unreachable"

    async with app.run_test() as pilot:
        app.push_screen(BtwScreen(answer, "why"))
        await pilot.pause()
        await pilot.press("escape")
        await asyncio.wait_for(cancelled.wait(), 2)
        await pilot.pause()
        assert not isinstance(app.screen, BtwScreen)


async def test_modal_error_is_plain_text() -> None:
    from textual.app import App

    app = App()
    answer = AsyncMock(side_effect=RuntimeError("bad [/tmp/file]"))
    async with app.run_test() as pilot:
        app.push_screen(BtwScreen(answer, "why"))
        await pilot.pause()
        assert app.screen.query_one("#btw-error", Static).content == "bad [/tmp/file]"
        await pilot.press("escape")
