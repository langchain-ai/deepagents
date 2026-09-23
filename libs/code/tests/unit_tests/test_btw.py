"""Side answers must never run tools or mutate the main conversation."""

from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient, MockTransport, Request, Response
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.runtime import ExecutionInfo, Runtime
from textual import events
from textual.containers import VerticalScroll
from textual.widgets import Markdown, Static, TextArea

from deepagents_code.btw import BtwOperation
from deepagents_code.client.remote_client import RemoteAgent
from deepagents_code.tui.modals.btw import BtwScreen
from deepagents_code.tui.widgets.messages import UserMessage

if TYPE_CHECKING:
    from pathlib import Path

    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import BaseMessage
    from langchain_core.runnables import RunnableConfig
    from langgraph.pregel import Pregel


async def test_tool_free_snapshot_keeps_state_and_uses_compaction() -> None:
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
async def test_effective_instructions_survive_restart_and_eviction(
    *,
    synchronous: bool,
) -> None:
    from langchain.agents import create_agent
    from langchain.agents.middleware import dynamic_prompt
    from langgraph.checkpoint.memory import InMemorySaver

    @dynamic_prompt
    def extension_prompt(_request: ModelRequest) -> str:
        return "Current model identity. Extension instruction: answer in Spanish."

    model = FakeMessagesListChatModel(responses=[AIMessage(content="main")])
    warm = BtwOperation(model, "Bootstrap instructions", None)
    graph = create_agent(
        model, middleware=[extension_prompt, warm], checkpointer=InMemorySaver()
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

    def answer(messages: list[BaseMessage], **_kwargs: object) -> AIMessage:
        prompts.append(messages[0].text)
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
        ) as create:
            assert await operation.answer("thread", state, "why") == "answer"
        if source == "checkpoint":
            create.assert_called_once_with(
                "openai:test-model",
                extra_kwargs=params,
                bind_preserved_thinking=False,
            )
        assert params == original
        defaults = model._get_request_payload([HumanMessage(content="main")])
        assert (defaults.get("extra_body") or defaults)["tools"] == options["tools"]


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
async def test_route_reads_busy_thread_without_writes(*, live_model: bool) -> None:
    from deepagents_code import offload_api

    operation = BtwOperation(
        FakeMessagesListChatModel(responses=[AIMessage(content="side answer")]),
        "system",
        None,
    )
    if live_model:
        operation._snapshots["thread"] = (
            FakeMessagesListChatModel(responses=[AIMessage(content="live answer")]),
            SystemMessage(content="system"),
            {"temperature": 0.2},
        )
    saved = FakeMessagesListChatModel(responses=[AIMessage(content="saved answer")])
    threads = MagicMock()
    threads.get_state = AsyncMock(
        return_value={
            "values": {
                "messages": [HumanMessage(content="Working")],
                "_model_spec": "provider:previous",
                "_model_params": {"output_config": {"effort": "high"}},
            },
            "next": ["tools"],
        }
    )
    with (
        patch(
            "deepagents_code.config.create_model",
            return_value=SimpleNamespace(model=saved),
        ) as create,
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
            offload_api,
            "_thread_client",
            return_value=SimpleNamespace(threads=threads),
        ),
    ):
        async with AsyncClient(
            transport=ASGITransport(app=offload_api.app), base_url="http://test"
        ) as client:
            result = await client.post(
                "/dcode/threads/thread/btw",
                json={
                    "question": "why",
                    "workspace": {"workspace_id": "1"},
                },
            )
    assert result.status_code == 200
    assert result.json() == {"text": "live answer" if live_model else "saved answer"}
    if live_model:
        create.assert_not_called()
    else:
        create.assert_called_once_with(
            "provider:previous",
            extra_kwargs={"output_config": {"effort": "high"}},
            bind_preserved_thinking=False,
        )
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
    config = {
        "configurable": {
            "thread_id": "thread",
            "model": "provider:selected",
            "model_params": {"temperature": 0.2},
        }
    }
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


async def test_follow_up_reaches_model_with_side_history_and_leaves_state_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from deepagents_code import offload_api

    state = {"messages": [HumanMessage(content="Main task")]}
    before = deepcopy(state)
    invoke = AsyncMock(return_value=AIMessage(content="Because it is faster."))
    monkeypatch.setattr(FakeMessagesListChatModel, "ainvoke", invoke)
    operation = BtwOperation(FakeMessagesListChatModel(responses=[]), "system", None)
    monkeypatch.setattr("deepagents_code.btw_api.require_thread_workspace", AsyncMock())
    monkeypatch.setattr(
        offload_api,
        "get_server_runtime",
        AsyncMock(
            return_value=SimpleNamespace(backend=SimpleNamespace(_dcode_btw=operation))
        ),
    )
    threads = MagicMock()
    threads.get_state = AsyncMock(return_value={"values": state})
    monkeypatch.setattr(
        offload_api, "_thread_client", lambda: SimpleNamespace(threads=threads)
    )
    async with AsyncClient(
        transport=ASGITransport(app=offload_api.app), base_url="http://test"
    ) as client:
        remote = RemoteAgent("http://test")

        async def post(path: str, *, json: dict[str, object]) -> object:
            response = await client.post(path, json=json)
            response.raise_for_status()
            return response.json()

        graph = SimpleNamespace(client=SimpleNamespace(http=SimpleNamespace(post=post)))
        monkeypatch.setattr(remote, "_get_graph", lambda: graph)
        monkeypatch.setattr(remote, "_workspace_for_thread", AsyncMock(return_value={}))
        monkeypatch.setattr(remote, "aensure_thread", AsyncMock())
        assert (
            await remote.abtw(
                "Why that option?",
                config={"configurable": {"thread_id": "thread"}},
                history=[("Which option?", "Use the cache.")],
            )
            == "Because it is faster."
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
@pytest.mark.parametrize("refresh_error", [RuntimeError("unavailable"), TimeoutError()])
async def test_checkpoint_reconciles_cost_when_accounting_fails(
    main_total: float, refresh_error: Exception, monkeypatch: pytest.MonkeyPatch
) -> None:
    from deepagents_code.app import DeepAgentsApp
    from deepagents_code.btw_cost import combine_session_cost
    from deepagents_code.cost_tracking import _empty_cost_breakdown

    remote = RemoteAgent("http://test")
    graph = MagicMock()
    remote._graph = graph
    side = _empty_cost_breakdown()
    side.update(total_cost_usd=0.5, request_count=1)
    graph.client.http.get = AsyncMock(
        return_value={"cost": combine_session_cost(1.0, None, side)}
    )
    app = DeepAgentsApp(agent=MagicMock(), thread_id="reconcile")
    monkeypatch.setattr(app, "_remote_agent", lambda: remote)
    main = _empty_cost_breakdown()
    main.update(total_cost_usd=main_total, request_count=2)
    state = {"_session_cost_usd": main_total, "_session_cost_breakdown": main}
    monkeypatch.setattr(app, "_get_thread_state_values", AsyncMock(return_value=state))
    config = {"configurable": {"thread_id": app._lc_thread_id}}
    await remote.aget_session_cost(config)
    app._set_session_cost(1.5)
    app._add_provisional_cost(1.0, request_id="missed-final-event")
    graph.client.http.get.side_effect = refresh_error

    await app._sync_session_cost_from_checkpoint()

    assert app._displayed_cost_usd == pytest.approx(main_total + 0.5)
    assert app._session_cost_breakdown is not None
    assert app._session_cost_breakdown["total_cost_usd"] == pytest.approx(
        main_total + 0.5
    )
    assert app._session_cost_breakdown["request_count"] == 3
    # A later side refresh must retain the graph total recovered from state.
    graph.client.http.get.side_effect = None
    refreshed = await remote.arefresh_side_cost(config)
    assert refreshed is not None
    assert refreshed["total"] == pytest.approx(main_total + 0.5)


async def test_cached_cost_without_checkpoint_preserves_provisional_spend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from deepagents_code.app import DeepAgentsApp

    remote = RemoteAgent("http://test")
    graph = MagicMock()
    remote._graph = graph
    graph.client.http.get = AsyncMock(
        return_value={"cost": {"total": 1.5, "breakdown": None}}
    )
    app = DeepAgentsApp(agent=MagicMock(), thread_id="reconcile")
    monkeypatch.setattr(app, "_remote_agent", lambda: remote)
    monkeypatch.setattr(app, "_get_thread_state_values", AsyncMock(return_value={}))
    await remote.aget_session_cost({"configurable": {"thread_id": app._lc_thread_id}})
    app._set_session_cost(1.5)
    app._add_provisional_cost(1.0, request_id="unfinished")
    graph.client.http.get.side_effect = TimeoutError()

    await app._sync_session_cost_from_checkpoint()

    assert app._displayed_cost_usd == pytest.approx(2.5)
    app._add_provisional_cost(-1.0, request_id="unfinished", is_correction=True)
    assert app._displayed_cost_usd == pytest.approx(1.5)


@pytest.mark.parametrize("main_total", [1.0, 2.0])
@pytest.mark.parametrize("provisional", [0.0, 0.2])
async def test_side_cost_survives_main_cancellation(
    main_total: float, provisional: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    from deepagents_code.app import DeepAgentsApp
    from deepagents_code.btw_cost import combine_session_cost
    from deepagents_code.cost_tracking import _empty_cost_breakdown

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
    monkeypatch.setattr(remote, "abtw", AsyncMock(return_value="Side answer"))
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
        graph.client.http.get = AsyncMock(
            return_value={"cost": combine_session_cost(main_total, None, None)}
        )
        await remote.aget_session_cost(config)
        app._set_session_cost(main_total)
        app._add_provisional_cost(provisional, request_id="unfinished")
        side = _empty_cost_breakdown()
        side.update(total_cost_usd=0.5, request_count=1)
        graph.client.http.get.return_value = {
            "cost": combine_session_cost(1.0, None, side)
        }
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


@pytest.mark.parametrize("question", ["", "Why this approach?"])
async def test_app_requires_a_message_before_btw(
    question: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from deepagents_code.app import DeepAgentsApp

    app = DeepAgentsApp(agent=MagicMock(), thread_id="btw-empty")
    monkeypatch.setattr(app, "_post_paint_init", AsyncMock())
    monkeypatch.setattr(app, "_get_thread_state_values", AsyncMock(return_value={}))
    remote = MagicMock(spec=RemoteAgent)
    monkeypatch.setattr(app, "_remote_agent", lambda: remote)
    notify = MagicMock()
    monkeypatch.setattr(app, "notify", notify)
    async with app.run_test() as pilot:
        await pilot.pause()
        app._connecting = False
        composer = app.query_one("#chat-input", TextArea)
        composer.focus()
        await pilot.press(*f"/btw {question}")
        await pilot.press("enter")
        await pilot.pause()

        assert not isinstance(app.screen, BtwScreen)
        remote.abtw.assert_not_called()
        notify.assert_called_once_with("Send a message before asking /btw.")


@pytest.mark.parametrize("question", ["", "Why this approach?"])
async def test_app_modal_while_main_run_continues(
    question: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from deepagents_code.app import DeepAgentsApp

    app = DeepAgentsApp(agent=MagicMock(), thread_id="btw-inflight")
    monkeypatch.setattr(app, "_post_paint_init", AsyncMock())
    monkeypatch.setattr(app, "_get_thread_state_values", AsyncMock(return_value={}))
    remote = MagicMock(spec=RemoteAgent)
    remote.arefresh_side_cost = AsyncMock(return_value=None)
    started = asyncio.Event()
    release = asyncio.Event()

    async def answer(
        question: str,
        *,
        config: dict[str, dict[str, object]],
        history: tuple[tuple[str, str], ...] = (),
    ) -> str:
        assert question == "Why this approach?"
        assert not history
        assert isinstance(config, dict)
        assert config == {"configurable": {"thread_id": app._lc_thread_id}}
        started.set()
        await release.wait()
        return "**Side answer**"

    remote.abtw = AsyncMock(side_effect=answer)
    monkeypatch.setattr(app, "_remote_agent", lambda: remote)
    async with app.run_test(size=(110, 36)) as pilot:
        await pilot.pause()
        app._agent_running = True
        app._agent_turn_started = True
        app._active_user_message = UserMessage("main")
        app._connecting = False
        before = app._message_store.get_all_messages()
        app._model_override = "provider:selected"
        app._model_params_override = {"temperature": 0.2}
        assert app._can_bypass_queue("/btw why")
        await app._submit_input(f"/btw {question}".strip(), "command")
        await pilot.pause()
        assert isinstance(app.screen, BtwScreen)
        if not question:
            app.screen.query_one(TextArea).load_text("Why this approach?")
            await pilot.press("enter")
        await asyncio.wait_for(started.wait(), 2)
        loading = app.screen.query_one("#btw-loading", Static)
        assert loading.display
        assert "Thinking..." in str(loading.content)
        assert app._agent_running
        release.set()
        await pilot.pause()
        assert not loading.display
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

    app = DeepAgentsApp(agent=MagicMock(), thread_id="btw-scroll")
    monkeypatch.setattr(app, "_post_paint_init", AsyncMock())
    monkeypatch.setattr(
        app,
        "_get_thread_state_values",
        AsyncMock(return_value={"messages": [HumanMessage(content="main")]}),
    )
    remote = MagicMock(spec=RemoteAgent)
    remote.arefresh_side_cost = AsyncMock(return_value=None)
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


async def test_app_follow_ups_preserve_exchanges_and_recover_after_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from deepagents_code.app import DeepAgentsApp

    app = DeepAgentsApp(agent=MagicMock(), thread_id="btw-follow-ups")
    monkeypatch.setattr(app, "_post_paint_init", AsyncMock())
    monkeypatch.setattr(
        app,
        "_get_thread_state_values",
        AsyncMock(return_value={"messages": [HumanMessage(content="Main task")]}),
    )
    remote = MagicMock(spec=RemoteAgent)
    remote.arefresh_side_cost = AsyncMock(return_value=None)
    remote.abtw = AsyncMock(
        side_effect=[
            "Use a cache.",
            RuntimeError("Try again"),
            "It is faster.",
            "New conversation.",
        ]
    )
    monkeypatch.setattr(app, "_remote_agent", lambda: remote)
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
        assert app.screen.query(".btw-error").last(Static).content == "Try again"
        await pilot.press(*"Why?", "enter")
        await pilot.pause()
        assert [
            widget.content
            for widget in app.screen.query(Static)
            if widget.has_class("btw-question")
        ] == [
            "Which option?",
            "Why?",
            "Why?",
        ]
        assert [
            widget._markdown for widget in app.screen.query(Markdown) if widget.display
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
        await app._handle_command("/btw Start over")
        await pilot.pause()
        assert remote.abtw.call_args.kwargs["history"] == ()


@pytest.mark.parametrize("size", [(110, 36), (80, 24)])
async def test_thinking_follows_current_question_and_prevents_duplicate_submits(
    size: tuple[int, int],
) -> None:
    from textual.app import App

    app = App()
    response: asyncio.Future[str] = asyncio.get_running_loop().create_future()

    async def answer(_question: str) -> str:
        return await response

    callback = AsyncMock(side_effect=answer)
    async with app.run_test(size=size) as pilot:
        app.push_screen(BtwScreen(callback, "First question"))
        await pilot.pause()
        for question in ("First question", "Follow-up question"):
            scroll = app.screen.query_one("#btw-scroll", VerticalScroll)
            loading = app.screen.query_one("#btw-loading", Static)
            latest = app.screen.query(".btw-question").last(Static)
            assert latest.content == question
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


async def test_modal_error_is_plain_text() -> None:
    from textual.app import App

    app = App()
    answer = AsyncMock(side_effect=RuntimeError("bad [/tmp/file]"))
    async with app.run_test() as pilot:
        app.push_screen(BtwScreen(answer, "why"))
        await pilot.pause()
        assert app.screen.query_one(".btw-error", Static).content == "bad [/tmp/file]"
        await pilot.press("escape")


@pytest.mark.parametrize("question", ["First line\nSecond line", "Pasted line\n" * 100])
async def test_modal_submits_complete_paste(question: str) -> None:
    """Pasted newlines stay in the editor and collapsed text expands on submit."""
    from textual.app import App

    app = App()
    answer = AsyncMock(return_value="Side answer")
    async with app.run_test() as pilot:
        app.push_screen(BtwScreen(answer))
        await pilot.pause()
        app.post_message(events.Paste(question))
        await pilot.pause()
        answer.assert_not_awaited()
        await pilot.press("enter")
        await pilot.pause()
        answer.assert_awaited_once_with(question.strip())
        assert app.screen.query_one(".btw-question", Static).content == question.strip()


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
        editor.load_text("x" * 16_000)
        await pilot.press("enter")
        await pilot.pause()
        answer.assert_awaited_once_with("x" * 16_000)


@pytest.mark.parametrize("size", [(110, 36), (80, 24)])
async def test_multiline_editor_keeps_help_visible(size: tuple[int, int]) -> None:
    """A long draft scrolls inside its editor without hiding modal controls."""
    from textual.app import App

    app = App()
    answer = AsyncMock()
    async with app.run_test(size=size) as pilot:
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


@pytest.mark.parametrize("size", [(110, 36), (80, 24)])
async def test_long_question_keeps_answer_and_dismissal_hint_visible(
    size: tuple[int, int],
) -> None:
    from textual.app import App

    app = App()
    question = "What does this mean? " * 125
    async with app.run_test(size=size) as pilot:
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
        question_widget = app.screen.query_one(".btw-question", Static)
        assert question_widget.region.y >= scroll.region.y
        assert question_widget.content == question
        await pilot.press("end")
        await pilot.pause()
        assert answer.region in scroll.content_region
        await pilot.press("escape")
        assert not isinstance(app.screen, BtwScreen)
