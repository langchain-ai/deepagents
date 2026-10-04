"""Tests for async subagent middleware functionality."""

import base64
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, TypeVar
from unittest.mock import MagicMock, patch

import httpx
import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.tools import ToolRuntime
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from langgraph_sdk.client import LangGraphClient, SyncLangGraphClient

from deepagents.backends import CompositeBackend, FilesystemBackend
from deepagents.middleware.async_subagents import (
    AsyncSubAgent,
    AsyncSubAgentMiddleware,
    AsyncSubAgentState,
    AsyncTask,
    _build_async_subagent_tools,
    _resolve_headers,
    _tasks_reducer,
)
from deepagents.middleware.filesystem import FilesystemMiddleware
from tests.unit_tests.chat_model import GenericFakeChatModel


def _make_spec(name: str = "test-agent", **overrides: Any) -> AsyncSubAgent:
    base: dict[str, Any] = {
        "name": name,
        "description": f"A test agent named {name}",
        "url": "http://localhost:8123",
        "graph_id": "my_graph",
    }
    base.update(overrides)
    return AsyncSubAgent(**base)  # type: ignore[typeddict-item]


def _make_runtime(tool_call_id: str = "tc_test") -> ToolRuntime:
    return ToolRuntime(
        state={},
        context=None,
        tool_call_id=tool_call_id,
        store=None,
        stream_writer=lambda _: None,
        config={},
    )


def _make_runtime_with_task(
    task_id: str = "thread_abc",
    agent_name: str = "test-agent",
    run_id: str = "run_xyz",
    status: str = "running",
    tool_call_id: str = "tc_test",
    created_at: str = "2024-01-15T10:30:00Z",
    last_checked_at: str = "2024-01-15T10:30:00Z",
    last_updated_at: str = "2024-01-15T10:30:00Z",
) -> ToolRuntime:
    """Create a runtime with a single tracked task in state."""
    tasks: dict[str, AsyncTask] = {
        task_id: {
            "task_id": task_id,
            "agent_name": agent_name,
            "thread_id": task_id,
            "run_id": run_id,
            "status": status,
            "created_at": created_at,
            "last_checked_at": last_checked_at,
            "last_updated_at": last_updated_at,
        },
    }
    return ToolRuntime(
        state={"async_tasks": tasks},
        context=None,
        tool_call_id=tool_call_id,
        store=None,
        stream_writer=lambda _: None,
        config={},
    )


def _get_tool(tools: list, name: str) -> Any:  # noqa: ANN401
    """Look up a tool by name from the built tools list."""
    for t in tools:
        if t.name == name:
            return t
    msg = f"Tool {name!r} not found"
    raise KeyError(msg)


class TestAsyncSubAgentMiddleware:
    def test_init_requires_at_least_one_agent(self) -> None:
        with pytest.raises(ValueError, match="At least one async subagent"):
            AsyncSubAgentMiddleware(async_subagents=[])

    def test_init_creates_five_tools(self) -> None:
        mw = AsyncSubAgentMiddleware(async_subagents=[_make_spec()])
        tool_names = {t.name for t in mw.tools}
        assert tool_names == {
            "start_async_task",
            "check_async_task",
            "update_async_task",
            "cancel_async_task",
            "list_async_tasks",
        }

    def test_system_prompt_includes_agent_descriptions(self) -> None:
        # The default is lean (no system prompt); when a prompt is supplied, the
        # available agent descriptions are appended to it.
        mw = AsyncSubAgentMiddleware(
            async_subagents=[
                _make_spec("alpha", description="Alpha agent"),
                _make_spec("beta", description="Beta agent"),
            ],
            system_prompt="Async subagent guidance.",
        )
        assert "alpha" in mw.system_prompt
        assert "beta" in mw.system_prompt
        assert "Alpha agent" in mw.system_prompt
        assert "Beta agent" in mw.system_prompt

    def test_system_prompt_can_be_disabled(self) -> None:
        mw = AsyncSubAgentMiddleware(async_subagents=[_make_spec()], system_prompt=None)
        assert mw.system_prompt is None

    def test_init_rejects_duplicate_names(self) -> None:
        with pytest.raises(ValueError, match="Duplicate async subagent names"):
            AsyncSubAgentMiddleware(async_subagents=[_make_spec("alpha"), _make_spec("alpha")])

    def test_state_schema_is_set(self) -> None:
        assert AsyncSubAgentMiddleware.state_schema is AsyncSubAgentState


class TestAsyncForkConfiguration:
    @pytest.mark.parametrize("mode", ["invalid", "handoff", None])
    def test_rejects_invalid_mode(self, mode: str | None) -> None:
        with pytest.raises(ValueError, match="expected 'isolated' or 'fork'"):
            AsyncSubAgentMiddleware(async_subagents=[_make_spec(mode=mode)])


class TestResolveHeaders:
    def test_adds_auth_scheme_by_default(self) -> None:
        spec = _make_spec()
        headers = _resolve_headers(spec)
        assert headers["x-auth-scheme"] == "langsmith"

    def test_preserves_custom_headers(self) -> None:
        spec = _make_spec(headers={"X-Custom": "value"})
        headers = _resolve_headers(spec)
        assert headers["x-auth-scheme"] == "langsmith"
        assert headers["X-Custom"] == "value"

    def test_does_not_override_explicit_auth_scheme(self) -> None:
        spec = _make_spec(headers={"x-auth-scheme": "custom"})
        headers = _resolve_headers(spec)
        assert headers["x-auth-scheme"] == "custom"


class TestTasksReducer:
    def test_merge_into_empty(self) -> None:
        task: AsyncTask = {
            "task_id": "t",
            "agent_name": "a",
            "thread_id": "t",
            "run_id": "r",
            "status": "running",
        }
        result = _tasks_reducer(None, {"t": task})
        assert result == {"t": task}

    def test_merge_updates_existing(self) -> None:
        old: AsyncTask = {
            "task_id": "t",
            "agent_name": "a",
            "thread_id": "t",
            "run_id": "r",
            "status": "running",
        }
        updated: AsyncTask = {**old, "status": "success"}
        result = _tasks_reducer({"t": old}, {"t": updated})
        assert result["t"]["status"] == "success"

    def test_merge_preserves_other_keys(self) -> None:
        task1: AsyncTask = {
            "task_id": "t1",
            "agent_name": "a",
            "thread_id": "t1",
            "run_id": "r1",
            "status": "running",
        }
        task2: AsyncTask = {
            "task_id": "t2",
            "agent_name": "a",
            "thread_id": "t2",
            "run_id": "r2",
            "status": "running",
        }
        result = _tasks_reducer({"t1": task1}, {"t2": task2})
        assert len(result) == 2
        assert "t1" in result
        assert "t2" in result


class TestBuildAsyncSubagentTools:
    def test_returns_five_tools(self) -> None:
        tools = _build_async_subagent_tools([_make_spec()])
        assert len(tools) == 5
        names = [t.name for t in tools]
        assert names == [
            "start_async_task",
            "check_async_task",
            "update_async_task",
            "cancel_async_task",
            "list_async_tasks",
        ]

    def test_launch_description_includes_agent_info(self) -> None:
        tools = _build_async_subagent_tools([_make_spec("researcher", description="Research agent")])
        launch_tool = tools[0]
        assert "researcher" in launch_tool.description
        assert "Research agent" in launch_tool.description


class TestLaunchTool:
    def test_launch_invalid_type_returns_error_string(self) -> None:
        tools = _build_async_subagent_tools([_make_spec("alpha")])
        launch = tools[0]
        result = launch.func(
            description="do something",
            subagent_type="nonexistent",
            runtime=_make_runtime(),
        )
        assert isinstance(result, str)
        assert "Unknown async subagent type" in result
        assert "`alpha`" in result

    @patch("deepagents.middleware.async_subagents.get_sync_client")
    def test_launch_returns_command_with_task(self, mock_get_client: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client.threads.create.return_value = {"thread_id": "thread_abc"}
        mock_client.runs.create.return_value = {"run_id": "run_xyz"}
        mock_get_client.return_value = mock_client

        tools = _build_async_subagent_tools([_make_spec("alpha")])
        launch = tools[0]
        result = launch.func(
            description="analyze data",
            subagent_type="alpha",
            runtime=_make_runtime("tc_launch"),
        )

        assert isinstance(result, Command)
        update = result.update
        assert "async_tasks" in update
        tasks = update["async_tasks"]
        assert "thread_abc" in tasks
        task = tasks["thread_abc"]
        assert task["task_id"] == "thread_abc"
        assert task["agent_name"] == "alpha"
        assert task["thread_id"] == "thread_abc"
        assert task["run_id"] == "run_xyz"
        assert task["status"] == "running"

        msgs = update["messages"]
        assert len(msgs) == 1
        assert msgs[0].tool_call_id == "tc_launch"
        assert "thread_abc" in msgs[0].content

        mock_get_client.assert_called_once_with(
            url="http://localhost:8123",
            headers={"x-auth-scheme": "langsmith"},
        )
        mock_client.threads.create.assert_called_once()
        mock_client.runs.create.assert_called_once_with(
            thread_id="thread_abc",
            assistant_id="my_graph",
            input={"messages": [{"role": "user", "content": "analyze data"}]},
        )


@pytest.mark.parametrize("invocation", ["sync", "async"])
@pytest.mark.parametrize(("mode", "summarized"), [(None, True), ("isolated", True), ("fork", False), ("fork", True)])
async def test_launch_context_snapshot(invocation: str, mode: str | None, *, summarized: bool) -> None:
    history = [
        HumanMessage(content="Original request"),
        AIMessage(content="Earlier answer"),
        HumanMessage(content=[{"type": "text", "text": "Investigate order 42"}]),
        AIMessage(content="", tool_calls=[{"name": "lookup", "args": {"order": 42}, "id": "lookup_call"}]),
        ToolMessage(content="Order 42 is delayed", tool_call_id="lookup_call", artifact=b"raw report bytes"),
        AIMessage(
            content="Delegating",
            tool_calls=[
                {"name": "start_async_task", "args": {"description": "Explain the delay", "subagent_type": "alpha"}, "id": "tc_launch"},
                {"name": "other_tool", "args": {}, "id": "sibling_call"},
            ],
        ),
    ]
    summary = HumanMessage(content="Summary of earlier context")
    runtime = _make_runtime("tc_launch")
    runtime.state.update({"messages": history, "files": {"private.txt": "not shared"}, "async_tasks": {}})
    if summarized:
        runtime.state["_summarization_event"] = {"cutoff_index": 2, "summary_message": summary, "file_path": None}
    initial_state = deepcopy(runtime.state)
    expected = deepcopy([summary, *history[2:-1]] if summarized else history[:-1])
    bodies: list[dict[str, Any]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/threads":
            assert runtime.state == initial_state
            history[2].content[0]["text"] = "Parent changed while creating child thread"
            summary.content = "Later summary"
            history.append(HumanMessage(content="Later parent message"))
            return httpx.Response(200, json={"thread_id": "new_child_thread"})
        assert request.url.path == "/threads/new_child_thread/runs"
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"run_id": "run_xyz", "status": "pending"})

    spec = _make_spec("alpha")
    if mode is not None:
        spec["mode"] = mode
    launch = _build_async_subagent_tools([spec])[0]
    transport = httpx.MockTransport(handle)
    if invocation == "sync":
        with (
            httpx.Client(base_url="http://remote", transport=transport) as http_client,
            patch("deepagents.middleware.async_subagents.get_sync_client", return_value=SyncLangGraphClient(http_client)),
        ):
            result = launch.func(description="Explain the delay", subagent_type="alpha", runtime=runtime)
    else:
        async with httpx.AsyncClient(base_url="http://remote", transport=transport) as http_client:
            with patch("deepagents.middleware.async_subagents.get_client", return_value=LangGraphClient(http_client)):
                result = await launch.coroutine(description="Explain the delay", subagent_type="alpha", runtime=runtime)

    assert isinstance(result, Command)
    assert result.update["async_tasks"]["new_child_thread"]["status"] == "running"
    assert result.update["messages"][0].tool_call_id == "tc_launch"
    assert len(bodies) == 1
    body = bodies[0]
    assert body["assistant_id"] == "my_graph"
    assert set(body["input"]) == {"messages"}
    if mode == "fork":
        child_messages = body["input"]["messages"]
        assert child_messages[:-1] == [message.model_dump(mode="json", exclude={"artifact"}) for message in expected]
        assert "artifact" not in child_messages[-2]
        assert child_messages[-1]["type"] == "human"
        assert child_messages[-1]["content"].endswith("Explain the delay")
    else:
        assert body["input"]["messages"] == [{"role": "user", "content": "Explain the delay"}]
    assert runtime.state["async_tasks"] == {}


@pytest.mark.filterwarnings("ignore:.*forked subagents.*:langchain_core._api.LangChainBetaWarning")
@pytest.mark.parametrize("invocation", ["sync", "async"])
@pytest.mark.parametrize("resume", [None, "new_turn", "approval"])
async def test_fork_rehydrates_offloaded_media(tmp_path: Path, invocation: str, resume: str | None) -> None:
    image = b"\x89PNG\r\n\x1a\n uploaded image"
    media = [
        {"type": "image", "mime_type": "image/png", "base64": base64.b64encode(image).decode("ascii")},
        {"type": "file", "mime_type": "application/pdf", "base64": base64.b64encode(b"%PDF-1.4 report").decode("ascii")},
    ]
    (tmp_path / "photo.png").write_bytes(image)
    responses = [AIMessage(content="", tool_calls=[{"name": "read_file", "args": {"file_path": "/photo.png"}, "id": "read"}])]
    if resume == "new_turn":
        responses.append(AIMessage(content="Media received"))
    responses.extend(
        [
            AIMessage(
                content="",
                tool_calls=[{"name": "start_async_task", "args": {"description": "Review the media", "subagent_type": "alpha"}, "id": "launch"}],
            ),
            AIMessage(content="Launched"),
        ]
    )
    model = GenericFakeChatModel(messages=iter(responses))
    agent = create_agent(
        model,
        middleware=[
            FilesystemMiddleware(
                backend=CompositeBackend(default=FilesystemBackend(root_dir=tmp_path, virtual_mode=True), routes={}, artifacts_root="/artifacts"),
                offload_binary_content=True,
            ),
            AsyncSubAgentMiddleware(async_subagents=[_make_spec("alpha", mode="fork")]),
            HumanInTheLoopMiddleware(interrupt_on={"start_async_task": True} if resume == "approval" else {}),
        ],
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "parent"}}
    payload = {"messages": [HumanMessage(content=deepcopy(media))]}
    if resume is not None:
        if invocation == "sync":
            paused = agent.invoke(payload, config)
        else:
            paused = await agent.ainvoke(payload, config)
        assert "_blob_payloads" not in agent.get_state(config).values
        (tmp_path / "photo.png").write_bytes(b"source changed after offloading")
        if resume == "approval":
            assert paused["__interrupt__"]
            payload = Command(resume={"decisions": [{"type": "approve"}]})
        else:
            payload = {"messages": [HumanMessage(content="Delegate the review")]}

    bodies: list[dict[str, Any]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/threads":
            return httpx.Response(200, json={"thread_id": "child"})
        assert request.url.path == "/threads/child/runs"
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"run_id": "run", "status": "pending"})

    transport = httpx.MockTransport(handle)
    if invocation == "sync":
        with (
            httpx.Client(base_url="http://remote", transport=transport) as http_client,
            patch("deepagents.middleware.async_subagents.get_sync_client", return_value=SyncLangGraphClient(http_client)),
        ):
            result = agent.invoke(payload, config)
    else:
        async with httpx.AsyncClient(base_url="http://remote", transport=transport) as http_client:
            with patch("deepagents.middleware.async_subagents.get_client", return_value=LangGraphClient(http_client)):
                result = await agent.ainvoke(payload, config)

    assert result["async_tasks"]["child"]["status"] == "running"
    assert len(bodies) == 1
    assert set(bodies[0]["input"]) == {"messages"}
    child_messages = bodies[0]["input"]["messages"]
    assert child_messages[0]["content"] == media
    child_tool = next(message for message in child_messages if message["type"] == "tool")
    assert child_tool["content"] == [media[0]]
    assert "deepagents_blob" not in json.dumps(child_messages)
    assert model.call_history[-2]["messages"][0].content == media
    assert "_blob_payloads" not in agent.get_state(config).values
    parent_messages = agent.get_state(config).values["messages"]
    parent_tool = next(message for message in parent_messages if message.type == "tool")
    for message in (parent_messages[0], parent_tool):
        assert all("deepagents_blob" in block and "base64" not in block for block in message.content)


@pytest.mark.filterwarnings("ignore:.*forked subagents.*:langchain_core._api.LangChainBetaWarning")
@pytest.mark.parametrize("invocation", ["sync", "async"])
async def test_fork_preserves_evicted_human_content(tmp_path: Path, invocation: str) -> None:
    content = "First section\n" * 500 + "Critical detail in the middle\n" + "Last section\n" * 500
    parent_backend = FilesystemBackend(root_dir=tmp_path / "parent", virtual_mode=True)
    parent_model = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[{"name": "start_async_task", "args": {"description": "Review details", "subagent_type": "alpha"}, "id": "launch"}],
                ),
                AIMessage(content="Launched"),
            ]
        )
    )
    parent = create_agent(
        parent_model,
        middleware=[
            FilesystemMiddleware(backend=parent_backend, human_message_token_limit_before_evict=1000),
            AsyncSubAgentMiddleware(async_subagents=[_make_spec("alpha", mode="fork")]),
        ],
    )
    bodies: list[dict[str, Any]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/threads":
            return httpx.Response(200, json={"thread_id": "child"})
        assert request.url.path == "/threads/child/runs"
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"run_id": "run", "status": "pending"})

    payload = {"messages": [HumanMessage(content=content, additional_kwargs={"source": "user-upload"})]}
    transport = httpx.MockTransport(handle)
    if invocation == "sync":
        with (
            httpx.Client(base_url="http://remote", transport=transport) as http_client,
            patch("deepagents.middleware.async_subagents.get_sync_client", return_value=SyncLangGraphClient(http_client)),
        ):
            result = parent.invoke(payload)
    else:
        async with httpx.AsyncClient(base_url="http://remote", transport=transport) as http_client:
            with patch("deepagents.middleware.async_subagents.get_client", return_value=LangGraphClient(http_client)):
                result = await parent.ainvoke(payload)

    parent_message = result["messages"][0]
    eviction_path = parent_message.additional_kwargs["lc_evicted_to"]
    assert parent_message.content == content
    assert parent_backend.download_files([eviction_path])[0].content == content.encode()
    assert "Critical detail in the middle" not in parent_model.call_history[0]["messages"][0].content

    child_backend = FilesystemBackend(root_dir=tmp_path / "child", virtual_mode=True)
    child_model = GenericFakeChatModel(messages=iter([AIMessage(content="Reviewed")]))
    child = create_agent(
        child_model,
        middleware=[FilesystemMiddleware(backend=child_backend, human_message_token_limit_before_evict=1000)],
    )
    assert child_backend.download_files([eviction_path])[0].error == "file_not_found"
    assert len(bodies) == 1
    if invocation == "sync":
        child.invoke(bodies[0]["input"])
    else:
        await child.ainvoke(bodies[0]["input"])
    received = child_model.call_history[0]["messages"][0]
    assert "Critical detail in the middle" in received.content
    assert received.content == content
    assert received.additional_kwargs == {"source": "user-upload"}
    assert parent_message.additional_kwargs == {"source": "user-upload", "lc_evicted_to": eviction_path}


class TestCheckTool:
    def _make_check_runtime(self, tool_call_id: str = "tc_check") -> ToolRuntime:
        """Create a runtime with a tracked task in state."""
        tasks: dict[str, AsyncTask] = {
            "thread_abc": {
                "task_id": "thread_abc",
                "agent_name": "test-agent",
                "thread_id": "thread_abc",
                "run_id": "run_xyz",
                "status": "running",
                "created_at": "2024-01-15T10:30:00Z",
                "last_checked_at": "2024-01-15T10:30:00Z",
                "last_updated_at": "2024-01-15T10:30:00Z",
            },
        }
        return ToolRuntime(
            state={"async_tasks": tasks},
            context=None,
            tool_call_id=tool_call_id,
            store=None,
            stream_writer=lambda _: None,
            config={},
        )

    @patch("deepagents.middleware.async_subagents.get_sync_client")
    def test_check_running_task(self, mock_get_client: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client.runs.get.return_value = {"run_id": "run_xyz", "status": "running"}
        mock_get_client.return_value = mock_client

        tools = _build_async_subagent_tools([_make_spec()])
        check = tools[1]
        result = check.func(
            task_id="thread_abc",
            runtime=self._make_check_runtime("tc_check"),
        )

        assert isinstance(result, Command)
        msgs = result.update["messages"]
        parsed = json.loads(msgs[0].content)
        assert parsed["status"] == "running"
        assert parsed["thread_id"] == "thread_abc"

        tasks = result.update["async_tasks"]
        assert tasks["thread_abc"]["status"] == "running"
        assert tasks["thread_abc"]["last_updated_at"] == "2024-01-15T10:30:00Z"

    @patch("deepagents.middleware.async_subagents.get_sync_client")
    def test_check_completed_task_returns_result(self, mock_get_client: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client.runs.get.return_value = {"run_id": "run_xyz", "status": "success"}
        mock_client.threads.get.return_value = {
            "values": {
                "messages": [
                    {"role": "assistant", "content": "Analysis complete: found 3 issues."},
                ]
            }
        }
        mock_get_client.return_value = mock_client

        tools = _build_async_subagent_tools([_make_spec()])
        check = tools[1]
        result = check.func(
            task_id="thread_abc",
            runtime=self._make_check_runtime("tc_check"),
        )

        assert isinstance(result, Command)
        parsed = json.loads(result.update["messages"][0].content)
        assert parsed["status"] == "success"
        assert parsed["result"] == "Analysis complete: found 3 issues."

        tasks = result.update["async_tasks"]
        assert tasks["thread_abc"]["status"] == "success"
        assert tasks["thread_abc"]["last_updated_at"] != "2024-01-15T10:30:00Z"

    @patch("deepagents.middleware.async_subagents.get_sync_client")
    def test_check_errored_task(self, mock_get_client: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client.runs.get.return_value = {"run_id": "run_xyz", "status": "error"}
        mock_get_client.return_value = mock_client

        tools = _build_async_subagent_tools([_make_spec()])
        check = tools[1]
        result = check.func(
            task_id="thread_abc",
            runtime=self._make_check_runtime("tc_check"),
        )

        assert isinstance(result, Command)
        parsed = json.loads(result.update["messages"][0].content)
        assert parsed["status"] == "error"
        assert "error" in parsed

        tasks = result.update["async_tasks"]
        assert tasks["thread_abc"]["status"] == "error"
        assert tasks["thread_abc"]["last_updated_at"] != "2024-01-15T10:30:00Z"


class TestUpdateTool:
    @patch("deepagents.middleware.async_subagents.get_sync_client")
    def test_update_returns_command_with_same_task_id(self, mock_get_client: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client.runs.create.return_value = {"run_id": "run_new"}
        mock_get_client.return_value = mock_client

        tools = _build_async_subagent_tools([_make_spec()])
        update = tools[2]
        tasks_state: dict[str, AsyncTask] = {
            "thread_abc": {
                "task_id": "thread_abc",
                "agent_name": "test-agent",
                "thread_id": "thread_abc",
                "run_id": "run_old",
                "status": "running",
                "created_at": "2024-01-15T10:30:00Z",
                "last_checked_at": "2024-01-15T10:30:00Z",
                "last_updated_at": "2024-01-15T10:30:00Z",
            },
        }
        rt = ToolRuntime(
            state={"async_tasks": tasks_state},
            context=None,
            tool_call_id="tc_update",
            store=None,
            stream_writer=lambda _: None,
            config={},
        )
        result = update.func(
            task_id="thread_abc",
            message="Focus on security issues only",
            runtime=rt,
        )

        assert isinstance(result, Command)
        tasks = result.update["async_tasks"]

        # Same task_id, updated run_id
        assert "thread_abc" in tasks
        assert len(tasks) == 1
        assert tasks["thread_abc"]["run_id"] == "run_new"
        assert tasks["thread_abc"]["status"] == "running"

        msgs = result.update["messages"]
        assert msgs[0].tool_call_id == "tc_update"
        assert "thread_abc" in msgs[0].content

        mock_client.runs.create.assert_called_once_with(
            thread_id="thread_abc",
            assistant_id="my_graph",
            input={"messages": [{"role": "user", "content": "Focus on security issues only"}]},
            multitask_strategy="interrupt",
        )


class TestListTasksTool:
    def test_empty_state_returns_no_tasks(self) -> None:
        tools = _build_async_subagent_tools([_make_spec()])
        list_tool = tools[4]
        rt = _make_runtime()
        result = list_tool.func(runtime=rt)
        assert "No async subagent tasks tracked" in result

    @patch("deepagents.middleware.async_subagents.get_sync_client")
    def test_returns_live_statuses(self, mock_get_client: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client.runs.get.side_effect = [
            {"run_id": "r1", "status": "success"},
            {"run_id": "r2", "status": "running"},
        ]
        mock_get_client.return_value = mock_client

        tools = _build_async_subagent_tools([_make_spec("test-agent")])
        list_tool = tools[4]
        tasks: dict[str, AsyncTask] = {
            "t1": {
                "task_id": "t1",
                "agent_name": "test-agent",
                "thread_id": "t1",
                "run_id": "r1",
                "status": "running",  # stale — SDK will return "success"
                "created_at": "2024-01-15T10:30:00Z",
                "last_checked_at": "2024-01-15T10:30:00Z",
                "last_updated_at": "2024-01-15T10:30:00Z",
            },
            "t2": {
                "task_id": "t2",
                "agent_name": "test-agent",
                "thread_id": "t2",
                "run_id": "r2",
                "status": "running",
                "created_at": "2024-01-15T10:31:00Z",
                "last_checked_at": "2024-01-15T10:31:00Z",
                "last_updated_at": "2024-01-15T10:31:00Z",
            },
        }
        rt = ToolRuntime(
            state={"async_tasks": tasks},
            context=None,
            tool_call_id="tc_list",
            store=None,
            stream_writer=lambda _: None,
            config={},
        )
        result = list_tool.func(runtime=rt)
        assert isinstance(result, Command)
        content = result.update["messages"][0].content
        assert "2 tracked task(s)" in content
        assert "t1" in content
        assert "t2" in content
        assert "success" in content
        assert "running" in content
        # state should be updated with fresh statuses
        updated = result.update["async_tasks"]
        assert updated["t1"]["status"] == "success"
        assert updated["t1"]["last_updated_at"] != "2024-01-15T10:30:00Z"
        assert updated["t2"]["status"] == "running"
        assert updated["t2"]["last_updated_at"] == "2024-01-15T10:31:00Z"

    @patch("deepagents.middleware.async_subagents.get_sync_client")
    def test_skips_sdk_call_for_terminal_statuses(self, mock_get_client: MagicMock) -> None:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        tools = _build_async_subagent_tools([_make_spec("test-agent")])
        list_tool = tools[4]
        tasks: dict[str, AsyncTask] = {
            "t1": {
                "task_id": "t1",
                "agent_name": "test-agent",
                "thread_id": "t1",
                "run_id": "r1",
                "status": "cancelled",
                "created_at": "2024-01-15T10:30:00Z",
                "last_checked_at": "2024-01-15T10:30:00Z",
                "last_updated_at": "2024-01-15T10:30:00Z",
            },
            "t2": {
                "task_id": "t2",
                "agent_name": "test-agent",
                "thread_id": "t2",
                "run_id": "r2",
                "status": "success",
                "created_at": "2024-01-15T10:31:00Z",
                "last_checked_at": "2024-01-15T10:31:00Z",
                "last_updated_at": "2024-01-15T10:31:00Z",
            },
            "t3": {
                "task_id": "t3",
                "agent_name": "test-agent",
                "thread_id": "t3",
                "run_id": "r3",
                "status": "error",
                "created_at": "2024-01-15T10:32:00Z",
                "last_checked_at": "2024-01-15T10:32:00Z",
                "last_updated_at": "2024-01-15T10:32:00Z",
            },
        }
        rt = ToolRuntime(
            state={"async_tasks": tasks},
            context=None,
            tool_call_id="tc_list",
            store=None,
            stream_writer=lambda _: None,
            config={},
        )
        result = list_tool.func(runtime=rt)
        assert isinstance(result, Command)
        mock_client.runs.get.assert_not_called()
        content = result.update["messages"][0].content
        assert "3 tracked task(s)" in content
        assert "cancelled" in content
        assert "success" in content
        assert "error" in content

    @patch("deepagents.middleware.async_subagents.get_sync_client")
    def test_status_filter_running(self, mock_get_client: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client.runs.get.return_value = {"run_id": "r1", "status": "running"}
        mock_get_client.return_value = mock_client

        tools = _build_async_subagent_tools([_make_spec("test-agent")])
        list_tool = tools[4]
        tasks: dict[str, AsyncTask] = {
            "t1": {
                "task_id": "t1",
                "agent_name": "test-agent",
                "thread_id": "t1",
                "run_id": "r1",
                "status": "running",
                "created_at": "2024-01-15T10:30:00Z",
                "last_checked_at": "2024-01-15T10:30:00Z",
                "last_updated_at": "2024-01-15T10:30:00Z",
            },
            "t2": {
                "task_id": "t2",
                "agent_name": "test-agent",
                "thread_id": "t2",
                "run_id": "r2",
                "status": "success",
                "created_at": "2024-01-15T10:31:00Z",
                "last_checked_at": "2024-01-15T10:31:00Z",
                "last_updated_at": "2024-01-15T10:31:00Z",
            },
        }
        rt = ToolRuntime(
            state={"async_tasks": tasks},
            context=None,
            tool_call_id="tc_list",
            store=None,
            stream_writer=lambda _: None,
            config={},
        )
        result = list_tool.func(runtime=rt, status_filter="running")
        assert isinstance(result, Command)
        content = result.update["messages"][0].content
        assert "1 tracked task(s)" in content
        assert "t1" in content
        assert "t2" not in content

    async def test_async_list_returns_no_tasks(self) -> None:
        tools = _build_async_subagent_tools([_make_spec()])
        list_tool = tools[4]
        rt = _make_runtime()
        result = await list_tool.coroutine(runtime=rt)
        assert "No async subagent tasks tracked" in result


_T = TypeVar("_T")


def _async_return(value: _T) -> Any:  # noqa: ANN401
    """Create an async function that returns a fixed value."""

    async def _inner(*_args: Any, **_kwargs: Any) -> _T:
        return value

    return _inner


@pytest.mark.allow_hosts(["127.0.0.1", "::1"])
class TestAsyncTools:
    @patch("deepagents.middleware.async_subagents.get_client")
    async def test_async_launch_returns_command(self, mock_get_client: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client.threads.create = _async_return({"thread_id": "thread_abc"})
        mock_client.runs.create = _async_return({"run_id": "run_xyz"})
        mock_get_client.return_value = mock_client

        tools = _build_async_subagent_tools([_make_spec("alpha")])
        launch = tools[0]
        result = await launch.coroutine(
            description="analyze data",
            subagent_type="alpha",
            runtime=_make_runtime("tc_async_launch"),
        )

        assert isinstance(result, Command)
        assert "thread_abc" in result.update["messages"][0].content
        tasks = result.update["async_tasks"]
        assert "thread_abc" in tasks

    @patch("deepagents.middleware.async_subagents.get_client")
    async def test_async_check_returns_command(self, mock_get_client: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client.runs.get = _async_return({"run_id": "run_xyz", "status": "success"})
        mock_client.threads.get = _async_return({"values": {"messages": [{"role": "assistant", "content": "Done!"}]}})
        mock_get_client.return_value = mock_client

        tools = _build_async_subagent_tools([_make_spec()])
        check = tools[1]
        tracked_tasks: dict[str, AsyncTask] = {
            "thread_abc": {
                "task_id": "thread_abc",
                "agent_name": "test-agent",
                "thread_id": "thread_abc",
                "run_id": "run_xyz",
                "status": "running",
                "created_at": "2024-01-15T10:30:00Z",
                "last_checked_at": "2024-01-15T10:30:00Z",
                "last_updated_at": "2024-01-15T10:30:00Z",
            },
        }
        rt = ToolRuntime(
            state={"async_tasks": tracked_tasks},
            context=None,
            tool_call_id="tc_async_check",
            store=None,
            stream_writer=lambda _: None,
            config={},
        )
        result = await check.coroutine(
            task_id="thread_abc",
            runtime=rt,
        )

        assert isinstance(result, Command)
        parsed = json.loads(result.update["messages"][0].content)
        assert parsed["status"] == "success"
        assert parsed["result"] == "Done!"
        assert result.update["async_tasks"]["thread_abc"]["status"] == "success"

    @patch("deepagents.middleware.async_subagents.get_client")
    async def test_async_update_returns_command(self, mock_get_client: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client.runs.create = _async_return({"run_id": "run_new"})
        mock_get_client.return_value = mock_client

        tools = _build_async_subagent_tools([_make_spec()])
        update = tools[2]
        tasks_state: dict[str, AsyncTask] = {
            "thread_abc": {
                "task_id": "thread_abc",
                "agent_name": "test-agent",
                "thread_id": "thread_abc",
                "run_id": "run_old",
                "status": "running",
                "created_at": "2024-01-15T10:30:00Z",
                "last_checked_at": "2024-01-15T10:30:00Z",
                "last_updated_at": "2024-01-15T10:30:00Z",
            },
        }
        rt = ToolRuntime(
            state={"async_tasks": tasks_state},
            context=None,
            tool_call_id="tc_async_update",
            store=None,
            stream_writer=lambda _: None,
            config={},
        )
        result = await update.coroutine(
            task_id="thread_abc",
            message="New instructions",
            runtime=rt,
        )

        assert isinstance(result, Command)
        assert "thread_abc" in result.update["async_tasks"]
        assert result.update["async_tasks"]["thread_abc"]["run_id"] == "run_new"

    @patch("deepagents.middleware.async_subagents.get_client")
    async def test_async_cancel_returns_command(self, mock_get_client: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client.runs.cancel = _async_return(None)
        mock_get_client.return_value = mock_client

        tools = _build_async_subagent_tools([_make_spec()])
        cancel = _get_tool(tools, "cancel_async_task")
        rt = _make_runtime_with_task(tool_call_id="tc_async_cancel")
        result = await cancel.coroutine(task_id="thread_abc", runtime=rt)

        assert isinstance(result, Command)
        assert result.update["async_tasks"]["thread_abc"]["status"] == "cancelled"


class TestCancelTool:
    @patch("deepagents.middleware.async_subagents.get_sync_client")
    def test_cancel_returns_command_with_cancelled_status(self, mock_get_client: MagicMock) -> None:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        tools = _build_async_subagent_tools([_make_spec()])
        cancel = _get_tool(tools, "cancel_async_task")
        rt = _make_runtime_with_task(tool_call_id="tc_cancel")
        result = cancel.func(task_id="thread_abc", runtime=rt)

        assert isinstance(result, Command)
        tasks = result.update["async_tasks"]
        assert tasks["thread_abc"]["status"] == "cancelled"
        assert tasks["thread_abc"]["last_updated_at"] != "2024-01-15T10:30:00Z"
        assert tasks["thread_abc"]["task_id"] == "thread_abc"
        msgs = result.update["messages"]
        assert msgs[0].tool_call_id == "tc_cancel"
        assert "thread_abc" in msgs[0].content
        mock_client.runs.cancel.assert_called_once_with(
            thread_id="thread_abc",
            run_id="run_xyz",
        )

    def test_cancel_unknown_task_returns_error(self) -> None:
        tools = _build_async_subagent_tools([_make_spec()])
        cancel = _get_tool(tools, "cancel_async_task")
        rt = _make_runtime("tc_cancel")
        result = cancel.func(task_id="nonexistent", runtime=rt)
        assert isinstance(result, str)
        assert "No tracked task found" in result

    @patch("deepagents.middleware.async_subagents.get_sync_client")
    def test_cancel_sdk_error_returns_error_string(self, mock_get_client: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client.runs.cancel.side_effect = RuntimeError("connection refused")
        mock_get_client.return_value = mock_client

        tools = _build_async_subagent_tools([_make_spec()])
        cancel = _get_tool(tools, "cancel_async_task")
        rt = _make_runtime_with_task(tool_call_id="tc_cancel")
        result = cancel.func(task_id="thread_abc", runtime=rt)
        assert isinstance(result, str)
        assert "Failed to cancel run" in result
        assert "connection refused" in result


class TestUnknownTaskId:
    """Tests that check/update/cancel return error strings for unknown task IDs."""

    def test_check_unknown_task_returns_error(self) -> None:
        tools = _build_async_subagent_tools([_make_spec()])
        check = _get_tool(tools, "check_async_task")
        rt = _make_runtime()
        result = check.func(task_id="nonexistent", runtime=rt)
        assert isinstance(result, str)
        assert "No tracked task found" in result

    def test_update_unknown_task_returns_error(self) -> None:
        tools = _build_async_subagent_tools([_make_spec()])
        update = _get_tool(tools, "update_async_task")
        rt = _make_runtime()
        result = update.func(task_id="nonexistent", message="hello", runtime=rt)
        assert isinstance(result, str)
        assert "No tracked task found" in result


class TestLaunchErrorHandling:
    @patch("deepagents.middleware.async_subagents.get_sync_client")
    def test_launch_sdk_error_returns_error_string(self, mock_get_client: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client.threads.create.side_effect = RuntimeError("connection refused")
        mock_get_client.return_value = mock_client

        tools = _build_async_subagent_tools([_make_spec("alpha")])
        launch = _get_tool(tools, "start_async_task")
        result = launch.func(
            description="do stuff",
            subagent_type="alpha",
            runtime=_make_runtime(),
        )
        assert isinstance(result, str)
        assert "Failed to launch" in result
        assert "connection refused" in result

    @patch("deepagents.middleware.async_subagents.get_sync_client")
    def test_update_sdk_error_returns_error_string(self, mock_get_client: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client.runs.create.side_effect = RuntimeError("timeout")
        mock_get_client.return_value = mock_client

        tools = _build_async_subagent_tools([_make_spec()])
        update = _get_tool(tools, "update_async_task")
        rt = _make_runtime_with_task(tool_call_id="tc_update")
        result = update.func(task_id="thread_abc", message="hello", runtime=rt)
        assert isinstance(result, str)
        assert "Failed to update" in result
        assert "timeout" in result


class TestCheckEdgeCases:
    @patch("deepagents.middleware.async_subagents.get_sync_client")
    def test_check_errored_task_includes_server_error(self, mock_get_client: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client.runs.get.return_value = {
            "run_id": "run_xyz",
            "status": "error",
            "error": "Tool 'search' raised ValueError: invalid query",
        }
        mock_get_client.return_value = mock_client

        tools = _build_async_subagent_tools([_make_spec()])
        check = _get_tool(tools, "check_async_task")
        rt = _make_runtime_with_task(tool_call_id="tc_check")
        result = check.func(task_id="thread_abc", runtime=rt)

        assert isinstance(result, Command)
        parsed = json.loads(result.update["messages"][0].content)
        assert parsed["status"] == "error"
        assert "ValueError" in parsed["error"]

    @patch("deepagents.middleware.async_subagents.get_sync_client")
    def test_check_success_empty_messages(self, mock_get_client: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client.runs.get.return_value = {"run_id": "run_xyz", "status": "success"}
        mock_client.threads.get.return_value = {"values": {"messages": []}}
        mock_get_client.return_value = mock_client

        tools = _build_async_subagent_tools([_make_spec()])
        check = _get_tool(tools, "check_async_task")
        rt = _make_runtime_with_task(tool_call_id="tc_check")
        result = check.func(task_id="thread_abc", runtime=rt)

        assert isinstance(result, Command)
        parsed = json.loads(result.update["messages"][0].content)
        assert parsed["status"] == "success"
        assert "no output" in parsed["result"].lower()

    @patch("deepagents.middleware.async_subagents.get_sync_client")
    def test_check_threads_get_failure_still_returns_status(self, mock_get_client: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client.runs.get.return_value = {"run_id": "run_xyz", "status": "success"}
        mock_client.threads.get.side_effect = RuntimeError("network error")
        mock_get_client.return_value = mock_client

        tools = _build_async_subagent_tools([_make_spec()])
        check = _get_tool(tools, "check_async_task")
        rt = _make_runtime_with_task(tool_call_id="tc_check")
        result = check.func(task_id="thread_abc", runtime=rt)

        assert isinstance(result, Command)
        parsed = json.loads(result.update["messages"][0].content)
        assert parsed["status"] == "success"
        # result should show empty-messages fallback since thread values couldn't be fetched
        assert "no output" in parsed["result"].lower()


class TestStaleStatusGuidanceInToolDescriptions:
    """The stale-status rule lives in the async status tool descriptions.

    It used to be in the (now-trimmed) async system prompt, so it is migrated
    into the always-visible `check_async_task` / `list_async_tasks` descriptions.
    """

    def _descriptions(self) -> dict[str, str]:
        tools = _build_async_subagent_tools([_make_spec()])
        return {t.name: (t.description or "") for t in tools}

    def test_check_async_task_warns_statuses_are_stale(self) -> None:
        assert "stale" in self._descriptions()["check_async_task"].lower()

    def test_list_async_tasks_warns_statuses_are_stale(self) -> None:
        assert "stale" in self._descriptions()["list_async_tasks"].lower()
