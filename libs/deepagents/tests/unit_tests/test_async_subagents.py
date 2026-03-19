"""Tests for async subagent middleware functionality."""

import json
from typing import Any, TypeVar
from unittest.mock import MagicMock, patch

import pytest
from langchain.tools import ToolRuntime
from langgraph.types import Command

from deepagents.middleware.async_subagents import (
    AsyncSubAgent,
    AsyncSubAgentJob,
    AsyncSubAgentMiddleware,
    AsyncSubAgentState,
    _build_async_subagent_tools,
    _jobs_reducer,
    _resolve_headers,
)


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


def _make_runtime_with_job(
    job_id: str = "thread_abc",
    agent_name: str = "test-agent",
    run_id: str = "run_xyz",
    status: str = "running",
    tool_call_id: str = "tc_test",
    created_at: str = "2024-01-15T10:30:00Z",
    last_checked_at: str = "2024-01-15T10:30:00Z",
    last_updated_at: str = "2024-01-15T10:30:00Z",
) -> ToolRuntime:
    """Create a runtime with a single tracked job in state."""
    jobs: dict[str, AsyncSubAgentJob] = {
        job_id: {
            "job_id": job_id,
            "agent_name": agent_name,
            "thread_id": job_id,
            "run_id": run_id,
            "status": status,
            "created_at": created_at,
            "last_checked_at": last_checked_at,
            "last_updated_at": last_updated_at,
        },
    }
    return ToolRuntime(
        state={"async_subagent_jobs": jobs},
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
            "launch_async_subagent",
            "check_async_subagent",
            "update_async_subagent",
            "cancel_async_subagent",
            "list_async_subagent_jobs",
        }

    def test_system_prompt_includes_agent_descriptions(self) -> None:
        mw = AsyncSubAgentMiddleware(
            async_subagents=[
                _make_spec("alpha", description="Alpha agent"),
                _make_spec("beta", description="Beta agent"),
            ]
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


class TestJobsReducer:
    def test_merge_into_empty(self) -> None:
        job: AsyncSubAgentJob = {
            "job_id": "t",
            "agent_name": "a",
            "thread_id": "t",
            "run_id": "r",
            "status": "running",
        }
        result = _jobs_reducer(None, {"t": job})
        assert result == {"t": job}

    def test_merge_updates_existing(self) -> None:
        old: AsyncSubAgentJob = {
            "job_id": "t",
            "agent_name": "a",
            "thread_id": "t",
            "run_id": "r",
            "status": "running",
        }
        updated: AsyncSubAgentJob = {**old, "status": "success"}
        result = _jobs_reducer({"t": old}, {"t": updated})
        assert result["t"]["status"] == "success"

    def test_merge_preserves_other_keys(self) -> None:
        job1: AsyncSubAgentJob = {
            "job_id": "t1",
            "agent_name": "a",
            "thread_id": "t1",
            "run_id": "r1",
            "status": "running",
        }
        job2: AsyncSubAgentJob = {
            "job_id": "t2",
            "agent_name": "a",
            "thread_id": "t2",
            "run_id": "r2",
            "status": "running",
        }
        result = _jobs_reducer({"t1": job1}, {"t2": job2})
        assert len(result) == 2
        assert "t1" in result
        assert "t2" in result


class TestBuildAsyncSubagentTools:
    def test_returns_five_tools(self) -> None:
        tools = _build_async_subagent_tools([_make_spec()])
        assert len(tools) == 5
        names = [t.name for t in tools]
        assert names == [
            "launch_async_subagent",
            "check_async_subagent",
            "update_async_subagent",
            "cancel_async_subagent",
            "list_async_subagent_jobs",
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
    def test_launch_returns_command_with_job(self, mock_get_client: MagicMock) -> None:
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
        assert "async_subagent_jobs" in update
        jobs = update["async_subagent_jobs"]
        assert "thread_abc" in jobs
        job = jobs["thread_abc"]
        assert job["job_id"] == "thread_abc"
        assert job["agent_name"] == "alpha"
        assert job["thread_id"] == "thread_abc"
        assert job["run_id"] == "run_xyz"
        assert job["status"] == "running"

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


class TestCheckTool:
    def _make_check_runtime(self, tool_call_id: str = "tc_check") -> ToolRuntime:
        """Create a runtime with a tracked job in state."""
        jobs: dict[str, AsyncSubAgentJob] = {
            "thread_abc": {
                "job_id": "thread_abc",
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
            state={"async_subagent_jobs": jobs},
            context=None,
            tool_call_id=tool_call_id,
            store=None,
            stream_writer=lambda _: None,
            config={},
        )

    @patch("deepagents.middleware.async_subagents.get_sync_client")
    def test_check_running_job(self, mock_get_client: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client.runs.get.return_value = {"run_id": "run_xyz", "status": "running"}
        mock_get_client.return_value = mock_client

        tools = _build_async_subagent_tools([_make_spec()])
        check = tools[1]
        result = check.func(
            job_id="thread_abc",
            runtime=self._make_check_runtime("tc_check"),
        )

        assert isinstance(result, Command)
        msgs = result.update["messages"]
        parsed = json.loads(msgs[0].content)
        assert parsed["status"] == "running"
        assert parsed["thread_id"] == "thread_abc"

        jobs = result.update["async_subagent_jobs"]
        assert jobs["thread_abc"]["status"] == "running"

    @patch("deepagents.middleware.async_subagents.get_sync_client")
    def test_check_completed_job_returns_result(self, mock_get_client: MagicMock) -> None:
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
            job_id="thread_abc",
            runtime=self._make_check_runtime("tc_check"),
        )

        assert isinstance(result, Command)
        parsed = json.loads(result.update["messages"][0].content)
        assert parsed["status"] == "success"
        assert parsed["result"] == "Analysis complete: found 3 issues."

        jobs = result.update["async_subagent_jobs"]
        assert jobs["thread_abc"]["status"] == "success"

    @patch("deepagents.middleware.async_subagents.get_sync_client")
    def test_check_errored_job(self, mock_get_client: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client.runs.get.return_value = {"run_id": "run_xyz", "status": "error"}
        mock_get_client.return_value = mock_client

        tools = _build_async_subagent_tools([_make_spec()])
        check = tools[1]
        result = check.func(
            job_id="thread_abc",
            runtime=self._make_check_runtime("tc_check"),
        )

        assert isinstance(result, Command)
        parsed = json.loads(result.update["messages"][0].content)
        assert parsed["status"] == "error"
        assert "error" in parsed

        jobs = result.update["async_subagent_jobs"]
        assert jobs["thread_abc"]["status"] == "error"


class TestUpdateTool:
    @patch("deepagents.middleware.async_subagents.get_sync_client")
    def test_update_returns_command_with_same_job_id(self, mock_get_client: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client.runs.create.return_value = {"run_id": "run_new"}
        mock_get_client.return_value = mock_client

        tools = _build_async_subagent_tools([_make_spec()])
        update = tools[2]
        jobs_state: dict[str, AsyncSubAgentJob] = {
            "thread_abc": {
                "job_id": "thread_abc",
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
            state={"async_subagent_jobs": jobs_state},
            context=None,
            tool_call_id="tc_update",
            store=None,
            stream_writer=lambda _: None,
            config={},
        )
        result = update.func(
            job_id="thread_abc",
            message="Focus on security issues only",
            runtime=rt,
        )

        assert isinstance(result, Command)
        jobs = result.update["async_subagent_jobs"]

        # Same job_id, updated run_id
        assert "thread_abc" in jobs
        assert len(jobs) == 1
        assert jobs["thread_abc"]["run_id"] == "run_new"
        assert jobs["thread_abc"]["status"] == "running"

        msgs = result.update["messages"]
        assert msgs[0].tool_call_id == "tc_update"
        assert "thread_abc" in msgs[0].content

        mock_client.runs.create.assert_called_once_with(
            thread_id="thread_abc",
            assistant_id="my_graph",
            input={"messages": [{"role": "user", "content": "Focus on security issues only"}]},
            multitask_strategy="interrupt",
        )


class TestListJobsTool:
    def test_empty_state_returns_no_jobs(self) -> None:
        tools = _build_async_subagent_tools([_make_spec()])
        list_tool = tools[4]
        rt = _make_runtime()
        result = list_tool.func(runtime=rt)
        assert "No async subagent jobs tracked" in result

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
        jobs: dict[str, AsyncSubAgentJob] = {
            "t1": {
                "job_id": "t1",
                "agent_name": "test-agent",
                "thread_id": "t1",
                "run_id": "r1",
                "status": "running",  # stale — SDK will return "success"
                "created_at": "2024-01-15T10:30:00Z",
                "last_checked_at": "2024-01-15T10:30:00Z",
                "last_updated_at": "2024-01-15T10:30:00Z",
            },
            "t2": {
                "job_id": "t2",
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
            state={"async_subagent_jobs": jobs},
            context=None,
            tool_call_id="tc_list",
            store=None,
            stream_writer=lambda _: None,
            config={},
        )
        result = list_tool.func(runtime=rt)
        assert isinstance(result, Command)
        content = result.update["messages"][0].content
        assert "2 tracked job(s)" in content
        assert "t1" in content
        assert "t2" in content
        assert "success" in content
        assert "running" in content
        # state should be updated with fresh statuses
        updated = result.update["async_subagent_jobs"]
        assert updated["t1"]["status"] == "success"
        assert updated["t2"]["status"] == "running"

    @patch("deepagents.middleware.async_subagents.get_sync_client")
    def test_skips_sdk_call_for_terminal_statuses(self, mock_get_client: MagicMock) -> None:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        tools = _build_async_subagent_tools([_make_spec("test-agent")])
        list_tool = tools[4]
        jobs: dict[str, AsyncSubAgentJob] = {
            "t1": {
                "job_id": "t1",
                "agent_name": "test-agent",
                "thread_id": "t1",
                "run_id": "r1",
                "status": "cancelled",
                "created_at": "2024-01-15T10:30:00Z",
                "last_checked_at": "2024-01-15T10:30:00Z",
                "last_updated_at": "2024-01-15T10:30:00Z",
            },
            "t2": {
                "job_id": "t2",
                "agent_name": "test-agent",
                "thread_id": "t2",
                "run_id": "r2",
                "status": "success",
                "created_at": "2024-01-15T10:31:00Z",
                "last_checked_at": "2024-01-15T10:31:00Z",
                "last_updated_at": "2024-01-15T10:31:00Z",
            },
            "t3": {
                "job_id": "t3",
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
            state={"async_subagent_jobs": jobs},
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
        assert "3 tracked job(s)" in content
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
        jobs: dict[str, AsyncSubAgentJob] = {
            "t1": {
                "job_id": "t1",
                "agent_name": "test-agent",
                "thread_id": "t1",
                "run_id": "r1",
                "status": "running",
                "created_at": "2024-01-15T10:30:00Z",
                "last_checked_at": "2024-01-15T10:30:00Z",
                "last_updated_at": "2024-01-15T10:30:00Z",
            },
            "t2": {
                "job_id": "t2",
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
            state={"async_subagent_jobs": jobs},
            context=None,
            tool_call_id="tc_list",
            store=None,
            stream_writer=lambda _: None,
            config={},
        )
        result = list_tool.func(runtime=rt, status_filter="running")
        assert isinstance(result, Command)
        content = result.update["messages"][0].content
        assert "1 tracked job(s)" in content
        assert "t1" in content
        assert "t2" not in content

    async def test_async_list_returns_no_jobs(self) -> None:
        tools = _build_async_subagent_tools([_make_spec()])
        list_tool = tools[4]
        rt = _make_runtime()
        result = await list_tool.coroutine(runtime=rt)
        assert "No async subagent jobs tracked" in result


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
        jobs = result.update["async_subagent_jobs"]
        assert "thread_abc" in jobs

    @patch("deepagents.middleware.async_subagents.get_client")
    async def test_async_check_returns_command(self, mock_get_client: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client.runs.get = _async_return({"run_id": "run_xyz", "status": "success"})
        mock_client.threads.get = _async_return({"values": {"messages": [{"role": "assistant", "content": "Done!"}]}})
        mock_get_client.return_value = mock_client

        tools = _build_async_subagent_tools([_make_spec()])
        check = tools[1]
        tracked_jobs: dict[str, AsyncSubAgentJob] = {
            "thread_abc": {
                "job_id": "thread_abc",
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
            state={"async_subagent_jobs": tracked_jobs},
            context=None,
            tool_call_id="tc_async_check",
            store=None,
            stream_writer=lambda _: None,
            config={},
        )
        result = await check.coroutine(
            job_id="thread_abc",
            runtime=rt,
        )

        assert isinstance(result, Command)
        parsed = json.loads(result.update["messages"][0].content)
        assert parsed["status"] == "success"
        assert parsed["result"] == "Done!"
        assert result.update["async_subagent_jobs"]["thread_abc"]["status"] == "success"

    @patch("deepagents.middleware.async_subagents.get_client")
    async def test_async_update_returns_command(self, mock_get_client: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client.runs.create = _async_return({"run_id": "run_new"})
        mock_get_client.return_value = mock_client

        tools = _build_async_subagent_tools([_make_spec()])
        update = tools[2]
        jobs_state: dict[str, AsyncSubAgentJob] = {
            "thread_abc": {
                "job_id": "thread_abc",
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
            state={"async_subagent_jobs": jobs_state},
            context=None,
            tool_call_id="tc_async_update",
            store=None,
            stream_writer=lambda _: None,
            config={},
        )
        result = await update.coroutine(
            job_id="thread_abc",
            message="New instructions",
            runtime=rt,
        )

        assert isinstance(result, Command)
        assert "thread_abc" in result.update["async_subagent_jobs"]
        assert result.update["async_subagent_jobs"]["thread_abc"]["run_id"] == "run_new"

    @patch("deepagents.middleware.async_subagents.get_client")
    async def test_async_cancel_returns_command(self, mock_get_client: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client.runs.cancel = _async_return(None)
        mock_get_client.return_value = mock_client

        tools = _build_async_subagent_tools([_make_spec()])
        cancel = _get_tool(tools, "cancel_async_subagent")
        rt = _make_runtime_with_job(tool_call_id="tc_async_cancel")
        result = await cancel.coroutine(job_id="thread_abc", runtime=rt)

        assert isinstance(result, Command)
        assert result.update["async_subagent_jobs"]["thread_abc"]["status"] == "cancelled"


class TestCancelTool:
    @patch("deepagents.middleware.async_subagents.get_sync_client")
    def test_cancel_returns_command_with_cancelled_status(self, mock_get_client: MagicMock) -> None:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        tools = _build_async_subagent_tools([_make_spec()])
        cancel = _get_tool(tools, "cancel_async_subagent")
        rt = _make_runtime_with_job(tool_call_id="tc_cancel")
        result = cancel.func(job_id="thread_abc", runtime=rt)

        assert isinstance(result, Command)
        jobs = result.update["async_subagent_jobs"]
        assert jobs["thread_abc"]["status"] == "cancelled"
        assert jobs["thread_abc"]["job_id"] == "thread_abc"
        msgs = result.update["messages"]
        assert msgs[0].tool_call_id == "tc_cancel"
        assert "thread_abc" in msgs[0].content
        mock_client.runs.cancel.assert_called_once_with(
            thread_id="thread_abc",
            run_id="run_xyz",
        )

    def test_cancel_unknown_job_returns_error(self) -> None:
        tools = _build_async_subagent_tools([_make_spec()])
        cancel = _get_tool(tools, "cancel_async_subagent")
        rt = _make_runtime("tc_cancel")
        result = cancel.func(job_id="nonexistent", runtime=rt)
        assert isinstance(result, str)
        assert "No tracked job found" in result

    @patch("deepagents.middleware.async_subagents.get_sync_client")
    def test_cancel_sdk_error_returns_error_string(self, mock_get_client: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client.runs.cancel.side_effect = RuntimeError("connection refused")
        mock_get_client.return_value = mock_client

        tools = _build_async_subagent_tools([_make_spec()])
        cancel = _get_tool(tools, "cancel_async_subagent")
        rt = _make_runtime_with_job(tool_call_id="tc_cancel")
        result = cancel.func(job_id="thread_abc", runtime=rt)
        assert isinstance(result, str)
        assert "Failed to cancel run" in result
        assert "connection refused" in result


class TestUnknownJobId:
    """Tests that check/update/cancel return error strings for unknown job IDs."""

    def test_check_unknown_job_returns_error(self) -> None:
        tools = _build_async_subagent_tools([_make_spec()])
        check = _get_tool(tools, "check_async_subagent")
        rt = _make_runtime()
        result = check.func(job_id="nonexistent", runtime=rt)
        assert isinstance(result, str)
        assert "No tracked job found" in result

    def test_update_unknown_job_returns_error(self) -> None:
        tools = _build_async_subagent_tools([_make_spec()])
        update = _get_tool(tools, "update_async_subagent")
        rt = _make_runtime()
        result = update.func(job_id="nonexistent", message="hello", runtime=rt)
        assert isinstance(result, str)
        assert "No tracked job found" in result


class TestLaunchErrorHandling:
    @patch("deepagents.middleware.async_subagents.get_sync_client")
    def test_launch_sdk_error_returns_error_string(self, mock_get_client: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client.threads.create.side_effect = RuntimeError("connection refused")
        mock_get_client.return_value = mock_client

        tools = _build_async_subagent_tools([_make_spec("alpha")])
        launch = _get_tool(tools, "launch_async_subagent")
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
        update = _get_tool(tools, "update_async_subagent")
        rt = _make_runtime_with_job(tool_call_id="tc_update")
        result = update.func(job_id="thread_abc", message="hello", runtime=rt)
        assert isinstance(result, str)
        assert "Failed to update" in result
        assert "timeout" in result


class TestCheckEdgeCases:
    @patch("deepagents.middleware.async_subagents.get_sync_client")
    def test_check_errored_job_includes_server_error(self, mock_get_client: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client.runs.get.return_value = {
            "run_id": "run_xyz",
            "status": "error",
            "error": "Tool 'search' raised ValueError: invalid query",
        }
        mock_get_client.return_value = mock_client

        tools = _build_async_subagent_tools([_make_spec()])
        check = _get_tool(tools, "check_async_subagent")
        rt = _make_runtime_with_job(tool_call_id="tc_check")
        result = check.func(job_id="thread_abc", runtime=rt)

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
        check = _get_tool(tools, "check_async_subagent")
        rt = _make_runtime_with_job(tool_call_id="tc_check")
        result = check.func(job_id="thread_abc", runtime=rt)

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
        check = _get_tool(tools, "check_async_subagent")
        rt = _make_runtime_with_job(tool_call_id="tc_check")
        result = check.func(job_id="thread_abc", runtime=rt)

        assert isinstance(result, Command)
        parsed = json.loads(result.update["messages"][0].content)
        assert parsed["status"] == "success"
        # result should show empty-messages fallback since thread values couldn't be fetched
        assert "no output" in parsed["result"].lower()
