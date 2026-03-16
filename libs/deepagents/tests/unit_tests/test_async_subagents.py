"""Tests for async subagent middleware functionality."""

import json
from typing import Any
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
            },
            "t2": {
                "job_id": "t2",
                "agent_name": "test-agent",
                "thread_id": "t2",
                "run_id": "r2",
                "status": "running",
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
            },
            "t2": {
                "job_id": "t2",
                "agent_name": "test-agent",
                "thread_id": "t2",
                "run_id": "r2",
                "status": "success",
            },
            "t3": {
                "job_id": "t3",
                "agent_name": "test-agent",
                "thread_id": "t3",
                "run_id": "r3",
                "status": "error",
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
        list_tool.func(runtime=rt)
        mock_client.runs.get.assert_not_called()

    async def test_async_list_returns_no_jobs(self) -> None:
        tools = _build_async_subagent_tools([_make_spec()])
        list_tool = tools[4]
        rt = _make_runtime()
        result = await list_tool.coroutine(runtime=rt)
        assert "No async subagent jobs tracked" in result


def _async_return(value: Any) -> Any:  # noqa: ANN401
    """Create an async function that returns a fixed value."""

    async def _inner(*_args: Any, **_kwargs: Any) -> Any:  # noqa: ANN401
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
