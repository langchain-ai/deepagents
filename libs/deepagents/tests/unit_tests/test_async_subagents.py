"""Tests for async subagent middleware functionality."""

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from deepagents.middleware.async_subagents import (
    AsyncSubAgent,
    AsyncSubAgentMiddleware,
    _build_async_subagent_tools,
    _parse_job_id,
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


class TestAsyncSubAgentMiddleware:
    def test_init_requires_at_least_one_agent(self) -> None:
        with pytest.raises(ValueError, match="At least one async subagent"):
            AsyncSubAgentMiddleware(async_subagents=[])

    def test_init_creates_three_tools(self) -> None:
        mw = AsyncSubAgentMiddleware(async_subagents=[_make_spec()])
        tool_names = {t.name for t in mw.tools}
        assert tool_names == {"launch_async_subagent", "check_async_subagent", "update_async_subagent"}

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


class TestParseJobId:
    def test_canonical_format(self) -> None:
        assert _parse_job_id("thread_abc::run_xyz") == ("thread_abc", "run_xyz")

    def test_full_launch_output(self) -> None:
        assert _parse_job_id("Launched async subagent. job_id: thread_abc::run_xyz") == (
            "thread_abc",
            "run_xyz",
        )

    def test_legacy_json_format(self) -> None:
        job_id = json.dumps({"thread_id": "thread_abc", "run_id": "run_xyz"})
        assert _parse_job_id(job_id) == ("thread_abc", "run_xyz")

    def test_whitespace_stripped(self) -> None:
        assert _parse_job_id("  thread_abc::run_xyz  ") == ("thread_abc", "run_xyz")

    def test_invalid_format_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid job_id format"):
            _parse_job_id("not-a-valid-job-id")


class TestBuildAsyncSubagentTools:
    def test_returns_three_tools(self) -> None:
        tools = _build_async_subagent_tools([_make_spec()])
        assert len(tools) == 3
        names = [t.name for t in tools]
        assert names == ["launch_async_subagent", "check_async_subagent", "update_async_subagent"]

    def test_launch_description_includes_agent_info(self) -> None:
        tools = _build_async_subagent_tools([_make_spec("researcher", description="Research agent")])
        launch_tool = tools[0]
        assert "researcher" in launch_tool.description
        assert "Research agent" in launch_tool.description


class TestLaunchTool:
    def test_launch_invalid_type_returns_error(self) -> None:
        tools = _build_async_subagent_tools([_make_spec("alpha")])
        launch = tools[0]
        result = launch.invoke({"description": "do something", "subagent_type": "nonexistent"})
        assert "Unknown async subagent type" in result
        assert "`alpha`" in result

    @patch("deepagents.middleware.async_subagents.get_sync_client")
    def test_launch_creates_thread_and_run(self, mock_get_client) -> None:
        mock_client = MagicMock()
        mock_client.threads.create.return_value = {"thread_id": "thread_abc"}
        mock_client.runs.create.return_value = {"run_id": "run_xyz"}
        mock_get_client.return_value = mock_client

        tools = _build_async_subagent_tools([_make_spec("alpha")])
        launch = tools[0]
        result = launch.invoke({"description": "analyze data", "subagent_type": "alpha"})

        mock_get_client.assert_called_once_with(
            url="http://localhost:8123",
            headers={"x-auth-scheme": "langsmith"},
        )

        assert "thread_abc::run_xyz" in result
        assert "job_id:" in result

        mock_client.threads.create.assert_called_once()
        mock_client.runs.create.assert_called_once_with(
            thread_id="thread_abc",
            assistant_id="my_graph",
            input={"messages": [{"role": "user", "content": "analyze data"}]},
        )


class TestCheckTool:
    @patch("deepagents.middleware.async_subagents.get_sync_client")
    def test_check_running_job(self, mock_get_client) -> None:
        mock_client = MagicMock()
        mock_client.runs.get.return_value = {
            "run_id": "run_xyz",
            "status": "running",
        }
        mock_get_client.return_value = mock_client

        tools = _build_async_subagent_tools([_make_spec()])
        check = tools[1]
        result = check.invoke({"job_id": "thread_abc::run_xyz"})

        parsed = json.loads(result)
        assert parsed["status"] == "running"
        assert parsed["thread_id"] == "thread_abc"

    @patch("deepagents.middleware.async_subagents.get_sync_client")
    def test_check_completed_job_returns_result(self, mock_get_client) -> None:
        mock_client = MagicMock()
        mock_client.runs.get.return_value = {
            "run_id": "run_xyz",
            "status": "success",
        }
        mock_client.threads.get_state.return_value = {
            "values": {
                "messages": [
                    {"role": "assistant", "content": "Analysis complete: found 3 issues."},
                ]
            }
        }
        mock_get_client.return_value = mock_client

        tools = _build_async_subagent_tools([_make_spec()])
        check = tools[1]
        result = check.invoke({"job_id": "thread_abc::run_xyz"})

        parsed = json.loads(result)
        assert parsed["status"] == "success"
        assert parsed["result"] == "Analysis complete: found 3 issues."

    @patch("deepagents.middleware.async_subagents.get_sync_client")
    def test_check_errored_job(self, mock_get_client) -> None:
        mock_client = MagicMock()
        mock_client.runs.get.return_value = {
            "run_id": "run_xyz",
            "status": "error",
        }
        mock_get_client.return_value = mock_client

        tools = _build_async_subagent_tools([_make_spec()])
        check = tools[1]
        result = check.invoke({"job_id": "thread_abc::run_xyz"})

        parsed = json.loads(result)
        assert parsed["status"] == "error"
        assert "error" in parsed


class TestUpdateTool:
    @patch("deepagents.middleware.async_subagents.get_sync_client")
    def test_update_sends_message(self, mock_get_client) -> None:
        mock_client = MagicMock()
        mock_client.threads.update_state.return_value = None
        mock_get_client.return_value = mock_client

        tools = _build_async_subagent_tools([_make_spec()])
        update = tools[2]
        result = update.invoke({"job_id": "thread_abc::run_xyz", "update": "Focus on security issues only"})

        parsed = json.loads(result)
        assert parsed["status"] == "updated"
        assert parsed["thread_id"] == "thread_abc"

        mock_client.threads.update_state.assert_called_once_with(
            thread_id="thread_abc",
            values={"messages": [{"role": "user", "content": "Focus on security issues only"}]},
        )


def _async_return(value):
    """Create an async function that returns a fixed value."""

    async def _inner(*_args: Any, **_kwargs: Any) -> Any:  # noqa: ANN401
        return value

    return _inner


@pytest.mark.allow_hosts(["127.0.0.1", "::1"])
class TestAsyncTools:
    @patch("deepagents.middleware.async_subagents.get_client")
    async def test_async_launch_creates_thread_and_run(self, mock_get_client) -> None:
        mock_client = MagicMock()
        mock_client.threads.create = _async_return({"thread_id": "thread_abc"})
        mock_client.runs.create = _async_return({"run_id": "run_xyz"})
        mock_get_client.return_value = mock_client

        tools = _build_async_subagent_tools([_make_spec("alpha")])
        launch = tools[0]
        result = await launch.ainvoke({"description": "analyze data", "subagent_type": "alpha"})

        assert "thread_abc::run_xyz" in result
        assert "job_id:" in result

    @patch("deepagents.middleware.async_subagents.get_client")
    async def test_async_check_completed_job(self, mock_get_client) -> None:
        mock_client = MagicMock()
        mock_client.runs.get = _async_return({"run_id": "run_xyz", "status": "success"})
        mock_client.threads.get_state = _async_return({"values": {"messages": [{"role": "assistant", "content": "Done!"}]}})
        mock_get_client.return_value = mock_client

        tools = _build_async_subagent_tools([_make_spec()])
        check = tools[1]
        result = await check.ainvoke({"job_id": "thread_abc::run_xyz"})

        parsed = json.loads(result)
        assert parsed["status"] == "success"
        assert parsed["result"] == "Done!"

    @patch("deepagents.middleware.async_subagents.get_client")
    async def test_async_update_sends_message(self, mock_get_client) -> None:
        mock_client = MagicMock()
        mock_client.threads.update_state = _async_return(None)
        mock_get_client.return_value = mock_client

        tools = _build_async_subagent_tools([_make_spec()])
        update = tools[2]
        result = await update.ainvoke({"job_id": "thread_abc::run_xyz", "update": "New instructions"})

        parsed = json.loads(result)
        assert parsed["status"] == "updated"
