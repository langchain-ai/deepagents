"""Tests for background middleware tools and model hook behavior."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import HumanMessage

from deepagents_cli.background_middleware import BackgroundMiddleware
from deepagents_cli.background_runtime import (
    BackgroundRuntime,
    BackgroundTaskRecord,
    BackgroundTaskStatus,
)

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool
    from langgraph.runtime import Runtime


def _find_tool(middleware: BackgroundMiddleware, name: str) -> BaseTool:
    for tool in middleware.tools:
        if tool.name == name:
            return tool
    msg = f"Tool not found: {name}"
    raise AssertionError(msg)


class TestBackgroundMiddleware:
    """Unit tests for BackgroundMiddleware."""

    async def test_tools_are_injected(self) -> None:
        runtime = BackgroundRuntime(require_hitl_for_shell=False)
        middleware = BackgroundMiddleware(runtime)
        names = {tool.name for tool in middleware.tools}
        assert names == {
            "submit_background_task",
            "list_background_tasks",
            "kill_background_task",
            "wait_background_task",
        }

    async def test_submit_tool_returns_started_status(self) -> None:
        runtime = BackgroundRuntime(require_hitl_for_shell=False)
        await runtime.start()
        try:
            middleware = BackgroundMiddleware(runtime)
            submit_tool = _find_tool(middleware, "submit_background_task")
            output = await submit_tool.ainvoke(cast("Any", {"command": "printf hi"}))
            assert "status=queued" in output
            assert "task_id=" in output
            task_id = output.split("task_id=", maxsplit=1)[1].split(",", maxsplit=1)[0]
            await runtime.wait_task(task_id, timeout_seconds=5)
        finally:
            await runtime.shutdown()

    async def test_wait_tool_calls_runtime_wait(self) -> None:
        runtime = BackgroundRuntime(require_hitl_for_shell=False)
        middleware = BackgroundMiddleware(runtime)
        wait_tool = _find_tool(middleware, "wait_background_task")

        wait_mock = AsyncMock(
            return_value=BackgroundTaskRecord(
                task_id="abc",
                command="printf x",
                status=BackgroundTaskStatus.SUCCEEDED,
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
                result_text="x",
                exit_code=0,
            )
        )
        runtime.wait_task = wait_mock  # type: ignore[method-assign]
        output = await wait_tool.ainvoke(cast("Any", {"task_id": "abc"}))
        wait_mock.assert_awaited_once_with("abc", timeout_seconds=30.0)
        assert "status=succeeded" in output

    async def test_wait_tool_reports_rejected_by_user(self) -> None:
        runtime = BackgroundRuntime(require_hitl_for_shell=False)
        middleware = BackgroundMiddleware(runtime)
        wait_tool = _find_tool(middleware, "wait_background_task")

        wait_mock = AsyncMock(
            return_value=BackgroundTaskRecord(
                task_id="abc",
                command="printf x",
                status=BackgroundTaskStatus.REJECTED,
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
                stderr_text="Rejected by user",
            )
        )
        runtime.wait_task = wait_mock  # type: ignore[method-assign]
        output = await wait_tool.ainvoke(cast("Any", {"task_id": "abc"}))
        assert "rejected by user before execution" in output.lower()
        assert "status=rejected" in output
        assert "reason:\nRejected by user" in output
        assert "completed" not in output.lower()

    async def test_wait_tool_timeout_reports_still_running(self) -> None:
        runtime = BackgroundRuntime(require_hitl_for_shell=False)
        middleware = BackgroundMiddleware(runtime)
        wait_tool = _find_tool(middleware, "wait_background_task")

        wait_mock = AsyncMock(side_effect=TimeoutError())
        runtime.wait_task = wait_mock  # type: ignore[method-assign]
        runtime.get_task = MagicMock(  # type: ignore[method-assign]
            return_value=BackgroundTaskRecord(
                task_id="abc",
                command="sleep 10",
                status=BackgroundTaskStatus.RUNNING,
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
            )
        )
        output = await wait_tool.ainvoke(cast("Any", {"task_id": "abc"}))
        wait_mock.assert_awaited_once_with("abc", timeout_seconds=30.0)
        assert "has not finished within 30s" in output
        assert "still running" in output

    async def test_before_model_injects_background_updates(self) -> None:
        runtime = BackgroundRuntime(require_hitl_for_shell=False)
        middleware = BackgroundMiddleware(runtime)
        runtime.consume_status_updates()
        runtime._pending_updates.append("Task `a1` succeeded.")

        result = middleware.before_model(state={"messages": []}, runtime=None)  # type: ignore[arg-type]
        assert result is not None
        messages = result.get("messages")
        assert isinstance(messages, list)
        assert isinstance(messages[0], HumanMessage)
        assert "[SYSTEM][BACKGROUND]" in str(messages[0].content)

    async def test_abefore_model_blocks_when_pending_hitl_exists(self) -> None:
        runtime = BackgroundRuntime(require_hitl_for_shell=False)
        middleware = BackgroundMiddleware(runtime)

        wait_for_no_pending_hitl = AsyncMock()
        runtime.pending_hitl_count = MagicMock(return_value=1)  # type: ignore[method-assign]
        runtime.wait_for_no_pending_hitl = wait_for_no_pending_hitl  # type: ignore[method-assign]
        runtime.consume_status_updates = MagicMock(  # type: ignore[method-assign]
            return_value=["Task `a1` succeeded."]
        )

        result = await middleware.abefore_model(
            state={"messages": []},
            runtime=cast("Runtime[Any]", MagicMock()),
        )

        wait_for_no_pending_hitl.assert_awaited_once()
        assert result is not None
        messages = result.get("messages")
        assert isinstance(messages, list)
        assert isinstance(messages[0], HumanMessage)

    async def test_abefore_model_skips_wait_when_no_pending_hitl(self) -> None:
        runtime = BackgroundRuntime(require_hitl_for_shell=False)
        middleware = BackgroundMiddleware(runtime)

        wait_for_no_pending_hitl = AsyncMock()
        runtime.pending_hitl_count = MagicMock(return_value=0)  # type: ignore[method-assign]
        runtime.wait_for_no_pending_hitl = wait_for_no_pending_hitl  # type: ignore[method-assign]
        runtime.consume_status_updates = MagicMock(return_value=[])  # type: ignore[method-assign]

        result = await middleware.abefore_model(
            state={"messages": []},
            runtime=cast("Runtime[Any]", MagicMock()),
        )

        wait_for_no_pending_hitl.assert_not_called()
        assert result is None
