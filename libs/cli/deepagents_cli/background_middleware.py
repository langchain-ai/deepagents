"""Agent middleware for background task tools and status reminders."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool, StructuredTool

if TYPE_CHECKING:
    from langgraph.runtime import Runtime

    from deepagents_cli.background_runtime import BackgroundRuntime


class BackgroundMiddlewareState(AgentState):
    """State schema placeholder for background middleware."""


class BackgroundMiddleware(AgentMiddleware):
    """Expose background-task management tools and inject task updates."""

    state_schema = BackgroundMiddlewareState

    def __init__(self, runtime: BackgroundRuntime) -> None:
        """Initialize middleware with a shared runtime instance."""
        self._runtime = runtime
        self.tools: list[BaseTool] = [
            self._build_submit_tool(),
            self._build_list_tool(),
            self._build_kill_tool(),
            self._build_wait_tool(),
        ]

    def before_model(
        self,
        state: BackgroundMiddlewareState,  # noqa: ARG002  # required by interface
        runtime: Runtime[Any],  # noqa: ARG002  # required by interface
    ) -> dict[str, Any] | None:
        """Inject runtime status updates into model input when available.

        Returns:
            Message update payload when updates are available, else `None`.
        """
        updates = self._runtime.consume_status_updates()
        if not updates:
            return None

        lines = ["[SYSTEM][BACKGROUND] Recent background task updates:"]
        lines.extend(f"- {line}" for line in updates)
        if any("running" in item.lower() for item in updates):
            lines.append(
                "[SYSTEM][BACKGROUND] If you depend on results, call "
                "`wait_background_task` before proceeding."
            )

        return {"messages": [HumanMessage(content="\n".join(lines))]}

    def _build_submit_tool(self) -> BaseTool:
        async def _submit_background_task(command: str) -> str:
            task_id = await self._runtime.submit_shell_task(command)
            return f"Background task started: task_id={task_id}, status=started"

        return StructuredTool.from_function(
            name="submit_background_task",
            description=(
                "Run a shell command as a background task and return task_id "
                "immediately."
            ),
            coroutine=_submit_background_task,
        )

    def _build_list_tool(self) -> BaseTool:
        def _list_background_tasks() -> str:
            records = self._runtime.list_tasks()
            if not records:
                return "No background tasks found."

            lines = ["Background tasks:"]
            lines.extend(
                f"- {record.task_id}: status={record.status}, command={record.command}"
                for record in records
            )
            return "\n".join(lines)

        return StructuredTool.from_function(
            name="list_background_tasks",
            description="List all known background tasks and their statuses.",
            func=_list_background_tasks,
        )

    def _build_kill_tool(self) -> BaseTool:
        async def _kill_background_task(task_id: str) -> str:
            killed = await self._runtime.kill_task(task_id)
            if not killed:
                return (
                    f"Background task {task_id} was not killed "
                    "(not found or already finished)."
                )
            return (
                f"Background task {task_id} marked as killed "
                "(best-effort cancellation)."
            )

        return StructuredTool.from_function(
            name="kill_background_task",
            description="Best-effort kill of a background task by task_id.",
            coroutine=_kill_background_task,
        )

    def _build_wait_tool(self) -> BaseTool:
        async def _wait_background_task(
            task_id: str,
            timeout_seconds: float | None = 30.0,
        ) -> str:
            try:
                record = await self._runtime.wait_task(
                    task_id,
                    timeout_seconds=timeout_seconds,
                )
            except TimeoutError:
                current = self._runtime.get_task(task_id)
                if current is None:
                    return f"Unknown background task: {task_id}"
                return (
                    f"Background task {task_id} has not finished within "
                    f"{timeout_seconds:.0f}s and is still {current.status}."
                )
            except ValueError as exc:
                return str(exc)

            summary = [f"Background task {record.task_id} completed."]
            summary.append(f"status={record.status}")
            if record.exit_code is not None:
                summary.append(f"exit_code={record.exit_code}")
            if record.result_text:
                summary.append(f"stdout:\n{record.result_text}")
            if record.error_text:
                summary.append(f"stderr:\n{record.error_text}")
            return "\n".join(summary)

        return StructuredTool.from_function(
            name="wait_background_task",
            description=(
                "Wait for a background task to complete and return final result/status."
            ),
            coroutine=_wait_background_task,
        )


__all__ = ["BackgroundMiddleware", "BackgroundMiddlewareState"]
