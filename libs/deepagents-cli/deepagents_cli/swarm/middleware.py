"""Middleware for injecting task board tools with session context."""

from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from langchain.tools import ToolRuntime
from langchain_core.tools import StructuredTool

from deepagents_cli.swarm.task_store import TaskStore
from deepagents_cli.swarm.types import TaskStatus


class TaskBoardMiddleware(AgentMiddleware):
    """Middleware that provides task board tools.

    The task board tools are scoped to a session_id which is extracted from
    the runtime configurable at invocation time.

    Usage:
        agent = create_cli_agent(...)
        result = agent.invoke(
            {"messages": [...]},
            config={"configurable": {"thread_id": "my-session-123"}}
        )
    """

    def __init__(self, task_store: TaskStore | None = None):
        super().__init__()
        self.task_store = task_store or TaskStore()

    @property
    def tools(self) -> list[StructuredTool]:
        """Return task board tools that extract session_id from runtime."""
        return _create_task_board_tools_with_runtime(self.task_store)


def _create_task_board_tools_with_runtime(task_store: TaskStore) -> list[StructuredTool]:
    """Create task board tools that get session_id from runtime config."""

    def _get_session_id(runtime: ToolRuntime) -> str:
        """Extract session_id from runtime configurable."""
        configurable = runtime.config.get("configurable", {})
        session_id = configurable.get("thread_id")
        if not session_id:
            return "default-session"
        return session_id

    async def task_create(
        subject: str,
        description: str,
        active_form: str | None = None,
        metadata: dict[str, Any] | None = None,
        runtime: ToolRuntime = None,
    ) -> str:
        """Create a new task on the task board. All tasks start as 'pending'.

        Args:
            subject: Brief title in imperative form (e.g., 'Fix auth bug')
            description: Detailed requirements for the task
            active_form: Present continuous form for display (e.g., 'Fixing auth bug')
            metadata: Arbitrary key-value data
        """
        session_id = _get_session_id(runtime)
        task = await task_store.create_task(
            session_id=session_id,
            subject=subject,
            description=description,
            active_form=active_form,
            metadata=metadata,
        )
        return f"Task #{task['id']} created successfully"

    async def task_get(
        task_id: str,
        runtime: ToolRuntime = None,
    ) -> str:
        """Get full details of a task.

        Args:
            task_id: The task ID to retrieve
        """
        session_id = _get_session_id(runtime)
        task = await task_store.get_task(session_id, task_id)
        if not task:
            return f"Task #{task_id} not found"

        lines = [
            f"Task #{task['id']}: {task['subject']}",
            f"Status: {task['status'].value}",
            f"Description: {task['description']}",
        ]
        if task["blocked_by"]:
            lines.append(
                f"Blocked by: {', '.join(f'#{t}' for t in task['blocked_by'])}"
            )
        if task["owner"]:
            lines.append(f"Owner: {task['owner']}")
        return "\n".join(lines)

    async def task_update(
        task_id: str,
        status: str | None = None,
        subject: str | None = None,
        description: str | None = None,
        active_form: str | None = None,
        owner: str | None = None,
        metadata: dict[str, Any] | None = None,
        add_blocks: list[str] | None = None,
        add_blocked_by: list[str] | None = None,
        runtime: ToolRuntime = None,
    ) -> str:
        """Update task status, details, or dependencies.

        Args:
            task_id: The task ID to update
            status: 'pending', 'in_progress', or 'completed'
            subject: New title
            description: New description
            active_form: New display text
            owner: Agent name to assign
            metadata: Metadata to merge
            add_blocks: Task IDs this task blocks
            add_blocked_by: Task IDs that block this task
        """
        session_id = _get_session_id(runtime)
        status_enum = TaskStatus(status) if status else None
        task = await task_store.update_task(
            session_id,
            task_id,
            status=status_enum,
            subject=subject,
            description=description,
            active_form=active_form,
            owner=owner,
            metadata=metadata,
            add_blocks=add_blocks,
            add_blocked_by=add_blocked_by,
        )
        return f"Task #{task_id} updated" if task else f"Task #{task_id} not found"

    async def task_list(
        runtime: ToolRuntime = None,
    ) -> str:
        """List all tasks with status."""
        session_id = _get_session_id(runtime)
        tasks = await task_store.list_tasks(session_id)
        if not tasks:
            return "No tasks"

        lines = []
        for task in tasks:
            blocked = ""
            if task["blocked_by"]:
                active_blockers = []
                for b_id in task["blocked_by"]:
                    b = await task_store.get_task(session_id, b_id)
                    if b and b["status"] != TaskStatus.COMPLETED:
                        active_blockers.append(b_id)
                if active_blockers:
                    blocked = (
                        f" [blocked by {', '.join(f'#{t}' for t in active_blockers)}]"
                    )

            lines.append(
                f"#{task['id']} [{task['status'].value}] {task['subject']}{blocked}"
            )

        return "\n".join(lines)

    return [
        StructuredTool.from_function(
            name="TaskCreate",
            coroutine=task_create,
            description="Create a new task on the task board for tracking work with dependencies.",
        ),
        StructuredTool.from_function(
            name="TaskGet",
            coroutine=task_get,
            description="Get full details of a task by ID.",
        ),
        StructuredTool.from_function(
            name="TaskUpdate",
            coroutine=task_update,
            description="Update task status/details. Workflow: pending → in_progress → completed",
        ),
        StructuredTool.from_function(
            name="TaskList",
            coroutine=task_list,
            description="List all tasks with status and blockers.",
        ),
    ]
