"""Task board tools for the main agent."""

from typing import Annotated, Any

from langchain_core.tools import StructuredTool

from deepagents_cli.swarm.task_store import TaskStore
from deepagents_cli.swarm.types import TaskStatus


def create_task_board_tools(
    task_store: TaskStore, session_id: str
) -> list[StructuredTool]:
    """Create TaskCreate, TaskGet, TaskUpdate, TaskList tools.

    Args:
        task_store: TaskStore instance for persistence
        session_id: Current session/thread ID

    Returns:
        List of 4 StructuredTool instances
    """

    async def task_create(
        subject: Annotated[
            str, "Brief title in imperative form (e.g., 'Fix auth bug')"
        ],
        description: Annotated[str, "Detailed requirements for the task"],
        active_form: Annotated[
            str | None,
            "Present continuous form for display (e.g., 'Fixing auth bug')",
        ] = None,
        metadata: Annotated[dict[str, Any] | None, "Arbitrary key-value data"] = None,
    ) -> str:
        """Create a new task on the task board. All tasks start as 'pending'."""
        task = await task_store.create_task(
            session_id=session_id,
            subject=subject,
            description=description,
            active_form=active_form,
            metadata=metadata,
        )
        return f"Task #{task['id']} created successfully"

    async def task_get(
        task_id: Annotated[str, "The task ID to retrieve"],
    ) -> str:
        """Get full details of a task."""
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
        task_id: Annotated[str, "The task ID to update"],
        status: Annotated[
            str | None, "'pending', 'in_progress', or 'completed'"
        ] = None,
        subject: Annotated[str | None, "New title"] = None,
        description: Annotated[str | None, "New description"] = None,
        active_form: Annotated[str | None, "New display text"] = None,
        owner: Annotated[str | None, "Agent name to assign"] = None,
        metadata: Annotated[dict[str, Any] | None, "Metadata to merge"] = None,
        add_blocks: Annotated[
            list[str] | None, "Task IDs this task blocks"
        ] = None,
        add_blocked_by: Annotated[
            list[str] | None, "Task IDs that block this task"
        ] = None,
    ) -> str:
        """Update task status, details, or dependencies."""
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

    async def task_list() -> str:
        """List all tasks with status."""
        tasks = await task_store.list_tasks(session_id)
        if not tasks:
            return "No tasks"

        lines = []
        for task in tasks:
            blocked = ""
            if task["blocked_by"]:
                # Show only incomplete blockers
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
            description="Create a new task on the task board.",
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
