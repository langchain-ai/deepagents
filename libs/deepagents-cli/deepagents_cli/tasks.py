"""Task management middleware with file-based persistence and dependencies.

This module implements a Task system that replaces the in-memory Todo system,
providing file-based persistence, task dependencies, and cross-session collaboration.

Inspired by Anthropic's Task system for Claude Code.

## Overview

Tasks are stored in `~/.deepagents/tasks/{task_list_id}.json` and can be shared
across multiple CLI sessions or subagents using the `DEEPAGENTS_TASK_LIST_ID`
environment variable.

## Usage

```python
from deepagents_cli.tasks import TaskMiddleware

# Auto-generate task list ID
middleware = TaskMiddleware()

# Or use a specific task list for sharing
middleware = TaskMiddleware(task_list_id="my-project-tasks")

# Pass to create_cli_agent()
agent = create_cli_agent(model, assistant_id, middleware=[middleware])
```

## Environment Variable

```bash
# Share tasks across CLI sessions
DEEPAGENTS_TASK_LIST_ID=my-project deepagents
```
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, cast

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langgraph.runtime import Runtime

from filelock import FileLock
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.types import Command
from typing_extensions import NotRequired, TypedDict

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
    OmitFromInput,
)
from langchain.tools import InjectedToolCallId

logger = logging.getLogger(__name__)


class Task(TypedDict):
    """A single task item with dependencies and metadata."""

    id: str
    """Unique identifier for this task (UUID)."""

    content: str
    """The content/description of the task."""

    status: Literal["pending", "in_progress", "completed", "blocked"]
    """Current status. 'blocked' indicates task is waiting on dependencies."""

    blocked_by: NotRequired[list[str]]
    """List of task IDs this task depends on. Task is blocked until these complete."""

    created_at: str
    """ISO 8601 timestamp when task was created."""

    updated_at: str
    """ISO 8601 timestamp when task was last modified."""


class TaskList(TypedDict):
    """Container for a shareable task list."""

    id: str
    """Unique identifier for this task list."""

    tasks: list[Task]
    """List of tasks in this task list."""

    created_at: str
    """ISO 8601 timestamp when task list was created."""

    updated_at: str
    """ISO 8601 timestamp when task list was last modified."""

    version: int
    """Optimistic locking version for concurrent updates."""


class TaskState(AgentState):
    """State schema for the task middleware."""

    tasks: Annotated[NotRequired[list[Task]], OmitFromInput]
    """Current snapshot of task list (for agent visibility)."""

    task_list_id: Annotated[NotRequired[str], OmitFromInput]
    """ID of the active task list."""


class TaskStorage:
    """Handles file-based task list persistence with locking."""

    def __init__(self, tasks_dir: Path) -> None:
        """Initialize task storage.

        Args:
            tasks_dir: Directory to store task list files.
        """
        self.tasks_dir = tasks_dir
        self.tasks_dir.mkdir(parents=True, exist_ok=True)

    def _get_file_path(self, task_list_id: str) -> Path:
        """Get the file path for a task list."""
        return self.tasks_dir / f"{task_list_id}.json"

    def _get_lock_path(self, task_list_id: str) -> Path:
        """Get the lock file path for a task list."""
        return self.tasks_dir / f"{task_list_id}.lock"

    def load(self, task_list_id: str) -> TaskList | None:
        """Load a task list from file.

        Args:
            task_list_id: The task list ID to load.

        Returns:
            TaskList if found, None otherwise.
        """
        file_path = self._get_file_path(task_list_id)
        if not file_path.exists():
            return None

        lock = FileLock(self._get_lock_path(task_list_id))
        with lock:
            content = file_path.read_text(encoding="utf-8")
            return cast(TaskList, json.loads(content))

    def save(self, task_list: TaskList) -> None:
        """Save task list to file with optimistic locking.

        Args:
            task_list: The task list to save.
        """
        file_path = self._get_file_path(task_list["id"])
        lock = FileLock(self._get_lock_path(task_list["id"]))

        with lock:
            # Increment version
            task_list["version"] = task_list.get("version", 0) + 1
            task_list["updated_at"] = datetime.now(timezone.utc).isoformat()

            file_path.write_text(
                json.dumps(task_list, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    def create(self, task_list_id: str) -> TaskList:
        """Create a new empty task list.

        Args:
            task_list_id: The ID for the new task list.

        Returns:
            The newly created TaskList.
        """
        now = datetime.now(timezone.utc).isoformat()
        task_list: TaskList = {
            "id": task_list_id,
            "tasks": [],
            "created_at": now,
            "updated_at": now,
            "version": 0,
        }
        self.save(task_list)
        return task_list


def _now_iso() -> str:
    """Get current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def _compute_blocked_status(task: Task, all_tasks: list[Task]) -> Literal["pending", "in_progress", "completed", "blocked"]:
    """Compute the effective status of a task based on dependencies.

    If a task has blocked_by dependencies that are not completed,
    its status is 'blocked' regardless of what was set.

    Args:
        task: The task to compute status for.
        all_tasks: All tasks in the list (for dependency lookup).

    Returns:
        The effective status.
    """
    blocked_by = task.get("blocked_by", [])
    if not blocked_by:
        return task["status"]

    # Build a lookup of task statuses
    status_lookup = {t["id"]: t["status"] for t in all_tasks}

    # Check if all dependencies are completed
    for dep_id in blocked_by:
        dep_status = status_lookup.get(dep_id)
        if dep_status != "completed":
            return "blocked"

    # All dependencies completed. Unblock tasks that were auto-blocked.
    if task["status"] == "blocked":
        return "pending"

    # Return actual status for in-progress or completed tasks.
    return task["status"]


def _validate_dependencies(tasks: list[Task]) -> str | None:
    """Validate task dependencies for cycles and invalid references.

    Args:
        tasks: List of tasks to validate.

    Returns:
        Error message if invalid, None if valid.
    """
    task_ids = {t["id"] for t in tasks}

    # Check for invalid references
    for task in tasks:
        blocked_by = task.get("blocked_by", [])
        for dep_id in blocked_by:
            if dep_id not in task_ids:
                return f"Task '{task['id']}' references non-existent dependency '{dep_id}'"
            if dep_id == task["id"]:
                return f"Task '{task['id']}' cannot depend on itself"

    # Check for cycles using DFS
    visited: set[str] = set()
    rec_stack: set[str] = set()

    def has_cycle(task_id: str) -> bool:
        visited.add(task_id)
        rec_stack.add(task_id)

        task = next((t for t in tasks if t["id"] == task_id), None)
        if task:
            for dep_id in task.get("blocked_by", []):
                if dep_id not in visited:
                    if has_cycle(dep_id):
                        return True
                elif dep_id in rec_stack:
                    return True

        rec_stack.remove(task_id)
        return False

    for task in tasks:
        if task["id"] not in visited:
            if has_cycle(task["id"]):
                return "Circular dependency detected in task list"

    return None


WRITE_TASKS_TOOL_DESCRIPTION = """Use this tool to create and manage a structured task list with dependencies.

Tasks are persisted to disk and can be shared across multiple CLI sessions or subagents.

## When to Use This Tool
Use this tool in these scenarios:

1. Complex multi-step tasks - When a task requires 3 or more distinct steps
2. Tasks with dependencies - When some tasks must complete before others can start
3. User explicitly requests task list - When the user directly asks you to create tasks
4. User provides multiple tasks - When users provide a list of things to be done
5. Coordinating with other sessions - When multiple agents need to work on the same project

## Task Structure

Each task has:
- `id`: Unique identifier (auto-generated UUID if not provided)
- `content`: Description of what needs to be done
- `status`: One of "pending", "in_progress", "completed", or "blocked"
- `blocked_by`: Optional list of task IDs that must complete first

## Task States

- **pending**: Task not yet started
- **in_progress**: Currently working on
- **completed**: Task finished successfully
- **blocked**: Waiting for dependencies to complete (auto-computed)

## Dependencies

Use `blocked_by` to specify task dependencies:
```
{"id": "task-2", "content": "Write tests", "blocked_by": ["task-1"], "status": "pending"}
```
Tasks with incomplete dependencies are automatically marked as "blocked".

## When NOT to Use This Tool

Skip this tool when:
1. There is only a single, straightforward task
2. The task can be completed in less than 3 steps
3. The task is purely conversational or informational

Being proactive with task management demonstrates attentiveness and ensures you complete all requirements successfully."""  # noqa: E501


WRITE_TASKS_SYSTEM_PROMPT = """## `write_tasks`

You have access to the `write_tasks` tool to help you manage and plan complex objectives with dependencies.
Use this tool for complex objectives to ensure that you are tracking each necessary step and giving the user visibility into your progress.

Tasks are stored on disk and can be shared across multiple CLI sessions using the same task list ID.

## Important Task Usage Notes
- The `write_tasks` tool should never be called multiple times in parallel.
- Use `blocked_by` to define dependencies between tasks.
- Tasks with incomplete dependencies are automatically marked as "blocked".
- Don't be afraid to revise the task list as you go. New information may reveal new tasks or make old tasks irrelevant.
- When first creating tasks for a complex objective, ask the user if the plan looks good before starting work."""  # noqa: E501


class TaskMiddleware(AgentMiddleware):
    """Middleware that provides file-based task management with dependencies.

    This middleware adds `write_tasks` and `get_tasks` tools that allow agents to
    create and manage structured task lists with dependencies. Tasks are persisted
    to disk and can be shared across multiple CLI sessions or subagents.

    Args:
        task_list_id: Optional task list ID. If not provided:
            1. Checks DEEPAGENTS_TASK_LIST_ID environment variable
            2. Auto-generates a new UUID
        tasks_dir: Optional directory for task storage. Defaults to ~/.deepagents/tasks/
        system_prompt: Custom system prompt override.
        tool_description: Custom description for the write_tasks tool.
    """

    state_schema = TaskState

    def __init__(
        self,
        *,
        task_list_id: str | None = None,
        tasks_dir: Path | None = None,
        system_prompt: str = WRITE_TASKS_SYSTEM_PROMPT,
        tool_description: str = WRITE_TASKS_TOOL_DESCRIPTION,
    ) -> None:
        """Initialize the TaskMiddleware.

        Args:
            task_list_id: Optional task list ID for sharing across sessions.
            tasks_dir: Directory to store task files. Defaults to ~/.deepagents/tasks/
            system_prompt: System prompt to inject.
            tool_description: Description for the write_tasks tool.
        """
        super().__init__()
        self.system_prompt = system_prompt
        self.tool_description = tool_description

        # Resolve tasks directory
        if tasks_dir is None:
            tasks_dir = Path.home() / ".deepagents" / "tasks"
        self._tasks_dir = tasks_dir
        self._storage = TaskStorage(tasks_dir)

        # Resolve task list ID
        import os

        if task_list_id is None:
            task_list_id = os.environ.get("DEEPAGENTS_TASK_LIST_ID")
        if task_list_id is None:
            task_list_id = str(uuid.uuid4())[:8]  # Short UUID for convenience

        self._task_list_id = task_list_id
        self._last_known_version = 0

        # Create tools
        storage = self._storage
        list_id = self._task_list_id

        @tool(description=self.tool_description)
        def write_tasks(
            tasks: list[Task],
            tool_call_id: Annotated[str, InjectedToolCallId],
        ) -> Command:
            """Create and manage a structured task list with dependencies."""
            now = _now_iso()

            # Ensure all tasks have IDs and timestamps
            for task in tasks:
                if "id" not in task or not task["id"]:
                    task["id"] = str(uuid.uuid4())[:8]
                if "created_at" not in task:
                    task["created_at"] = now
                task["updated_at"] = now

            # Validate dependencies
            error = _validate_dependencies(tasks)
            if error:
                return Command(
                    update={
                        "messages": [
                            ToolMessage(
                                content=f"Error: {error}",
                                tool_call_id=tool_call_id,
                                status="error",
                            )
                        ]
                    }
                )

            # Compute effective statuses based on dependencies
            for task in tasks:
                effective_status = _compute_blocked_status(task, tasks)
                task["status"] = effective_status

            # Load or create task list
            task_list = storage.load(list_id)
            if task_list is None:
                task_list = storage.create(list_id)

            # Update task list
            task_list["tasks"] = tasks
            storage.save(task_list)

            # Format response
            task_summary = []
            for t in tasks:
                status_icon = {
                    "pending": "[ ]",
                    "in_progress": "[*]",
                    "completed": "[x]",
                    "blocked": "[!]",
                }.get(t["status"], "[ ]")
                deps = f" (blocked by: {', '.join(t.get('blocked_by', []))})" if t.get("blocked_by") else ""
                task_summary.append(f"{status_icon} {t['content']}{deps}")

            summary = "\n".join(task_summary)
            return Command(
                update={
                    "tasks": tasks,
                    "task_list_id": list_id,
                    "messages": [
                        ToolMessage(
                            content=f"Updated task list '{list_id}':\n{summary}",
                            tool_call_id=tool_call_id,
                        )
                    ],
                }
            )

        @tool(description="Refresh and return the current task list from storage. Use this to sync with updates from other sessions.")
        def get_tasks(
            tool_call_id: Annotated[str, InjectedToolCallId],
        ) -> Command:
            """Refresh and return the current task list from storage."""
            task_list = storage.load(list_id)
            if task_list is None:
                return Command(
                    update={
                        "tasks": [],
                        "task_list_id": list_id,
                        "messages": [
                            ToolMessage(
                                content=f"Task list '{list_id}' is empty or does not exist.",
                                tool_call_id=tool_call_id,
                            )
                        ],
                    }
                )

            tasks = task_list["tasks"]

            # Format response
            if not tasks:
                summary = "(No tasks)"
            else:
                task_summary = []
                for t in tasks:
                    status_icon = {
                        "pending": "[ ]",
                        "in_progress": "[*]",
                        "completed": "[x]",
                        "blocked": "[!]",
                    }.get(t["status"], "[ ]")
                    deps = f" (blocked by: {', '.join(t.get('blocked_by', []))})" if t.get("blocked_by") else ""
                    task_summary.append(f"{status_icon} {t['content']}{deps}")
                summary = "\n".join(task_summary)

            return Command(
                update={
                    "tasks": tasks,
                    "task_list_id": list_id,
                    "messages": [
                        ToolMessage(
                            content=f"Current task list '{list_id}':\n{summary}",
                            tool_call_id=tool_call_id,
                        )
                    ],
                }
            )

        self.tools = [write_tasks, get_tasks]

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """Update the system message to include the task system prompt."""
        if request.system_message is not None:
            new_system_content = [
                *request.system_message.content_blocks,
                {"type": "text", "text": f"\n\n{self.system_prompt}"},
            ]
        else:
            new_system_content = [{"type": "text", "text": self.system_prompt}]
        new_system_message = SystemMessage(
            content=cast("list[str | dict[str, str]]", new_system_content)
        )
        return handler(request.override(system_message=new_system_message))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Update the system message to include the task system prompt (async version)."""
        if request.system_message is not None:
            new_system_content = [
                *request.system_message.content_blocks,
                {"type": "text", "text": f"\n\n{self.system_prompt}"},
            ]
        else:
            new_system_content = [{"type": "text", "text": self.system_prompt}]
        new_system_message = SystemMessage(
            content=cast("list[str | dict[str, str]]", new_system_content)
        )
        return await handler(request.override(system_message=new_system_message))

    def after_model(
        self,
        state: AgentState,
        runtime: Runtime,  # noqa: ARG002
    ) -> dict[str, Any] | None:
        """Check for parallel write_tasks tool calls and return errors if detected.

        Args:
            state: The current agent state containing messages.
            runtime: The LangGraph runtime instance.

        Returns:
            A dict containing error ToolMessages if multiple parallel calls detected.
        """
        messages = state["messages"]
        if not messages:
            return None

        last_ai_msg = next((msg for msg in reversed(messages) if isinstance(msg, AIMessage)), None)
        if not last_ai_msg or not last_ai_msg.tool_calls:
            return None

        # Count write_tasks tool calls
        write_tasks_calls = [tc for tc in last_ai_msg.tool_calls if tc["name"] == "write_tasks"]

        if len(write_tasks_calls) > 1:
            error_messages = [
                ToolMessage(
                    content=(
                        "Error: The `write_tasks` tool should never be called multiple times "
                        "in parallel. Please call it only once per model invocation."
                    ),
                    tool_call_id=tc["id"],
                    status="error",
                )
                for tc in write_tasks_calls
            ]
            return {"messages": error_messages}

        return None

    async def aafter_model(
        self,
        state: AgentState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        """Async version of after_model."""
        return self.after_model(state, runtime)

    def before_model(
        self,
        state: TaskState,
        runtime: Runtime,  # noqa: ARG002
    ) -> dict[str, Any] | None:
        """Check for task list updates from other sessions before each model call.

        Args:
            state: Current agent state.
            runtime: The LangGraph runtime instance.

        Returns:
            State update with refreshed tasks if the file was modified.
        """
        task_list = self._storage.load(self._task_list_id)
        if task_list is None:
            return None

        current_version = task_list.get("version", 0)
        if current_version > self._last_known_version:
            self._last_known_version = current_version
            logger.debug(f"Task list '{self._task_list_id}' updated by another session (v{current_version})")
            return {
                "tasks": task_list["tasks"],
                "task_list_id": self._task_list_id,
            }

        return None

    async def abefore_model(
        self,
        state: TaskState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        """Async version of before_model."""
        return self.before_model(state, runtime)

    @property
    def task_list_id(self) -> str:
        """Get the current task list ID."""
        return self._task_list_id
