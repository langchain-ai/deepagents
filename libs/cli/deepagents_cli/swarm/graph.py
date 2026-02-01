"""Dependency graph utilities for swarm task execution."""

from collections import deque

from deepagents_cli.swarm.types import SwarmTask


class CycleError(Exception):
    """Raised when a cycle is detected in the dependency graph."""

    def __init__(self, cycle: list[str]):
        self.cycle = cycle
        cycle_str = " -> ".join(cycle + [cycle[0]])
        super().__init__(f"Circular dependency detected: {cycle_str}")


class DependencyGraph:
    """Graph representation of task dependencies.

    Tracks which tasks depend on which other tasks and provides utilities
    for dependency-aware scheduling.
    """

    def __init__(self) -> None:
        # Map task_id -> set of task_ids that this task depends on
        self._dependencies: dict[str, set[str]] = {}
        # Map task_id -> set of task_ids that depend on this task
        self._dependents: dict[str, set[str]] = {}
        # All task IDs in the graph
        self._tasks: set[str] = set()

    @classmethod
    def from_tasks(cls, tasks: list[SwarmTask]) -> "DependencyGraph":
        """Build a dependency graph from a list of tasks.

        Args:
            tasks: List of SwarmTask objects.

        Returns:
            DependencyGraph instance.

        Raises:
            CycleError: If circular dependencies are detected.
        """
        graph = cls()

        # First pass: register all tasks
        for task in tasks:
            graph._tasks.add(task["id"])
            graph._dependencies[task["id"]] = set()
            graph._dependents[task["id"]] = set()

        # Second pass: build dependency edges
        for task in tasks:
            task_id = task["id"]
            blocked_by = task.get("blocked_by", [])

            for dep_id in blocked_by:
                graph._dependencies[task_id].add(dep_id)
                graph._dependents[dep_id].add(task_id)

        # Validate no cycles
        graph._detect_cycles()

        return graph

    def _detect_cycles(self) -> None:
        """Detect cycles in the dependency graph using DFS.

        Raises:
            CycleError: If a cycle is detected.
        """
        # Track visit state: 0 = unvisited, 1 = in progress, 2 = done
        state: dict[str, int] = {task_id: 0 for task_id in self._tasks}
        path: list[str] = []

        def dfs(task_id: str) -> None:
            if state[task_id] == 2:
                return
            if state[task_id] == 1:
                # Found cycle - extract it from path
                cycle_start = path.index(task_id)
                raise CycleError(path[cycle_start:])

            state[task_id] = 1
            path.append(task_id)

            for dep_id in self._dependents.get(task_id, set()):
                dfs(dep_id)

            path.pop()
            state[task_id] = 2

        for task_id in self._tasks:
            if state[task_id] == 0:
                dfs(task_id)

    def get_dependencies(self, task_id: str) -> set[str]:
        """Get task IDs that this task depends on (blocked_by).

        Args:
            task_id: ID of the task.

        Returns:
            Set of task IDs that must complete before this task.
        """
        return self._dependencies.get(task_id, set()).copy()

    def get_dependents(self, task_id: str) -> set[str]:
        """Get task IDs that depend on this task.

        Args:
            task_id: ID of the task.

        Returns:
            Set of task IDs that are blocked by this task.
        """
        return self._dependents.get(task_id, set()).copy()

    def get_all_downstream(self, task_id: str) -> set[str]:
        """Get all tasks that transitively depend on this task.

        Uses BFS to find all downstream tasks.

        Args:
            task_id: ID of the task.

        Returns:
            Set of all task IDs that directly or transitively depend on this task.
        """
        downstream: set[str] = set()
        queue = deque(self._dependents.get(task_id, set()))

        while queue:
            current = queue.popleft()
            if current in downstream:
                continue
            downstream.add(current)
            queue.extend(self._dependents.get(current, set()) - downstream)

        return downstream

    def get_ready_tasks(self, tasks: list[SwarmTask], completed: set[str]) -> list[str]:
        """Get task IDs that are ready to execute.

        A task is ready if all its dependencies are in the completed set.

        Args:
            tasks: List of all tasks.
            completed: Set of task IDs that have already completed.

        Returns:
            List of task IDs ready for execution.
        """
        ready = []
        for task in tasks:
            task_id = task["id"]
            if task_id in completed:
                continue
            deps = self._dependencies.get(task_id, set())
            if deps.issubset(completed):
                ready.append(task_id)
        return ready

    def get_independent_tasks(self) -> list[str]:
        """Get task IDs with no dependencies.

        Returns:
            List of task IDs that can start immediately.
        """
        return [
            task_id
            for task_id in self._tasks
            if not self._dependencies.get(task_id)
        ]

    def topological_sort(self) -> list[str]:
        """Return task IDs in topological order (dependencies first).

        Uses Kahn's algorithm.

        Returns:
            List of task IDs in execution order.
        """
        # Compute in-degree for each task
        in_degree = {
            task_id: len(self._dependencies.get(task_id, set()))
            for task_id in self._tasks
        }

        # Start with tasks that have no dependencies
        queue = deque(task_id for task_id, degree in in_degree.items() if degree == 0)
        result: list[str] = []

        while queue:
            current = queue.popleft()
            result.append(current)

            for dependent in self._dependents.get(current, set()):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        return result

    def __len__(self) -> int:
        """Return the number of tasks in the graph."""
        return len(self._tasks)

    def __contains__(self, task_id: str) -> bool:
        """Check if a task ID is in the graph."""
        return task_id in self._tasks
