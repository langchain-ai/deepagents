"""Unit tests for dependency graph utilities."""

import pytest

from deepagents_cli.swarm.graph import CycleError, DependencyGraph
from deepagents_cli.swarm.types import SwarmTask


def make_task(id: str, blocked_by: list[str] | None = None) -> SwarmTask:
    """Helper to create SwarmTask for testing."""
    task: SwarmTask = {"id": id, "description": f"Task {id}"}
    if blocked_by:
        task["blocked_by"] = blocked_by
    return task


class TestDependencyGraphConstruction:
    def test_creates_from_tasks(self):
        tasks = [
            make_task("1"),
            make_task("2"),
            make_task("3", blocked_by=["1", "2"]),
        ]

        graph = DependencyGraph.from_tasks(tasks)

        assert len(graph) == 3
        assert "1" in graph
        assert "2" in graph
        assert "3" in graph

    def test_empty_tasks(self):
        graph = DependencyGraph.from_tasks([])

        assert len(graph) == 0


class TestDependencyTracking:
    def test_get_dependencies(self):
        tasks = [
            make_task("1"),
            make_task("2"),
            make_task("3", blocked_by=["1", "2"]),
        ]

        graph = DependencyGraph.from_tasks(tasks)

        assert graph.get_dependencies("1") == set()
        assert graph.get_dependencies("2") == set()
        assert graph.get_dependencies("3") == {"1", "2"}

    def test_get_dependents(self):
        tasks = [
            make_task("1"),
            make_task("2"),
            make_task("3", blocked_by=["1", "2"]),
        ]

        graph = DependencyGraph.from_tasks(tasks)

        assert graph.get_dependents("1") == {"3"}
        assert graph.get_dependents("2") == {"3"}
        assert graph.get_dependents("3") == set()

    def test_get_all_downstream(self):
        """Test transitive downstream dependencies."""
        tasks = [
            make_task("1"),
            make_task("2", blocked_by=["1"]),
            make_task("3", blocked_by=["2"]),
            make_task("4", blocked_by=["3"]),
        ]

        graph = DependencyGraph.from_tasks(tasks)

        # Task 1 transitively blocks 2, 3, and 4
        assert graph.get_all_downstream("1") == {"2", "3", "4"}
        assert graph.get_all_downstream("2") == {"3", "4"}
        assert graph.get_all_downstream("3") == {"4"}
        assert graph.get_all_downstream("4") == set()


class TestIndependentTasks:
    def test_get_independent_tasks(self):
        tasks = [
            make_task("1"),
            make_task("2"),
            make_task("3", blocked_by=["1"]),
        ]

        graph = DependencyGraph.from_tasks(tasks)

        independent = graph.get_independent_tasks()

        assert set(independent) == {"1", "2"}

    def test_all_independent(self):
        tasks = [make_task("1"), make_task("2"), make_task("3")]

        graph = DependencyGraph.from_tasks(tasks)

        assert set(graph.get_independent_tasks()) == {"1", "2", "3"}


class TestReadyTasks:
    def test_ready_tasks_initial(self):
        tasks = [
            make_task("1"),
            make_task("2"),
            make_task("3", blocked_by=["1", "2"]),
        ]

        graph = DependencyGraph.from_tasks(tasks)

        ready = graph.get_ready_tasks(tasks, completed=set())

        assert set(ready) == {"1", "2"}

    def test_ready_tasks_after_completion(self):
        tasks = [
            make_task("1"),
            make_task("2"),
            make_task("3", blocked_by=["1", "2"]),
        ]

        graph = DependencyGraph.from_tasks(tasks)

        # After completing task 1
        ready = graph.get_ready_tasks(tasks, completed={"1"})
        assert "2" in ready
        assert "3" not in ready  # Still blocked by 2

        # After completing both 1 and 2
        ready = graph.get_ready_tasks(tasks, completed={"1", "2"})
        assert "3" in ready


class TestTopologicalSort:
    def test_topological_sort_simple(self):
        tasks = [
            make_task("3", blocked_by=["1", "2"]),
            make_task("1"),
            make_task("2"),
        ]

        graph = DependencyGraph.from_tasks(tasks)
        order = graph.topological_sort()

        # Task 3 must come after 1 and 2
        assert order.index("3") > order.index("1")
        assert order.index("3") > order.index("2")

    def test_topological_sort_chain(self):
        tasks = [
            make_task("4", blocked_by=["3"]),
            make_task("3", blocked_by=["2"]),
            make_task("2", blocked_by=["1"]),
            make_task("1"),
        ]

        graph = DependencyGraph.from_tasks(tasks)
        order = graph.topological_sort()

        assert order == ["1", "2", "3", "4"]


class TestCycleDetection:
    def test_detects_direct_cycle(self):
        tasks = [
            make_task("1", blocked_by=["2"]),
            make_task("2", blocked_by=["1"]),
        ]

        with pytest.raises(CycleError) as exc:
            DependencyGraph.from_tasks(tasks)

        # Cycle should include both tasks
        assert "1" in exc.value.cycle
        assert "2" in exc.value.cycle

    def test_detects_indirect_cycle(self):
        tasks = [
            make_task("1", blocked_by=["3"]),
            make_task("2", blocked_by=["1"]),
            make_task("3", blocked_by=["2"]),
        ]

        with pytest.raises(CycleError):
            DependencyGraph.from_tasks(tasks)

    def test_detects_self_cycle(self):
        tasks = [make_task("1", blocked_by=["1"])]

        with pytest.raises(CycleError):
            DependencyGraph.from_tasks(tasks)

    def test_no_cycle_error_message(self):
        tasks = [
            make_task("A", blocked_by=["B"]),
            make_task("B", blocked_by=["A"]),
        ]

        with pytest.raises(CycleError) as exc:
            DependencyGraph.from_tasks(tasks)

        assert "Circular dependency" in str(exc.value)
