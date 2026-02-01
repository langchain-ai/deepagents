"""Task board and swarm execution systems for multi-agent coordination."""

from deepagents_cli.swarm.enrichment import (
    EnrichmentError,
    create_enrichment_tasks,
    merge_enrichment_results,
    parse_csv_for_enrichment,
    parse_enrichment_output,
    write_enriched_csv,
)
from deepagents_cli.swarm.executor import SwarmExecutor, get_default_output_dir
from deepagents_cli.swarm.graph import CycleError, DependencyGraph
from deepagents_cli.swarm.middleware import SwarmMiddleware, TaskBoardMiddleware
from deepagents_cli.swarm.parser import TaskFileError, parse_task_file
from deepagents_cli.swarm.task_board import create_task_board_tools
from deepagents_cli.swarm.task_store import TaskStore
from deepagents_cli.swarm.types import (
    SwarmProgress,
    SwarmResult,
    SwarmResultStatus,
    SwarmSummary,
    SwarmTask,
    Task,
    TaskStatus,
)

__all__ = [
    # Task Board (manual coordination)
    "TaskStore",
    "Task",
    "TaskStatus",
    "create_task_board_tools",
    "TaskBoardMiddleware",
    # Swarm Execution (batch parallel execution)
    "SwarmTask",
    "SwarmResult",
    "SwarmResultStatus",
    "SwarmProgress",
    "SwarmSummary",
    "SwarmMiddleware",
    "SwarmExecutor",
    "DependencyGraph",
    "CycleError",
    "TaskFileError",
    "parse_task_file",
    "get_default_output_dir",
    # CSV Enrichment
    "EnrichmentError",
    "parse_csv_for_enrichment",
    "create_enrichment_tasks",
    "parse_enrichment_output",
    "merge_enrichment_results",
    "write_enriched_csv",
]
