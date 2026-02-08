"""Minimal swarm execution primitives for parallel JSONL task runs."""

from deepagents_cli.swarm.executor import (
    SwarmExecutionError,
    SwarmExecutor,
    generate_swarm_run_id,
    get_default_output_dir,
)
from deepagents_cli.swarm.middleware import SwarmMiddleware
from deepagents_cli.swarm.parser import TaskFileError, parse_task_file
from deepagents_cli.swarm.types import (
    SwarmProgress,
    SwarmResult,
    SwarmResultStatus,
    SwarmSummary,
    SwarmTask,
)

__all__ = [
    "SwarmExecutionError",
    "SwarmExecutor",
    "SwarmMiddleware",
    "SwarmProgress",
    "SwarmResult",
    "SwarmResultStatus",
    "SwarmSummary",
    "SwarmTask",
    "TaskFileError",
    "generate_swarm_run_id",
    "get_default_output_dir",
    "parse_task_file",
]
