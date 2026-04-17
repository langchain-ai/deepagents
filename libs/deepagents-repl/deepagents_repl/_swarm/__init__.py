"""Swarm: parallel subagent fan-out invoked from the REPL.

Ported from ``libs/deepagents/src/swarm/*`` in deepagentsjs.
"""

from deepagents_repl._swarm.compile import compile_subagents
from deepagents_repl._swarm.executor import SwarmExecutionOptions, execute_swarm
from deepagents_repl._swarm.parse import (
    parse_tasks_jsonl,
    serialize_results_jsonl,
    serialize_tasks_jsonl,
)
from deepagents_repl._swarm.types import (
    DEFAULT_CONCURRENCY,
    MAX_CONCURRENCY,
    TASK_TIMEOUT_SECONDS,
    FailedTaskInfo,
    SwarmExecutionSummary,
    SwarmTaskResult,
    SwarmTaskSpec,
)
from deepagents_repl._swarm.virtual_table import (
    VirtualTableInput,
    VirtualTableResolution,
    resolve_virtual_table_tasks,
)

__all__ = [
    "DEFAULT_CONCURRENCY",
    "MAX_CONCURRENCY",
    "TASK_TIMEOUT_SECONDS",
    "FailedTaskInfo",
    "SwarmExecutionOptions",
    "SwarmExecutionSummary",
    "SwarmTaskResult",
    "SwarmTaskSpec",
    "VirtualTableInput",
    "VirtualTableResolution",
    "compile_subagents",
    "execute_swarm",
    "parse_tasks_jsonl",
    "resolve_virtual_table_tasks",
    "serialize_results_jsonl",
    "serialize_tasks_jsonl",
]
