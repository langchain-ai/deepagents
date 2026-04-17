"""Virtual-table form: synthesize SwarmTaskSpec per file resolved from the backend.

Ported from ``libs/deepagents/src/swarm/virtual-table.ts``. The JS port
uses ``backend.glob`` (the V2 protocol). In Python we use the async
``backend.aglob`` from :class:`deepagents.backends.protocol.BackendProtocol`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

from deepagents_repl._swarm.parse import serialize_tasks_jsonl
from deepagents_repl._swarm.types import SwarmTaskSpec

if TYPE_CHECKING:
    from deepagents.backends.protocol import BackendProtocol


@dataclass
class VirtualTableInput:
    """Resolved input from the swarm call for the virtual-table form."""

    instruction: str
    """Shared instruction prepended to each file's description."""

    file_paths: list[str] | None = None
    """Explicit file paths to process."""

    glob: str | list[str] | None = None
    """Glob pattern(s) to match files."""

    subagent_type: str | None = None
    """Subagent type for all synthesized tasks."""


@dataclass
class VirtualTableResolution:
    """Result of the virtual-table resolver.

    On success both ``tasks`` and ``tasks_jsonl`` are populated; on
    failure ``error`` is populated and the others are ``None``. Mirrors
    the discriminated-union in the JS port without the awkwardness of
    a Python union type.
    """

    tasks: list[SwarmTaskSpec] | None = None
    tasks_jsonl: str | None = None
    error: str | None = None


async def resolve_virtual_table_tasks(
    input_: VirtualTableInput,
    backend: BackendProtocol,
) -> VirtualTableResolution:
    """Resolve the virtual-table input form into SwarmTaskSpec list.

    Steps:
        1. Resolve file paths from explicit ``file_paths`` and/or
           ``glob`` patterns.
        2. Synthesize one SwarmTaskSpec per file.

    Returns a :class:`VirtualTableResolution` with ``error`` set (rather
    than raising) so the swarm tool handler can pass the string through
    to the orchestrator as normal tool output.
    """
    resolved: set[str] = set()

    if input_.file_paths:
        resolved.update(input_.file_paths)

    if input_.glob is not None:
        patterns = [input_.glob] if isinstance(input_.glob, str) else list(input_.glob)
        for pattern in patterns:
            glob_result = await backend.aglob(pattern)
            if glob_result.error is not None:
                return VirtualTableResolution(
                    error=f'Glob pattern "{pattern}" failed: {glob_result.error}'
                )
            for match in glob_result.matches or []:
                resolved.add(match["path"])

    if not resolved:
        if input_.glob is not None:
            pattern_desc = f"glob pattern(s): {_json_like(input_.glob)}"
        else:
            pattern_desc = f"files: {_json_like(input_.file_paths)}"
        return VirtualTableResolution(error=f"No files matched {pattern_desc}")

    sorted_paths = sorted(resolved)
    task_ids = _build_task_ids(sorted_paths)
    tasks = [
        SwarmTaskSpec(
            id=task_ids[path],
            description=f"{input_.instruction}\n\nFile: {path}",
            subagent_type=input_.subagent_type,
        )
        for path in sorted_paths
    ]
    return VirtualTableResolution(tasks=tasks, tasks_jsonl=serialize_tasks_jsonl(tasks))


def _build_task_ids(file_paths: list[str]) -> dict[str, str]:
    """Build a unique task ID from each file path.

    Disambiguates basename collisions by prepending the parent directory
    name (``en/readme.md`` → ``en-readme.md`` if ``fr/readme.md`` also
    exists in the input). Mirrors the JS port's semantics.
    """
    basename_counts: dict[str, int] = {}
    for path in file_paths:
        base = os.path.basename(path)
        basename_counts[base] = basename_counts.get(base, 0) + 1

    ids: dict[str, str] = {}
    for path in file_paths:
        base = os.path.basename(path)
        if basename_counts[base] > 1:
            parent = os.path.basename(os.path.dirname(path))
            ids[path] = f"{parent}-{base}"
        else:
            ids[path] = base
    return ids


def _json_like(value: object) -> str:
    """Render a str/list the same way JSON.stringify does for error messages."""
    import json

    return json.dumps(value)
