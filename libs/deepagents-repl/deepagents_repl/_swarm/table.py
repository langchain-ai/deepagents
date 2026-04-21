"""Table materialisation — build a JSONL table from files or inline rows."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from deepagents_repl._swarm.parse import serialize_table_jsonl

if TYPE_CHECKING:
    from deepagents.backends.protocol import BackendProtocol

    from deepagents_repl._swarm.types import CreateTableSource


WriteCallback = Callable[[str, str], None]
"""Synchronous callback that persists ``(path, content)``.

Abstracts over direct backend writes vs. session pending-writes so the
same ``create_table`` / ``execute_swarm`` logic works in either context.
"""


def _build_task_ids(file_paths: list[str]) -> dict[str, str]:
    """Build a unique task ID from each file path.

    Disambiguates basename collisions by prepending the parent directory
    name (``en/readme.md`` → ``en-readme.md`` if ``fr/readme.md`` also
    exists in the input).
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


async def _resolve_file_paths(
    source: CreateTableSource,
    backend: BackendProtocol,
) -> list[str]:
    """Collect unique file paths from explicit paths + glob patterns.

    Returns a sorted list. Raises ``ValueError`` if nothing matches or
    if a glob pattern errors out on the backend.
    """
    resolved: set[str] = set()

    if source.file_paths:
        resolved.update(source.file_paths)

    if source.glob is not None:
        patterns = [source.glob] if isinstance(source.glob, str) else list(source.glob)
        for raw in patterns:
            # Strip leading slashes so the pattern is interpreted
            # relative to the backend's root.
            pattern = raw.lstrip("/")
            result = await backend.aglob(pattern)
            if result.error is not None:
                msg = f'Glob pattern "{pattern}" failed: {result.error}'
                raise ValueError(msg)
            for match in result.matches or []:
                resolved.add(match["path"])

    if not resolved:
        import json as _json

        desc = (
            f"glob: {_json.dumps(source.glob)}"
            if source.glob is not None
            else f"file_paths: {_json.dumps(source.file_paths)}"
        )
        msg = f"No files matched {desc}"
        raise ValueError(msg)

    return sorted(resolved)


async def create_table(
    file: str,
    source: CreateTableSource,
    backend: BackendProtocol,
    write: WriteCallback,
) -> None:
    """Materialise a JSONL table at ``file`` from a :class:`CreateTableSource`.

    ``glob`` / ``file_paths`` produce rows shaped ``{id, file}`` where
    ``id`` is the file's basename (disambiguated by parent dir when
    basenames collide). ``tasks`` passes rows through as-is but requires
    each to carry a string ``id``.

    The resulting JSONL is written through ``write`` so the caller can
    route it into a pending-writes buffer (for read-after-write within a
    single REPL eval) or directly to the backend.
    """
    has_glob = source.glob is not None
    has_file_paths = bool(source.file_paths)
    has_tasks = bool(source.tasks)

    if not (has_glob or has_file_paths or has_tasks):
        msg = (
            "swarm.create: source must provide at least one of "
            "`glob`, `file_paths`, or `tasks`."
        )
        raise ValueError(msg)

    rows: list[dict[str, Any]]
    if has_tasks:
        assert source.tasks is not None  # noqa: S101
        missing_id = [
            idx for idx, t in enumerate(source.tasks) if not isinstance(t.get("id"), str)
        ]
        if missing_id:
            idx_str = ", ".join(str(i) for i in missing_id)
            msg = (
                f"swarm.create: tasks at index {idx_str} missing required "
                "`id` field (must be a string)."
            )
            raise ValueError(msg)
        rows = list(source.tasks)
    else:
        paths = await _resolve_file_paths(source, backend)
        task_ids = _build_task_ids(paths)
        rows = [{"id": task_ids[p], "file": p} for p in paths]

    write(file, serialize_table_jsonl(rows))
