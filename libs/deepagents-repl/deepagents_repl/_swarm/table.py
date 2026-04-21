"""Table materialisation — helpers for building table rows from file sources."""

from __future__ import annotations

import os


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
