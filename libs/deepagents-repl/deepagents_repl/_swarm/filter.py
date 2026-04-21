"""Declarative filter clauses over JSONL table rows.

Supports dotted-path column access, equality / membership / existence
tests, and ``and`` / ``or`` combinators. Filter objects arrive from JS
as plain dicts — validation happens here at evaluation time.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

SwarmFilter = Mapping[str, Any]
"""A filter clause. One of:

- ``{"column": str, "equals": Any}``
- ``{"column": str, "notEquals": Any}``
- ``{"column": str, "in": list[Any]}``
- ``{"column": str, "exists": bool}``
- ``{"and": list[SwarmFilter]}``
- ``{"or": list[SwarmFilter]}``
"""

MISSING: Any = object()
"""Sentinel returned by :func:`read_column` when a path is unreachable.

Distinguishes "column not present" from "column present but null",
which matters for interpolation (error vs. render ``"null"``) but
not for ``exists`` filters (both treated as "not present").
"""


def read_column(row: Mapping[str, Any], path: str) -> Any:
    """Read a dotted path from a row object.

    Traverses nested mappings by splitting on ``.``. Returns
    :data:`MISSING` for missing segments or non-mapping intermediates;
    returns the actual value (including ``None``) when the path resolves.
    """
    cursor: Any = row
    for segment in path.split("."):
        if not isinstance(cursor, Mapping) or segment not in cursor:
            return MISSING
        cursor = cursor[segment]
    return cursor


def _deep_equal(a: Any, b: Any) -> bool:
    """Structural equality check for filter comparisons.

    Primitives compare by identity/value; containers compare by JSON
    serialization, which is good enough for filter-literal shapes (no
    cycles, all JSON-safe values).
    """
    if a == b:
        return True
    if type(a) != type(b):  # noqa: E721 — strict type comparison intentional
        return False
    if isinstance(a, (dict, list)):
        try:
            return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
        except (TypeError, ValueError):
            return False
    return False


def evaluate_filter(clause: SwarmFilter, row: Mapping[str, Any]) -> bool:
    """Evaluate a :class:`SwarmFilter` clause against a single row.

    Combinators (``and`` / ``or``) recurse; leaf clauses compare values
    read via :func:`read_column` against the clause operand.
    """
    if "and" in clause:
        sub = clause["and"]
        if not isinstance(sub, list):
            return False
        return all(evaluate_filter(c, row) for c in sub)

    if "or" in clause:
        sub = clause["or"]
        if not isinstance(sub, list):
            return False
        return any(evaluate_filter(c, row) for c in sub)

    column = clause.get("column")
    if not isinstance(column, str):
        return False
    value = read_column(row, column)
    # For equality-family operators, a missing path compares as None —
    # the caller almost always writes `{column, equals: null}` to mean
    # "missing or null" and we match that intent.
    compared = None if value is MISSING else value

    if "equals" in clause:
        return _deep_equal(compared, clause["equals"])
    if "notEquals" in clause:
        return not _deep_equal(compared, clause["notEquals"])
    if "in" in clause:
        choices = clause["in"]
        if not isinstance(choices, list):
            return False
        return any(_deep_equal(compared, c) for c in choices)
    if "exists" in clause:
        present = value is not MISSING and value is not None
        return bool(clause["exists"]) == present
    return False
