"""Generic table JSONL helpers.

Each non-empty line must be a JSON object. Row semantics (required
fields, value shapes) are enforced at dispatch time, not here.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any


def parse_table_jsonl(content: str) -> list[dict[str, Any]]:
    """Parse a generic JSONL table into a list of row objects.

    Raises:
        ValueError: with line-numbered diagnostics on any malformed line.
    """
    rows: list[dict[str, Any]] = []
    errors: list[str] = []

    for idx, raw in enumerate(content.split("\n"), start=1):
        if not raw.strip():
            continue
        try:
            parsed: Any = json.loads(raw)
        except json.JSONDecodeError:
            errors.append(f"Line {idx}: invalid JSON")
            continue
        if not isinstance(parsed, dict):
            kind = "array" if isinstance(parsed, list) else type(parsed).__name__
            errors.append(f"Line {idx}: expected a JSON object, got {kind}")
            continue
        rows.append(parsed)

    if errors:
        msg = "Table parse failed:\n" + "\n".join(errors)
        raise ValueError(msg)
    return rows


def serialize_table_jsonl(rows: Iterable[dict[str, Any]]) -> str:
    """Serialize rows to JSONL with a trailing newline (empty → ``""``)."""
    lines = [json.dumps(r, separators=(",", ":")) for r in rows]
    if not lines:
        return ""
    return "\n".join(lines) + "\n"
