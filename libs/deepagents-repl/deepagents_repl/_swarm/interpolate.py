"""Template interpolation for per-row instruction synthesis.

Replaces ``{column}`` / ``{dotted.path}`` placeholders in an instruction
template with values read from a JSONL row. Strings are inserted bare;
booleans / numbers are stringified; objects and arrays are
JSON-serialised so they round-trip through the subagent's prompt.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

from deepagents_repl._swarm.filter import MISSING, read_column

_PLACEHOLDER_RE = re.compile(r"\{([^{}]+)\}")


def interpolate_instruction(template: str, row: Mapping[str, Any]) -> str:
    """Interpolate ``{column}`` placeholders in ``template`` from ``row``.

    Raises:
        ValueError: listing every missing column. Caller treats this as
            a per-task failure rather than aborting the whole swarm.
    """
    missing: list[str] = []

    def _substitute(match: re.Match[str]) -> str:
        path = match.group(1).strip()
        value = read_column(row, path)
        if value is MISSING:
            missing.append(path)
            return match.group(0)
        if value is None:
            return "null"
        if isinstance(value, str):
            return value
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return str(value)
        try:
            return json.dumps(value, default=str)
        except (TypeError, ValueError):
            return str(value)

    output = _PLACEHOLDER_RE.sub(_substitute, template)
    if missing:
        msg = f"Missing column(s) in row: {', '.join(missing)}"
        raise ValueError(msg)
    return output
