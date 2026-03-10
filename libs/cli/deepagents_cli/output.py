"""Machine-readable JSON output for CLI subcommands.

This module is imported by subcommand handlers when `--output-format json`
is active.  It deliberately avoids heavy imports (no SDK, no Rich, no
langchain) so that it stays fast and does not pull in unnecessary
dependency trees.
"""

from __future__ import annotations

import json
import sys
from typing import Literal

OutputFormat = Literal["text", "json"]
"""Accepted values for the `--output-format` CLI flag."""


def write_json(command: str, data: list | dict) -> None:
    """Write a JSON envelope to stdout and flush.

    The envelope is a single-line JSON object with a stable schema:

    ```json
    {"version": 1, "command": "...", "data": ...}
    ```

    Args:
        command: Self-documenting command name (e.g. `'list'`,
            `'threads list'`).
        data: Payload — typically a list for listing commands or a dict
            for action/info commands.

            `default=str` is used so that `Path` and `datetime` objects
            serialize without error.
    """
    envelope = {"version": 1, "command": command, "data": data}
    sys.stdout.write(json.dumps(envelope, default=str) + "\n")
    sys.stdout.flush()
