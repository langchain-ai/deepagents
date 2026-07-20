"""Validate that an agent impl names a graph registered in langgraph.json.

`_harbor_run.yml` calls this so an invalid `agent_impl` can never reach
`harbor run --agent-kwarg graph=...`. langgraph.json's graph keys are
repo-controlled, so membership is a strict allowlist. The impl is read from the
`AGENT_IMPL` environment variable, never interpolated into this source.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("::error::usage: AGENT_IMPL=<impl> validate_agent_graph.py <langgraph.json>")
        return 2
    impl = os.environ.get("AGENT_IMPL", "")
    try:
        graphs = json.loads(Path(argv[0]).read_text()).get("graphs", {})
    except (OSError, ValueError) as exc:
        print(f"::error::cannot read agent registry {argv[0]}: {exc}")
        return 2
    if impl not in graphs:
        print(
            f"::error::Unknown agent implementation {impl!r}; not a graph in "
            f"langgraph.json (have: {sorted(graphs)})"
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
