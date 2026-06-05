"""Swarm: parallel subagent dispatch for the QuickJS code interpreter.

Two entry points share one dispatch core (``build_swarm_dispatch``):

- :func:`swarm` — an interpreter extension that makes the handle-based table
  API (``create`` / ``run`` / ``rows``) importable from guest JS as
  ``import { ... } from "swarm"``, with no PTC wiring.
- :func:`create_swarm_task_tool` — a PTC-only ``BaseTool`` for the legacy
  ``tools.swarmTask`` path.
"""

from __future__ import annotations

from langchain_quickjs._swarm._extension import SwarmExtension, swarm
from langchain_quickjs._swarm._task import (
    SwarmDispatch,
    SwarmSubAgent,
    SwarmTaskMode,
    VariantCache,
    build_swarm_dispatch,
    create_swarm_task_tool,
)

__all__ = [
    "SwarmDispatch",
    "SwarmExtension",
    "SwarmSubAgent",
    "SwarmTaskMode",
    "VariantCache",
    "build_swarm_dispatch",
    "create_swarm_task_tool",
    "swarm",
]
