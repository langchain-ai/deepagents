"""Async subagent configuration loading for Talon.

Talon is an experimental runtime and is subject to change or removal at any time.
"""

from __future__ import annotations

import logging
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from deepagents.middleware.async_subagents import AsyncSubAgent

logger = logging.getLogger(__name__)


def load_async_subagents(config_path: Path | None = None) -> list[AsyncSubAgent]:
    """Load async subagent definitions from `config.toml`.

    Reads the `[async_subagents]` section where each sub-table defines a remote
    LangGraph deployment.

    Args:
        config_path: Path to config file. Defaults to `~/.deepagents/config.toml`.

    Returns:
        List of async subagent specs, or an empty list when absent or invalid.
    """
    if config_path is None:
        config_path = Path.home() / ".deepagents" / "config.toml"

    if not config_path.exists():
        return []

    try:
        with config_path.open("rb") as file:
            data = tomllib.load(file)
    except (tomllib.TOMLDecodeError, PermissionError, OSError) as exc:
        logger.warning("Could not read async subagents from %s: %s", config_path, exc)
        return []

    section = data.get("async_subagents")
    if not isinstance(section, dict):
        return []

    agents: list[AsyncSubAgent] = []
    for name, spec in section.items():
        agent = _parse_async_subagent(name, spec)
        if agent is not None:
            agents.append(agent)
    return agents


def _parse_async_subagent(name: object, spec: object) -> AsyncSubAgent | None:
    if not isinstance(name, str):
        logger.warning("Skipping async subagent with non-string name: %r", name)
        return None
    if not isinstance(spec, dict):
        logger.warning("Skipping async subagent '%s': expected a table", name)
        return None

    data = cast("dict[str, object]", spec)
    missing = {"description", "graph_id"} - data.keys()
    if missing:
        logger.warning("Skipping async subagent '%s': missing fields %s", name, missing)
        return None

    description = data["description"]
    graph_id = data["graph_id"]
    if not isinstance(description, str) or not isinstance(graph_id, str):
        logger.warning(
            "Skipping async subagent '%s': description and graph_id must be strings",
            name,
        )
        return None

    agent: AsyncSubAgent = {
        "name": name,
        "description": description,
        "graph_id": graph_id,
    }
    url = data.get("url")
    if isinstance(url, str):
        agent["url"] = url
    headers = data.get("headers")
    if isinstance(headers, dict):
        agent["headers"] = {
            key: value
            for key, value in headers.items()
            if isinstance(key, str) and isinstance(value, str)
        }
    return agent
