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


def load_async_subagents(
    config_path: Path | None = None, *, strict: bool = False
) -> list[AsyncSubAgent]:
    """Load async subagent definitions from `config.toml`.

    Reads the `[async_subagents]` section where each sub-table defines a remote
    LangGraph deployment.

    Args:
        config_path: Path to config file. Defaults to `~/.deepagents/config.toml`.
        strict: Reject an invalid configuration instead of returning a partial list.

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
        if strict:
            msg = "Could not read async subagent configuration"
            raise ValueError(msg) from None
        logger.warning(
            "Could not read async subagents from %s (%s)", config_path, type(exc).__name__
        )
        return []

    section = data.get("async_subagents")
    if not isinstance(section, dict):
        if strict and section is not None:
            msg = "Async subagents must be a table"
            raise ValueError(msg)
        return []

    agents: list[AsyncSubAgent] = []
    for name, spec in section.items():
        agent = _parse_async_subagent(name, spec)
        if strict and agent is None:
            msg = "Invalid async subagent definition"
            raise ValueError(msg)
        if agent is not None:
            agents.append(agent)
    return agents


def _parse_async_subagent(name: object, spec: object) -> AsyncSubAgent | None:
    if not isinstance(name, str) or not name.strip():
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
    if (
        not isinstance(description, str)
        or not description.strip()
        or not isinstance(graph_id, str)
        or not graph_id.strip()
    ):
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
    if not _valid_connection_fields(data):
        return None
    url = data.get("url")
    if isinstance(url, str):
        agent["url"] = url
    headers = data.get("headers")
    if isinstance(headers, dict):
        agent["headers"] = cast("dict[str, str]", headers).copy()
    return agent


def _valid_connection_fields(data: dict[str, object]) -> bool:
    url = data.get("url")
    if url is not None and (not isinstance(url, str) or not url.strip()):
        logger.warning("Skipping async subagent: url must be a nonempty string")
        return False
    headers = data.get("headers")
    if headers is not None and (
        not isinstance(headers, dict)
        or any(
            not isinstance(key, str) or not isinstance(value, str) for key, value in headers.items()
        )
    ):
        logger.warning("Skipping async subagent: headers must map strings to strings")
        return False
    return True
