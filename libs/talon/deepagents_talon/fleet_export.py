"""Fleet export validation and MCP tool requirement extraction.

Talon is an experimental runtime and is subject to change or removal at any time.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast
from urllib.parse import urlsplit, urlunsplit

if TYPE_CHECKING:
    from pathlib import Path

_REQUIRED_EXPORT_FILES = ("AGENTS.md", "config.json")
_KNOWN_FLEET_AUTH_PATHS = frozenset({"builtin", "headers", "oauth"})


class FleetExportValidationError(ValueError):
    """Raised when a Fleet export cannot be imported by Talon."""


@dataclass(frozen=True, slots=True)
class FleetExportToolEntry:
    """MCP tool requirement entry declared by a Fleet export.

    Args:
        scope: Fleet scope declaring the requirement, such as `root` or
            `subagent:<name>`.
        tool_name: Requested MCP tool name.
        mcp_server_url: MCP server URL with query string and fragment removed.
        auth_path: Fleet authentication path for the server.
    """

    scope: str
    tool_name: str
    mcp_server_url: str
    auth_path: str


def validate_fleet_export(fleet_dir: Path) -> None:
    """Validate required Fleet export files and optional `tools.json` content.

    Args:
        fleet_dir: Operator-unzipped Fleet export directory.

    Raises:
        FleetExportValidationError: If the export is missing required files or
            contains malformed Fleet MCP tool metadata.
    """
    if not fleet_dir.is_dir():
        msg = f"Fleet export {fleet_dir} must be a directory"
        raise FleetExportValidationError(msg)

    for filename in _REQUIRED_EXPORT_FILES:
        path = fleet_dir / filename
        if not path.is_file():
            msg = f"Fleet export {fleet_dir} is missing required file {filename}"
            raise FleetExportValidationError(msg)

    fleet_tool_entries(fleet_dir)


def fleet_tool_entries(fleet_dir: Path) -> list[FleetExportToolEntry]:
    """Return validated Fleet MCP tool entries from root and subagent exports.

    Args:
        fleet_dir: Operator-unzipped Fleet export directory.

    Returns:
        Deterministic Fleet MCP tool entries.

    Raises:
        FleetExportValidationError: If a present `tools.json` file is malformed.
    """
    entries: list[FleetExportToolEntry] = []
    _extend_tool_entries(entries, fleet_dir / "tools.json", scope="root")

    subagents_dir = fleet_dir / "subagents"
    try:
        subagent_dirs = sorted(path for path in subagents_dir.glob("*") if path.is_dir())
    except FileNotFoundError:
        return entries
    except OSError:
        return entries

    for subagent_dir in subagent_dirs:
        _extend_tool_entries(
            entries,
            subagent_dir / "tools.json",
            scope=f"subagent:{subagent_dir.name}",
        )
    return entries


def _extend_tool_entries(
    entries: list[FleetExportToolEntry],
    path: Path,
    *,
    scope: str,
) -> None:
    if not path.is_file():
        return

    data = _load_tools_file(path)
    raw_tools = data.get("tools")
    if not isinstance(raw_tools, list):
        msg = f"Fleet export {path} field 'tools' must be a list"
        raise FleetExportValidationError(msg)

    for index, raw_tool in enumerate(raw_tools):
        if not isinstance(raw_tool, Mapping):
            msg = f"Fleet export {path} tools[{index}] must be an object"
            raise FleetExportValidationError(msg)
        tool = cast("Mapping[str, object]", raw_tool)
        name = _required_tool_str(tool, "name", path=path, index=index)
        server_url = _required_tool_str(tool, "mcp_server_url", path=path, index=index)
        entries.append(
            FleetExportToolEntry(
                scope=scope,
                tool_name=name,
                mcp_server_url=strip_url_query_and_fragment(server_url),
                auth_path=_fleet_auth_path(tool),
            )
        )


def _load_tools_file(path: Path) -> Mapping[str, object]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        msg = f"Fleet export {path} is not valid JSON"
        raise FleetExportValidationError(msg) from error
    except OSError as error:
        msg = f"Could not read Fleet export {path}"
        raise FleetExportValidationError(msg) from error

    if not isinstance(data, Mapping):
        msg = f"Fleet export {path} must contain a JSON object"
        raise FleetExportValidationError(msg)
    return cast("Mapping[str, object]", data)


def _required_tool_str(
    tool: Mapping[str, object],
    key: str,
    *,
    path: Path,
    index: int,
) -> str:
    value = tool.get(key)
    if isinstance(value, str) and value:
        return value
    msg = f"Fleet export {path} tools[{index}].{key} must be a non-empty string"
    raise FleetExportValidationError(msg)


def _fleet_auth_path(tool: Mapping[str, object]) -> str:
    raw_auth = tool.get("auth_type") or tool.get("auth")
    if isinstance(raw_auth, str) and raw_auth in _KNOWN_FLEET_AUTH_PATHS:
        return raw_auth
    if "headers" in tool:
        return "headers"
    return "unknown"


def strip_url_query_and_fragment(value: str) -> str:
    """Return a URL without query string or fragment components.

    Args:
        value: URL-like value from a Fleet export.

    Returns:
        Sanitized URL-like value safe to persist in local manifests.
    """
    try:
        parts = urlsplit(value)
    except ValueError:
        return value.partition("?")[0].partition("#")[0]
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
