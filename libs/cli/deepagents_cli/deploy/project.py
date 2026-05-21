"""Parse a Managed Deep Agents project directory into a structured value.

Layout (canonical, all paths relative to the project root):

    agent.json              required — top-level config
    AGENTS.md               required — system prompt
    tools.json              optional — verbatim ToolsConfig
    skills/<name>/SKILL.md  optional — frontmatter + body
    skills/<name>/<file>    optional — siblings of SKILL.md → files map
    subagents/<name>/agent.json   required if subagent dir exists
    subagents/<name>/AGENTS.md    required if subagent dir exists
    subagents/<name>/tools.json   optional

The result is plain Python data — no I/O happens after `load()` returns. The
payload builder (`payload.py`) consumes this dataclass.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_AGENT_JSON = "agent.json"
_AGENTS_MD = "AGENTS.md"
_TOOLS_JSON = "tools.json"
_SKILLS_DIR = "skills"
_SUBAGENTS_DIR = "subagents"
_SKILL_FILE = "SKILL.md"

_VALID_BACKEND_TYPES = frozenset(
    {"default", "thread_scoped_sandbox", "agent_scoped_sandbox"}
)
_VALID_IDENTITY = frozenset({"personal", "shared"})
_VALID_VISIBILITY = frozenset({"tenant", "user"})
_VALID_TENANT_ACCESS = frozenset({"read", "run", "write"})


class ProjectError(ValueError):
    """Raised when the on-disk project is malformed."""


@dataclass
class Skill:
    """A skill discovered under `skills/<name>/`."""

    name: str
    description: str
    instructions: str
    files: dict[str, str] = field(default_factory=dict)


@dataclass
class Subagent:
    """A subagent discovered under `subagents/<name>/`."""

    name: str
    description: str | None
    model_id: str | None
    instructions: str
    tools: dict[str, Any] | None = None
    extra_files: dict[str, str] = field(default_factory=dict)
    """Subagent-local skills, keyed by path under `subagents/<name>/`."""


@dataclass
class Project:
    """In-memory view of the on-disk project."""

    root: Path
    name: str
    description: str | None
    system_prompt: str
    runtime: dict[str, Any] | None
    permissions: dict[str, Any] | None
    extras: dict[str, Any] | None
    tools: dict[str, Any] | None
    skills: list[Skill]
    subagents: list[Subagent]

    @classmethod
    def load(cls, root: Path) -> Project:
        """Read the project at *root*; raise `ProjectError` on any problem."""
        root = root.resolve()
        if not root.is_dir():
            msg = f"Project root is not a directory: {root}"
            raise ProjectError(msg)

        agent_data = _read_agent_json(root)
        system_prompt = _read_agents_md(root)

        return cls(
            root=root,
            name=agent_data["name"],
            description=agent_data.get("description"),
            system_prompt=system_prompt,
            runtime=agent_data.get("runtime"),
            permissions=agent_data.get("permissions"),
            extras=agent_data.get("extras"),
            tools=_read_tools_json(root),
            skills=[],          # task 8
            subagents=[],       # task 9
        )


def _read_agent_json(root: Path) -> dict[str, Any]:
    path = root / _AGENT_JSON
    if not path.is_file():
        msg = f"agent.json is required but not found in {root}."
        raise ProjectError(msg)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        msg = f"Invalid JSON in {path}: {exc}"
        raise ProjectError(msg) from exc
    if not isinstance(data, dict):
        msg = f"{path} must contain a JSON object."
        raise ProjectError(msg)

    name = data.get("name")
    if not isinstance(name, str) or not name.strip():
        msg = f"`name` (non-empty string) is required in {path}."
        raise ProjectError(msg)

    runtime = data.get("runtime")
    if runtime is not None:
        backend_type = runtime.get("backend_type")
        if backend_type is not None and backend_type not in _VALID_BACKEND_TYPES:
            msg = (
                f"runtime.backend_type {backend_type!r} not in "
                f"{sorted(_VALID_BACKEND_TYPES)}"
            )
            raise ProjectError(msg)

    permissions = data.get("permissions")
    if permissions is not None:
        if (ident := permissions.get("identity")) and ident not in _VALID_IDENTITY:
            msg = f"permissions.identity {ident!r} not in {sorted(_VALID_IDENTITY)}"
            raise ProjectError(msg)
        if (vis := permissions.get("visibility")) and vis not in _VALID_VISIBILITY:
            msg = f"permissions.visibility {vis!r} not in {sorted(_VALID_VISIBILITY)}"
            raise ProjectError(msg)
        if (lvl := permissions.get("tenant_access_level")) and lvl not in _VALID_TENANT_ACCESS:
            msg = (
                f"permissions.tenant_access_level {lvl!r} not in "
                f"{sorted(_VALID_TENANT_ACCESS)}"
            )
            raise ProjectError(msg)

    return data


def _read_agents_md(root: Path) -> str:
    path = root / _AGENTS_MD
    if not path.is_file():
        msg = f"AGENTS.md is required but not found in {root}."
        raise ProjectError(msg)
    return path.read_text(encoding="utf-8")


def _read_tools_json(root: Path) -> dict[str, Any] | None:
    path = root / _TOOLS_JSON
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        msg = f"Invalid JSON in {path}: {exc}"
        raise ProjectError(msg) from exc
    if not isinstance(data, dict):
        msg = f"{path} must contain a JSON object."
        raise ProjectError(msg)
    tools = data.get("tools")
    if not isinstance(tools, list):
        msg = f"{path}: `tools` must be an array."
        raise ProjectError(msg)
    for idx, tool in enumerate(tools):
        if not isinstance(tool, dict):
            msg = f"{path}: tools[{idx}] must be an object."
            raise ProjectError(msg)
        if not isinstance(tool.get("name"), str) or not tool["name"]:
            msg = f"{path}: tools[{idx}].name is required."
            raise ProjectError(msg)
        if not isinstance(tool.get("mcp_server_url"), str) or not tool["mcp_server_url"]:
            msg = f"{path}: tools[{idx}].mcp_server_url is required."
            raise ProjectError(msg)
    interrupt_config = data.get("interrupt_config")
    if interrupt_config is not None and not isinstance(interrupt_config, dict):
        msg = f"{path}: `interrupt_config` must be an object."
        raise ProjectError(msg)
    return data
