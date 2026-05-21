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
import re as _re
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

        _check_no_legacy_files(root)

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
            skills=_read_skills(root),
            subagents=_read_subagents(root),
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


_FRONTMATTER_RE = _re.compile(
    r"^---\n(?P<fm>.*?)\n---\n(?P<body>.*)$", _re.DOTALL
)


def _parse_skill_frontmatter(text: str, *, source: Path) -> tuple[dict[str, str], str]:
    match = _FRONTMATTER_RE.match(text)
    if not match:
        msg = f"{source}: YAML frontmatter (--- ... ---) is required."
        raise ProjectError(msg)
    frontmatter: dict[str, str] = {}
    for line in match.group("fm").splitlines():
        if not line.strip() or ":" not in line:
            continue
        key, _, value = line.partition(":")
        frontmatter[key.strip()] = value.strip().strip('"').strip("'")
    if "name" not in frontmatter or not frontmatter["name"]:
        msg = f"{source}: frontmatter is missing required key `name`."
        raise ProjectError(msg)
    if "description" not in frontmatter or not frontmatter["description"]:
        msg = f"{source}: frontmatter is missing required key `description`."
        raise ProjectError(msg)
    return frontmatter, match.group("body").strip()


def _read_skills(root: Path) -> list[Skill]:
    skills_dir = root / _SKILLS_DIR
    if not skills_dir.is_dir():
        return []
    result: list[Skill] = []
    seen: set[str] = set()
    for entry in sorted(skills_dir.iterdir()):
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        skill_file = entry / _SKILL_FILE
        if not skill_file.is_file():
            msg = f"{entry}: missing SKILL.md"
            raise ProjectError(msg)
        frontmatter, body = _parse_skill_frontmatter(
            skill_file.read_text(encoding="utf-8"), source=skill_file
        )
        name = frontmatter["name"]
        if name in seen:
            msg = f"duplicate skill name {name!r} in {skills_dir}"
            raise ProjectError(msg)
        seen.add(name)
        files: dict[str, str] = {}
        for child in sorted(entry.iterdir()):
            if child.is_file() and child.name != _SKILL_FILE and not child.name.startswith("."):
                files[child.name] = child.read_text(encoding="utf-8")
        result.append(
            Skill(
                name=name,
                description=frontmatter["description"],
                instructions=body,
                files=files,
            )
        )
    return result


def _read_subagents(root: Path) -> list[Subagent]:
    sa_dir = root / _SUBAGENTS_DIR
    if not sa_dir.is_dir():
        return []
    result: list[Subagent] = []
    seen: set[str] = set()
    for entry in sorted(sa_dir.iterdir()):
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        agent_json = entry / _AGENT_JSON
        agents_md = entry / _AGENTS_MD
        if not agent_json.is_file():
            msg = f"{entry}: missing agent.json"
            raise ProjectError(msg)
        if not agents_md.is_file():
            msg = f"{entry}: missing AGENTS.md"
            raise ProjectError(msg)
        try:
            data = json.loads(agent_json.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            msg = f"Invalid JSON in {agent_json}: {exc}"
            raise ProjectError(msg) from exc
        if not isinstance(data, dict):
            msg = f"{agent_json} must contain a JSON object."
            raise ProjectError(msg)
        name = entry.name
        key = name.lower()
        if key in seen:
            msg = f"duplicate subagent name {name!r} (case-insensitive)"
            raise ProjectError(msg)
        seen.add(key)

        tools = _read_tools_json(entry)
        extra_files: dict[str, str] = {}
        local_skills_dir = entry / _SKILLS_DIR
        if local_skills_dir.is_dir():
            for f in sorted(local_skills_dir.rglob("*")):
                if f.is_file() and not f.name.startswith("."):
                    rel = f.relative_to(entry).as_posix()
                    extra_files[rel] = f.read_text(encoding="utf-8")

        result.append(
            Subagent(
                name=name,
                description=data.get("description"),
                model_id=data.get("model_id"),
                instructions=agents_md.read_text(encoding="utf-8"),
                tools=tools,
                extra_files=extra_files,
            )
        )
    return result


_LEGACY_TOML_HINT = """\
Found legacy deepagents.toml in {root}. The migrated `deepagents deploy`
expects the new layout. Quick mapping:

  [agent]                       → agent.json (top-level keys: name, description)
  [agent].model                 → agent.json runtime.model.model_id
  [sandbox].scope               → agent.json runtime.backend_type
                                  ("thread_scoped_sandbox" or "agent_scoped_sandbox")
  [auth], [memories], [frontend]→ remove; managed by the platform now

Then run `deepagents init --force` to refresh scaffolding or migrate by hand.
"""


_LEGACY_MCP_HINT = """\
Found legacy `mcp.json` in {root}. MCP servers are now workspace-level resources:

  deepagents mcp-servers add --url <url> --header KEY=VALUE [--name <name>]

Then reference the server in tools.json by mcp_server_url.
"""


def _check_no_legacy_files(root: Path) -> None:
    if (root / "deepagents.toml").is_file():
        raise ProjectError(_LEGACY_TOML_HINT.format(root=root))
    if (root / "mcp.json").is_file():
        raise ProjectError(_LEGACY_MCP_HINT.format(root=root))
