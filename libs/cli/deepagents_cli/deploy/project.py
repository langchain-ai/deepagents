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
from typing import Any, cast

_AGENT_JSON = "agent.json"
_AGENTS_MD = "AGENTS.md"
_TOOLS_JSON = "tools.json"
_SKILLS_DIR = "skills"
_SUBAGENTS_DIR = "subagents"
_SKILL_FILE = "SKILL.md"

_BACKEND_TYPE_STATE = "state"
_BACKEND_TYPE_SANDBOX = "sandbox"
_BACKEND_TYPE_THREAD_SCOPED_SANDBOX = "thread_scoped_sandbox"
_BACKEND_TYPE_AGENT_SCOPED_SANDBOX = "agent_scoped_sandbox"
_LEGACY_BACKEND_TYPE_DEFAULT = "default"
_VALID_BACKEND_TYPES = frozenset(
    {
        _BACKEND_TYPE_STATE,
        _BACKEND_TYPE_SANDBOX,
        _BACKEND_TYPE_THREAD_SCOPED_SANDBOX,
        _BACKEND_TYPE_AGENT_SCOPED_SANDBOX,
    }
)
_SANDBOX_INTEGER_FIELDS = frozenset({"idle_ttl_seconds", "delete_after_stop_seconds"})
_VALID_SANDBOX_SCOPES = frozenset({"thread", "agent"})
_VALID_IDENTITY = frozenset({"personal", "shared"})
_VALID_VISIBILITY = frozenset({"tenant", "user"})
_VALID_TENANT_ACCESS = frozenset({"read", "run", "write"})


class ProjectError(ValueError):
    """Raised when the on-disk project is malformed."""


def _symlink_error(path: Path) -> ProjectError:
    msg = f"{path}: symlinks are not allowed in deploy project inputs."
    return ProjectError(msg)


def _ensure_project_contained(
    root: Path,
    path: Path,
    *,
    container: Path | None = None,
) -> None:
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root.resolve(strict=True))
        if container is not None:
            resolved.relative_to(container.resolve(strict=True))
    except FileNotFoundError as exc:
        msg = f"{path}: file does not exist."
        raise ProjectError(msg) from exc
    except ValueError as exc:
        msg = f"{path}: path escapes the deploy project directory."
        raise ProjectError(msg) from exc


def _read_project_text(
    root: Path,
    path: Path,
    *,
    missing_msg: str | None = None,
    container: Path | None = None,
) -> str:
    if path.is_symlink():
        raise _symlink_error(path)
    if not path.is_file():
        msg = missing_msg or f"{path}: expected a regular file."
        raise ProjectError(msg)
    _ensure_project_contained(root, path, container=container)
    return path.read_text(encoding="utf-8")


def _is_project_file(root: Path, path: Path) -> bool:
    if path.is_symlink():
        raise _symlink_error(path)
    if not path.is_file():
        return False
    _ensure_project_contained(root, path)
    return True


def _is_project_dir(root: Path, path: Path) -> bool:
    if path.is_symlink():
        raise _symlink_error(path)
    if not path.is_dir():
        return False
    _ensure_project_contained(root, path)
    return True


@dataclass
class Skill:
    """A skill discovered under `skills/<name>/`."""

    name: str
    description: str
    instructions: str
    skill_file: str
    files: dict[str, str] = field(default_factory=dict)


@dataclass
class Subagent:
    """A subagent discovered under `subagents/<name>/`."""

    name: str
    description: str | None
    model_id: str | None
    instructions: str
    tools: dict[str, Any] | None = None
    tools_text: str | None = None
    extra_files: dict[str, str] = field(default_factory=dict)
    """Subagent-local skills, keyed by path under `subagents/<name>/`."""


@dataclass
class Project:
    """In-memory view of the on-disk project."""

    root: Path
    agent_id: str | None
    name: str
    description: str | None
    model: str | None
    system_prompt: str
    runtime: dict[str, Any] | None
    backend: dict[str, Any] | None
    permissions: dict[str, Any] | None
    extras: dict[str, Any] | None
    tools: dict[str, Any] | None
    tools_text: str | None
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
        tools, tools_text = _read_tools_json(root)

        return cls(
            root=root,
            agent_id=agent_data.get("agent_id"),
            name=agent_data["name"],
            description=agent_data.get("description"),
            model=agent_data.get("model"),
            system_prompt=system_prompt,
            runtime=agent_data.get("runtime"),
            backend=agent_data.get("backend"),
            permissions=agent_data.get("permissions"),
            extras=agent_data.get("extras"),
            tools=tools,
            tools_text=tools_text,
            skills=_read_skills(root),
            subagents=_read_subagents(root),
        )


def _read_agent_json(root: Path) -> dict[str, Any]:
    path = root / _AGENT_JSON
    try:
        data = json.loads(
            _read_project_text(
                root,
                path,
                missing_msg=f"agent.json is required but not found in {root}.",
            )
        )
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

    agent_id = data.get("agent_id")
    if agent_id is not None:
        if not isinstance(agent_id, str) or not agent_id.strip():
            msg = f"{path}: `agent_id` must be a non-empty string when provided."
            raise ProjectError(msg)
        data["agent_id"] = agent_id.strip()

    model = data.get("model")
    if model is not None:
        if not isinstance(model, str) or not model.strip():
            msg = f"{path}: `model` must be a non-empty string when provided."
            raise ProjectError(msg)
        data["model"] = model.strip()

    runtime = data.get("runtime")
    if runtime is not None:
        if not isinstance(runtime, dict):
            msg = f"{path}: `runtime` must be an object."
            raise ProjectError(msg)
        if model is not None:
            msg = f"{path}: use either top-level `model` or `runtime`, not both."
            raise ProjectError(msg)
        backend_type = runtime.get("backend_type")
        if backend_type is not None:
            msg = (
                f"{path}: `runtime.backend_type` is no longer supported. "
                'Use top-level `backend`: {"type": "sandbox", '
                '"sandbox_config": {"scope": "thread"}} instead.'
            )
            raise ProjectError(msg)

    backend = data.get("backend")
    if backend is not None:
        data["backend"] = _normalize_backend(backend, source=path)

    permissions = data.get("permissions")
    if permissions is not None:
        if not isinstance(permissions, dict):
            msg = f"{path}: `permissions` must be an object."
            raise ProjectError(msg)
        if (ident := permissions.get("identity")) and ident not in _VALID_IDENTITY:
            msg = f"permissions.identity {ident!r} not in {sorted(_VALID_IDENTITY)}"
            raise ProjectError(msg)
        if (vis := permissions.get("visibility")) and vis not in _VALID_VISIBILITY:
            msg = f"permissions.visibility {vis!r} not in {sorted(_VALID_VISIBILITY)}"
            raise ProjectError(msg)
        lvl = permissions.get("tenant_access_level")
        if lvl and lvl not in _VALID_TENANT_ACCESS:
            msg = (
                f"permissions.tenant_access_level {lvl!r} not in "
                f"{sorted(_VALID_TENANT_ACCESS)}"
            )
            raise ProjectError(msg)

    return data


def _normalize_backend(backend: object, *, source: Path) -> dict[str, Any]:
    if not isinstance(backend, dict):
        msg = f"{source}: `backend` must be an object."
        raise ProjectError(msg)
    data = dict(cast("dict[str, Any]", backend))
    backend_type = data.get("type")
    if backend_type == _LEGACY_BACKEND_TYPE_DEFAULT:
        backend_type = _BACKEND_TYPE_STATE
        data["type"] = backend_type
    if backend_type is not None and backend_type not in _VALID_BACKEND_TYPES:
        msg = f"backend.type {backend_type!r} not in {sorted(_VALID_BACKEND_TYPES)}"
        raise ProjectError(msg)

    if backend_type == _BACKEND_TYPE_THREAD_SCOPED_SANDBOX:
        return _normalize_sandbox_backend(data, source=source, default_scope="thread")
    if backend_type == _BACKEND_TYPE_AGENT_SCOPED_SANDBOX:
        return _normalize_sandbox_backend(data, source=source, default_scope="agent")
    if backend_type == _BACKEND_TYPE_SANDBOX:
        return _normalize_sandbox_backend(data, source=source, default_scope="thread")

    if data.get("sandbox") is not None or data.get("sandbox_config") is not None:
        msg = (
            f"{source}: sandbox settings require `backend.type` to be "
            "`thread_scoped_sandbox`, `agent_scoped_sandbox`, or `sandbox`."
        )
        raise ProjectError(msg)
    return data


def _normalize_sandbox_backend(
    backend: dict[str, Any],
    *,
    source: Path,
    default_scope: str,
) -> dict[str, Any]:
    data = dict(backend)
    sandbox = data.pop("sandbox", None)
    sandbox_config = data.pop("sandbox_config", None)

    config: dict[str, Any] = {}
    if sandbox is not None:
        config = _normalize_sandbox_config(sandbox, field="sandbox", source=source)
    if sandbox_config is not None:
        config = {
            **config,
            **_normalize_sandbox_config(
                sandbox_config, field="sandbox_config", source=source
            ),
        }
    config.setdefault("scope", default_scope)

    data["type"] = _BACKEND_TYPE_SANDBOX
    data["sandbox_config"] = config
    return data


def _normalize_sandbox_config(
    sandbox: object,
    *,
    field: str,
    source: Path,
) -> dict[str, Any]:
    if not isinstance(sandbox, dict):
        msg = f"{source}: `backend.{field}` must be an object."
        raise ProjectError(msg)
    data = dict(cast("dict[str, Any]", sandbox))
    scope = data.get("scope")
    if scope is not None and scope not in _VALID_SANDBOX_SCOPES:
        msg = f"{source}: `backend.{field}.scope` must be `thread` or `agent`."
        raise ProjectError(msg)
    policies = data.get("policy_ids")
    if policies is not None and (
        not isinstance(policies, list)
        or not all(isinstance(policy, str) for policy in policies)
    ):
        msg = f"{source}: `backend.{field}.policy_ids` must be an array of strings."
        raise ProjectError(msg)
    for key in _SANDBOX_INTEGER_FIELDS:
        value = data.get(key)
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool)
        ):
            msg = f"{source}: `backend.{field}.{key}` must be an integer."
            raise ProjectError(msg)
    return data


def _read_agents_md(root: Path) -> str:
    path = root / _AGENTS_MD
    return _read_project_text(
        root,
        path,
        missing_msg=f"AGENTS.md is required but not found in {root}.",
    )


def _read_tools_json(root: Path) -> tuple[dict[str, Any] | None, str | None]:
    path = root / _TOOLS_JSON
    if not _is_project_file(root, path):
        return None, None
    text = _read_project_text(root, path)
    try:
        data = json.loads(text)
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
        tool_data = cast("dict[str, Any]", tool)
        name = tool_data.get("name")
        if not isinstance(name, str) or not name:
            msg = f"{path}: tools[{idx}].name is required."
            raise ProjectError(msg)
        url_val = tool_data.get("mcp_server_url")
        if not isinstance(url_val, str) or not url_val:
            msg = f"{path}: tools[{idx}].mcp_server_url is required."
            raise ProjectError(msg)
    interrupt_config = data.get("interrupt_config")
    if interrupt_config is not None and not isinstance(interrupt_config, dict):
        msg = f"{path}: `interrupt_config` must be an object."
        raise ProjectError(msg)
    return data, text


_FRONTMATTER_RE = _re.compile(r"^---\n(?P<fm>.*?)\n---\n(?P<body>.*)$", _re.DOTALL)


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
    if not _is_project_dir(root, skills_dir):
        return []
    result: list[Skill] = []
    seen: set[str] = set()
    for entry in sorted(skills_dir.iterdir()):
        if entry.name.startswith("."):
            continue
        if entry.is_symlink():
            raise _symlink_error(entry)
        if not entry.is_dir():
            continue
        _ensure_project_contained(root, entry)
        skill_file = entry / _SKILL_FILE
        skill_text = _read_project_text(
            root,
            skill_file,
            missing_msg=f"{entry}: missing SKILL.md",
            container=entry,
        )
        frontmatter, body = _parse_skill_frontmatter(skill_text, source=skill_file)
        name = frontmatter["name"]
        if name in seen:
            msg = f"duplicate skill name {name!r} in {skills_dir}"
            raise ProjectError(msg)
        seen.add(name)
        files: dict[str, str] = {}
        for child in sorted(entry.rglob("*")):
            if child.is_symlink():
                raise _symlink_error(child)
            if not child.is_file() or child.name == _SKILL_FILE:
                continue
            rel = child.relative_to(entry)
            if any(part.startswith(".") for part in rel.parts):
                continue
            files[rel.as_posix()] = _read_project_text(root, child, container=entry)
        result.append(
            Skill(
                name=name,
                description=frontmatter["description"],
                instructions=body,
                skill_file=skill_text,
                files=files,
            )
        )
    return result


def _read_subagent_model(data: dict[str, Any], *, source: Path) -> str | None:
    """Read a subagent's model from `model`, falling back to legacy `model_id`.

    The top-level `agent.json` and the SDK's `SubAgent` spec both spell this key
    `model`, so subagents accept the same. `model_id` is still honored for
    backward compatibility but should be migrated to `model`.

    Args:
        data: The parsed subagent `agent.json` object.
        source: Path to the subagent `agent.json`, used in error messages.

    Returns:
        The stripped model identifier, or `None` when unspecified.

    Raises:
        ProjectError: If both keys are set, or the value is not a non-empty
            string.
    """
    model = data.get("model")
    legacy = data.get("model_id")
    if model is not None and legacy is not None:
        msg = f"{source}: use either `model` or `model_id`, not both."
        raise ProjectError(msg)
    value = model if model is not None else legacy
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        msg = f"{source}: `model` must be a non-empty string when provided."
        raise ProjectError(msg)
    return value.strip()


def _read_subagents(root: Path) -> list[Subagent]:
    sa_dir = root / _SUBAGENTS_DIR
    if not _is_project_dir(root, sa_dir):
        return []
    result: list[Subagent] = []
    seen: set[str] = set()
    for entry in sorted(sa_dir.iterdir()):
        if entry.name.startswith("."):
            continue
        if entry.is_symlink():
            raise _symlink_error(entry)
        if not entry.is_dir():
            continue
        _ensure_project_contained(root, entry)
        agent_json = entry / _AGENT_JSON
        agents_md = entry / _AGENTS_MD
        try:
            data = json.loads(
                _read_project_text(
                    root,
                    agent_json,
                    missing_msg=f"{entry}: missing agent.json",
                    container=entry,
                )
            )
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

        tools, tools_text = _read_tools_json(entry)
        extra_files: dict[str, str] = {}
        local_skills_dir = entry / _SKILLS_DIR
        if _is_project_dir(root, local_skills_dir):
            for f in sorted(local_skills_dir.rglob("*")):
                if f.is_symlink():
                    raise _symlink_error(f)
                if not f.is_file():
                    continue
                rel = f.relative_to(entry)
                if any(part.startswith(".") for part in rel.parts):
                    continue
                extra_files[rel.as_posix()] = _read_project_text(
                    root,
                    f,
                    container=entry,
                )

        result.append(
            Subagent(
                name=name,
                description=data.get("description"),
                model_id=_read_subagent_model(data, source=agent_json),
                instructions=_read_project_text(
                    root,
                    agents_md,
                    missing_msg=f"{entry}: missing AGENTS.md",
                    container=entry,
                ),
                tools=tools,
                tools_text=tools_text,
                extra_files=extra_files,
            )
        )
    return result


_LEGACY_TOML_HINT = """\
Found legacy deepagents.toml in {root}. The migrated `deepagents deploy`
expects the new layout. Quick mapping:

  [agent]                       → agent.json (top-level keys: name, description)
  [agent].model                 → agent.json model
  [sandbox].scope               → agent.json backend.sandbox_config.scope
  [auth], [memories], [frontend]→ remove; managed by the platform now

Then run `deepagents init --force` to refresh scaffolding or migrate by hand.
"""


_LEGACY_MCP_HINT = """\
Found legacy `mcp.json` in {root}. MCP servers are now workspace-level resources:

  deepagents mcp-servers add --url <url> --header <api-key-name>=<value> --name <name>

Then reference the server in tools.json by mcp_server_url.
"""


def _check_no_legacy_files(root: Path) -> None:
    if (root / "deepagents.toml").is_file():
        raise ProjectError(_LEGACY_TOML_HINT.format(root=root))
    if (root / "mcp.json").is_file():
        raise ProjectError(_LEGACY_MCP_HINT.format(root=root))
