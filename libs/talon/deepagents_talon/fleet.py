"""Fleet export loading for Talon.

Talon is an experimental runtime and is subject to change or removal at any time.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlsplit

from fleet_deepagents_export import StaticSkillsLoader, load_agent_components

from deepagents_talon.fleet_export import strip_url_query_and_fragment, validate_fleet_export
from deepagents_talon.observability import log_event

if TYPE_CHECKING:
    from deepagents import AsyncSubAgent, CompiledSubAgent, SubAgent
    from langchain.agents.middleware import InterruptOnConfig
    from langchain.agents.middleware.types import AgentMiddleware
    from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)
_KNOWN_FLEET_AUTH_PATHS = frozenset({"builtin", "headers", "oauth"})


@dataclass(frozen=True, slots=True)
class FleetAgentComponents:
    """Components returned by a Fleet export loader.

    Args:
        model: Chat model id from Fleet `config.json`.
        system_prompt: System prompt from Fleet `AGENTS.md`.
        tools: Resolved Fleet MCP tools.
        subagents: Fleet subagent specs.
        interrupt_on: Fleet human-in-the-loop config passed to the Deep Agents
            graph so Talon can surface tool approval over its channels.
        skills: Skill source paths for `create_deep_agent`.
        middleware: Middleware required by Fleet-loaded components.
    """

    model: str
    system_prompt: str
    tools: tuple[BaseTool | Callable[..., object], ...]
    subagents: tuple[SubAgent | CompiledSubAgent | AsyncSubAgent, ...]
    interrupt_on: Mapping[str, bool | InterruptOnConfig] | None
    skills: tuple[str, ...] = ()
    middleware: tuple[AgentMiddleware[Any, Any, Any], ...] = ()


async def load_fleet_agent_components(
    fleet_dir: Path,
    *,
    env: Mapping[str, str] | None = None,
) -> FleetAgentComponents:
    """Load a Fleet export directory through `fleet-deepagents-export`.

    Args:
        fleet_dir: Operator-unzipped Fleet export directory.
        env: Optional environment values to expose while loading the Fleet
            components. This lets embedding hosts pass TalonConfig-derived
            LangSmith settings even when they are not already in `os.environ`.

    Returns:
        Validated Fleet components ready for Talon's runtime wiring.

    Raises:
        FleetExportValidationError: If the Fleet export is missing required
            files or contains malformed Fleet MCP tool metadata.
        TypeError: If the library returns an unexpected component shape.
    """
    validate_fleet_export(fleet_dir)
    with _patched_environ(env):
        raw = await load_agent_components(fleet_dir)
    components = _coerce_components(raw)
    components = _with_static_skills_loader(components, fleet_dir=fleet_dir, env=env)
    _log_fleet_mcp_surface(fleet_dir, components, env=env)
    return components


def _coerce_components(raw: object) -> FleetAgentComponents:
    if not isinstance(raw, Mapping):
        msg = "Fleet loader returned a non-mapping component payload"
        raise TypeError(msg)

    data = cast("Mapping[str, object]", raw)
    model = _required_str(data, "model")
    system_prompt = _required_str(data, "system_prompt")
    tools = cast(
        "tuple[BaseTool | Callable[..., object], ...]",
        _optional_sequence(data.get("tools"), "tools"),
    )
    subagents = cast(
        "tuple[SubAgent | CompiledSubAgent | AsyncSubAgent, ...]",
        _optional_sequence(data.get("subagents"), "subagents"),
    )
    interrupt_on = cast(
        "Mapping[str, bool | InterruptOnConfig] | None",
        _optional_mapping(data.get("interrupt_on"), "interrupt_on"),
    )
    return FleetAgentComponents(
        model=model,
        system_prompt=system_prompt,
        tools=tools,
        subagents=subagents,
        interrupt_on=interrupt_on,
    )


def _with_static_skills_loader(
    components: FleetAgentComponents,
    *,
    fleet_dir: Path,
    env: Mapping[str, str] | None,
) -> FleetAgentComponents:
    loader = StaticSkillsLoader(_static_skill_sources(fleet_dir, env=env))
    if not loader.skill_paths:
        return components
    return replace(
        components,
        skills=tuple(loader.skill_paths),
        middleware=(loader,),
    )


def _static_skill_sources(
    fleet_dir: Path,
    *,
    env: Mapping[str, str] | None,
) -> list[tuple[Path, str]]:
    sources: list[tuple[Path, str]] = []
    seen: set[str] = set()
    _append_skill_source(sources, seen, fleet_dir / "skills")
    values = os.environ if env is None else env
    for raw in _split_path_env(
        values.get("DEEPAGENTS_TALON_SKILLS_DIRS") or values.get("SKILLS_DIRS"),
    ):
        _append_skill_source(sources, seen, Path(raw).expanduser())
    return sources


def _append_skill_source(
    sources: list[tuple[Path, str]],
    seen: set[str],
    path: Path,
) -> None:
    marker = str(path)
    if marker in seen:
        return
    seen.add(marker)
    sources.append((path, marker))


def _required_str(data: Mapping[str, object], key: str) -> str:
    value = data.get(key)
    if isinstance(value, str):
        return value
    msg = f"Fleet loader returned invalid {key!r}; expected string"
    raise TypeError(msg)


def _optional_sequence(value: object, key: str) -> tuple[object, ...]:
    if value is None:
        return ()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    msg = f"Fleet loader returned invalid {key!r}; expected sequence"
    raise TypeError(msg)


def _optional_mapping(value: object, key: str) -> Mapping[str, object] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        msg = f"Fleet loader returned invalid {key!r}; expected mapping"
        raise TypeError(msg)

    result: dict[str, object] = {}
    for raw_key, raw_value in value.items():
        if not isinstance(raw_key, str):
            msg = f"Fleet loader returned invalid {key!r}; expected string keys"
            raise TypeError(msg)
        result[raw_key] = raw_value
    return result


def _split_path_env(raw: str | None) -> list[str]:
    if not raw:
        return []
    separator = ";" if ";" in raw else os.pathsep
    return [str(Path(part).expanduser()) for part in raw.split(separator) if part.strip()]


@dataclass(frozen=True, slots=True)
class _FleetToolEntry:
    name: str
    server_url: str
    auth_path: str
    scope: str


def _log_fleet_mcp_surface(
    fleet_dir: Path,
    components: FleetAgentComponents,
    *,
    env: Mapping[str, str] | None,
) -> None:
    entries = _load_fleet_tool_entries(fleet_dir, env=env)
    records = _fleet_mcp_records(entries, components, env=env)
    log_event(
        logger,
        "fleet.mcp_surface",
        server_count=len(records),
        servers=records,
    )


def _load_fleet_tool_entries(
    fleet_dir: Path,
    *,
    env: Mapping[str, str] | None,
) -> list[_FleetToolEntry]:
    entries: list[_FleetToolEntry] = []
    _extend_fleet_tool_entries(entries, fleet_dir / "tools.json", scope="root", env=env)

    subagents_dir = fleet_dir / "subagents"
    try:
        subagent_dirs = sorted(path for path in subagents_dir.iterdir() if path.is_dir())
    except OSError:
        return entries

    for subagent_dir in subagent_dirs:
        _extend_fleet_tool_entries(
            entries,
            subagent_dir / "tools.json",
            scope=f"subagent:{subagent_dir.name}",
            env=env,
        )
    return entries


def _extend_fleet_tool_entries(
    entries: list[_FleetToolEntry],
    path: Path,
    *,
    scope: str,
    env: Mapping[str, str] | None,
) -> None:
    if not path.is_file():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        logger.warning("Could not read Fleet MCP tool entries for startup logging")
        return
    if not isinstance(data, Mapping):
        return

    raw_tools = data.get("tools")
    if not isinstance(raw_tools, Sequence) or isinstance(raw_tools, (str, bytes, bytearray)):
        return

    for raw_tool in raw_tools:
        if not isinstance(raw_tool, Mapping):
            continue
        tool = cast("Mapping[str, object]", raw_tool)
        name = tool.get("name")
        server_url = tool.get("mcp_server_url")
        if not isinstance(name, str) or not name:
            continue
        if not isinstance(server_url, str) or not server_url:
            continue
        entries.append(
            _FleetToolEntry(
                name=name,
                server_url=server_url,
                auth_path=_fleet_auth_path(tool, server_url, env=env),
                scope=scope,
            )
        )


def _fleet_auth_path(
    tool: Mapping[str, object],
    server_url: str,
    *,
    env: Mapping[str, str] | None,
) -> str:
    raw_auth = tool.get("auth_type") or tool.get("auth")
    if isinstance(raw_auth, str) and raw_auth in _KNOWN_FLEET_AUTH_PATHS:
        return raw_auth
    if _is_builtin_fleet_server(server_url, env=env):
        return "builtin"
    if "headers" in tool:
        return "headers"
    return "unknown"


def _fleet_mcp_records(
    entries: Sequence[_FleetToolEntry],
    components: FleetAgentComponents,
    *,
    env: Mapping[str, str] | None,
) -> list[dict[str, object]]:
    loaded_tool_names = _component_tool_names(components)
    grouped: dict[str, list[_FleetToolEntry]] = {}
    for entry in entries:
        grouped.setdefault(_normalize_url(entry.server_url), []).append(entry)

    records: list[dict[str, object]] = []
    for group in grouped.values():
        requested_tools = sorted({entry.name for entry in group})
        loaded_tools = [name for name in requested_tools if name in loaded_tool_names]
        auth_paths = sorted({entry.auth_path for entry in group})
        auth_path = auth_paths[0] if len(auth_paths) == 1 else "mixed"
        status = "loaded" if loaded_tools else "skipped"
        record: dict[str, object] = {
            "auth_path": auth_path,
            "endpoint": _fleet_log_endpoint(group[0].server_url, auth_path, env=env),
            "loaded_tool_count": len(loaded_tools),
            "loaded_tools": loaded_tools,
            "requested_tool_count": len(requested_tools),
            "requested_tools": requested_tools,
            "scopes": sorted({entry.scope for entry in group}),
            "status": status,
        }
        if status == "skipped":
            record["skip_reason"] = (
                "no requested tools loaded; server may be unresolved, "
                "unauthenticated, or missing requested tools"
            )
        records.append(record)
    return sorted(records, key=lambda record: str(record["endpoint"]))


def _component_tool_names(components: FleetAgentComponents) -> set[str]:
    names: set[str] = set()
    names.update(_tool_names(components.tools))
    for subagent in components.subagents:
        names.update(_tool_names(_subagent_tools(subagent)))
    return names


def _subagent_tools(subagent: object) -> Sequence[object]:
    if isinstance(subagent, Mapping):
        tools = cast("Mapping[str, object]", subagent).get("tools")
    else:
        tools = getattr(subagent, "tools", None)
    if isinstance(tools, Sequence) and not isinstance(tools, (str, bytes, bytearray)):
        return tools
    return ()


def _tool_names(tools: Sequence[object]) -> set[str]:
    names: set[str] = set()
    for tool in tools:
        name = getattr(tool, "name", None)
        if not isinstance(name, str):
            name = getattr(tool, "__name__", None)
        if isinstance(name, str) and name:
            names.add(name)
    return names


def _fleet_log_endpoint(
    server_url: str,
    auth_path: str,
    *,
    env: Mapping[str, str] | None,
) -> str:
    if auth_path != "builtin":
        return server_url
    values = os.environ if env is None else env
    return strip_url_query_and_fragment(values.get("BUILTIN_MCP_URL") or server_url)


def _is_builtin_fleet_server(
    server_url: str,
    *,
    env: Mapping[str, str] | None,
) -> bool:
    values = os.environ if env is None else env
    builtin = values.get("BUILTIN_MCP_URL")
    if not builtin:
        return False
    return _url_host(server_url) == _url_host(builtin)


def _url_host(value: str) -> str | None:
    try:
        return urlsplit(value).hostname
    except ValueError:
        return None


def _normalize_url(value: str) -> str:
    return value.rstrip("/").lower()


@contextmanager
def _patched_environ(env: Mapping[str, str] | None) -> Iterator[None]:
    if not env:
        yield
        return

    previous = {key: os.environ.get(key) for key in env}
    os.environ.update(env)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
