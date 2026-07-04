"""Fleet export import helpers for Talon.

Talon is an experimental runtime and is subject to change or removal at any time.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import zipfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import cast
from urllib.parse import urlsplit, urlunsplit

from deepagents_talon.config import TalonConfig

_ZIP_FILE_TYPE_MASK = 0o170000
_ZIP_SYMLINK_TYPE = 0o120000


class FleetImportError(ValueError):
    """Raised when a Fleet export cannot be imported."""


@dataclass(frozen=True, slots=True)
class FleetMCPServerNote:
    """Fleet MCP server notes written for local MCP follow-up.

    Args:
        scope: Root agent or subagent scope that requested the tools.
        server_name: Fleet MCP server name, or a sanitized fallback.
        endpoint: Sanitized Fleet MCP server endpoint.
        auth_path: Fleet authentication path, when known.
        tool_names: Fleet tools requested from this server in this scope.
        interrupt_tools: Tools Fleet configured with interrupt approval.
    """

    scope: str
    server_name: str
    endpoint: str
    auth_path: str
    tool_names: tuple[str, ...]
    interrupt_tools: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FleetImportSummary:
    """Result of importing a Fleet export into an assistant directory.

    Args:
        fleet_source: Fleet export zip file or directory imported by the command.
        assistant_id: Assistant id used to namespace Talon state.
        agent_dir: Materialized Talon agent directory.
        mcp_config_path: Root `.mcp.json` path written by the import.
        tool_count: Number of Fleet-requested MCP tools summarized.
        server_count: Number of Fleet MCP server notes written.
        interrupt_tool_count: Number of Fleet tools with interrupt approval enabled.
        mcp_server_notes: Local MCP follow-up notes grouped by scope and server.
    """

    fleet_source: Path
    assistant_id: str
    agent_dir: Path
    mcp_config_path: Path
    tool_count: int
    server_count: int
    interrupt_tool_count: int
    mcp_server_notes: tuple[FleetMCPServerNote, ...]


@dataclass(frozen=True, slots=True)
class _FleetToolEntry:
    scope: str
    name: str
    endpoint: str
    server_name: str
    auth_path: str
    interrupt: bool


def import_fleet_manifest(
    fleet_dir: Path,
    *,
    assistant_id: str,
    env: Mapping[str, str] | None = None,
) -> FleetImportSummary:
    """Materialize a Fleet export into a Talon assistant directory.

    Args:
        fleet_dir: Fleet export zip file or directory.
        assistant_id: Talon assistant id whose home receives the materialized agent.
        env: Environment mapping used to resolve Talon state locations.

    Returns:
        Summary of the import result.

    Raises:
        FleetImportError: If required Fleet export files are missing or malformed.
    """
    values = dict(os.environ if env is None else env)
    values["DEEPAGENTS_TALON_ASSISTANT_ID"] = assistant_id
    config = TalonConfig.from_env(values)

    config.ensure_home()
    target = config.manifest_dir
    mcp_config_path = target / ".mcp.json"
    source_path = _resolve_source_path(fleet_dir)

    with _fleet_source(source_path) as source:
        entries = _fleet_tool_entries(source)
        notes = _mcp_server_notes(entries)
        _materialize_agent_directory(source, target)
        _write_mcp_config_notes(
            mcp_config_path,
            fleet_source=source_path,
            assistant_id=config.assistant_id,
            agent_dir=target,
            notes=notes,
        )

    return FleetImportSummary(
        fleet_source=source_path,
        assistant_id=config.assistant_id,
        agent_dir=target,
        mcp_config_path=mcp_config_path,
        tool_count=len(entries),
        server_count=len(notes),
        interrupt_tool_count=sum(len(note.interrupt_tools) for note in notes),
        mcp_server_notes=notes,
    )


def _resolve_source_path(path: Path) -> Path:
    source = path.expanduser()
    try:
        return source.resolve(strict=True)
    except OSError as exc:
        msg = f"Fleet export does not exist: {path}"
        raise FleetImportError(msg) from exc


@contextmanager
def _fleet_source(path: Path) -> Iterator[Path]:
    if path.is_dir():
        _validate_fleet_root(path)
        yield path
        return

    if not path.is_file() or not zipfile.is_zipfile(path):
        msg = f"Fleet export must be a .zip file or directory: {path}"
        raise FleetImportError(msg)

    with tempfile.TemporaryDirectory(prefix="deepagents-talon-fleet-") as tmp:
        root = Path(tmp) / "export"
        root.mkdir(mode=0o700)
        _extract_zip(path, root)
        source = _zip_export_root(root)
        _validate_fleet_root(source)
        yield source


def _extract_zip(path: Path, target: Path) -> None:
    try:
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                if _is_zip_symlink(info):
                    msg = f"Fleet export zip contains unsupported symlink: {info.filename}"
                    raise FleetImportError(msg)
                destination = _safe_zip_destination(target, info.filename)
                if info.is_dir():
                    destination.mkdir(mode=0o700, parents=True, exist_ok=True)
                    continue
                destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                with archive.open(info) as source, destination.open("wb") as output:
                    shutil.copyfileobj(source, output)
                destination.chmod(0o600)
    except zipfile.BadZipFile as exc:
        msg = f"Fleet export is not a valid zip file: {path}"
        raise FleetImportError(msg) from exc
    except OSError as exc:
        msg = f"Could not extract Fleet export zip: {path}"
        raise FleetImportError(msg) from exc


def _safe_zip_destination(root: Path, name: str) -> Path:
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts:
        msg = f"Fleet export zip contains unsafe path: {name}"
        raise FleetImportError(msg)
    if any(part in {"", ".", ".."} for part in path.parts):
        msg = f"Fleet export zip contains unsafe path: {name}"
        raise FleetImportError(msg)
    return root.joinpath(*path.parts)


def _is_zip_symlink(info: zipfile.ZipInfo) -> bool:
    return ((info.external_attr >> 16) & _ZIP_FILE_TYPE_MASK) == _ZIP_SYMLINK_TYPE


def _zip_export_root(root: Path) -> Path:
    if (root / "AGENTS.md").is_file():
        return root

    try:
        children = [
            path
            for path in root.iterdir()
            if path.is_dir() and path.name not in {"__MACOSX", ".DS_Store"}
        ]
    except OSError as exc:
        msg = "Could not inspect extracted Fleet export"
        raise FleetImportError(msg) from exc

    if len(children) == 1 and (children[0] / "AGENTS.md").is_file():
        return children[0]
    return root


def _validate_fleet_root(source: Path) -> None:
    if not source.is_dir():
        msg = f"Fleet export root is not a directory: {source}"
        raise FleetImportError(msg)
    agents_md = source / "AGENTS.md"
    if not agents_md.is_file():
        msg = "Fleet export is missing required AGENTS.md"
        raise FleetImportError(msg)


def _materialize_agent_directory(source: Path, target: Path) -> None:
    """Copy Fleet prompts and skills into Talon's assistant directory.

    Args:
        source: Validated Fleet export root.
        target: Talon assistant `agent` directory.
    """
    target.mkdir(mode=0o700, parents=True, exist_ok=True)
    target.chmod(0o700)
    shutil.copy2(source / "AGENTS.md", target / "AGENTS.md")

    _copy_tree(source / "skills", target / "skills")
    _copy_subagents(source / "subagents", target / "subagents")


def _copy_subagents(source: Path, target: Path) -> None:
    if not source.is_dir():
        return
    target.mkdir(mode=0o700, parents=True, exist_ok=True)
    target.chmod(0o700)
    for child in sorted(path for path in source.iterdir() if path.is_dir()):
        prompt = child / "AGENTS.md"
        if not prompt.is_file():
            continue
        destination = target / child.name
        destination.mkdir(mode=0o700, parents=True, exist_ok=True)
        destination.chmod(0o700)
        shutil.copy2(prompt, destination / "AGENTS.md")
        _copy_tree(child / "skills", destination / "skills")


def _copy_tree(source: Path, target: Path) -> None:
    if source.is_dir():
        shutil.copytree(source, target, dirs_exist_ok=True)


def _write_mcp_config_notes(
    path: Path,
    *,
    fleet_source: Path,
    assistant_id: str,
    agent_dir: Path,
    notes: Sequence[FleetMCPServerNote],
) -> None:
    existing = _read_existing_mcp_config(path)
    payload = {
        "mcpServers": existing.get("mcpServers", {}),
        "_fleet_import": {
            "source": "fleet",
            "fleet_export": str(fleet_source),
            "assistant_id": assistant_id,
            "agent_dir": str(agent_dir),
            "notes": [
                "Configure MCP servers under mcpServers before relying on Fleet tools.",
                (
                    "Tools listed in interrupt_tools had Fleet interrupt approval enabled "
                    "and should be considered for human-in-the-loop configuration."
                ),
            ],
            "servers": [asdict(note) for note in notes],
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    path.chmod(0o600)


def _read_existing_mcp_config(path: Path) -> Mapping[str, object]:
    if not path.is_file():
        return {}
    data = _read_json_object(path, label="existing Talon MCP config")
    servers = data.get("mcpServers")
    if servers is None:
        return {}
    if not isinstance(servers, Mapping):
        msg = "Malformed existing Talon MCP config: mcpServers must be an object"
        raise FleetImportError(msg)
    return {"mcpServers": dict(cast("Mapping[str, object]", servers))}


def _fleet_tool_entries(fleet_dir: Path) -> tuple[_FleetToolEntry, ...]:
    entries: list[_FleetToolEntry] = []
    _extend_tool_entries(entries, fleet_dir / "tools.json", scope="root")

    subagents = fleet_dir / "subagents"
    if not subagents.is_dir():
        return tuple(entries)
    for child in sorted(path for path in subagents.iterdir() if path.is_dir()):
        _extend_tool_entries(
            entries,
            child / "tools.json",
            scope=f"subagent:{child.name}",
        )
    return tuple(sorted(entries, key=lambda entry: (entry.scope, entry.server_name, entry.name)))


def _extend_tool_entries(
    entries: list[_FleetToolEntry],
    path: Path,
    *,
    scope: str,
) -> None:
    if not path.is_file():
        return
    data = _read_json_object(path, label=f"Fleet {path.name}")
    raw = data.get("tools")
    if raw is None:
        return
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        msg = f"Malformed Fleet {path.name}: tools must be an array"
        raise FleetImportError(msg)

    interrupts = _interrupt_config(data, path=path)
    for index, entry in enumerate(raw):
        if not isinstance(entry, Mapping):
            msg = f"Malformed Fleet {path.name}: tools[{index}] must be an object"
            raise FleetImportError(msg)
        tool = cast("Mapping[str, object]", entry)
        name = _required_tool_str(tool, "name", path=path, index=index)
        raw_endpoint = _required_tool_str(tool, "mcp_server_url", path=path, index=index)
        endpoint = _safe_endpoint(raw_endpoint)
        server_name = _server_name(tool, endpoint=endpoint)
        entries.append(
            _FleetToolEntry(
                scope=scope,
                name=name,
                endpoint=endpoint,
                server_name=server_name,
                auth_path=_fleet_auth_path(tool),
                interrupt=_tool_interrupt_enabled(
                    tool,
                    interrupts=interrupts,
                    interrupt_keys=_interrupt_keys(
                        raw_endpoint,
                        endpoint,
                        name,
                        server_name,
                    ),
                ),
            )
        )


def _mcp_server_notes(entries: Sequence[_FleetToolEntry]) -> tuple[FleetMCPServerNote, ...]:
    grouped: dict[tuple[str, str, str, str], list[_FleetToolEntry]] = {}
    for entry in entries:
        grouped.setdefault(
            (entry.scope, entry.server_name, entry.endpoint, entry.auth_path),
            [],
        ).append(entry)

    notes: list[FleetMCPServerNote] = []
    for (scope, server_name, endpoint, auth_path), tools in sorted(grouped.items()):
        tool_names = tuple(sorted({tool.name for tool in tools}))
        interrupt_tools = tuple(sorted({tool.name for tool in tools if tool.interrupt}))
        notes.append(
            FleetMCPServerNote(
                scope=scope,
                server_name=server_name,
                endpoint=endpoint,
                auth_path=auth_path,
                tool_names=tool_names,
                interrupt_tools=interrupt_tools,
            )
        )
    return tuple(notes)


def _read_json_object(path: Path, *, label: str) -> Mapping[str, object]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        msg = f"Could not read {label}: {path}"
        raise FleetImportError(msg) from exc
    except json.JSONDecodeError as exc:
        msg = f"Malformed {label}: {path}"
        raise FleetImportError(msg) from exc
    if not isinstance(data, Mapping):
        msg = f"Malformed {label}: expected a JSON object"
        raise FleetImportError(msg)
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
    msg = f"Malformed Fleet {path.name}: tools[{index}].{key} is required"
    raise FleetImportError(msg)


def _interrupt_config(data: Mapping[str, object], *, path: Path) -> Mapping[str, object]:
    value = data.get("interrupt_config")
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        msg = f"Malformed Fleet {path.name}: interrupt_config must be an object"
        raise FleetImportError(msg)
    return cast("Mapping[str, object]", value)


def _server_name(tool: Mapping[str, object], *, endpoint: str) -> str:
    for key in (
        "mcp_server_name",
        "server_name",
        "server_registry_name",
        "mcp_server_registry_name",
        "registry_name",
        "server_display_name",
        "mcp_server_display_name",
    ):
        value = tool.get(key)
        if isinstance(value, str) and value:
            return value
    parsed = urlsplit(endpoint)
    return parsed.hostname or endpoint


def _fleet_auth_path(tool: Mapping[str, object]) -> str:
    raw = tool.get("auth_type") or tool.get("auth")
    if isinstance(raw, str) and raw in {"builtin", "headers", "oauth"}:
        return raw
    if "headers" in tool:
        return "headers"
    return "unknown"


def _tool_interrupt_enabled(
    tool: Mapping[str, object],
    *,
    interrupts: Mapping[str, object],
    interrupt_keys: Sequence[str],
) -> bool:
    direct = tool.get("interrupt_on")
    if isinstance(direct, bool):
        return direct
    direct = tool.get("interrupt")
    if isinstance(direct, bool):
        return direct

    for key in interrupt_keys:
        value = interrupts.get(key)
        if isinstance(value, bool):
            return value
    return False


def _interrupt_keys(
    raw_endpoint: str,
    endpoint: str,
    name: str,
    server_name: str,
) -> tuple[str, ...]:
    values = {
        f"{raw_endpoint}::{name}::{server_name}",
        f"{endpoint}::{name}::{server_name}",
        name,
    }
    return tuple(sorted(values))


def _safe_endpoint(value: str) -> str:
    try:
        parsed = urlsplit(value)
    except ValueError:
        return value.partition("?")[0].partition("#")[0]
    if parsed.scheme and parsed.netloc:
        return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))
    return value.partition("?")[0].partition("#")[0]
