"""Fleet zip import support for Talon local agent directories."""

from __future__ import annotations

import json
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING, Any, Self, cast
from urllib.parse import urlsplit, urlunsplit

from deepagents_talon.runtime import INTERRUPT_ON_TOOLS_ENV_KEY

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

_AGENT_ID_PATTERN = re.compile(r"[A-Za-z0-9_.-]{1,128}")
_MCP_CONFIG_FILENAME = ".mcp.json"
_SETUP_FILENAME = ".mcp.json.setup"
_ZIP_FILE_TYPE_MASK = 0o170000
_ZIP_SYMLINK_TYPE = 0o120000
_SUBAGENT_FILE_PARTS = 3
_MAX_ZIP_ENTRY_COUNT = 10_000
_MAX_ZIP_UNCOMPRESSED_BYTES = 256 * 1024 * 1024
_MAX_ZIP_COMPRESSION_RATIO = 100
_COPY_CHUNK_SIZE = 1024 * 1024
_SECRET_PATH_PATTERN = re.compile(
    r"(?:"
    r"bearer[-_a-z0-9]*|"
    r"token[-_a-z0-9]*|"
    r"key[-_a-z0-9]*|"
    r"secret[-_a-z0-9]*|"
    r"cookie[-_a-z0-9]*|"
    r"oauth[-_a-z0-9]*|"
    r"sk-[A-Za-z0-9]{20,}|"
    r"gh[opu]_[A-Za-z0-9]{20,}|"
    r"lsv2_pt_[A-Za-z0-9]+"
    r")",
    re.IGNORECASE,
)
_SECRET_PATH_MARKER_PATTERN = re.compile(
    r"(?:"
    r"bearer|"
    r"token|"
    r"access[-_]?token|"
    r"refresh[-_]?token|"
    r"api[-_]?key|"
    r"key|"
    r"secret|"
    r"cookie|"
    r"oauth"
    r")",
    re.IGNORECASE,
)


class FleetImportError(ValueError):
    """Raised when a Fleet zip cannot be materialized into a Talon agent directory."""


@dataclass(frozen=True, slots=True)
class FleetImportResult:
    """Summary of a completed Fleet zip import.

    Args:
        target_dir: Directory that received the materialized Talon agent files.
        root_prompt_count: Number of root prompt files written.
        subagent_prompt_count: Number of subagent prompt files written.
        config_ignored: Whether the Fleet zip contained a root `config.json`.
        mcp_notes: Human-readable MCP setup notes written to `.mcp.json.setup`.
        interrupt_tools: Tool names recommended for Talon interrupt configuration.
    """

    target_dir: Path
    root_prompt_count: int
    subagent_prompt_count: int
    config_ignored: bool
    mcp_notes: str | None
    interrupt_tools: tuple[str, ...]


@dataclass(slots=True)
class _ToolRequest:
    name: str
    server_url: str
    server_name: str
    scope: str
    interrupt: bool


@dataclass(slots=True)
class _ServerSummary:
    server_url: str
    server_name: str
    scopes: set[str] = field(default_factory=set)
    tools: set[str] = field(default_factory=set)
    interrupt_tools: set[str] = field(default_factory=set)

    @classmethod
    def from_tool(cls, tool: _ToolRequest) -> Self:
        """Create a server summary initialized from one tool request."""
        summary = cls(server_url=tool.server_url, server_name=tool.server_name)
        summary.add(tool)
        return summary

    def add(self, tool: _ToolRequest) -> None:
        """Record a requested tool for this MCP server."""
        self.scopes.add(tool.scope)
        self.tools.add(tool.name)
        if tool.interrupt:
            self.interrupt_tools.add(tool.name)


def import_fleet_zip(
    zip_path: Path,
    *,
    target_dir: Path,
    assistant_home: Path | None = None,
) -> FleetImportResult:
    """Materialize a Fleet zip export into a Talon local agent directory.

    Args:
        zip_path: Fleet export zip file to import.
        target_dir: Talon assistant directory to refresh with materialized files.
        assistant_home: Assistant state directory that should receive local
            subagents. Defaults to `target_dir`, keeping all writes under the
            explicit target.

    Returns:
        Summary of the materialized files and generated MCP configuration.

    Raises:
        FleetImportError: If the zip is structurally unsafe, missing required
            prompts, contains malformed `tools.json`, or cannot be written.
    """
    source = zip_path.expanduser()
    target = target_dir.expanduser()
    home = assistant_home.expanduser() if assistant_home is not None else target
    try:
        with zipfile.ZipFile(source) as archive:
            entries = _validated_entries(archive)
            if "AGENTS.md" not in entries:
                msg = "AGENTS.md: missing required root prompt"
                raise FleetImportError(msg)
            with tempfile.TemporaryDirectory(prefix="deepagents-talon-import-") as raw:
                staging = Path(raw)
                _materialize_staging(archive, entries, staging)
                summaries = _mcp_summaries(staging, source)
                notes = _format_setup_notes(source.name, summaries)
                mcp_config = _format_mcp_config(summaries)
                config_ignored = (staging / "config.json").is_file()
                _refresh_target(staging, target, home, notes, mcp_config)
    except zipfile.BadZipFile as exc:
        msg = f"{source}: invalid zip file"
        raise FleetImportError(msg) from exc
    except OSError as exc:
        msg = f"{target}: {exc}"
        raise FleetImportError(msg) from exc

    return FleetImportResult(
        target_dir=target,
        root_prompt_count=1,
        subagent_prompt_count=len(_subagent_prompt_paths(home)),
        config_ignored=config_ignored,
        mcp_notes=notes,
        interrupt_tools=tuple(
            sorted({tool for summary in summaries for tool in summary.interrupt_tools})
        ),
    )


def format_import_stdout(result: FleetImportResult) -> str:
    """Render a concise user-facing import summary.

    Args:
        result: Completed import summary.

    Returns:
        Text suitable for printing to stdout.
    """
    lines = [
        "Fleet import complete.",
        f"Agent files imported to: {result.target_dir}",
        f"Root prompts written: {result.root_prompt_count}",
        f"Subagent prompts written: {result.subagent_prompt_count}",
        f"config.json: {'ignored' if result.config_ignored else 'not present'}",
        "",
        "Next steps:",
    ]
    if result.mcp_notes is None:
        lines.append("- No Fleet MCP tool requirements were found.")
        lines.append("- Add MCP servers to .mcp.json if this assistant needs local tools.")
    else:
        lines.append("- Review .mcp.json before running Talon.")
        lines.append("- Review .mcp.json.setup for requested tools and setup details.")
    if result.interrupt_tools:
        lines.append(
            f"- Add HITL for sensitive tools with "
            f"{INTERRUPT_ON_TOOLS_ENV_KEY}={','.join(result.interrupt_tools)}.",
        )
    return "\n".join(lines) + "\n"


def _validated_entries(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    entries: dict[str, zipfile.ZipInfo] = {}
    total_size = 0
    for info in archive.infolist():
        name = _normalized_zip_name(info.filename)
        if name is None:
            continue
        if _is_unsafe_zip_path(name):
            msg = f"{info.filename}: unsafe zip path"
            raise FleetImportError(msg)
        if _is_symlink(info):
            msg = f"{name}: symlink entries are not supported"
            raise FleetImportError(msg)
        if info.is_dir():
            continue
        _validate_zip_entry_size(name, info)
        if len(entries) >= _MAX_ZIP_ENTRY_COUNT:
            msg = f"{archive.filename}: too many zip entries"
            raise FleetImportError(msg)
        total_size += info.file_size
        if total_size > _MAX_ZIP_UNCOMPRESSED_BYTES:
            msg = f"{archive.filename}: zip uncompressed size exceeds limit"
            raise FleetImportError(msg)
        entries[name] = info
    return entries


def _normalized_zip_name(name: str) -> str | None:
    normalized = name.replace("\\", "/")
    if not normalized or normalized.endswith("/"):
        return None
    return normalized


def _is_unsafe_zip_path(name: str) -> bool:
    posix = PurePosixPath(name)
    windows = PureWindowsPath(name)
    return (
        posix.is_absolute()
        or windows.is_absolute()
        or windows.drive != ""
        or any(part in {"", ".", ".."} for part in posix.parts)
    )


def _is_symlink(info: zipfile.ZipInfo) -> bool:
    file_type = (info.external_attr >> 16) & _ZIP_FILE_TYPE_MASK
    return file_type == _ZIP_SYMLINK_TYPE


def _validate_zip_entry_size(name: str, info: zipfile.ZipInfo) -> None:
    if info.file_size > _MAX_ZIP_UNCOMPRESSED_BYTES:
        msg = f"{name}: zip entry uncompressed size exceeds limit"
        raise FleetImportError(msg)
    if info.compress_size == 0:
        return
    if info.file_size > info.compress_size * _MAX_ZIP_COMPRESSION_RATIO:
        msg = f"{name}: zip entry compression ratio exceeds limit"
        raise FleetImportError(msg)


def _materialize_staging(
    archive: zipfile.ZipFile,
    entries: Mapping[str, zipfile.ZipInfo],
    staging: Path,
) -> None:
    _copy_zip_file(archive, entries["AGENTS.md"], staging / "AGENTS.md")

    for name, info in entries.items():
        if name.startswith("skills/"):
            _copy_zip_file(archive, info, staging / name)
        elif _is_subagent_prompt_path(name):
            subagent = PurePosixPath(name).parts[1]
            _validate_agent_name(subagent, name)
            _copy_zip_file(archive, info, staging / "agents" / subagent / "AGENTS.md")
        elif name in {"config.json", "tools.json"}:
            _copy_zip_file(archive, info, staging / name)
        elif _is_subagent_tools_path(name):
            subagent = PurePosixPath(name).parts[1]
            _validate_agent_name(subagent, name)
            _copy_zip_file(archive, info, staging / "agents" / subagent / "tools.json")


def _copy_zip_file(archive: zipfile.ZipFile, info: zipfile.ZipInfo, target: Path) -> None:
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    copied = 0
    with archive.open(info) as src, target.open("wb") as dst:
        while chunk := src.read(_COPY_CHUNK_SIZE):
            copied += len(chunk)
            if copied > info.file_size or copied > _MAX_ZIP_UNCOMPRESSED_BYTES:
                msg = f"{info.filename}: zip entry expanded beyond declared size"
                raise FleetImportError(msg)
            dst.write(chunk)
    target.chmod(0o600)


def _validate_agent_name(name: str, path: str) -> None:
    if not _AGENT_ID_PATTERN.fullmatch(name) or name in {".", ".."}:
        msg = f"{path}: unsafe subagent name {name!r}"
        raise FleetImportError(msg)


def _is_subagent_prompt_path(name: str) -> bool:
    parts = PurePosixPath(name).parts
    return (
        len(parts) == _SUBAGENT_FILE_PARTS and parts[0] == "subagents" and parts[2] == "AGENTS.md"
    )


def _is_subagent_tools_path(name: str) -> bool:
    parts = PurePosixPath(name).parts
    return (
        len(parts) == _SUBAGENT_FILE_PARTS and parts[0] == "subagents" and parts[2] == "tools.json"
    )


def _mcp_summaries(staging: Path, source: Path) -> list[_ServerSummary]:
    grouped: dict[tuple[str, str], _ServerSummary] = {}
    for path, scope in _tools_json_paths(staging):
        for tool in _load_tool_requests(path, scope, source):
            key = (tool.server_url, tool.server_name)
            summary = grouped.get(key)
            if summary is None:
                grouped[key] = _ServerSummary.from_tool(tool)
            else:
                summary.add(tool)
    return sorted(grouped.values(), key=lambda item: (item.server_name.lower(), item.server_url))


def _tools_json_paths(staging: Path) -> list[tuple[Path, str]]:
    paths: list[tuple[Path, str]] = []
    root = staging / "tools.json"
    if root.is_file():
        paths.append((root, "root"))
    agents = staging / "agents"
    if agents.is_dir():
        for child in sorted(agents.iterdir(), key=lambda item: item.name):
            path = child / "tools.json"
            if path.is_file():
                paths.append((path, child.name))
    return paths


def _load_tool_requests(path: Path, scope: str, source: Path) -> list[_ToolRequest]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        msg = f"{source}: {_display_path(path)}: malformed tools.json: {exc.msg}"
        raise FleetImportError(msg) from exc
    except OSError as exc:
        msg = f"{source}: {_display_path(path)}: {exc}"
        raise FleetImportError(msg) from exc
    if not isinstance(data, dict):
        msg = f"{source}: {_display_path(path)}: malformed tools.json: expected object"
        raise FleetImportError(msg)

    raw_tools = data.get("tools")
    if not isinstance(raw_tools, list):
        msg = f"{source}: {_display_path(path)}: malformed tools.json: expected tools list"
        raise FleetImportError(msg)
    interrupt_config = data.get("interrupt_config")
    interrupts = interrupt_config if isinstance(interrupt_config, dict) else {}

    requests: list[_ToolRequest] = []
    for index, item in enumerate(raw_tools):
        if not isinstance(item, dict):
            msg = f"{source}: {_display_path(path)}: tools[{index}] must be an object"
            raise FleetImportError(msg)
        tool = cast("Mapping[str, object]", item)
        name = _required_str(tool, "name", path, index, source)
        server_url = _sanitize_server_url(
            _required_str(tool, "mcp_server_url", path, index, source)
        )
        server_name = _required_str(tool, "mcp_server_name", path, index, source)
        requests.append(
            _ToolRequest(
                name=name,
                server_url=server_url,
                server_name=server_name,
                scope=scope,
                interrupt=_tool_interrupt_enabled(tool, interrupts, name, server_url, server_name),
            )
        )
    return requests


def _required_str(
    item: Mapping[str, object],
    key: str,
    path: Path,
    index: int,
    source: Path,
) -> str:
    value = item.get(key)
    if not isinstance(value, str) or not value:
        msg = f"{source}: {_display_path(path)}: tools[{index}].{key} must be a non-empty string"
        raise FleetImportError(msg)
    return value


def _tool_interrupt_enabled(
    item: Mapping[str, object],
    interrupts: Mapping[object, object],
    name: str,
    server_url: str,
    server_name: str,
) -> bool:
    if item.get("interrupt_config") is True:
        return True
    keys = (
        f"{server_url}::{name}::{server_name}",
        f"{_unsanitized_url(item)}::{name}::{server_name}",
        name,
    )
    return any(interrupts.get(key) is True for key in keys)


def _unsanitized_url(item: Mapping[str, object]) -> str:
    value = item.get("mcp_server_url")
    return value if isinstance(value, str) else ""


def _sanitize_server_url(raw: str) -> str:
    parts = urlsplit(raw)
    hostname = parts.hostname or ""
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    port = f":{parts.port}" if parts.port is not None else ""
    path = _sanitize_url_path(parts.path)
    return urlunsplit((parts.scheme, f"{hostname}{port}", path, "", ""))


def _sanitize_url_path(path: str) -> str:
    parts = [part for part in path.split("/") if part]
    if not parts:
        return ""
    sanitized: list[str] = []
    redact_next = False
    for part in parts:
        marker = _SECRET_PATH_MARKER_PATTERN.fullmatch(part) is not None
        secret = _SECRET_PATH_PATTERN.search(part) is not None
        if redact_next or marker or secret:
            sanitized.append("<secret-redacted>")
        else:
            sanitized.append(part)
        redact_next = marker
    return "/" + "/".join(sanitized)


def _format_setup_notes(source_name: str, summaries: Sequence[_ServerSummary]) -> str | None:
    if not summaries:
        return None
    lines = [
        f"Fleet MCP setup notes for {source_name}",
        "",
        "Generated .mcp.json contains the suggested server configuration.",
    ]
    for summary in summaries:
        lines.extend(_format_server_summary(summary))
    return "\n".join(lines) + "\n"


def _format_server_summary(summary: _ServerSummary) -> list[str]:
    tools = sorted(summary.tools)
    interrupts = sorted(summary.interrupt_tools)
    fragment = _suggested_config_fragment(summary, tools)
    return [
        "",
        f"Server: {summary.server_name}",
        f"URL: {summary.server_url}",
        f"Tool count: {len(tools)}",
        f"Scopes: {', '.join(sorted(summary.scopes))}",
        "Requested tools:",
        *[f"- {tool}" for tool in tools],
        f"Interrupt-enabled tools: {', '.join(interrupts) if interrupts else 'none'}",
        "",
        "Suggested .mcp.json fragment:",
        json.dumps(fragment, indent=2, sort_keys=True),
    ]


def _format_mcp_config(summaries: Sequence[_ServerSummary]) -> str | None:
    if not summaries:
        return None
    servers: dict[str, dict[str, Any]] = {}
    for summary in summaries:
        server_id = _unique_server_id(_server_id(summary), servers)
        servers[server_id] = _server_config(summary, sorted(summary.tools))
    return json.dumps({"mcpServers": servers}, indent=2, sort_keys=True) + "\n"


def _suggested_config_fragment(summary: _ServerSummary, tools: Sequence[str]) -> dict[str, Any]:
    server_id = _server_id(summary)
    return {server_id: _server_config(summary, tools)}


def _server_config(summary: _ServerSummary, tools: Sequence[str]) -> dict[str, Any]:
    return {
        "type": "http",
        "url": summary.server_url,
        "auth": "oauth",
        "allowedTools": list(tools),
    }


def _unique_server_id(server_id: str, servers: Mapping[str, object]) -> str:
    if server_id not in servers:
        return server_id
    suffix = 2
    while f"{server_id}-{suffix}" in servers:
        suffix += 1
    return f"{server_id}-{suffix}"


def _server_id(summary: _ServerSummary) -> str:
    raw = summary.server_name or urlsplit(summary.server_url).hostname or "server"
    normalized = re.sub(r"[^A-Za-z0-9_-]+", "-", raw.strip().lower()).strip("-")
    return normalized or "server"


def _refresh_target(
    staging: Path,
    target: Path,
    assistant_home: Path,
    notes: str | None,
    mcp_config: str | None,
) -> None:
    assistant_home.mkdir(mode=0o700, parents=True, exist_ok=True)
    assistant_home.chmod(0o700)
    target.mkdir(mode=0o700, parents=True, exist_ok=True)
    target.chmod(0o700)
    _replace_file(staging / "AGENTS.md", target / "AGENTS.md")

    _replace_tree(staging / "skills", target / "skills")
    _replace_tree(staging / "agents", assistant_home / "agents")
    if assistant_home != target:
        _remove_path(target / "agents")
    _remove_path(target / "subagents")

    setup_path = target / _SETUP_FILENAME
    if notes is None:
        if setup_path.exists():
            setup_path.unlink()
    else:
        setup_path.write_text(notes, encoding="utf-8")
        setup_path.chmod(0o600)

    config_path = target / _MCP_CONFIG_FILENAME
    if mcp_config is None:
        if config_path.exists():
            config_path.unlink()
    else:
        config_path.write_text(mcp_config, encoding="utf-8")
        config_path.chmod(0o600)


def _replace_file(source: Path, target: Path) -> None:
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temp = target.with_name(f".{target.name}.tmp")
    shutil.copy2(source, temp)
    temp.chmod(0o600)
    temp.replace(target)


def _replace_tree(source: Path, target: Path) -> None:
    if target.exists():
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
    if source.is_dir():
        shutil.copytree(source, target)


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _subagent_prompt_paths(assistant_home: Path) -> list[Path]:
    agents = assistant_home / "agents"
    if not agents.is_dir():
        return []
    return sorted(path for path in agents.glob("*/AGENTS.md") if path.is_file())


def _display_path(path: Path) -> str:
    parts = path.parts
    for directory in ("subagents", "agents"):
        if directory in parts:
            index = parts.index(directory)
            return "/".join(parts[index:])
    return path.name
