"""Agent Plugins v1.0 validation and MCP adaptation."""

from __future__ import annotations

import ipaddress
import json
import logging
import os
import re
from pathlib import Path, PureWindowsPath
from typing import TYPE_CHECKING, Never
from urllib.parse import urlsplit

from deepagents_code.mcp_config import (
    MCP_ENV_RESOLUTION_DISABLED,
    MCP_REDIRECTS_DISABLED,
)
from deepagents_code.plugins._json import json_object, json_value

if TYPE_CHECKING:
    from deepagents_code.plugins.models import JsonObject

logger = logging.getLogger(__name__)

AGENT_PLUGIN_FORMAT = "agent-plugins-v1"
AGENT_PLUGIN_MANIFEST_SCHEMA = (
    "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json"
)
AGENT_PLUGIN_MCP_SCHEMA = "https://agent-plugins.org/schemas/1.0.0/mcp.schema.json"

_MANIFEST_FIELDS = {
    "$schema",
    "author",
    "description",
    "extensions",
    "homepage",
    "keywords",
    "license",
    "name",
    "repository",
    "version",
}
_MANIFEST_STRING_FIELDS = {
    "description",
    "homepage",
    "license",
    "repository",
    "version",
}
_AUTHOR_FIELDS = {"email", "name", "url"}
_AGENT_PLUGIN_NAME_RE = re.compile(
    r"^(?!.*(?:--|\.\.))[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?$"
)
_HEADER_NAME_RE = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")
_REG_NAME_RE = re.compile(r"^(?:[A-Za-z0-9._~!$&'()*+,;=-]|%[0-9A-Fa-f]{2})+$")
_INVALID_PERCENT_RE = re.compile(r"%(?![0-9A-Fa-f]{2})")
_PLUGIN_PLACEHOLDER_RE = re.compile(r"\$\{PLUGIN_(ROOT|DATA)\}")
_RESERVED_ENV_NAMES = {"PLUGIN_DATA", "PLUGIN_ROOT"}
_STDIO_FIELDS = {"args", "command", "cwd", "env", "type"}
_REMOTE_FIELDS = {"headers", "type", "url"}
_MAX_PLUGIN_NAME_LENGTH = 64
_MAX_PORT = 65535
_ASCII_CONTROL_LIMIT = 32
_ASCII_DELETE = 127


class AgentPluginError(ValueError):
    """Raised when an Agent Plugins document is invalid."""


def _reject_json_constant(value: str) -> Never:
    msg = f"invalid JSON constant {value!r}"
    raise AgentPluginError(msg)


def _decode_json(value: str) -> object:
    try:
        return json.loads(value, parse_constant=_reject_json_constant)
    except json.JSONDecodeError as exc:
        msg = f"invalid JSON syntax: {exc}"
        raise AgentPluginError(msg) from exc


def _raise_field(field: str, detail: str) -> Never:
    msg = f"Agent Plugins field {field} {detail}"
    raise AgentPluginError(msg)


def _validate_manifest_name(value: object) -> str:
    if (
        not isinstance(value, str)
        or not 1 <= len(value) <= _MAX_PLUGIN_NAME_LENGTH
        or _AGENT_PLUGIN_NAME_RE.fullmatch(value) is None
    ):
        _raise_field(
            "name",
            "must be 1-64 lowercase alphanumeric, hyphen, or period characters",
        )
    return value


def validate_agent_plugin_manifest(
    decoded: object,
) -> tuple[JsonObject, tuple[str, ...]]:
    """Validate an Agent Plugins v1.0 manifest.

    Args:
        decoded: Parsed JSON manifest value.

    Returns:
        Validated fields and non-fatal warnings.

    Raises:
        AgentPluginError: If a fatal manifest rule is violated.
    """
    if not isinstance(decoded, dict):
        msg = "Agent Plugins manifest must be a JSON object"
        raise AgentPluginError(msg)
    raw = json_object(decoded)
    if raw.get("$schema") != AGENT_PLUGIN_MANIFEST_SCHEMA:
        _raise_field("$schema", f"must equal {AGENT_PLUGIN_MANIFEST_SCHEMA!r}")

    filtered = {key: value for key, value in raw.items() if key in _MANIFEST_FIELDS}
    warnings = tuple(
        f"ignoring unknown Agent Plugins manifest field {key!r}"
        for key in raw
        if key not in _MANIFEST_FIELDS
    )
    filtered["name"] = _validate_manifest_name(filtered.get("name"))

    for field in _MANIFEST_STRING_FIELDS:
        value = filtered.get(field)
        if value is not None and not isinstance(value, str):
            _raise_field(field, "must be a string")

    author = filtered.get("author")
    if author is not None:
        if not isinstance(author, dict):
            _raise_field("author", "must be an object")
        unknown = set(author) - _AUTHOR_FIELDS
        if unknown:
            _raise_field("author", f"contains unknown fields {sorted(unknown)!r}")
        if any(not isinstance(value, str) for value in author.values()):
            _raise_field("author", "values must be strings")

    keywords = filtered.get("keywords")
    if keywords is not None and (
        not isinstance(keywords, list)
        or any(not isinstance(value, str) for value in keywords)
    ):
        _raise_field("keywords", "must be an array of strings")

    extensions = filtered.get("extensions")
    if extensions is not None and not isinstance(extensions, dict):
        filtered.pop("extensions", None)
        warnings = (*warnings, "ignoring non-object Agent Plugins extensions field")
    elif isinstance(extensions, dict) and any(
        not isinstance(value, dict) for value in extensions.values()
    ):
        _raise_field("extensions", "values must be objects")

    return filtered, warnings


def load_agent_plugin_manifest(path: Path) -> tuple[JsonObject, tuple[str, ...]]:
    """Read and validate an Agent Plugins v1.0 manifest.

    Args:
        path: Manifest path.

    Returns:
        Validated fields and non-fatal warnings.

    Raises:
        AgentPluginError: If the manifest cannot be read or is invalid.
    """
    try:
        decoded = _decode_json(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as exc:
        msg = f"could not read Agent Plugins manifest {path}: {exc}"
        raise AgentPluginError(msg) from exc
    return validate_agent_plugin_manifest(decoded)


def _path_within(path: Path, root: Path, field: str) -> Path:
    try:
        resolved = path.resolve()
        boundary = root.resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        msg = f"could not resolve Agent Plugins {field}: {exc}"
        raise AgentPluginError(msg) from exc
    if not resolved.is_relative_to(boundary):
        msg = f"Agent Plugins {field} escapes its permitted root"
        raise AgentPluginError(msg)
    return resolved


def _is_windows_absolute(value: str) -> bool:
    parsed = PureWindowsPath(value)
    return bool(parsed.drive or parsed.root)


def _expand_plugin_value(value: str, *, plugin_root: Path, plugin_data: Path) -> str:
    replacements = {"ROOT": str(plugin_root), "DATA": str(plugin_data)}
    return _PLUGIN_PLACEHOLDER_RE.sub(
        lambda match: replacements[match.group(1)],
        value,
    )


def _string_list(value: object, field: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        _raise_field(field, "must be an array of strings")
    return [item for item in value if isinstance(item, str)]


def _string_map(value: object, field: str) -> dict[str, str]:
    if not isinstance(value, dict) or any(
        not isinstance(key, str) or not isinstance(item, str)
        for key, item in value.items()
    ):
        _raise_field(field, "must be an object with string values")
    return {
        key: item
        for key, item in value.items()
        if isinstance(key, str) and isinstance(item, str)
    }


def _stdio_command(value: object, plugin_root: Path) -> str:
    if not isinstance(value, str) or not value:
        _raise_field("mcpServers.*.command", "must be a nonempty string")
    if value.startswith("./"):
        relative = value[2:]
        if not relative or _is_windows_absolute(relative):
            _raise_field("mcpServers.*.command", "must stay inside the plugin root")
        return str(
            _path_within(
                plugin_root / relative,
                plugin_root,
                "mcpServers command",
            )
        )
    if (
        _is_windows_absolute(value)
        or any(character.isspace() for character in value)
        or any(separator in value for separator in ("/", "\\"))
    ):
        _raise_field(
            "mcpServers.*.command",
            "must be a bare executable name or a path beginning with './'",
        )
    return value


def _stdio_cwd(value: object, *, plugin_root: Path, plugin_data: Path) -> str:
    if value is None:
        return str(plugin_root)
    if not isinstance(value, str):
        _raise_field("mcpServers.*.cwd", "must be a string")
    if value.startswith("./"):
        relative = value[2:]
        if not relative:
            return str(plugin_root)
        if _is_windows_absolute(relative):
            _raise_field("mcpServers.*.cwd", "must stay inside the plugin root")
        return str(_path_within(plugin_root / relative, plugin_root, "mcpServers cwd"))
    for placeholder, root in (
        ("${PLUGIN_ROOT}", plugin_root),
        ("${PLUGIN_DATA}", plugin_data),
    ):
        if value == placeholder or value.startswith(f"{placeholder}/"):
            expanded = _expand_plugin_value(
                value, plugin_root=plugin_root, plugin_data=plugin_data
            )
            return str(_path_within(Path(expanded), root, "mcpServers cwd"))
    _raise_field(
        "mcpServers.*.cwd",
        "must begin with './', '${PLUGIN_ROOT}', or '${PLUGIN_DATA}'",
    )


def _validate_header_value(value: str) -> None:
    if any(
        (ord(character) < _ASCII_CONTROL_LIMIT and character != "\t")
        or ord(character) == _ASCII_DELETE
        for character in value
    ):
        _raise_field("mcpServers.*.headers", "contains an invalid header value")


def _remote_url(value: object) -> str:
    if not isinstance(value, str) or not value:
        _raise_field("mcpServers.*.url", "must be a nonempty string")
    if (
        "\\" in value
        or _INVALID_PERCENT_RE.search(value) is not None
        or any(
            character.isspace() or ord(character) < _ASCII_CONTROL_LIMIT
            for character in value
        )
    ):
        _raise_field("mcpServers.*.url", "must be a valid absolute URL")
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        msg = "Agent Plugins MCP URL is invalid"
        raise AgentPluginError(msg) from exc
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or (port is not None and not 0 < port <= _MAX_PORT)
    ):
        _raise_field(
            "mcpServers.*.url",
            "must be an absolute HTTP(S) URL without user information or a fragment",
        )
    _validate_url_host(parsed.hostname)
    if parsed.scheme == "http" and not _is_loopback_host(parsed.hostname):
        _raise_field("mcpServers.*.url", "must use HTTPS for non-loopback hosts")
    return value


def _validate_url_host(host: str) -> None:
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        return
    try:
        ascii_host = host.encode("idna").decode("ascii")
    except UnicodeError as exc:
        msg = "Agent Plugins MCP URL contains an invalid host"
        raise AgentPluginError(msg) from exc
    if _REG_NAME_RE.fullmatch(ascii_host) is None:
        _raise_field("mcpServers.*.url", "contains an invalid host")


def _is_loopback_host(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _remote_headers(value: object) -> dict[str, str]:
    if value is None:
        return {}
    headers = _string_map(value, "mcpServers.*.headers")
    normalized: set[str] = set()
    for name, item in headers.items():
        lower = name.lower()
        if _HEADER_NAME_RE.fullmatch(name) is None:
            _raise_field("mcpServers.*.headers", f"contains invalid name {name!r}")
        if lower in normalized:
            _raise_field(
                "mcpServers.*.headers",
                f"contains duplicate case-insensitive name {name!r}",
            )
        normalized.add(lower)
        _validate_header_value(item)
    return headers


def _validate_keys(server: dict[str, object], allowed: set[str]) -> None:
    unknown = set(server) - allowed
    if unknown:
        _raise_field("mcpServers.*", f"contains unknown fields {sorted(unknown)!r}")


def _adapt_stdio(
    server: dict[str, object], *, plugin_root: Path, plugin_data: Path
) -> JsonObject:
    _validate_keys(server, _STDIO_FIELDS)
    command = _stdio_command(server.get("command"), plugin_root)
    args = _string_list(server.get("args", []), "mcpServers.*.args")
    env = _string_map(server.get("env", {}), "mcpServers.*.env")
    reserved = (
        {name.upper() for name in _RESERVED_ENV_NAMES}
        if os.name == "nt"
        else _RESERVED_ENV_NAMES
    )
    configured = {name.upper() for name in env} if os.name == "nt" else set(env)
    if configured & reserved:
        _raise_field("mcpServers.*.env", "must not override PLUGIN_ROOT or PLUGIN_DATA")
    adapted: JsonObject = {
        "type": "stdio",
        "command": command,
        "args": [
            _expand_plugin_value(item, plugin_root=plugin_root, plugin_data=plugin_data)
            for item in args
        ],
        "env": {
            name: _expand_plugin_value(
                item, plugin_root=plugin_root, plugin_data=plugin_data
            )
            for name, item in env.items()
        },
        "cwd": _stdio_cwd(
            server.get("cwd"), plugin_root=plugin_root, plugin_data=plugin_data
        ),
        MCP_ENV_RESOLUTION_DISABLED: True,
    }
    return adapted


def _adapt_remote(server: dict[str, object], server_type: str) -> JsonObject:
    _validate_keys(server, _REMOTE_FIELDS)
    adapted: JsonObject = {
        "type": server_type,
        "url": _remote_url(server.get("url")),
        MCP_ENV_RESOLUTION_DISABLED: True,
        MCP_REDIRECTS_DISABLED: True,
    }
    headers = _remote_headers(server.get("headers"))
    if headers:
        adapted["headers"] = json_value(headers)
    return adapted


def _adapt_server(value: object, *, plugin_root: Path, plugin_data: Path) -> JsonObject:
    if not isinstance(value, dict):
        _raise_field("mcpServers.*", "must be an object")
    server = dict(value)
    server_type = server.get("type")
    if server_type == "stdio":
        return _adapt_stdio(server, plugin_root=plugin_root, plugin_data=plugin_data)
    if server_type in {"streamable-http", "sse"}:
        return _adapt_remote(server, server_type)
    _raise_field(
        "mcpServers.*.type",
        "must be 'stdio', 'streamable-http', or 'sse'",
    )


def load_agent_plugin_mcp(
    path: Path, *, plugin_root: Path, plugin_data: Path
) -> JsonObject:
    """Load and adapt an Agent Plugins v1.0 MCP document.

    Args:
        path: MCP document path.
        plugin_root: Installed plugin root.
        plugin_data: Writable plugin data directory.

    Returns:
        Valid server entries keyed by their declared names.
    """
    try:
        decoded = _decode_json(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, AgentPluginError) as exc:
        logger.warning("Skipping Agent Plugins MCP document %s: %s", path, exc)
        return {}
    if not isinstance(decoded, dict):
        logger.warning("Skipping Agent Plugins MCP document %s: expected object", path)
        return {}
    if set(decoded) != {"$schema", "mcpServers"}:
        logger.warning(
            "Skipping Agent Plugins MCP document %s: invalid top-level fields", path
        )
        return {}
    if decoded.get("$schema") != AGENT_PLUGIN_MCP_SCHEMA:
        logger.warning(
            "Skipping Agent Plugins MCP document %s: unsupported schema", path
        )
        return {}
    servers = decoded.get("mcpServers")
    if not isinstance(servers, dict):
        logger.warning(
            "Skipping Agent Plugins MCP document %s: mcpServers must be an object",
            path,
        )
        return {}

    adapted: JsonObject = {}
    for name, server in servers.items():
        if not isinstance(name, str):
            logger.warning("Skipping Agent Plugins MCP entry with a non-string name")
            continue
        try:
            adapted[name] = _adapt_server(
                server,
                plugin_root=plugin_root,
                plugin_data=plugin_data,
            )
        except AgentPluginError as exc:
            logger.warning("Skipping Agent Plugins MCP server %s: %s", name, exc)
    return adapted
