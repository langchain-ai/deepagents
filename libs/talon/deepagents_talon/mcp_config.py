"""Narrow, redacted access to Talon's operator-selected MCP configuration."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import stat
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, cast

from langchain_core.tools import tool

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping

    from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)

MCP_CONFIG_AUTO_APPROVE_ENV = "DEEPAGENTS_TALON_MCP_CONFIG_AUTO_APPROVE"
MCP_CONFIG_UPDATE_TOOL = "update_mcp_server"
# Duplicated from runtime._WORKSPACE_ENV: importing it would pull the runtime, and the
# public alias lives in channels, whose package imports every channel driver eagerly.
WORKSPACE_ENV = "DEEPAGENTS_TALON_WORKSPACE"
_REDACTED = "<redacted>"
_REFERENCE = re.compile(r"\$\{[A-Za-z_][A-Za-z0-9_]*\}")
_NAME = re.compile(r"[A-Za-z0-9_-]+")
_FIELDS = frozenset(
    {
        "transport",
        "type",
        "command",
        "args",
        "env",
        "url",
        "headers",
        "auth",
        "allowedTools",
        "disabledTools",
    }
)
_ENUMS = {
    "transport": {"stdio", "http", "sse", "streamable_http", "streamable-http"},
    "type": {"stdio", "http", "sse", "streamable_http", "streamable-http"},
    "auth": {"oauth"},
}


def agent_workspace_root(env: Mapping[str, str] | None = None) -> Path:
    """Return the directory the agent's shell backend reaches with relative paths.

    Args:
        env: Runtime environment to read. Falls back to the process environment,
            which is what the default backend itself reads.

    Returns:
        The configured workspace root, or the current working directory, where
        `LocalShellBackend(root_dir=None)` runs agent commands.
    """
    configured = (env or {}).get(WORKSPACE_ENV) or os.environ.get(WORKSPACE_ENV)
    return Path(configured).expanduser().resolve() if configured else Path.cwd().resolve()


def warn_agent_workspace_path(path: Path, agent_root: Path | None, *, subject: str) -> Path:
    """Return `path` resolved, warning when it sits inside the agent workspace.

    This reports placement, not secrecy. Talon's default backend runs real shell
    commands with `virtual_mode=False`, so a path outside the workspace is still
    readable by absolute path; keeping it out of the workspace only removes the
    relative-path route and keeps it out of the model's working tree. Because it
    buys no present protection, a bad placement is not worth failing a start over
    -- with no workspace configured the root is the working directory, so Talon
    launched from `$HOME` puts both default paths inside it.

    Args:
        path: Operator-selected path to check.
        agent_root: Workspace root. Falls back to `agent_workspace_root()`.
        subject: Name of the file, used in the warning.

    Returns:
        The resolved path, with the final component left unresolved.
    """
    resolved = path.parent.resolve() / path.name
    root = agent_workspace_root() if agent_root is None else agent_root
    # Round 2: raise here instead, once the runtime deny rules for the token
    # directory and the configuration path make the placement worth enforcing.
    if resolved.is_relative_to(root):
        logger.warning(
            "%s %s is inside the agent workspace %s; move it elsewhere, or point %s "
            "at a directory that does not contain it",
            subject,
            resolved,
            root,
            WORKSPACE_ENV,
        )
    return resolved


class MCPConfigStore:
    """Manage one fixed file without exposing stored credentials to tools.

    Nothing here is a confidentiality boundary. Placement is checked and warned
    about, not enforced, and against Talon's default shell backend `cat` and
    `echo >` on the absolute path bypass redaction, the revision CAS, the
    `O_NOFOLLOW` open, and the `update_mcp_server` approval interrupt alike. So
    `update_mcp_server` mediates change; it cannot keep a configured secret from
    the model. Only `${ENV_VAR}` references, which are never expanded here, do
    that. Enforcing placement, denying the path to the agent's filesystem tools,
    and moving credentials out of a readable file are tracked separately.

    Args:
        path: Operator-selected configuration path, ideally outside the agent
            workspace; a path inside it is warned about, not rejected.
        on_update: Schedule a reload after a successful write.
        agent_root: Agent workspace root. Defaults to the process workspace.
    """

    def __init__(
        self,
        path: Path,
        on_update: Callable[[], None],
        *,
        agent_root: Path | None = None,
    ) -> None:
        """Capture the operator path and a process-local revision key."""
        self._path = warn_agent_workspace_path(path, agent_root, subject="MCP configuration")
        self._on_update = on_update
        self._revision_key = secrets.token_bytes(32)

    def tools(self) -> tuple[BaseTool, BaseTool]:
        """Return the read and single-server update capabilities."""

        @tool
        def get_mcp_configuration() -> dict[str, object]:
            """View MCP server configuration and its revision without reading files.

            Stored strings are <redacted>, except transport/auth enums and exact
            ${ENV_VAR} references. Values are never expanded. Use this revision
            with update_mcp_server. This tool and update_mcp_server are the
            supported way to read and change MCP settings; do not read or edit
            the configuration file with filesystem or shell tools.
            """
            return self._view()

        @tool
        def update_mcp_server(
            server_name: str, server: dict[str, object] | None, expected_revision: str
        ) -> dict[str, str]:
            """Add, replace, or remove one MCP server; normally requires approval.

            Args:
                server_name: Server name from mcpServers, or a new name.
                server: Complete replacement settings, or null to remove. Copy
                    <redacted> at the same field to preserve its stored value.
                    Omitted fields are removed. Use ${ENV_VAR} references for
                    credentials; never request or include literal secrets.
                expected_revision: Revision returned by get_mcp_configuration.

            Changes can execute commands or send credentials to configured URLs.
            Saved changes activate only after a successful reload before the next turn.
            Use get_agent_tools to verify activation. Running tasks retain old capabilities.
            """
            return self._update(server_name, server, expected_revision)

        return get_mcp_configuration, update_mcp_server

    def _revision(self, raw: bytes | None) -> str:
        return hmac.new(
            self._revision_key, b"missing" if raw is None else b"file:" + raw, hashlib.sha256
        ).hexdigest()

    def _read(self) -> tuple[bytes | None, dict[str, object]]:
        _require_posix()
        try:
            descriptor = os.open(self._path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        except FileNotFoundError:
            return None, {"mcpServers": {}}
        with os.fdopen(descriptor, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                msg = "MCP configuration must be a regular file"
                raise ValueError(msg)
            raw = stream.read()
        document = json.loads(raw)
        if not isinstance(document, dict) or not isinstance(document.get("mcpServers"), dict):
            msg = "Invalid MCP configuration"
            raise TypeError(msg)
        return raw, document

    def _view(self) -> dict[str, object]:
        try:
            raw, document = self._read()
            servers = cast("dict[str, object]", document["mcpServers"])
            return {
                "revision": self._revision(raw),
                "mcpServers": {name: _redact_server(server) for name, server in servers.items()},
            }
        except (OSError, ValueError, TypeError, RecursionError):
            return {"status": "error", "message": "Cannot read MCP configuration."}

    def _update(self, name: str, server: dict[str, object] | None, revision: str) -> dict[str, str]:
        try:
            if not _NAME.fullmatch(name):
                return {"status": "error", "message": "Invalid server name."}
            with self._locked():
                raw, document = self._read()
                if not hmac.compare_digest(self._revision(raw), revision):
                    return {"status": "conflict", "message": "Read the configuration again."}
                self._replace_server(document, name, server)
                _atomic_write(self._path, document)
        except (OSError, ValueError, TypeError, RecursionError):
            return {
                "status": "error",
                "message": "Cannot update MCP configuration; check settings.",
            }
        self._on_update()
        return {"status": "updated", "available": "after_successful_reload"}

    @contextmanager
    def _locked(self) -> Iterator[None]:
        _require_posix()
        import fcntl  # noqa: PLC0415  # Keep Talon importable on non-POSIX hosts.

        self._path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        descriptor = os.open(
            self._path.with_name(self._path.name + ".lock"),
            os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW,
            0o600,
        )
        with os.fdopen(descriptor, "rb") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _replace_server(
        self, document: dict[str, object], name: str, server: dict[str, object] | None
    ) -> None:
        servers = cast("dict[str, object]", document["mcpServers"])
        if server is None:
            servers.pop(name, None)
            return
        replacement = cast("dict[str, object]", _restore(server, servers.get(name)))
        _validate_server(replacement)
        servers[name] = replacement


def _require_posix() -> None:
    if os.name != "posix":
        msg = "MCP configuration management requires POSIX file locking"
        raise OSError(msg)


def _redact_server(server: object) -> object:
    if not isinstance(server, dict):
        return _redact(server)
    return {
        key: value if isinstance(value, str) and value in _ENUMS.get(key, set()) else _redact(value)
        for key, value in server.items()
    }


def _redact(value: object) -> object:
    if isinstance(value, dict):
        return {key: _redact(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, str):
        return value if _REFERENCE.fullmatch(value) else _REDACTED
    return value


def _restore(value: object, previous: object) -> object:
    if value == _REDACTED:
        if not isinstance(previous, str):
            msg = "No stored value for redacted field"
            raise ValueError(msg)
        return previous
    if isinstance(value, dict):
        old = previous if isinstance(previous, dict) else {}
        return {key: _restore(item, old.get(key)) for key, item in value.items()}
    if isinstance(value, list):
        old_list = previous if isinstance(previous, list) else []
        return [
            _restore(item, old_list[i] if i < len(old_list) else None)
            for i, item in enumerate(value)
        ]
    return value


def _validate_server(server: dict[str, object]) -> None:
    # Reuse pure loader validation; never resolve environment values or open sessions.
    from deepagents_talon.mcp import _filter_tools, _stdio_connection  # noqa: PLC0415

    if server.keys() - _FIELDS:
        msg = "Unsupported MCP settings"
        raise ValueError(msg)
    _validate_fields(server)
    _validate_references(server)
    transport = server.get(
        "transport", server.get("type", "stdio" if "command" in server else "http")
    )
    for field, choices in _ENUMS.items():
        if field in server and server[field] not in choices:
            msg = "Invalid MCP transport or authentication"
            raise ValueError(msg)
    if transport == "stdio":
        _stdio_connection("server", server)
    else:
        _validate_remote(server)
    _filter_tools("server", server, ())


def _validate_fields(server: dict[str, object]) -> None:
    for field in ("command", "url", "auth"):
        if field in server and not isinstance(server[field], str):
            msg = "MCP settings must contain strings"
            raise TypeError(msg)
    args = server.get("args", [])
    if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
        msg = "MCP args must contain strings"
        raise TypeError(msg)
    for field in ("env", "headers"):
        values = server.get(field, {})
        if not isinstance(values, dict) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in values.items()
        ):
            msg = "MCP environment and headers must contain strings"
            raise TypeError(msg)


def _validate_references(server: dict[str, object]) -> None:
    from deepagents_talon.mcp import _ENV_REF  # noqa: PLC0415  # Avoid a module import cycle.

    values = [server.get(field, "") for field in ("command", "url")]
    values.extend(cast("list[str]", server.get("args", [])))
    for field in ("env", "headers"):
        values.extend(cast("dict[str, str]", server.get(field, {})).values())
    if any(isinstance(value, str) and "${" in _ENV_REF.sub("", value) for value in values):
        msg = "Malformed MCP environment reference"
        raise ValueError(msg)


def _validate_remote(server: dict[str, object]) -> None:
    url = server.get("url")
    headers = cast("dict[str, str]", server.get("headers", {}))
    if not isinstance(url, str) or not url:
        msg = "Remote MCP server requires a URL"
        raise ValueError(msg)
    if server.get("auth") == "oauth" and any(key.lower() == "authorization" for key in headers):
        msg = "OAuth cannot be combined with an Authorization header"
        raise ValueError(msg)


def _atomic_write(path: Path, document: dict[str, object]) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=".mcp-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(document, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if path.is_symlink():
            msg = "MCP configuration must not be a symlink"
            raise ValueError(msg)
        Path(temporary).replace(path)
    finally:
        Path(temporary).unlink(missing_ok=True)
