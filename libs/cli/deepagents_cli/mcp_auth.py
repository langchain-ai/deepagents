"""OAuth + bearer-token auth helpers for MCP servers."""

from __future__ import annotations

import json
import os
import re
import stat
from collections.abc import Awaitable, Callable
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from mcp.client.auth import OAuthClientProvider, TokenStorage
from mcp.shared.auth import AnyUrl, OAuthClientInformationFull, OAuthClientMetadata, OAuthToken


_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

_STORAGE_VERSION = 1


def resolve_headers(
    headers: dict[str, str],
    *,
    server_name: str | None = None,
) -> dict[str, str]:
    """Resolve `${VAR}` env-var references in header values.

    `$${VAR}` is the escape form and collapses to the literal `${VAR}` with no
    lookup. A dollar sign not followed by `{` or `$` is left untouched.

    Raises:
        TypeError: If a header value is not a string.
        RuntimeError: If a `${VAR}` references an unset environment variable.
    """
    resolved: dict[str, str] = {}
    for name, value in headers.items():
        if not isinstance(value, str):
            where = f"mcpServers.{server_name}.headers.{name}" if server_name else name
            msg = f"{where} must be a string, got {type(value).__name__}"
            raise TypeError(msg)
        resolved[name] = _interpolate(value, header=name, server_name=server_name)
    return resolved


def _interpolate(s: str, *, header: str, server_name: str | None) -> str:
    out: list[str] = []
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == "$" and i + 1 < len(s):
            nxt = s[i + 1]
            if nxt == "$":
                # `$${...}` → literal `${...}` if a brace group follows;
                # otherwise literal `$$`.
                if i + 2 < len(s) and s[i + 2] == "{":
                    end = s.find("}", i + 3)
                    if end != -1:
                        out.append("$" + s[i + 2 : end + 1])
                        i = end + 1
                        continue
                out.append("$$")
                i += 2
                continue
            if nxt == "{":
                end = s.find("}", i + 2)
                if end != -1:
                    var_name = s[i + 2 : end]
                    if _IDENT_RE.fullmatch(var_name):
                        val = os.environ.get(var_name)
                        if val is None:
                            where = (
                                f"mcpServers.{server_name}.headers.{header}"
                                if server_name
                                else header
                            )
                            msg = (
                                f"{where} references unset env var {var_name}. "
                                f"Set {var_name} in the environment or remove "
                                "the reference."
                            )
                            raise RuntimeError(msg)
                        out.append(val)
                        i = end + 1
                        continue
        out.append(ch)
        i += 1
    return "".join(out)


def _tokens_dir() -> Path:
    """Resolve the mcp-tokens directory.

    Re-reads `$HOME` on every call so tests that override it via
    `monkeypatch.setenv("HOME", ...)` transparently redirect token files.
    Falls back to the module-level `DEFAULT_CONFIG_DIR` when `$HOME` is unset.
    """
    home_override = os.environ.get("HOME")
    if home_override:
        return Path(home_override) / ".deepagents" / "mcp-tokens"
    from deepagents_cli.model_config import DEFAULT_CONFIG_DIR

    return DEFAULT_CONFIG_DIR / "mcp-tokens"


class FileTokenStorage(TokenStorage):
    """File-backed `TokenStorage` under `~/.deepagents/mcp-tokens/<server>.json`.

    Atomic-write via tmp-file + `os.replace` + chmod(0o600). Directory is
    created with mode 0o700. Permission bits are POSIX-only.
    """

    def __init__(self, server_name: str) -> None:
        self._server_name = server_name

    @property
    def _path(self) -> Path:
        return _tokens_dir() / f"{self._server_name}.json"

    async def get_tokens(self) -> OAuthToken | None:
        data = self._read()
        if data is None:
            return None
        raw = data.get("tokens")
        if raw is None:
            return None
        return OAuthToken.model_validate(raw)

    async def set_tokens(self, tokens: OAuthToken) -> None:
        data = self._read() or {}
        data["version"] = _STORAGE_VERSION
        data["tokens"] = json.loads(tokens.model_dump_json(exclude_none=True))
        self._write(data)

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        data = self._read()
        if data is None:
            return None
        raw = data.get("client_info")
        if raw is None:
            return None
        return OAuthClientInformationFull.model_validate(raw)

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        data = self._read() or {}
        data["version"] = _STORAGE_VERSION
        data["client_info"] = json.loads(
            client_info.model_dump_json(exclude_none=True)
        )
        self._write(data)

    def _read(self) -> dict | None:
        path = self._path
        if not path.exists():
            return None
        try:
            raw = path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except (OSError, json.JSONDecodeError) as exc:
            msg = (
                f"Failed to read MCP token file {path}: {exc}. "
                "Delete the file and re-run `deepagents mcp login "
                f"{self._server_name}` if it is corrupt."
            )
            raise RuntimeError(msg) from exc
        if data.get("version") != _STORAGE_VERSION:
            msg = (
                f"MCP token file {path} has unsupported version "
                f"{data.get('version')!r} (expected {_STORAGE_VERSION}). "
                "Delete it and re-run `deepagents mcp login "
                f"{self._server_name}`."
            )
            raise RuntimeError(msg)
        return data

    def _write(self, data: dict) -> None:
        path = self._path
        path.parent.mkdir(parents=True, exist_ok=True)
        # Tighten directory perms on POSIX (best effort on Windows).
        if hasattr(os, "chmod"):
            try:
                os.chmod(path.parent, stat.S_IRWXU)  # 0o700
            except OSError:
                pass
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
        try:
            os.replace(tmp, path)
        except Exception:
            # Clean up the tmp file so no partial state leaks at the final path.
            try:
                tmp.unlink()
            except OSError:
                pass
            raise
        if hasattr(os, "chmod"):
            try:
                os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)  # 0o600
            except OSError:
                pass


RedirectHandler = Callable[[str], Awaitable[None]]
CallbackHandler = Callable[[], Awaitable[tuple[str, str | None]]]


def _make_paste_back_handlers() -> tuple[RedirectHandler, CallbackHandler]:
    """Create paste-back redirect and callback handlers for OAuth.

    Returns:
        A tuple of (redirect_handler, callback_handler) functions that implement
        the paste-back OAuth flow for interactive CLI use.
    """

    async def redirect(auth_url: str) -> None:
        print(  # noqa: T201 - intentional user-facing prompt
            "\nOpen this URL in a browser, approve access, then paste the full "
            "callback URL back here:\n"
            f"\n  {auth_url}\n"
        )

    async def callback() -> tuple[str, str | None]:
        url = input("Callback URL: ").strip()
        params = parse_qs(urlparse(url).query)
        if "code" not in params or not params["code"]:
            msg = "Callback URL is missing the 'code' parameter."
            raise RuntimeError(msg)
        return params["code"][0], (params.get("state") or [None])[0]

    return redirect, callback


def build_oauth_provider(
    *,
    server_name: str,  # noqa: ARG001 - reserved for future error-message use
    server_url: str,
    storage: TokenStorage,
) -> OAuthClientProvider:
    """Construct a paste-back `OAuthClientProvider` for an MCP server.

    The metadata defaults match what most public MCP servers accept under
    Dynamic Client Registration; servers are expected to advertise scopes
    via their OAuth metadata document.

    Args:
        server_name: The name of the MCP server (for future error messages).
        server_url: The MCP server's URL (e.g., "https://mcp.notion.com/mcp").
        storage: A TokenStorage instance for persisting OAuth credentials.

    Returns:
        An OAuthClientProvider configured for paste-back OAuth flow.
    """
    redirect, callback = _make_paste_back_handlers()
    metadata = OAuthClientMetadata(
        client_name="deepagents-cli",
        redirect_uris=[AnyUrl("http://localhost/callback")],
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
    )
    return OAuthClientProvider(
        server_url=server_url,
        client_metadata=metadata,
        storage=storage,
        redirect_handler=redirect,
        callback_handler=callback,
    )


async def _drive_handshake(
    connections: dict, storage: FileTokenStorage
) -> None:
    """Open a one-shot MCP session to trigger the OAuth handshake.

    The MCP SDK's `OAuthClientProvider` hooks into session startup and runs
    discovery → DCR → paste-back handlers → token exchange. Tokens land in
    `storage` via `set_tokens` before this returns.
    """
    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient(connections=connections)
    server_name = next(iter(connections))
    async with client.session(server_name):
        # Entering the session drives the full flow.
        pass


async def login(
    *,
    server_name: str,
    server_config: dict,
) -> None:
    """Drive OAuth login for `server_name`, persisting tokens on success.

    Raises:
        ValueError: If `server_config` isn't an OAuth http/sse server.
    """
    from langchain_mcp_adapters.sessions import (
        SSEConnection,
        StreamableHttpConnection,
    )

    if server_config.get("auth") != "oauth":
        msg = (
            f"Server '{server_name}' does not use OAuth "
            "(set \"auth\": \"oauth\" in mcpServers)."
        )
        raise ValueError(msg)
    transport = server_config.get("type") or server_config.get("transport", "stdio")
    if transport not in {"http", "sse"}:
        msg = (
            f"Server '{server_name}' uses {transport!r} transport; "
            "OAuth login is only valid for http/sse."
        )
        raise ValueError(msg)

    storage = FileTokenStorage(server_name)
    provider = build_oauth_provider(
        server_name=server_name,
        server_url=server_config["url"],
        storage=storage,
    )
    conn: dict
    if transport == "http":
        conn = StreamableHttpConnection(
            transport="streamable_http", url=server_config["url"], auth=provider
        )
    else:
        conn = SSEConnection(transport="sse", url=server_config["url"], auth=provider)

    await _drive_handshake({server_name: conn}, storage)
    print(  # noqa: T201 - user-facing confirmation
        f"Logged in to MCP server '{server_name}'. "
        f"Tokens saved to {storage._path}."
    )
