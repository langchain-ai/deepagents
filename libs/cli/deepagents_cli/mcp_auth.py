"""OAuth + bearer-token auth helpers for MCP servers."""

from __future__ import annotations

import contextlib
import json
import os
import re
import stat
from collections.abc import Awaitable, Callable
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from mcp.client.auth import OAuthClientProvider, TokenStorage
from mcp.shared.auth import (
    AnyUrl,
    OAuthClientInformationFull,
    OAuthClientMetadata,
    OAuthToken,
)

_REF_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

_STORAGE_VERSION = 1


# Slack's MCP server rejects Dynamic Client Registration — every client must
# use a pre-registered Slack app ID. See Slack's MCP "App Identity" docs.
# https://localhost is the redirect URI registered on the Slack app; the CLI's
# paste-back flow lets the user copy the final URL after OAuth approval.
_SLACK_MCP_CLIENT_ID = "4518649543379.10944517634130"
_SLACK_REDIRECT_URI = "https://localhost"


def _is_slack_mcp_url(url: str) -> bool:
    """Return True when *url* points at a Slack-hosted MCP endpoint.

    Matches any `*.slack.com` or `slack.com` host so the special-case stays
    robust across subdomain changes (e.g., `mcp.slack.com` → `api.slack.com`).
    """
    host = urlparse(url).hostname or ""
    return host == "slack.com" or host.endswith(".slack.com")


# GitHub's Remote MCP Server rejects DCR and its token endpoint requires a
# client_secret — unsafe to distribute in a public CLI. RFC 8628 Device Flow
# works with client_id only, so the CLI obtains a GitHub access token that
# way and pre-seeds it into storage; the MCP SDK then treats auth as already
# complete and attaches `Authorization: Bearer <token>` to every request.
_GITHUB_MCP_CLIENT_ID = "Iv23libxz8qOApH0WQL3"
_GITHUB_DEVICE_CODE_URL = "https://github.com/login/device/code"
_GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"  # noqa: S105 — OAuth endpoint URL, not a secret


def _is_github_mcp_url(url: str) -> bool:
    """Return True when *url* points at GitHub's remote MCP endpoint."""
    return (urlparse(url).hostname or "") == "api.githubcopilot.com"


def _prompt_slack_team() -> str | None:
    """Interactively ask the user which Slack workspace to install into.

    Slack's `/oauth/v2/authorize` accepts an optional `team=<team_id>`
    parameter that pre-selects the workspace. When the user leaves the prompt
    blank, Slack shows its own workspace picker on the authorization page,
    which is the right default for first-time or single-workspace users.

    Returns:
        The entered team ID, or ``None`` if the user left the prompt blank.
    """
    raw = input(
        "Slack team ID to install the app into "
        "(e.g. T01234567 — leave blank to pick on Slack's page): "
    ).strip()
    return raw or None


def resolve_headers(
    headers: dict[str, str],
    *,
    server_name: str | None = None,
) -> dict[str, str]:
    """Resolve `${VAR}` env-var references in header values.

    A dollar sign not followed by `{<ident>}` is left untouched.

    Returns:
        A new dict with all env-var references resolved to their current values.

    Raises:
        TypeError: If a header value is not a string.
        RuntimeError: If a `${VAR}` references an unset environment variable
            (propagated from the inner `_interpolate` helper).
    """  # noqa: DOC502 — RuntimeError is raised via `_interpolate`
    resolved: dict[str, str] = {}
    for name, value in headers.items():
        if not isinstance(value, str):
            where = f"mcpServers.{server_name}.headers.{name}" if server_name else name
            msg = f"{where} must be a string, got {type(value).__name__}"
            raise TypeError(msg)
        resolved[name] = _interpolate(value, header=name, server_name=server_name)
    return resolved


def _interpolate(s: str, *, header: str, server_name: str | None) -> str:
    def replace(m: re.Match[str]) -> str:
        var_name = m.group(1)
        val = os.environ.get(var_name)
        if val is None:
            where = (
                f"mcpServers.{server_name}.headers.{header}" if server_name else header
            )
            msg = (
                f"{where} references unset env var {var_name}. "
                f"Set {var_name} in the environment or remove the reference."
            )
            raise RuntimeError(msg)
        return val

    return _REF_RE.sub(replace, s)


def _tokens_dir() -> Path:
    """Return `~/.deepagents/mcp-tokens/`. `Path.home()` re-reads `$HOME`."""
    return Path.home() / ".deepagents" / "mcp-tokens"


class FileTokenStorage(TokenStorage):
    """File-backed `TokenStorage` under `~/.deepagents/mcp-tokens/<server>.json`.

    Atomic-write via tmp-file + `os.replace` + chmod(0o600). Directory is
    created with mode 0o700. Permission bits are POSIX-only.
    """

    def __init__(self, server_name: str) -> None:
        """Bind this storage to *server_name* under `~/.deepagents/mcp-tokens/`."""
        self._server_name = server_name

    @property
    def _path(self) -> Path:
        return _tokens_dir() / f"{self._server_name}.json"

    async def get_tokens(self) -> OAuthToken | None:
        """Return the stored `OAuthToken`, or `None` if none is persisted."""
        data = self._read()
        if data is None:
            return None
        raw = data.get("tokens")
        if raw is None:
            return None
        return OAuthToken.model_validate(raw)

    async def set_tokens(self, tokens: OAuthToken) -> None:
        """Persist *tokens* to disk, merging with any existing client_info."""
        data = self._read() or {}
        data["version"] = _STORAGE_VERSION
        data["tokens"] = json.loads(tokens.model_dump_json(exclude_none=True))
        self._write(data)

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        """Return the stored client registration, or `None` if none is persisted."""
        data = self._read()
        if data is None:
            return None
        raw = data.get("client_info")
        if raw is None:
            return None
        return OAuthClientInformationFull.model_validate(raw)

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        """Persist *client_info* to disk, merging with any existing tokens."""
        data = self._read() or {}
        data["version"] = _STORAGE_VERSION
        data["client_info"] = json.loads(client_info.model_dump_json(exclude_none=True))
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
            with contextlib.suppress(OSError):
                Path(path.parent).chmod(stat.S_IRWXU)  # 0o700
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
        try:
            Path(tmp).replace(path)
        except Exception:
            # Clean up the tmp file so no partial state leaks at the final path.
            with contextlib.suppress(OSError):
                tmp.unlink()
            raise
        if hasattr(os, "chmod"):
            with contextlib.suppress(OSError):
                Path(path).chmod(stat.S_IRUSR | stat.S_IWUSR)  # 0o600


RedirectHandler = Callable[[str], Awaitable[None]]
CallbackHandler = Callable[[], Awaitable[tuple[str, str | None]]]


class MCPReauthRequiredError(RuntimeError):
    """Raised when an MCP server needs interactive re-authentication.

    The runtime OAuth provider (used during agent tool calls) raises this
    instead of driving the paste-back authorization-code flow — that flow
    calls `input()`, which blocks the event loop in Textual and trips
    blockbuster in LangGraph dev. Callers catch this, invalidate the cached
    session, and surface a `ToolException` instructing the user to run
    `deepagents mcp login <server>`.
    """

    def __init__(self, server_name: str) -> None:
        """Build with *server_name* so the message tells the user what to fix."""
        self.server_name = server_name
        super().__init__(
            f"MCP server {server_name!r} needs re-authentication. "
            f"Run: deepagents mcp login {server_name}"
        )


def _make_reauth_required_handlers(
    server_name: str,
) -> tuple[RedirectHandler, CallbackHandler]:
    """Return OAuth handlers that refuse to prompt and raise instead.

    Attached by `build_oauth_provider(..., interactive=False)` on the runtime
    path so a mid-session 401 → SDK auth-flow fallback surfaces as a clean
    `MCPReauthRequiredError` rather than blocking on `input()`.
    """

    async def redirect(_auth_url: str) -> None:  # noqa: RUF029 — MCP SDK requires async handler
        raise MCPReauthRequiredError(server_name)

    async def callback() -> tuple[str, str | None]:  # noqa: RUF029 — MCP SDK requires async handler
        raise MCPReauthRequiredError(server_name)

    return redirect, callback


def _make_paste_back_handlers(
    *, extra_auth_params: dict[str, str] | None = None
) -> tuple[RedirectHandler, CallbackHandler]:
    """Create paste-back redirect and callback handlers for OAuth.

    Args:
        extra_auth_params: Extra query-string parameters to merge into the
            authorization URL before showing it to the user. Used to thread
            vendor-specific hints (e.g., Slack's ``team=<id>``) without
            changing the MCP SDK's URL-builder.

    Returns:
        A tuple of (redirect_handler, callback_handler) functions that implement
        the paste-back OAuth flow for interactive CLI use.
    """
    extras = dict(extra_auth_params or {})

    async def redirect(auth_url: str) -> None:  # noqa: RUF029 — MCP SDK requires async handler
        final_url = _append_query_params(auth_url, extras) if extras else auth_url
        print(  # noqa: T201 - intentional user-facing prompt
            "\nOpen this URL in a browser, approve access, then paste the full "
            "callback URL back here:\n"
            f"\n  {final_url}\n"
        )

    async def callback() -> tuple[str, str | None]:  # noqa: RUF029 — MCP SDK requires async handler
        url = input("Callback URL: ").strip()  # noqa: ASYNC250 — interactive paste-back is intentional
        params = parse_qs(urlparse(url).query)
        if "code" not in params or not params["code"]:
            msg = "Callback URL is missing the 'code' parameter."
            raise RuntimeError(msg)
        return params["code"][0], (params.get("state") or [None])[0]

    return redirect, callback


def _append_query_params(url: str, params: dict[str, str]) -> str:
    """Return *url* with *params* merged into its query string.

    New keys override any existing value for the same key. Preserves the
    original scheme, netloc, path, and fragment.
    """
    from urllib.parse import urlencode, urlunparse

    parsed = urlparse(url)
    existing = dict(parse_qs(parsed.query, keep_blank_values=True))
    for key, value in params.items():
        existing[key] = [value]
    new_query = urlencode(existing, doseq=True)
    return urlunparse(parsed._replace(query=new_query))


def build_oauth_provider(
    *,
    server_name: str,
    server_url: str,
    storage: TokenStorage,
    extra_auth_params: dict[str, str] | None = None,
    interactive: bool = True,
) -> OAuthClientProvider:
    """Construct an `OAuthClientProvider` for an MCP server.

    The metadata defaults match what most public MCP servers accept under
    Dynamic Client Registration; servers are expected to advertise scopes
    via their OAuth metadata document. Slack is special-cased — it rejects
    DCR and requires a pre-registered app ID (see `_SLACK_MCP_CLIENT_ID`),
    so the metadata uses Slack's registered `https://localhost` redirect URI.

    Args:
        server_name: The name of the MCP server — used in re-auth error
            messages when ``interactive`` is false.
        server_url: The MCP server's URL (e.g., "https://mcp.notion.com/mcp").
        storage: A TokenStorage instance for persisting OAuth credentials.
        extra_auth_params: Extra query params appended to the authorization
            URL before the user is prompted to open it. Used for Slack's
            ``team=<id>`` workspace selector. Ignored when
            ``interactive=False``.
        interactive: When true (the `deepagents mcp login` path), attach
            paste-back handlers that prompt on stdin. When false (the
            runtime tool-loading path), attach handlers that raise
            `MCPReauthRequiredError` instead of prompting — prompting from a
            tool call blocks the Textual event loop and trips blockbuster
            in LangGraph dev.

    Returns:
        An OAuthClientProvider configured for the chosen flow.
    """
    if interactive:
        redirect, callback = _make_paste_back_handlers(
            extra_auth_params=extra_auth_params
        )
    else:
        redirect, callback = _make_reauth_required_handlers(server_name=server_name)
    if _is_slack_mcp_url(server_url):
        metadata = OAuthClientMetadata(
            client_name="deepagents-cli",
            redirect_uris=[AnyUrl(_SLACK_REDIRECT_URI)],
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
            token_endpoint_auth_method="none",  # noqa: S106 — OAuth method literal, not a password
        )
    else:
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


async def _preseed_slack_client_info(storage: FileTokenStorage) -> None:
    """Write the hardcoded Slack `client_info` to *storage* if not already set.

    The MCP SDK's `OAuthClientProvider` calls `storage.get_client_info()` at
    the start of the flow and, when it returns non-None, skips Dynamic Client
    Registration entirely (see `mcp/client/auth/oauth2.py:561`). Pre-seeding
    storage is therefore how we opt out of DCR for Slack.
    """
    existing = await storage.get_client_info()
    if existing is not None and existing.client_id == _SLACK_MCP_CLIENT_ID:
        return
    await storage.set_client_info(
        OAuthClientInformationFull(
            client_id=_SLACK_MCP_CLIENT_ID,
            redirect_uris=[AnyUrl(_SLACK_REDIRECT_URI)],
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
            token_endpoint_auth_method="none",  # noqa: S106 — OAuth method literal, not a password
        )
    )


async def _run_device_flow(
    *,
    device_code_url: str,
    token_url: str,
    client_id: str,
    scope: str | None = None,
) -> OAuthToken:
    """Run OAuth 2.0 Device Authorization Grant (RFC 8628) and return the token.

    Generic across providers that implement RFC 8628. Omits `client_secret` —
    only use for public clients (CLIs, desktop apps). Callers pass provider-
    specific endpoints and the registered public `client_id`. The `scope`
    param is provider-dependent: GitHub Apps ignore it (permissions come from
    app config), OAuth Apps and most other IdPs require it.

    Returns:
        The issued `OAuthToken` (bearer access token, refresh token when
        supported, and expiry).

    Raises:
        RuntimeError: If the provider returns a terminal error, or the user
            doesn't complete authorization before the device code expires.
    """
    import asyncio

    import httpx

    init_data = {"client_id": client_id}
    if scope is not None:
        init_data["scope"] = scope

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            device_code_url,
            data=init_data,
            headers={"Accept": "application/json"},
        )
        resp.raise_for_status()
        dev = resp.json()

        print(  # noqa: T201 — user-facing prompt
            f"\nVisit {dev['verification_uri']} and enter code: "
            f"{dev['user_code']}\n(code expires in {dev['expires_in']}s)\n"
        )

        interval = max(int(dev.get("interval", 5)), 1)
        loop = asyncio.get_event_loop()
        deadline = loop.time() + int(dev["expires_in"])
        while loop.time() < deadline:
            await asyncio.sleep(interval)
            tok = await client.post(
                token_url,
                data={
                    "client_id": client_id,
                    "device_code": dev["device_code"],
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                },
                headers={"Accept": "application/json"},
            )
            tok.raise_for_status()
            body = tok.json()
            err = body.get("error")
            if err == "authorization_pending":
                continue
            if err == "slow_down":
                interval += 5
                continue
            if err:
                msg = f"Device flow failed: {err}: {body.get('error_description', '')}"
                raise RuntimeError(msg)
            return OAuthToken.model_validate(body)

    msg = "Device flow timed out; re-run `deepagents mcp login <server>`."
    raise RuntimeError(msg)


async def _preseed_github_auth(storage: FileTokenStorage) -> None:
    """Run GitHub Device Flow and persist the token + stub client_info.

    GitHub Apps ignore the ``scope`` param — permissions come from the app's
    configured permissions, so we don't send one. The client_info stub exists
    purely so the MCP SDK skips its DCR branch on subsequent sessions; token
    refresh against GitHub requires a client_secret and isn't supported in
    this flow, so callers must re-run ``mcp login <server>`` when tokens
    expire.
    """
    token = await _run_device_flow(
        device_code_url=_GITHUB_DEVICE_CODE_URL,
        token_url=_GITHUB_TOKEN_URL,
        client_id=_GITHUB_MCP_CLIENT_ID,
    )
    await storage.set_tokens(token)
    await storage.set_client_info(
        OAuthClientInformationFull(
            client_id=_GITHUB_MCP_CLIENT_ID,
            redirect_uris=[AnyUrl("http://localhost/callback")],
            grant_types=["urn:ietf:params:oauth:grant-type:device_code"],
            response_types=["code"],
            token_endpoint_auth_method="none",  # noqa: S106 — OAuth method literal, not a password
        )
    )


def find_reauth_required(exc: BaseException) -> MCPReauthRequiredError | None:
    """Find an `MCPReauthRequiredError` anywhere inside *exc*'s exception tree.

    The MCP SDK's auth flow runs inside anyio `TaskGroup`s, so an exception
    raised by a `callback_handler` arrives wrapped in one or more
    `BaseExceptionGroup` layers by the time it reaches a tool-call site.
    Shape mirrors `_find_http_status_error`.

    Returns:
        The innermost `MCPReauthRequiredError` found, or `None` if the
        exception tree doesn't contain one.
    """
    if isinstance(exc, MCPReauthRequiredError):
        return exc
    sub_exceptions = getattr(exc, "exceptions", None)
    if sub_exceptions:
        for sub in sub_exceptions:
            found = find_reauth_required(sub)
            if found is not None:
                return found
    cause = exc.__cause__ or exc.__context__
    if cause is not None and cause is not exc:
        return find_reauth_required(cause)
    return None


async def _drive_handshake(connections: dict) -> None:
    """Open a one-shot MCP session to trigger the OAuth handshake.

    The MCP SDK's `OAuthClientProvider` hooks into session startup and runs
    discovery → DCR → paste-back handlers → token exchange. Tokens land in
    storage via `set_tokens` before this returns.
    """
    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient(connections=connections)
    server_name = next(iter(connections))
    async with client.session(server_name):
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
            '(set "auth": "oauth" in mcpServers).'
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

    # GitHub's MCP server doesn't implement DCR, and its token endpoint
    # requires a client_secret that a public CLI can't safely distribute.
    # Route through Device Flow (no secret) and persist the bearer token so
    # regular MCP sessions pick it up via FileTokenStorage.get_tokens().
    if _is_github_mcp_url(server_config["url"]):
        await _preseed_github_auth(storage)
        print(  # noqa: T201 — user-facing confirmation
            f"Logged in to MCP server '{server_name}'. Tokens saved to {storage._path}."
        )
        return

    extra_auth_params: dict[str, str] = {}

    # Slack forbids Dynamic Client Registration; pre-seed storage with the
    # hardcoded app's client_info so the MCP SDK skips its DCR branch. Also
    # prompt for a team ID so users on multiple workspaces can pick which one
    # the app gets installed into; Slack's `/oauth/v2/authorize` accepts
    # `team=<team_id>` to pre-select, or shows a workspace picker when absent.
    if _is_slack_mcp_url(server_config["url"]):
        await _preseed_slack_client_info(storage)
        team_id = _prompt_slack_team()
        if team_id:
            extra_auth_params["team"] = team_id

    provider = build_oauth_provider(
        server_name=server_name,
        server_url=server_config["url"],
        storage=storage,
        extra_auth_params=extra_auth_params or None,
    )
    conn: StreamableHttpConnection | SSEConnection
    if transport == "http":
        conn = StreamableHttpConnection(
            transport="streamable_http", url=server_config["url"], auth=provider
        )
    else:
        conn = SSEConnection(transport="sse", url=server_config["url"], auth=provider)

    await _drive_handshake({server_name: conn})
    print(  # noqa: T201 - user-facing confirmation
        f"Logged in to MCP server '{server_name}'. Tokens saved to {storage._path}."
    )
