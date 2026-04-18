"""OAuth + bearer-token auth helpers for MCP servers."""

from __future__ import annotations

import contextlib
import json
import os
import re
import stat
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import parse_qs, urlparse

from mcp.client.auth import OAuthClientProvider, TokenStorage
from mcp.shared.auth import (
    AnyUrl,
    OAuthClientInformationFull,
    OAuthClientMetadata,
    OAuthToken,
)

if TYPE_CHECKING:
    import httpx

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

_STORAGE_VERSION = 1

# Cap HTTP response bodies (and replays) shown in error messages so a
# misconfigured server can't flood the user's terminal with MBs of HTML.
_BODY_TRUNCATE_AT = 2000


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


_SLACK_TEAM_ID_RE = re.compile(r"^T[A-Z0-9]{2,}$")


def _prompt_slack_team() -> str | None:
    """Interactively ask the user which Slack workspace to install into.

    Slack's `/oauth/v2/authorize` accepts an optional `team=<team_id>`
    parameter that pre-selects the workspace. When the user leaves the prompt
    blank, Slack shows its own workspace picker on the authorization page,
    which is the right default for first-time or single-workspace users.

    Returns:
        The entered team ID (e.g., ``T01234567``), or ``None`` if the user
        left the prompt blank.

    Raises:
        RuntimeError: If the entered ID isn't a valid Slack team ID format.
    """
    raw = input(
        "Slack team ID to install the app into "
        "(e.g. T01234567 — leave blank to pick on Slack's page): "
    ).strip()
    if not raw:
        return None
    if not _SLACK_TEAM_ID_RE.fullmatch(raw):
        msg = (
            f"Invalid Slack team ID {raw!r}. Team IDs start with 'T' and "
            "contain only uppercase letters and digits (e.g. 'T01234567')."
        )
        raise RuntimeError(msg)
    return raw


def resolve_headers(
    headers: dict[str, str],
    *,
    server_name: str | None = None,
) -> dict[str, str]:
    """Resolve `${VAR}` env-var references in header values.

    `$${VAR}` is the escape form and collapses to the literal `${VAR}` with no
    lookup. A dollar sign not followed by `{` or `$` is left untouched.

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

    Returns:
        Absolute path to the per-user mcp-tokens directory.
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
    server_name: str,  # noqa: ARG001 - reserved for future error-message use
    server_url: str,
    storage: TokenStorage,
    extra_auth_params: dict[str, str] | None = None,
) -> OAuthClientProvider:
    """Construct a paste-back `OAuthClientProvider` for an MCP server.

    The metadata defaults match what most public MCP servers accept under
    Dynamic Client Registration; servers are expected to advertise scopes
    via their OAuth metadata document. Slack is special-cased — it rejects
    DCR and requires a pre-registered app ID (see `_SLACK_MCP_CLIENT_ID`),
    so the metadata uses Slack's registered `https://localhost` redirect URI.

    Args:
        server_name: The name of the MCP server (for future error messages).
        server_url: The MCP server's URL (e.g., "https://mcp.notion.com/mcp").
        storage: A TokenStorage instance for persisting OAuth credentials.
        extra_auth_params: Extra query params appended to the authorization
            URL before the user is prompted to open it. Used for Slack's
            ``team=<id>`` workspace selector.

    Returns:
        An OAuthClientProvider configured for paste-back OAuth flow.
    """
    redirect, callback = _make_paste_back_handlers(extra_auth_params=extra_auth_params)
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


def _find_http_status_error(exc: BaseException) -> httpx.HTTPStatusError | None:
    """Find the first `httpx.HTTPStatusError` wrapped inside *exc*.

    Recursively searches *exc* and any `BaseExceptionGroup` / chained
    `__cause__` / `__context__` it wraps. The MCP streamable-HTTP client runs
    inside an anyio `TaskGroup`, so HTTP errors from `raise_for_status` arrive
    wrapped in one or more `BaseExceptionGroup` layers.

    Returns:
        The innermost `httpx.HTTPStatusError` found, or `None` if the
        exception tree doesn't contain one.
    """
    import httpx

    if isinstance(exc, httpx.HTTPStatusError):
        return exc
    sub_exceptions = getattr(exc, "exceptions", None)
    if sub_exceptions:
        for sub in sub_exceptions:
            found = _find_http_status_error(sub)
            if found is not None:
                return found
    cause = exc.__cause__ or exc.__context__
    if cause is not None and cause is not exc:
        return _find_http_status_error(cause)
    return None


async def _drive_handshake(
    connections: dict,
    storage: FileTokenStorage,  # noqa: ARG001 — documents caller contract; SDK writes via the provider
) -> None:
    """Open a one-shot MCP session to trigger the OAuth handshake.

    The MCP SDK's `OAuthClientProvider` hooks into session startup and runs
    discovery → DCR → paste-back handlers → token exchange. Tokens land in
    `storage` via `set_tokens` before this returns.

    When the post-OAuth MCP init request fails with an HTTP error, the
    response body almost always carries the real reason (invalid_scope,
    missing_team, etc.). We unwrap the ExceptionGroup-wrapped error so that
    body gets surfaced to the user instead of being swallowed.

    Raises:
        RuntimeError: If the MCP server rejects the initialize request with
            an HTTP error; the message includes status line, headers, and the
            response body when recoverable.
    """
    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient(connections=connections)
    server_name = next(iter(connections))
    try:
        async with client.session(server_name):
            # Entering the session drives the full flow.
            pass
    except BaseException as exc:
        status_error = _find_http_status_error(exc)
        if status_error is None:
            raise
        response = status_error.response
        request = status_error.request

        # Streamed responses have already been consumed by `raise_for_status`;
        # try several paths to recover the body for the user.
        body = ""
        for reader in (
            lambda: response.text,
            lambda: response.content.decode(errors="replace"),
        ):
            try:
                body = reader()
                if body:
                    break
            except Exception:  # noqa: BLE001, S112 — best-effort diagnostic, swallow and try next
                continue
        if not body:
            try:
                await response.aread()
                body = response.text
            except Exception:  # noqa: BLE001, S110 — best-effort diagnostic, body stays empty
                pass

        lines = [
            f"MCP server '{server_name}' rejected the initialize request:",
            (
                f"  HTTP {response.status_code} {response.reason_phrase} "
                f"from {request.method} {request.url}"
            ),
        ]
        content_type = response.headers.get("content-type")
        if content_type:
            lines.append(f"  Content-Type: {content_type}")
        slack_err = response.headers.get("x-slack-error")
        if slack_err:
            lines.append(f"  X-Slack-Error: {slack_err}")
        www_auth = response.headers.get("www-authenticate")
        if www_auth:
            lines.append(f"  WWW-Authenticate: {www_auth}")

        if body:
            truncated = (
                body
                if len(body) <= _BODY_TRUNCATE_AT
                else body[:_BODY_TRUNCATE_AT] + "…[truncated]"
            )
            lines.extend(("\nResponse body:", truncated))
        else:
            # Streamed responses in the MCP SDK are consumed by raise_for_status
            # before we can read the body. Replay the request manually as a
            # non-streaming POST so we can see what the server actually said.
            replay = await _replay_request_for_body(request)
            if replay is not None:
                lines.extend(("\n[Replayed request to capture response body]", replay))
            else:
                lines.append(
                    "\n(Response body was empty on both the original request "
                    "and a replay. Check the Slack app's OAuth scopes and the "
                    "MCP-Protocol-Version the server expects.)"
                )

        raise RuntimeError("\n".join(lines)) from status_error


async def _replay_request_for_body(request: httpx.Request) -> str | None:
    """Resend *request* as a non-streaming POST so we can read the body.

    The MCP SDK streams responses, which makes failed 4xx/5xx bodies
    unavailable after `raise_for_status` closes the stream. Replaying the
    same request outside of streaming mode is the simplest way to recover
    the body for diagnostics.

    Returns:
        A human-readable summary (status line + body, truncated) or ``None``
        if the replay itself failed.
    """
    import httpx

    try:
        body_bytes = request.content
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.request(
                request.method,
                str(request.url),
                content=body_bytes,
                headers=dict(request.headers),
            )
        preview = resp.text
        if len(preview) > _BODY_TRUNCATE_AT:
            preview = preview[:_BODY_TRUNCATE_AT] + "…[truncated]"
        return (
            f"HTTP {resp.status_code} {resp.reason_phrase}\n"
            f"Content-Type: {resp.headers.get('content-type', '(none)')}\n\n"
            f"{preview}"
        )
    except Exception as exc:  # noqa: BLE001 — best-effort diagnostic
        return f"(replay failed: {type(exc).__name__}: {exc})"


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

    await _drive_handshake({server_name: conn}, storage)
    print(  # noqa: T201 - user-facing confirmation
        f"Logged in to MCP server '{server_name}'. Tokens saved to {storage._path}."
    )
