"""Codex OAuth authentication - PKCE login, refresh, logout."""

from __future__ import annotations

import base64
import hashlib
import http.server
import json
import logging
import secrets
import socket
import threading
import time
import urllib.parse
import webbrowser
from typing import Any

import httpx

from deepagents_codex.errors import CodexAuthError
from deepagents_codex.status import CodexAuthInfo, CodexAuthStatus
from deepagents_codex.store import CodexAuthStore, CodexCredentials

logger = logging.getLogger(__name__)

# OpenAI Auth0 configuration (public client)
_AUTH0_DOMAIN = "auth0.openai.com"
_CLIENT_ID = "pdlLIX2Y72MIl2rhLhTE9VV9bN905kBh"
_AUDIENCE = "https://api.openai.com/v1"
_SCOPES = "openid profile email offline_access"
_CALLBACK_PORT_START = 8484
_CALLBACK_PORT_END = 8584
_CALLBACK_PATH = "/callback"


def _generate_pkce_pair() -> tuple[str, str]:
    """Generate a PKCE code verifier and challenge pair."""
    verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


def _find_available_port() -> int:
    """Find an available port for the OAuth callback server."""
    for port in range(_CALLBACK_PORT_START, _CALLBACK_PORT_END):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("localhost", port))
            except OSError:
                continue
            else:
                return port
    msg = f"No available ports in range {_CALLBACK_PORT_START}-{_CALLBACK_PORT_END}"
    raise CodexAuthError(msg)


class _OAuthCallbackHandler(http.server.BaseHTTPRequestHandler):
    """HTTP handler that captures the OAuth authorization code."""

    authorization_code: str | None = None
    error: str | None = None

    def do_GET(self) -> None:
        """Handle the OAuth callback GET request."""
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)

        if parsed.path != _CALLBACK_PATH:
            self.send_response(404)
            self.end_headers()
            return

        if "error" in params:
            _OAuthCallbackHandler.error = params["error"][0]
            self._send_html("Authentication failed. You can close this tab.")
            return

        code = params.get("code", [None])[0]
        if code:
            _OAuthCallbackHandler.authorization_code = code
            self._send_html("Authentication successful! You can close this tab.")
        else:
            _OAuthCallbackHandler.error = "No authorization code received"
            self._send_html("Authentication failed. You can close this tab.")

    def _send_html(self, message: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        html = f"<html><body><h2>{message}</h2></body></html>"
        self.wfile.write(html.encode())

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        """Suppress default HTTP server logging."""


def _exchange_code_for_tokens(
    code: str,
    verifier: str,
    redirect_uri: str,
) -> dict[str, Any]:
    """Exchange authorization code for tokens."""
    with httpx.Client(timeout=30) as client:
        resp = client.post(
            f"https://{_AUTH0_DOMAIN}/oauth/token",
            json={
                "grant_type": "authorization_code",
                "client_id": _CLIENT_ID,
                "code_verifier": verifier,
                "code": code,
                "redirect_uri": redirect_uri,
            },
        )
        if resp.status_code != 200:  # noqa: PLR2004
            msg = f"Token exchange failed: {resp.text}"
            raise CodexAuthError(msg)
        return resp.json()


def login(
    *, headless: bool = False, store: CodexAuthStore | None = None
) -> CodexCredentials:
    """Authenticate with the Codex backend via browser-based OAuth PKCE flow.

    Args:
        headless: If True, print the URL for manual copy instead of opening browser.
        store: Optional credential store (defaults to standard location).

    Returns:
        The obtained credentials.

    Raises:
        CodexAuthError: If authentication fails.
    """
    _store = store or CodexAuthStore()
    verifier, challenge = _generate_pkce_pair()

    port = _find_available_port()
    redirect_uri = f"http://localhost:{port}{_CALLBACK_PATH}"

    auth_url = (
        f"https://{_AUTH0_DOMAIN}/authorize?"
        + urllib.parse.urlencode({
            "response_type": "code",
            "client_id": _CLIENT_ID,
            "redirect_uri": redirect_uri,
            "scope": _SCOPES,
            "audience": _AUDIENCE,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        })
    )

    # Reset handler state
    _OAuthCallbackHandler.authorization_code = None
    _OAuthCallbackHandler.error = None

    server = http.server.HTTPServer(("localhost", port), _OAuthCallbackHandler)
    server_thread = threading.Thread(target=server.handle_request, daemon=True)
    server_thread.start()

    if headless:
        print(f"\nOpen this URL in your browser to authenticate:\n\n{auth_url}\n")  # noqa: T201
    else:
        print("Opening browser for authentication...")  # noqa: T201
        webbrowser.open(auth_url)

    # Wait for callback (timeout after 120 seconds)
    server_thread.join(timeout=120)
    server.server_close()

    if _OAuthCallbackHandler.error:
        msg = f"Authentication failed: {_OAuthCallbackHandler.error}"
        raise CodexAuthError(msg)

    if not _OAuthCallbackHandler.authorization_code:
        msg = "Authentication timed out. No callback received within 120 seconds."
        raise CodexAuthError(msg)

    # Exchange code for tokens
    token_data = _exchange_code_for_tokens(
        _OAuthCallbackHandler.authorization_code, verifier, redirect_uri
    )

    access_token = token_data.get("access_token")
    refresh_token = token_data.get("refresh_token")
    expires_in = token_data.get("expires_in", 3600)

    if not access_token or not refresh_token:
        msg = "Token response missing access_token or refresh_token"
        raise CodexAuthError(msg)

    # Try to extract email from ID token (best effort)
    user_email = _extract_email_from_id_token(token_data.get("id_token"))

    creds = CodexCredentials(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_at=time.time() + expires_in,
        user_email=user_email,
    )
    _store.save(creds)
    return creds


def _extract_email_from_id_token(id_token: str | None) -> str | None:
    """Best-effort extraction of email from JWT id_token (no verification)."""
    if not id_token:
        return None
    try:
        # JWT is header.payload.signature -- we just need the payload
        parts = id_token.split(".")
        if len(parts) < 2:  # noqa: PLR2004
            return None
        # Add padding
        payload = parts[1]
        payload += "=" * (4 - len(payload) % 4)
        data = json.loads(base64.urlsafe_b64decode(payload))
        return data.get("email")
    except (ValueError, KeyError, UnicodeDecodeError, json.JSONDecodeError):
        logger.debug("Could not extract email from id_token", exc_info=True)
        return None


def logout(*, store: CodexAuthStore | None = None) -> bool:
    """Remove stored Codex credentials.

    Args:
        store: Optional credential store.

    Returns:
        True if credentials were deleted, False if none existed.
    """
    _store = store or CodexAuthStore()
    return _store.delete()


def get_auth_status(*, store: CodexAuthStore | None = None) -> CodexAuthInfo:
    """Check the current Codex authentication status.

    Args:
        store: Optional credential store.

    Returns:
        Structured auth status info.
    """
    _store = store or CodexAuthStore()
    creds = _store.load()
    if creds is None:
        if _store.path.exists():
            return CodexAuthInfo(
                status=CodexAuthStatus.CORRUPT,
                message="Credentials file exists but is malformed",
            )
        return CodexAuthInfo(
            status=CodexAuthStatus.NOT_AUTHENTICATED,
            message="Not authenticated. Run 'deepagents auth login --provider codex'",
        )
    if creds.is_expired:
        return CodexAuthInfo(
            status=CodexAuthStatus.EXPIRED,
            user_email=creds.user_email,
            expires_at=creds.expires_at,
            message="Session expired. Run 'deepagents auth login --provider codex'",
        )
    return CodexAuthInfo(
        status=CodexAuthStatus.AUTHENTICATED,
        user_email=creds.user_email,
        expires_at=creds.expires_at,
        message="Authenticated",
    )


def refresh_access_token(
    creds: CodexCredentials, *, store: CodexAuthStore | None = None
) -> CodexCredentials:
    """Refresh an expired access token.

    Args:
        creds: Current credentials with a valid refresh token.
        store: Optional credential store.

    Returns:
        New credentials with a fresh access token.

    Raises:
        CodexAuthError: If refresh fails.
    """
    _store = store or CodexAuthStore()
    with httpx.Client(timeout=30) as client:
        resp = client.post(
            f"https://{_AUTH0_DOMAIN}/oauth/token",
            json={
                "grant_type": "refresh_token",
                "client_id": _CLIENT_ID,
                "refresh_token": creds.refresh_token,
            },
        )
        if resp.status_code != 200:  # noqa: PLR2004
            msg = f"Token refresh failed: {resp.text}"
            raise CodexAuthError(msg)
        data = resp.json()

    new_creds = CodexCredentials(
        access_token=data["access_token"],
        refresh_token=data.get("refresh_token", creds.refresh_token),
        expires_at=time.time() + data.get("expires_in", 3600),
        user_email=creds.user_email,
    )
    _store.save(new_creds)
    return new_creds
