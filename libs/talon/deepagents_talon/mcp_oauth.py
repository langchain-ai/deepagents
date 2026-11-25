"""Validated OAuth client settings for MCP servers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Literal, cast
from urllib.parse import urlparse

from pydantic import AnyUrl

_ASCII_CONTROL_LIMIT = 32
_ASCII_DELETE = 127
_MAX_PORT = 65535
_FIELDS = frozenset(
    {"client_id", "client_secret", "token_endpoint_auth_method", "callback_url", "scopes"}
)
SECRET_REFERENCE = re.compile(r"\$\{[A-Za-z_][A-Za-z0-9_]*\}")
"""The only accepted spelling of `client_secret` in stored configuration."""

TokenEndpointAuthMethod = Literal["client_secret_basic", "client_secret_post"]
_AUTH_METHODS: frozenset[str] = frozenset({"client_secret_basic", "client_secret_post"})


class MCPOAuthConfigError(ValueError):
    """OAuth settings are invalid, with a credential-safe diagnostic."""


@dataclass(frozen=True)
class MCPOAuthConfig:
    """Explicit client settings, independent of stored registration.

    A `client_secret` makes this a confidential client. It is held only in
    memory: it is excluded from `repr`, from the storage identity, and from the
    credential file.
    """

    client_id: str | None = None
    callback_url: str | None = None
    scopes: tuple[str, ...] | None = None
    client_secret: str | None = field(default=None, repr=False)
    token_endpoint_auth_method: TokenEndpointAuthMethod | None = None

    def storage_identity(self) -> str:
        """Return the configuration identity used to isolate credentials.

        The secret and its auth method are left out so rotating a secret, or
        upgrading a public client to a confidential one, keeps the grant.
        """
        return json.dumps([self.client_id, self.callback_url, self.scopes])


def require_secret_reference(value: object) -> None:
    """Reject an `oauth.client_secret` that is not a whole `${VAR}` reference.

    Call this on raw configuration, before environment resolution. A literal
    secret in `mcp.json` would sit in a file the agent can read and edit, and a
    `${VAR:-default}` fallback is a literal in disguise.

    Args:
        value: The raw `oauth` setting of one server.

    Raises:
        MCPOAuthConfigError: `client_secret` is present but not a reference.
    """
    if not isinstance(value, dict) or "client_secret" not in value:
        return
    secret = value["client_secret"]
    if not isinstance(secret, str) or SECRET_REFERENCE.fullmatch(secret) is None:
        msg = (
            "MCP oauth.client_secret must be an environment reference such as "
            "${MY_SERVER_CLIENT_SECRET}; set the secret in the environment, not in the config."
        )
        raise MCPOAuthConfigError(msg)


def parse_oauth_config(value: object) -> MCPOAuthConfig:
    """Validate optional client settings without reflecting input values."""
    if value is None:
        return MCPOAuthConfig()
    if not isinstance(value, dict):
        msg = "MCP oauth must be an object."
        raise MCPOAuthConfigError(msg)
    if value.keys() - _FIELDS:
        msg = (
            "Supported MCP oauth fields: client_id, client_secret, "
            "token_endpoint_auth_method, callback_url, scopes."
        )
        raise MCPOAuthConfigError(msg)
    client_id = value.get("client_id")
    if "client_id" in value and (
        not isinstance(client_id, str) or not client_id.strip() or _has_controls(client_id)
    ):
        msg = "MCP oauth.client_id must be a non-empty string without control characters."
        raise MCPOAuthConfigError(msg)
    scopes = _scopes(value)
    if scopes is not None and client_id is None:
        msg = (
            "MCP oauth.scopes requires an explicit client_id; omit scopes for dynamic registration."
        )
        raise MCPOAuthConfigError(msg)
    secret, method = _confidential(value)
    return MCPOAuthConfig(
        client_id=cast("str | None", client_id),
        callback_url=_callback_url(value),
        scopes=scopes,
        client_secret=secret,
        token_endpoint_auth_method=method,
    )


def _confidential(
    value: dict[str, object],
) -> tuple[str | None, TokenEndpointAuthMethod | None]:
    secret = value.get("client_secret")
    method = value.get("token_endpoint_auth_method")
    if "client_secret" in value:
        if not isinstance(secret, str) or not secret or _has_controls(secret):
            msg = "MCP oauth.client_secret must be a non-empty string without control characters."
            raise MCPOAuthConfigError(msg)
        if "client_id" not in value:
            msg = "MCP oauth.client_secret requires the client_id it was issued with."
            raise MCPOAuthConfigError(msg)
    if "token_endpoint_auth_method" in value:
        if secret is None:
            msg = "MCP oauth.token_endpoint_auth_method requires client_secret."
            raise MCPOAuthConfigError(msg)
        if method not in _AUTH_METHODS:
            msg = (
                "MCP oauth.token_endpoint_auth_method must be client_secret_basic "
                "or client_secret_post."
            )
            raise MCPOAuthConfigError(msg)
    if secret is None:
        return None, None
    return cast("str", secret), cast("TokenEndpointAuthMethod", method or "client_secret_basic")


def _has_controls(value: str) -> bool:
    return any(
        ord(character) < _ASCII_CONTROL_LIMIT or ord(character) == _ASCII_DELETE
        for character in value
    )


def _callback_url(value: dict[str, object]) -> str | None:
    if "callback_url" not in value:
        return None
    url = value.get("callback_url")
    msg = (
        "MCP oauth.callback_url must be an HTTP loopback URL with an explicit port and path, "
        "without credentials, query, or fragment."
    )
    if not isinstance(url, str) or _has_controls(url) or any(c.isspace() for c in url):
        raise MCPOAuthConfigError(msg)
    try:
        parsed = urlparse(url)
        valid = (
            parsed.scheme == "http"
            and parsed.hostname in {"localhost", "127.0.0.1", "::1"}
            and parsed.port is not None
            and 1 <= parsed.port <= _MAX_PORT
            and parsed.username is None
            and parsed.password is None
            and parsed.path.startswith("/")
            and not parsed.params
            and "?" not in url
            and "#" not in url
        )
    except ValueError:
        raise MCPOAuthConfigError(msg) from None
    if not valid:
        raise MCPOAuthConfigError(msg)
    try:
        canonical = str(AnyUrl(url))
    except ValueError:
        raise MCPOAuthConfigError(msg) from None
    if canonical != url:
        msg = (
            "MCP oauth callback URL must use canonical spelling and a non-default port; "
            "register that exact URL with the server."
        )
        raise MCPOAuthConfigError(msg)
    return url


def _scopes(value: dict[str, object]) -> tuple[str, ...] | None:
    if "scopes" not in value:
        return None
    scopes = value["scopes"]
    if not isinstance(scopes, list) or not all(
        isinstance(scope, str) and re.fullmatch(r"[\x21\x23-\x5b\x5d-\x7e]+", scope)
        for scope in scopes
    ):
        msg = "MCP oauth.scopes must be a list of non-empty OAuth scope tokens."
        raise MCPOAuthConfigError(msg)
    return tuple(dict.fromkeys(cast("list[str]", scopes)))
