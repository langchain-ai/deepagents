"""Validated public-client OAuth settings for MCP servers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import cast
from urllib.parse import urlparse

from pydantic import AnyUrl

_ASCII_CONTROL_LIMIT = 32
_ASCII_DELETE = 127
_MAX_PORT = 65535


class MCPOAuthConfigError(ValueError):
    """OAuth settings are invalid, with a credential-safe diagnostic."""


@dataclass(frozen=True)
class MCPOAuthConfig:
    """Explicit public-client settings, independent of stored registration."""

    client_id: str | None = None
    callback_url: str | None = None
    scopes: tuple[str, ...] | None = None

    def storage_identity(self) -> str:
        """Return the configuration identity used to isolate credentials."""
        return json.dumps([self.client_id, self.callback_url, self.scopes])


def parse_oauth_config(value: object) -> MCPOAuthConfig:
    """Validate optional public-client settings without reflecting input values."""
    if value is None:
        return MCPOAuthConfig()
    if not isinstance(value, dict):
        msg = "MCP oauth must be an object."
        raise MCPOAuthConfigError(msg)
    if value.keys() - {"client_id", "callback_url", "scopes"}:
        msg = "Supported MCP oauth fields: client_id, callback_url, scopes; no secret is required."
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
    return MCPOAuthConfig(
        client_id=cast("str | None", client_id),
        callback_url=_callback_url(value),
        scopes=scopes,
    )


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
