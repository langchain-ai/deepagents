"""Resolve MCP server URLs in a payload to workspace-registered server IDs.

The deploy command does not auto-create MCP servers. Instead it validates that
every `mcp_server_url` the payload references already exists at the
`/v1/deepagents/mcp-servers` endpoint, and surfaces a friendly hint if not.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from deepagents_cli.deploy.api_client import ApiClient


class _ServerResolutionError(RuntimeError):
    """Base error for MCP server resolution failures."""

    def __init__(self, msg: str, *, urls: Iterable[str]) -> None:
        super().__init__(msg)
        self.urls = tuple(urls)


class UnresolvedServersError(_ServerResolutionError):
    """Raised when one or more `mcp_server_url`s aren't registered."""


class UninvokableServersError(_ServerResolutionError):
    """Raised when referenced MCP servers exist but cannot be invoked."""


def _normalize_url(url: str) -> str:
    return url.strip().rstrip("/").lower()


def _collect_referenced_urls(payload: dict[str, Any]) -> set[str]:
    urls: set[str] = set()
    for tool in (payload.get("tools") or {}).get("tools", []):
        u = tool.get("mcp_server_url")
        if isinstance(u, str) and u:
            urls.add(_normalize_url(u))
    for sa in payload.get("subagents", []):
        for tool in (sa.get("tools") or {}).get("tools", []):
            u = tool.get("mcp_server_url")
            if isinstance(u, str) and u:
                urls.add(_normalize_url(u))
    return urls


def resolve_referenced_servers(
    client: ApiClient,
    payload: dict[str, Any],
    *,
    cache: dict[str, str],
) -> dict[str, str]:
    """Return `{normalized_url → mcp_server_id}` for every URL in *payload*.

    Always hits the list endpoint for referenced URLs so stale local state
    cannot authorize a deploy against deleted, changed, or unavailable MCP
    servers. Raises `UnresolvedServersError` if any URL is unresolved after
    the list call and `UninvokableServersError` if a referenced server exists
    but this identity cannot invoke it.
    """
    _ = cache
    referenced = _collect_referenced_urls(payload)
    if not referenced:
        return {}

    out: dict[str, str] = {}
    uninvokable: set[str] = set()
    connect_hints: list[str] = []
    for server in client.list_mcp_servers():
        url = server.get("url")
        if isinstance(url, str) and url:
            key = _normalize_url(url)
            server_id = server.get("id")
            if key in referenced and server.get("can_invoke") is False:
                uninvokable.add(key)
                if (
                    server.get("auth_type") == "oauth"
                    and isinstance(server_id, str)
                    and server_id
                ):
                    connect_hints.append(
                        f"  deepagents mcp-servers connect {server_id}"
                    )
            elif key in referenced and isinstance(server_id, str):
                out[key] = server_id

    if uninvokable:
        urls = tuple(sorted(uninvokable))
        listed = "\n".join(f"  - {u}" for u in urls)
        msg = (
            "The following MCP server URLs referenced in your tools are "
            f"registered but this identity cannot invoke them:\n{listed}\n\n"
        )
        if connect_hints:
            hints = "\n".join(sorted(set(connect_hints)))
            msg += (
                "If these servers use OAuth, connect your user credentials with:\n"
                f"{hints}\n\n"
            )
        msg += (
            "Otherwise update MCP server permissions or reference a server this "
            "identity can invoke."
        )
        raise UninvokableServersError(msg, urls=urls)

    still_missing = referenced - out.keys()
    if still_missing:
        urls = tuple(sorted(still_missing))
        listed = "\n".join(f"  - {u}" for u in urls)
        msg = (
            f"The following MCP server URLs referenced in your tools "
            f"are not registered in this workspace:\n{listed}\n\n"
            f"Register each with:\n"
            f"  deepagents mcp-servers add --url <url> "
            f"--name <name> --header <api-key-name>=<value>"
        )
        raise UnresolvedServersError(msg, urls=urls)
    return out
