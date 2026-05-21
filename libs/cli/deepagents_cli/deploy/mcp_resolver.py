"""Resolve MCP server URLs in a payload to workspace-registered server IDs.

The deploy command does not auto-create MCP servers. Instead it validates that
every `mcp_server_url` the payload references already exists at the
`/v1/deepagents/mcp-servers` endpoint, and surfaces a friendly hint if not.
"""

from __future__ import annotations

from typing import Any

from deepagents_cli.deploy.api_client import ApiClient


class UnresolvedServersError(RuntimeError):
    """Raised when one or more `mcp_server_url`s aren't registered."""


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

    Uses `cache` as a starting point and only hits the list endpoint if any
    URL is missing. Raises `UnresolvedServersError` if any URL is still
    unresolved after the list call.
    """
    referenced = _collect_referenced_urls(payload)
    out: dict[str, str] = {url: cache[url] for url in referenced if url in cache}
    missing = referenced - out.keys()
    if not missing:
        return out

    for server in client.list_mcp_servers():
        url = server.get("url")
        if isinstance(url, str) and url:
            key = _normalize_url(url)
            if key in missing:
                out[key] = server["id"]

    still_missing = referenced - out.keys()
    if still_missing:
        listed = "\n".join(f"  - {u}" for u in sorted(still_missing))
        msg = (
            f"The following MCP server URLs referenced in your tools "
            f"are not registered in this workspace:\n{listed}\n\n"
            f"Register each with:\n"
            f"  deepagents mcp-servers add --url <url> "
            f"--header KEY=VALUE [--name <name>]"
        )
        raise UnresolvedServersError(msg)
    return out
