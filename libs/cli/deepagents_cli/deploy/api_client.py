"""HTTP client for the Managed Deep Agents `/v1/deepagents/*` surface.

Thin wrapper around `httpx.Client` that:

- Resolves auth from `LANGSMITH_API_KEY` (preferred) or `LANGCHAIN_API_KEY`
  and sends it as `X-Api-Key`.
- Resolves the endpoint from `LANGSMITH_ENDPOINT` / `LANGCHAIN_ENDPOINT`,
  defaulting to `https://api.smith.langchain.com`.
- Parses 4xx responses into `ApiError` with the platform's `ErrorResponse`
  shape (`type`/`code`/`detail`/`status`).
- Retries 5xx responses once with a short backoff before raising.

Agents and MCP-servers CRUD methods are layered on top in subsequent tasks.
"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, cast
from urllib.parse import urlsplit

import httpx

_DEFAULT_ENDPOINT = "https://api.smith.langchain.com"
_DEPLOY_PATH = "/v1/deepagents"
_HUB_PATH = "/v1/platform/hub"
_RETRY_SLEEP_SECONDS = 1.0


@dataclass
class ApiError(Exception):
    """Surface the platform's `ErrorResponse` envelope as a Python exception."""

    status: int
    code: str = ""
    detail: str = ""
    type_: str = ""

    def __str__(self) -> str:  # noqa: D105
        bits = [f"HTTP {self.status}"]
        if self.code:
            bits.append(self.code)
        if self.detail:
            bits.append(self.detail)
        return " — ".join(bits)


def _normalize_endpoint(endpoint: str) -> str:
    endpoint = endpoint.strip().rstrip("/")
    parsed = urlsplit(endpoint)
    if parsed.scheme != "https" or not parsed.netloc:
        msg = "Error: LANGSMITH_ENDPOINT / LANGCHAIN_ENDPOINT must be an HTTPS URL.\n"
        sys.stderr.write(msg)
        raise SystemExit(1)
    if parsed.username or parsed.password:
        msg = (
            "Error: LANGSMITH_ENDPOINT / LANGCHAIN_ENDPOINT must not include "
            "userinfo.\n"
        )
        sys.stderr.write(msg)
        raise SystemExit(1)
    return endpoint


class ApiClient:
    """HTTP client for `/v1/deepagents/*`."""

    def __init__(
        self,
        *,
        endpoint: str,
        api_key: str,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """Initialise the client with an endpoint and API key."""
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self._client = httpx.Client(
            base_url=self.endpoint,
            transport=transport,
            trust_env=False,
            timeout=httpx.Timeout(30.0, connect=10.0),
            headers={"X-Api-Key": api_key, "Content-Type": "application/json"},
        )

    @classmethod
    def from_env(
        cls,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> ApiClient:
        """Build a client from `LANGSMITH_*` / `LANGCHAIN_*` env vars.

        Endpoint resolution is env var > `_DEFAULT_ENDPOINT`. Project-local
        deploy state is intentionally ignored because it can be repository
        controlled and must not steer authenticated requests.

        Exits non-zero with a friendly message if the API key is missing.
        """
        api_key = (
            os.environ.get("LANGSMITH_API_KEY")
            or os.environ.get("LANGCHAIN_API_KEY")
            or ""
        ).strip()
        if not api_key:
            sys.stderr.write(
                "Error: set LANGSMITH_API_KEY in your .env or environment.\n"
            )
            raise SystemExit(1)
        endpoint = (
            os.environ.get("LANGSMITH_ENDPOINT")
            or os.environ.get("LANGCHAIN_ENDPOINT")
            or _DEFAULT_ENDPOINT
        )
        endpoint = _normalize_endpoint(endpoint)
        return cls(endpoint=endpoint, api_key=api_key, transport=transport)

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        self._client.close()

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> Any:  # noqa: ANN401
        last_status = 0
        last_text = ""
        for attempt in range(2):
            response = self._client.request(method, path, json=json, params=params)
            last_status = response.status_code
            last_text = response.text
            if 200 <= response.status_code < 300:  # noqa: PLR2004
                if response.status_code == 204 or not response.content:  # noqa: PLR2004
                    return None
                return response.json()
            if 400 <= response.status_code < 500:  # noqa: PLR2004
                raise self._build_error(response)
            # 5xx: retry once
            if attempt == 0:
                time.sleep(_RETRY_SLEEP_SECONDS)
                continue
        raise ApiError(status=last_status, detail=last_text[:500])

    @staticmethod
    def _build_error(response: httpx.Response) -> ApiError:
        try:
            payload = response.json()
        except ValueError:
            payload = {}
        return ApiError(
            status=response.status_code,
            code=str(payload.get("code") or ""),
            detail=str(payload.get("detail") or response.text[:500]),
            type_=str(payload.get("type") or ""),
        )

    # --- agents ----------------------------------------------------------

    def create_agent(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Create a new agent and return the created resource."""
        return self._request("POST", f"{_DEPLOY_PATH}/agents", json=payload)

    def get_agent(
        self,
        agent_id: str,
        *,
        include_files: bool = False,
    ) -> dict[str, Any]:
        """Fetch a single agent by ID."""
        params = {"include_files": "true"} if include_files else None
        return self._request("GET", f"{_DEPLOY_PATH}/agents/{agent_id}", params=params)

    def iter_agents(
        self,
        *,
        page_size: int = 50,
        name: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield AgentSummary objects across all pages."""
        cursor: str | None = None
        while True:
            params: dict[str, Any] = {"page_size": page_size}
            if cursor:
                params["cursor"] = cursor
            if name:
                params["name"] = name
            body = self._request("GET", f"{_DEPLOY_PATH}/agents", params=params)
            yield from body.get("items", [])
            cursor = body.get("next_cursor")
            if not cursor:
                return

    def patch_agent(self, agent_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Partially update an agent by ID."""
        return self._request("PATCH", f"{_DEPLOY_PATH}/agents/{agent_id}", json=payload)

    def delete_agent(self, agent_id: str) -> None:
        """Delete an agent by ID."""
        self._request("DELETE", f"{_DEPLOY_PATH}/agents/{agent_id}")

    # --- mcp-servers -----------------------------------------------------

    def list_mcp_servers(self) -> list[dict[str, Any]]:
        """Return all registered MCP servers in this workspace."""
        body = self._request("GET", f"{_DEPLOY_PATH}/mcp-servers")
        if isinstance(body, list):
            return [cast("dict[str, Any]", item) for item in body]
        if isinstance(body, dict) and isinstance(body.get("servers"), list):
            return [cast("dict[str, Any]", item) for item in body["servers"]]
        msg = "Unexpected MCP server list response."
        raise ApiError(status=0, detail=msg)

    def get_mcp_server(self, mcp_server_id: str) -> dict[str, Any]:
        """Fetch a single MCP server by ID."""
        return self._request("GET", f"{_DEPLOY_PATH}/mcp-servers/{mcp_server_id}")

    def create_mcp_server(
        self,
        *,
        name: str,
        url: str,
        headers: list[dict[str, str]] | None = None,
        auth_type: str = "headers",
        oauth_mode: str | None = None,
    ) -> dict[str, Any]:
        """Register a new MCP server in this workspace."""
        payload: dict[str, Any] = {
            "name": name,
            "url": url,
            "auth_type": auth_type,
        }
        if headers:
            payload["headers"] = headers
        if oauth_mode is not None:
            payload["oauth_mode"] = oauth_mode
        return self._request("POST", f"{_DEPLOY_PATH}/mcp-servers", json=payload)

    def update_mcp_server(
        self,
        mcp_server_id: str,
        *,
        url: str | None = None,
        headers: list[dict[str, str]] | None = None,
        auth_type: str | None = None,
    ) -> dict[str, Any]:
        """Update an existing MCP server in this workspace."""
        payload: dict[str, Any] = {}
        if url is not None:
            payload["url"] = url
        if headers is not None:
            payload["headers"] = headers
        if auth_type is not None:
            payload["auth_type"] = auth_type
        return self._request(
            "PATCH", f"{_DEPLOY_PATH}/mcp-servers/{mcp_server_id}", json=payload
        )

    def delete_mcp_server(self, mcp_server_id: str) -> None:
        """Delete an MCP server by ID."""
        self._request("DELETE", f"{_DEPLOY_PATH}/mcp-servers/{mcp_server_id}")

    def list_mcp_server_tools(
        self,
        url: str,
        *,
        oauth_provider_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return the tools exposed by a registered MCP server.

        Backed by `GET /v1/deepagents/mcp/tools`, which resolves the server by
        URL (cache-first, with a remote MCP `tools/list` fallback).

        Args:
            url: The registered MCP server URL.
            oauth_provider_id: OAuth provider id; required for OAuth servers.

        Returns:
            MCP tool definitions, each with `name`, `description`, and
            `inputSchema`.
        """
        params: dict[str, Any] = {"url": url}
        if oauth_provider_id:
            params["oauth_provider_id"] = oauth_provider_id
        body = self._request("GET", f"{_DEPLOY_PATH}/mcp/tools", params=params)
        if isinstance(body, dict) and isinstance(body.get("tools"), list):
            return list(body["tools"])
        msg = "Unexpected MCP tools response."
        raise ApiError(status=0, detail=msg)

    def register_mcp_oauth_provider(self, mcp_server_id: str) -> dict[str, Any]:
        """Register the caller's per-user OAuth provider for an MCP server."""
        return self._request(
            "POST",
            f"{_DEPLOY_PATH}/mcp-servers/{mcp_server_id}/oauth-provider",
            json={},
        )

    def create_auth_session(
        self,
        *,
        provider_id: str,
        scopes: list[str],
        strategy: str,
    ) -> dict[str, Any]:
        """Start an OAuth authorization session for the caller."""
        return self._request(
            "POST",
            f"{_DEPLOY_PATH}/auth-sessions",
            json={
                "provider_id": provider_id,
                "scopes": scopes,
                "strategy": strategy,
            },
        )

    def get_auth_session(
        self,
        session_id: str,
        *,
        wait_seconds: int,
    ) -> dict[str, Any]:
        """Fetch or long-poll an OAuth authorization session."""
        return self._request(
            "GET",
            f"{_DEPLOY_PATH}/auth-sessions/{session_id}",
            params={"wait_seconds": wait_seconds},
        )

    # --- hub directories -------------------------------------------------

    def get_agent_directory(self, agent_id: str) -> dict[str, Any]:
        """Fetch the Hub directory backing a managed deep agent."""
        return self._request("GET", f"{_HUB_PATH}/repos/-/{agent_id}/directories")

    def commit_agent_directory(
        self,
        agent_id: str,
        *,
        files: dict[str, dict[str, str] | None],
        parent_commit: str | None,
    ) -> dict[str, Any]:
        """Commit file updates to the Hub directory backing an agent."""
        payload: dict[str, Any] = {"files": files}
        if parent_commit:
            payload["parent_commit"] = parent_commit
        return self._request(
            "POST",
            f"{_HUB_PATH}/repos/-/{agent_id}/directories/commits",
            json=payload,
        )
