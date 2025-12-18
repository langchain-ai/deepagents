"""HTTP client for the agent filesystem API.

This module provides a client for interacting with the remote agent filesystem
server to push, pull, and list agents.
"""

from dataclasses import dataclass
from http import HTTPStatus
from typing import Any

import requests


class AgentFilesystemError(Exception):
    """Base exception for agent filesystem errors."""


class AuthenticationError(AgentFilesystemError):
    """Raised when authentication fails or is required."""


class AgentNotFoundError(AgentFilesystemError):
    """Raised when an agent is not found."""


class AgentConflictError(AgentFilesystemError):
    """Raised when there's a conflict (e.g., duplicate public agent name)."""


@dataclass
class AgentFile:
    """Represents a file in an agent."""

    path: str
    content: str
    size: int


@dataclass
class AgentInfo:
    """Metadata about an agent."""

    name: str
    owner_id: str
    is_public: bool
    created_at: str
    latest_version: int


@dataclass
class PushResponse:
    """Response from pushing an agent."""

    name: str
    version: int
    files_count: int


@dataclass
class PullResponse:
    """Response from pulling an agent."""

    name: str
    version: int
    files: list[AgentFile]


class AgentFilesystemClient:
    """Client for interacting with the agent filesystem API."""

    def __init__(self, base_url: str, api_key: str | None = None) -> None:
        """Initialize the client.

        Args:
            base_url: Base URL of the agent filesystem server
            api_key: Optional API key for authentication
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = 30  # seconds

    def _headers(self) -> dict[str, str]:
        """Build request headers."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _handle_response(self, response: requests.Response) -> dict[str, Any] | list | None:
        """Handle API response and raise appropriate errors.

        Args:
            response: The response from the API

        Returns:
            The JSON response data

        Raises:
            AuthenticationError: If authentication fails
            AgentNotFoundError: If agent is not found
            AgentConflictError: If there's a conflict
            AgentFilesystemError: For other errors
        """
        if response.status_code == HTTPStatus.UNAUTHORIZED:
            msg = "Authentication required. Set AGENT_FS_API_KEY."
            raise AuthenticationError(msg)
        if response.status_code == HTTPStatus.FORBIDDEN:
            msg = "Permission denied. Check your API key."
            raise AuthenticationError(msg)
        if response.status_code == HTTPStatus.NOT_FOUND:
            msg = "Agent not found."
            raise AgentNotFoundError(msg)
        if response.status_code == HTTPStatus.CONFLICT:
            msg = "Agent name conflict. A public agent with this name already exists."
            raise AgentConflictError(msg)

        try:
            data = response.json()
        except requests.exceptions.JSONDecodeError:
            if not response.ok:
                msg = f"API error: {response.status_code} - {response.text}"
                raise AgentFilesystemError(msg) from None
            return None

        if not response.ok:
            error_msg = data.get("detail", data.get("error", str(data)))
            msg = f"API error: {error_msg}"
            raise AgentFilesystemError(msg)

        return data

    def push(
        self,
        name: str,
        files: list[dict[str, Any]],
        *,
        is_public: bool = False,
    ) -> PushResponse:
        """Push an agent to the remote filesystem.

        Args:
            name: Agent name
            files: List of files with path, content, and size
            is_public: Whether the agent should be public

        Returns:
            PushResponse with name, version, and files_count

        Raises:
            AuthenticationError: If not authenticated
            AgentConflictError: If public agent name already exists
            AgentFilesystemError: For other errors
        """
        url = f"{self.base_url}/v1/profiles/{name}/push"
        params = {"is_public": str(is_public).lower()}

        response = requests.post(
            url,
            headers=self._headers(),
            params=params,
            json={"files": files},
            timeout=self.timeout,
        )

        data = self._handle_response(response)
        return PushResponse(
            name=data["name"],
            version=data["version"],
            files_count=data["files_count"],
        )

    def pull(
        self,
        name: str,
        *,
        version: int | None = None,
        is_public: bool | None = None,
    ) -> PullResponse:
        """Pull an agent from the remote filesystem.

        Args:
            name: Agent name
            version: Specific version to pull (default: latest)
            is_public: Force lookup type (True=public only, False=private only, None=auto)

        Returns:
            PullResponse with name, version, and files

        Raises:
            AgentNotFoundError: If agent is not found
            AuthenticationError: If private agent and not authenticated
            AgentFilesystemError: For other errors
        """
        url = f"{self.base_url}/v1/profiles/{name}/pull"
        params: dict[str, Any] = {}
        if version is not None:
            params["version"] = version
        if is_public is not None:
            params["is_public"] = str(is_public).lower()

        response = requests.get(
            url,
            headers=self._headers(),
            params=params,
            timeout=self.timeout,
        )

        data = self._handle_response(response)
        files = [
            AgentFile(
                path=f["path"],
                content=f["content"],
                size=f["size"],
            )
            for f in data["files"]
        ]
        return PullResponse(
            name=data["name"],
            version=data["version"],
            files=files,
        )

    def list_agents(self, *, is_public: bool | None = None) -> list[AgentInfo]:
        """List all accessible agents.

        Returns all public agents and (if authenticated) the user's private agents.

        Args:
            is_public: Filter by visibility (True=public only, False=private only, None=all)

        Returns:
            List of AgentInfo objects

        Raises:
            AgentFilesystemError: For API errors
        """
        url = f"{self.base_url}/v1/profiles"
        params: dict[str, Any] = {}
        if is_public is not None:
            params["is_public"] = str(is_public).lower()

        response = requests.get(
            url,
            headers=self._headers(),
            params=params,
            timeout=self.timeout,
        )

        data = self._handle_response(response)
        return [
            AgentInfo(
                name=p["name"],
                owner_id=p["owner_id"],
                is_public=p["is_public"],
                created_at=p["created_at"],
                latest_version=p["latest_version"],
            )
            for p in data
        ]

    def get_agent(
        self,
        name: str,
        *,
        is_public: bool | None = None,
    ) -> AgentInfo:
        """Get information about a specific agent.

        Args:
            name: Agent name
            is_public: Force lookup type (True=public only, False=private only, None=auto)

        Returns:
            AgentInfo object

        Raises:
            AgentNotFoundError: If agent is not found
            AgentFilesystemError: For other errors
        """
        url = f"{self.base_url}/v1/profiles/{name}"
        params: dict[str, Any] = {}
        if is_public is not None:
            params["is_public"] = str(is_public).lower()

        response = requests.get(
            url,
            headers=self._headers(),
            params=params,
            timeout=self.timeout,
        )

        data = self._handle_response(response)
        return AgentInfo(
            name=data["name"],
            owner_id=data["owner_id"],
            is_public=data["is_public"],
            created_at=data["created_at"],
            latest_version=data["latest_version"],
        )
