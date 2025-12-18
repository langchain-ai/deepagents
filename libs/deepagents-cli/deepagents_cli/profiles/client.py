"""HTTP client for the agent filesystem API.

This module provides a client for interacting with the remote agent filesystem
server to push, pull, and list agent profiles.
"""

from dataclasses import dataclass
from http import HTTPStatus
from typing import Any

import requests


class AgentFilesystemError(Exception):
    """Base exception for agent filesystem errors."""


class AuthenticationError(AgentFilesystemError):
    """Raised when authentication fails or is required."""


class ProfileNotFoundError(AgentFilesystemError):
    """Raised when a profile is not found."""


class ProfileConflictError(AgentFilesystemError):
    """Raised when there's a conflict (e.g., duplicate public profile name)."""


@dataclass
class ProfileFile:
    """Represents a file in a profile."""

    path: str
    content: str
    size: int


@dataclass
class ProfileInfo:
    """Metadata about a profile."""

    name: str
    owner_id: str
    is_public: bool
    created_at: str
    latest_version: int


@dataclass
class PushResponse:
    """Response from pushing a profile."""

    name: str
    version: int
    files_count: int


@dataclass
class PullResponse:
    """Response from pulling a profile."""

    name: str
    version: int
    files: list[ProfileFile]


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
            ProfileNotFoundError: If profile is not found
            ProfileConflictError: If there's a conflict
            AgentFilesystemError: For other errors
        """
        if response.status_code == HTTPStatus.UNAUTHORIZED:
            msg = "Authentication required. Set AGENT_FS_API_KEY."
            raise AuthenticationError(msg)
        if response.status_code == HTTPStatus.FORBIDDEN:
            msg = "Permission denied. Check your API key."
            raise AuthenticationError(msg)
        if response.status_code == HTTPStatus.NOT_FOUND:
            msg = "Profile not found."
            raise ProfileNotFoundError(msg)
        if response.status_code == HTTPStatus.CONFLICT:
            msg = "Profile name conflict. A public profile with this name already exists."
            raise ProfileConflictError(msg)

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
        """Push a profile to the remote filesystem.

        Args:
            name: Profile name
            files: List of files with path, content, and size
            is_public: Whether the profile should be public

        Returns:
            PushResponse with name, version, and files_count

        Raises:
            AuthenticationError: If not authenticated
            ProfileConflictError: If public profile name already exists
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
        """Pull a profile from the remote filesystem.

        Args:
            name: Profile name
            version: Specific version to pull (default: latest)
            is_public: Force lookup type (True=public only, False=private only, None=auto)

        Returns:
            PullResponse with name, version, and files

        Raises:
            ProfileNotFoundError: If profile is not found
            AuthenticationError: If private profile and not authenticated
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
            ProfileFile(
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

    def list_profiles(self, *, is_public: bool | None = None) -> list[ProfileInfo]:
        """List all accessible profiles.

        Returns all public profiles and (if authenticated) the user's private profiles.

        Args:
            is_public: Filter by visibility (True=public only, False=private only, None=all)

        Returns:
            List of ProfileInfo objects

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
            ProfileInfo(
                name=p["name"],
                owner_id=p["owner_id"],
                is_public=p["is_public"],
                created_at=p["created_at"],
                latest_version=p["latest_version"],
            )
            for p in data
        ]

    def get_profile(
        self,
        name: str,
        *,
        is_public: bool | None = None,
    ) -> ProfileInfo:
        """Get information about a specific profile.

        Args:
            name: Profile name
            is_public: Force lookup type (True=public only, False=private only, None=auto)

        Returns:
            ProfileInfo object

        Raises:
            ProfileNotFoundError: If profile is not found
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
        return ProfileInfo(
            name=data["name"],
            owner_id=data["owner_id"],
            is_public=data["is_public"],
            created_at=data["created_at"],
            latest_version=data["latest_version"],
        )
