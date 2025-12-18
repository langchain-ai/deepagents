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
    path: str
    content: str
    size: int


@dataclass
class AgentInfo:
    name: str
    owner_id: str
    is_public: bool
    created_at: str
    latest_version: int


@dataclass
class PushResponse:
    name: str
    version: int
    files_count: int


@dataclass
class PullResponse:
    name: str
    version: int
    files: list[AgentFile]


class AgentFilesystemClient:
    def __init__(self, base_url: str, api_key: str | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = 30  # seconds

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _handle_response(self, response: requests.Response) -> dict[str, Any] | list | None:
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

