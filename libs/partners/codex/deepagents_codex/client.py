"""Codex HTTP client with automatic token refresh on 401."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import httpx

from deepagents_codex.errors import CodexAPIError, CodexAuthError
from deepagents_codex.store import CodexAuthStore

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

logger = logging.getLogger(__name__)

_API_BASE = "https://api.openai.com/v1"

_HTTP_OK = 200
_HTTP_UNAUTHORIZED = 401


class CodexClient:
    """HTTP client for the Codex API with automatic 401 retry-refresh.

    On receiving a 401 response, refreshes the access token once and
    retries the request. If still 401, raises CodexAuthError.
    """

    def __init__(self, *, store: CodexAuthStore | None = None) -> None:
        """Initialize the Codex client.

        Args:
            store: Optional credential store (defaults to standard location).
        """
        self._store = store or CodexAuthStore()

    def _get_headers(self) -> dict[str, str]:
        """Get authorization headers from stored credentials."""
        creds = self._store.load()
        if creds is None:
            msg = (
                "Not authenticated. "
                "Run 'deepagents auth login --provider codex'"
            )
            raise CodexAuthError(msg)
        return {
            "Authorization": f"Bearer {creds.access_token}",
            "Content-Type": "application/json",
        }

    def _refresh_and_get_headers(self) -> dict[str, str]:
        """Refresh the access token and return new headers."""
        from deepagents_codex.auth import refresh_access_token

        creds = self._store.load()
        if creds is None:
            msg = (
                "Not authenticated. "
                "Run 'deepagents auth login --provider codex'"
            )
            raise CodexAuthError(msg)
        new_creds = refresh_access_token(creds, store=self._store)
        return {
            "Authorization": f"Bearer {new_creds.access_token}",
            "Content-Type": "application/json",
        }

    def _build_payload(
        self,
        messages: list[dict[str, Any]],
        model: str,
        *,
        stream: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build the API request payload."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }
        payload.update(kwargs)
        return payload

    def chat_completions(
        self,
        messages: list[dict[str, Any]],
        model: str,
        *,
        stream: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any] | Iterator[dict[str, Any]]:
        """Send a chat completion request.

        Args:
            messages: List of message dicts.
            model: Model ID to use.
            stream: Whether to stream the response.
            **kwargs: Additional API parameters.

        Returns:
            Response dict or iterator of event dicts (streaming).

        Raises:
            CodexAuthError: If authentication fails after refresh.
            CodexAPIError: If the API returns a non-200/non-401 error.
        """
        payload = self._build_payload(
            messages, model, stream=stream, **kwargs,
        )
        headers = self._get_headers()
        url = f"{_API_BASE}/chat/completions"

        with httpx.Client(timeout=120) as client:
            resp = client.post(url, json=payload, headers=headers)

            if resp.status_code == _HTTP_UNAUTHORIZED:
                logger.debug("Got 401, attempting token refresh")
                headers = self._refresh_and_get_headers()
                resp = client.post(url, json=payload, headers=headers)
                if resp.status_code == _HTTP_UNAUTHORIZED:
                    msg = "Authentication failed after token refresh"
                    raise CodexAuthError(msg)

            if resp.status_code != _HTTP_OK:
                raise CodexAPIError(resp.status_code, resp.text)

            if stream:
                return self._iter_sse(resp)
            return resp.json()

    def _iter_sse(
        self, resp: httpx.Response,
    ) -> Iterator[dict[str, Any]]:
        """Parse SSE events from a response."""
        from deepagents_codex.sse import parse_sse_stream

        yield from parse_sse_stream(iter(resp.text.splitlines()))

    async def achat_completions(
        self,
        messages: list[dict[str, Any]],
        model: str,
        *,
        stream: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any] | AsyncIterator[dict[str, Any]]:
        """Async version of chat_completions.

        Args:
            messages: List of message dicts.
            model: Model ID to use.
            stream: Whether to stream the response.
            **kwargs: Additional API parameters.

        Returns:
            Response dict or async iterator of event dicts (streaming).

        Raises:
            CodexAuthError: If authentication fails after refresh.
            CodexAPIError: If the API returns a non-200/non-401 error.
        """
        payload = self._build_payload(
            messages, model, stream=stream, **kwargs,
        )
        headers = self._get_headers()
        url = f"{_API_BASE}/chat/completions"

        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                url, json=payload, headers=headers,
            )

            if resp.status_code == _HTTP_UNAUTHORIZED:
                logger.debug("Got 401, attempting token refresh")
                headers = self._refresh_and_get_headers()
                resp = await client.post(
                    url, json=payload, headers=headers,
                )
                if resp.status_code == _HTTP_UNAUTHORIZED:
                    msg = "Authentication failed after token refresh"
                    raise CodexAuthError(msg)

            if resp.status_code != _HTTP_OK:
                raise CodexAPIError(resp.status_code, resp.text)

            if stream:
                return self._aiter_sse(resp)
            return resp.json()

    async def _aiter_sse(
        self, resp: httpx.Response,
    ) -> AsyncIterator[dict[str, Any]]:
        """Parse SSE events from an async response."""
        from deepagents_codex.sse import aparse_sse_stream

        async def _lines() -> AsyncIterator[str]:
            for line in resp.text.splitlines():
                yield line

        async for event in aparse_sse_stream(_lines()):
            yield event
