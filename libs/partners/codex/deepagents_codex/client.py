"""Codex HTTP client with automatic token refresh on 401."""

from __future__ import annotations

import base64
import json
import logging
from typing import TYPE_CHECKING, Any

import httpx

from deepagents_codex.errors import CodexAPIError, CodexAuthError
from deepagents_codex.store import CodexAuthStore

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

logger = logging.getLogger(__name__)

_API_BASE = "https://chatgpt.com/backend-api"
_CODEX_RESPONSES_PATH = "/codex/responses"

_HTTP_OK = 200
_HTTP_UNAUTHORIZED = 401


def _extract_account_id(access_token: str) -> str | None:
    """Extract chatgpt_account_id from the OAuth JWT access token.

    The account ID is in the JWT claim at
    ``["https://api.openai.com/auth"]["chatgpt_account_id"]``.

    Args:
        access_token: The OAuth access token (JWT format).

    Returns:
        The account ID string, or None if extraction fails.
    """
    try:
        parts = access_token.split(".")
        if len(parts) < 2:  # noqa: PLR2004
            return None
        payload = parts[1]
        payload += "=" * (4 - len(payload) % 4)
        data = json.loads(base64.urlsafe_b64decode(payload))
        auth_claim = data.get("https://api.openai.com/auth", {})
        return auth_claim.get("chatgpt_account_id")
    except (ValueError, KeyError, UnicodeDecodeError, json.JSONDecodeError):
        logger.debug("Could not extract account_id from JWT", exc_info=True)
        return None


class CodexClient:
    """HTTP client for the Codex Responses API.

    Sends requests to ``chatgpt.com/backend-api/codex/responses`` using
    OAuth Bearer tokens. On 401, refreshes the token once and retries.
    """

    def __init__(self, *, store: CodexAuthStore | None = None) -> None:
        """Initialize the Codex client.

        Args:
            store: Optional credential store (defaults to standard location).
        """
        self._store = store or CodexAuthStore()

    def _get_headers(self) -> dict[str, str]:
        """Get authorization and required Codex headers."""
        creds = self._store.load()
        if creds is None:
            msg = "Not authenticated. Run 'deepagents auth login --provider codex'"
            raise CodexAuthError(msg)

        headers = {
            "Authorization": f"Bearer {creds.access_token}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

        account_id = _extract_account_id(creds.access_token)
        if account_id:
            headers["ChatGPT-Account-Id"] = account_id

        return headers

    def _refresh_and_get_headers(self) -> dict[str, str]:
        """Refresh the access token and return new headers."""
        from deepagents_codex.auth import refresh_access_token

        creds = self._store.load()
        if creds is None:
            msg = "Not authenticated. Run 'deepagents auth login --provider codex'"
            raise CodexAuthError(msg)
        new_creds = refresh_access_token(creds, store=self._store)

        headers = {
            "Authorization": f"Bearer {new_creds.access_token}",
            "Content-Type": "application/json",
            "OpenAI-Beta": "responses=experimental",
            "originator": "codex_cli_rs",
        }

        account_id = _extract_account_id(new_creds.access_token)
        if account_id:
            headers["chatgpt-account-id"] = account_id

        return headers

    def _build_payload(
        self,
        input_items: list[dict[str, Any]],
        model: str,
        *,
        instructions: str = "",
        stream: bool = True,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build the Responses API request payload."""
        payload: dict[str, Any] = {
            "model": model,
            "instructions": instructions,
            "input": input_items,
            "stream": stream,
            "store": False,
            "include": ["reasoning.encrypted_content"],
        }
        if tools:
            payload["tools"] = [self._convert_tool(t) for t in tools]
            payload["tool_choice"] = "auto"
        payload.update(kwargs)
        return payload

    @staticmethod
    def _convert_tool(tool: dict[str, Any]) -> dict[str, Any]:
        """Convert a tool from Chat Completions format to Responses API format.

        Chat Completions: nested ``function`` key with ``name``, etc.
        Responses API: flat with top-level ``name``, ``parameters``.

        If the tool is already in Responses API format (has top-level "name"),
        it is returned as-is.
        """
        if "function" in tool and "name" not in tool:
            func = tool["function"]
            return {
                "type": "function",
                "name": func["name"],
                **({k: v for k, v in func.items() if k != "name"}),
            }
        return tool

    def create_response(
        self,
        input_items: list[dict[str, Any]],
        model: str,
        *,
        instructions: str = "",
        stream: bool = True,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Iterator[dict[str, Any]]:
        """Send a request to the Codex Responses API.

        Args:
            input_items: List of input items (Responses API format).
            model: Model ID to use.
            instructions: System instructions.
            stream: Whether to stream (always True for Codex).
            tools: Optional tool definitions.
            **kwargs: Additional API parameters.

        Yields:
            Parsed SSE event dicts.

        Raises:
            CodexAuthError: If authentication fails after refresh.
            CodexAPIError: If the API returns an error.
        """
        payload = self._build_payload(
            input_items,
            model,
            instructions=instructions,
            stream=stream,
            tools=tools,
            **kwargs,
        )
        headers = self._get_headers()
        url = f"{_API_BASE}{_CODEX_RESPONSES_PATH}"

        with (
            httpx.Client(timeout=300) as client,
            client.stream(
                "POST",
                url,
                json=payload,
                headers=headers,
            ) as resp,
        ):
            if resp.status_code == _HTTP_UNAUTHORIZED:
                resp.close()
                logger.debug("Got 401, attempting token refresh")
                headers = self._refresh_and_get_headers()
                with client.stream(
                    "POST",
                    url,
                    json=payload,
                    headers=headers,
                ) as resp2:
                    if resp2.status_code == _HTTP_UNAUTHORIZED:
                        msg = "Authentication failed after token refresh"
                        raise CodexAuthError(msg)
                    if resp2.status_code != _HTTP_OK:
                        raise CodexAPIError(
                            resp2.status_code,
                            resp2.read().decode(),
                        )
                    yield from self._parse_sse(resp2)
                return

            if resp.status_code != _HTTP_OK:
                raise CodexAPIError(
                    resp.status_code,
                    resp.read().decode(),
                )
            yield from self._parse_sse(resp)

    def _parse_sse(
        self,
        resp: httpx.Response,
    ) -> Iterator[dict[str, Any]]:
        """Parse SSE events from a streaming response."""
        for line in resp.iter_lines():
            stripped = line.strip()
            if not stripped or not stripped.startswith("data: "):
                continue
            data = stripped[6:]
            if data == "[DONE]":
                return
            try:
                yield json.loads(data)
            except json.JSONDecodeError:
                logger.warning("Malformed SSE event: %s", data)

    async def acreate_response(
        self,
        input_items: list[dict[str, Any]],
        model: str,
        *,
        instructions: str = "",
        stream: bool = True,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        """Async version of create_response."""
        payload = self._build_payload(
            input_items,
            model,
            instructions=instructions,
            stream=stream,
            tools=tools,
            **kwargs,
        )
        headers = self._get_headers()
        url = f"{_API_BASE}{_CODEX_RESPONSES_PATH}"

        async with (
            httpx.AsyncClient(timeout=300) as client,
            client.stream(
                "POST",
                url,
                json=payload,
                headers=headers,
            ) as resp,
        ):
            if resp.status_code == _HTTP_UNAUTHORIZED:
                await resp.aclose()
                logger.debug("Got 401, attempting token refresh")
                headers = self._refresh_and_get_headers()
                async with client.stream(
                    "POST",
                    url,
                    json=payload,
                    headers=headers,
                ) as resp2:
                    if resp2.status_code == _HTTP_UNAUTHORIZED:
                        msg = "Authentication failed after token refresh"
                        raise CodexAuthError(msg)
                    if resp2.status_code != _HTTP_OK:
                        body = await resp2.aread()
                        raise CodexAPIError(
                            resp2.status_code,
                            body.decode(),
                        )
                    async for event in self._aparse_sse(resp2):
                        yield event
                return

            if resp.status_code != _HTTP_OK:
                body = await resp.aread()
                raise CodexAPIError(
                    resp.status_code,
                    body.decode(),
                )
            async for event in self._aparse_sse(resp):
                yield event

    async def _aparse_sse(
        self,
        resp: httpx.Response,
    ) -> AsyncIterator[dict[str, Any]]:
        """Parse SSE events from an async streaming response."""
        async for line in resp.aiter_lines():
            stripped = line.strip()
            if not stripped or not stripped.startswith("data: "):
                continue
            data = stripped[6:]
            if data == "[DONE]":
                return
            try:
                yield json.loads(data)
            except json.JSONDecodeError:
                logger.warning("Malformed SSE event: %s", data)
