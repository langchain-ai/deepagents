"""Server-Sent Events (SSE) stream parser for Codex API responses."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

logger = logging.getLogger(__name__)


def parse_sse_stream(raw_lines: Iterator[str]) -> Iterator[dict[str, Any]]:
    """Parse an SSE stream from raw text lines into event dicts.

    Handles the OpenAI SSE format: ``data: {json}`` lines,
    terminated by ``data: [DONE]``.

    Args:
        raw_lines: Iterator of raw text lines from the HTTP response.

    Yields:
        Parsed JSON event dicts.
    """
    for raw_line in raw_lines:
        stripped = raw_line.strip()
        if not stripped:
            continue
        if not stripped.startswith("data: "):
            continue
        data = stripped[6:]  # Strip "data: " prefix
        if data == "[DONE]":
            return
        try:
            yield json.loads(data)
        except json.JSONDecodeError:
            logger.warning("Malformed SSE event: %s", data)


async def aparse_sse_stream(
    raw_lines: AsyncIterator[str],
) -> AsyncIterator[dict[str, Any]]:
    """Async version of parse_sse_stream.

    Args:
        raw_lines: Async iterator of raw text lines from the HTTP response.

    Yields:
        Parsed JSON event dicts.
    """
    async for raw_line in raw_lines:
        stripped = raw_line.strip()
        if not stripped:
            continue
        if not stripped.startswith("data: "):
            continue
        data = stripped[6:]
        if data == "[DONE]":
            return
        try:
            yield json.loads(data)
        except json.JSONDecodeError:
            logger.warning("Malformed SSE event: %s", data)
