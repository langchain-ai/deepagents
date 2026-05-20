"""Per-thread persistence for `_context_tokens` (client-side bookkeeping).

`_context_tokens` is a dcode-client display value, not anything the agent or
the deepagents SDK reads. Historically it was pushed into graph state via
`aupdate_state` so it would survive across sessions — but for remote agents
(`langgraph dev` / LangGraph Platform), every `aupdate_state` call traverses
HTTP and the server emits its own `UpdateState` LangSmith trace as a
root-level run. The client's `tracing_context(enabled=False)` cannot reach
across the HTTP boundary, and as of langgraph-sdk 0.3 / langgraph-api 0.8
there is no upstream mechanism to suppress server-side traces for
`threads.update_state` (no `langsmith_tracing` parameter, no honored header,
no server-side tracing context for state updates). Storing the value locally
sidesteps the HTTP round-trip and the trace entirely.

Long term, graph state is still the right home — it would carry the count
across machines on thread resume. Once upstream exposes a way to suppress
tracing on `threads.update_state` (or the field moves to a non-traced
channel), this module should be deleted and `_context_tokens` should move
back to graph state.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from deepagents_code.model_config import DEFAULT_STATE_DIR

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

CONTEXT_TOKENS_DIR = DEFAULT_STATE_DIR / "context_tokens"
"""Directory holding one JSON file per thread (`{thread_id}.json`)."""


def _safe_path_for(thread_id: str) -> Path | None:
    """Return the cache path for `thread_id`, or `None` if it would escape.

    Thread IDs are UUID7 strings in normal flow; this guard is defensive
    against a future caller passing untrusted input.
    """
    if (
        not thread_id
        or "/" in thread_id
        or "\\" in thread_id
        or thread_id in {".", ".."}
    ):
        logger.debug(
            "Refusing context-tokens path for unsafe thread_id: %r", thread_id
        )
        return None
    return CONTEXT_TOKENS_DIR / f"{thread_id}.json"


def read_context_tokens(thread_id: str) -> int | None:
    """Return the cached `_context_tokens` for a thread, or `None` if absent.

    `None` signals cache miss — distinct from a legitimate persisted `0`, so
    callers can fall back to a secondary source (e.g. graph state from older
    builds) only when the cache truly has nothing.

    Args:
        thread_id: Thread identifier.

    Returns:
        The persisted token count, or `None` when no usable cache entry
        exists (missing file, corrupt JSON, unexpected shape, negative value).
    """
    path = _safe_path_for(thread_id)
    if path is None:
        return None
    try:
        if not path.exists():
            return None
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        logger.debug(
            "Could not read context-tokens cache for %s",
            thread_id,
            exc_info=True,
        )
        return None
    if not isinstance(raw, dict):
        logger.debug(
            "Unexpected shape in context-tokens cache for %s: %r", thread_id, raw
        )
        return None
    value = raw.get("tokens")
    if not isinstance(value, int) or value < 0:
        logger.debug(
            "Unexpected `tokens` field in context-tokens cache for %s: %r",
            thread_id,
            value,
        )
        return None
    return value


def write_context_tokens(thread_id: str, tokens: int) -> None:
    """Best-effort write of `_context_tokens` for a thread.

    Failures are logged at DEBUG and swallowed — a stale count on resume is
    acceptable.

    Args:
        thread_id: Thread identifier.
        tokens: Token count to persist (negative values are clamped to 0).
    """
    path = _safe_path_for(thread_id)
    if path is None:
        return
    safe_tokens = max(int(tokens), 0)
    try:
        CONTEXT_TOKENS_DIR.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps({"tokens": safe_tokens}), encoding="utf-8")
        tmp.replace(path)
    except OSError:
        logger.debug(
            "Could not write context-tokens cache for %s=%d at %s",
            thread_id,
            safe_tokens,
            path,
            exc_info=True,
        )


def delete_context_tokens(thread_id: str) -> None:
    """Remove the cached `_context_tokens` for a thread, if present.

    Args:
        thread_id: Thread identifier.
    """
    path = _safe_path_for(thread_id)
    if path is None:
        return
    try:
        path.unlink(missing_ok=True)
    except OSError:
        logger.debug(
            "Could not delete context-tokens cache for %s at %s",
            thread_id,
            path,
            exc_info=True,
        )
