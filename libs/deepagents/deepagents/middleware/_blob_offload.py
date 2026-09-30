"""Content-addressed offload of binary `read_file` blocks to the backend.

Binary blocks are written to `{artifacts_root}/blobs/<sha256>` and replaced in
state with a `deepagents_blob` reference. Model requests are rehydrated from the
backend, so checkpoints never carry the base64 payload.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import logging
import re
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Final

from langchain_core.messages import AIMessage, AnyMessage, BaseMessage, HumanMessage, ToolMessage

if TYPE_CHECKING:
    from collections.abc import Sequence

    from langgraph.types import Command

    from deepagents.backends.protocol import BackendProtocol, FileDownloadResponse

logger = logging.getLogger(__name__)

_BLOB_REF_KEY: Final = "deepagents_blob"
"""Content block key holding the SHA-256 digest of an offloaded payload."""

_MISSING_BLOB_TEXT: Final = "[Binary content from an earlier read_file call is no longer available. Re-read the file if you still need it.]"

_BLOB_PAYLOADS_KEY: Final = "_blob_payloads"
"""Private, untracked state key mapping digests to base64 payloads for the current run."""

_DIGEST_RE: Final = re.compile(r"[0-9a-f]{64}")


def _blob_path(prefix: str, digest: str) -> str:
    """Return the backend path for `digest` under `prefix`."""
    return f"{prefix}/{digest}"


def _pending_blobs(messages: Sequence[Any]) -> dict[str, tuple[str, bytes]]:
    """Map each inline base64 payload to its digest and decoded bytes."""
    pending: dict[str, tuple[str, bytes]] = {}
    for message in messages:
        if not isinstance(message, BaseMessage) or not isinstance(message.content, list):
            continue
        for block in message.content:
            payload = block.get("base64") if isinstance(block, dict) else None
            if not isinstance(payload, str) or payload in pending:
                continue
            try:
                raw = base64.b64decode(payload, validate=True)
            except (ValueError, binascii.Error):
                continue
            pending[payload] = (hashlib.sha256(raw).hexdigest(), raw)
    return pending


def _stub_messages(messages: Sequence[Any], digests: dict[str, str]) -> list[Any]:
    """Replace payloads found in `digests` with blob references."""

    def stub(block: Any) -> Any:  # noqa: ANN401
        if not isinstance(block, dict) or block.get("base64") not in digests:
            return block
        return {key: value for key, value in block.items() if key != "base64"} | {_BLOB_REF_KEY: digests[block["base64"]]}

    result = []
    for message in messages:
        if isinstance(message, BaseMessage) and isinstance(message.content, list):
            content = [stub(block) for block in message.content]
            if content != message.content:
                message = message.model_copy(update={"content": content})  # noqa: PLW2901
        result.append(message)
    return result


def _stored_digests(pending: dict[str, tuple[str, bytes]], errors: Sequence[str | None], cache: dict[str, str]) -> dict[str, str]:
    stored: dict[str, str] = {}
    for (payload, (digest, _)), error in zip(pending.items(), errors, strict=True):
        if error is None:
            stored[payload] = digest
            cache[digest] = payload
    return stored


def _offload_messages(messages: Sequence[Any], backend: BackendProtocol, prefix: str, cache: dict[str, str]) -> list[Any]:
    """Upload inline binary payloads and return messages carrying blob references.

    Payloads that fail to upload stay inline.
    """
    pending = _pending_blobs(messages)
    if not pending:
        return list(messages)
    try:
        responses = backend.upload_files([(_blob_path(prefix, digest), raw) for digest, raw in pending.values()])
    except Exception:  # noqa: BLE001 -- offload is best-effort; the payload stays inline
        logger.warning("Failed to offload binary read_file content; keeping it inline", exc_info=True)
        return list(messages)
    return _stub_messages(messages, _stored_digests(pending, [r.error for r in responses], cache))


async def _aoffload_messages(messages: Sequence[Any], backend: BackendProtocol, prefix: str, cache: dict[str, str]) -> list[Any]:
    """Async version of `_offload_messages`."""
    pending = _pending_blobs(messages)
    if not pending:
        return list(messages)
    try:
        responses = await backend.aupload_files([(_blob_path(prefix, digest), raw) for digest, raw in pending.values()])
    except Exception:  # noqa: BLE001 -- offload is best-effort; the payload stays inline
        logger.warning("Failed to offload binary read_file content; keeping it inline", exc_info=True)
        return list(messages)
    return _stub_messages(messages, _stored_digests(pending, [r.error for r in responses], cache))


def _offload_tool_result(result: ToolMessage | Command, backend: BackendProtocol, prefix: str, cache: dict[str, str]) -> ToolMessage | Command:
    """Apply `_offload_messages` to a tool result."""
    if isinstance(result, ToolMessage):
        return _offload_messages([result], backend, prefix, cache)[0]
    if isinstance(result.update, dict) and isinstance(result.update.get("messages"), list):
        return replace(result, update={**result.update, "messages": _offload_messages(result.update["messages"], backend, prefix, cache)})
    return result


async def _aoffload_tool_result(result: ToolMessage | Command, backend: BackendProtocol, prefix: str, cache: dict[str, str]) -> ToolMessage | Command:
    """Async version of `_offload_tool_result`."""
    if isinstance(result, ToolMessage):
        return (await _aoffload_messages([result], backend, prefix, cache))[0]
    if isinstance(result.update, dict) and isinstance(result.update.get("messages"), list):
        messages = await _aoffload_messages(result.update["messages"], backend, prefix, cache)
        return replace(result, update={**result.update, "messages": messages})
    return result


def _human_candidates(state_messages: Sequence[Any], pending: Sequence[Any]) -> list[HumanMessage]:
    """Return `HumanMessage`s with ids after the last `AIMessage`, preferring versions already queued in `pending`."""
    last_ai = max((i for i, m in enumerate(state_messages) if isinstance(m, AIMessage)), default=-1)
    queued = {m.id: m for m in pending if isinstance(m, BaseMessage) and m.id is not None}
    return [queued.get(m.id, m) for m in state_messages[last_ai + 1 :] if isinstance(m, HumanMessage) and m.id is not None]


def _merge_replacements(pending: Sequence[Any], offloaded: Sequence[Any], candidates: Sequence[Any]) -> list[Any]:
    replaced = [new for new, old in zip(offloaded, candidates, strict=True) if new is not old]
    replaced_ids = {m.id for m in replaced}
    return [*(m for m in pending if getattr(m, "id", None) not in replaced_ids), *replaced]


def _offload_human_messages(
    state_messages: Sequence[Any], pending: Sequence[Any], backend: BackendProtocol, prefix: str, cache: dict[str, str]
) -> list[Any]:
    """Return `pending` plus stubbed replacements for `HumanMessage`s in state carrying inline payloads."""
    candidates = _human_candidates(state_messages, pending)
    return _merge_replacements(pending, _offload_messages(candidates, backend, prefix, cache), candidates)


async def _aoffload_human_messages(
    state_messages: Sequence[Any], pending: Sequence[Any], backend: BackendProtocol, prefix: str, cache: dict[str, str]
) -> list[Any]:
    """Async version of `_offload_human_messages`."""
    candidates = _human_candidates(state_messages, pending)
    return _merge_replacements(pending, await _aoffload_messages(candidates, backend, prefix, cache), candidates)


def _referenced_digests(messages: Sequence[BaseMessage]) -> list[str]:
    digests: dict[str, None] = {}
    for message in messages:
        if not isinstance(message.content, list):
            continue
        for block in message.content:
            ref = block.get(_BLOB_REF_KEY) if isinstance(block, dict) else None
            if isinstance(ref, str) and _DIGEST_RE.fullmatch(ref):
                digests[ref] = None
    return list(digests)


def _cached_payloads(digests: list[str], cache: dict[str, str]) -> tuple[dict[str, str], list[str]]:
    payloads = {digest: payload for digest in digests if (payload := cache.get(digest)) is not None}
    return payloads, [digest for digest in digests if digest not in payloads]


def _accept_downloads(missing: list[str], responses: Sequence[FileDownloadResponse], payloads: dict[str, str], cache: dict[str, str]) -> None:
    for digest, response in zip(missing, responses, strict=False):
        # Blobs live on an agent-writable filesystem, so verify before trusting them.
        if response.error is not None or response.content is None or hashlib.sha256(response.content).hexdigest() != digest:
            continue
        payload = base64.b64encode(response.content).decode("ascii")
        cache[digest] = payload
        payloads[digest] = payload


def _message_has_refs(message: BaseMessage) -> bool:
    return isinstance(message.content, list) and any(isinstance(block, dict) and _BLOB_REF_KEY in block for block in message.content)


def _restore_payloads(messages: Sequence[AnyMessage], payloads: dict[str, str]) -> list[AnyMessage]:
    def hydrate(block: Any) -> Any:  # noqa: ANN401
        if not isinstance(block, dict) or _BLOB_REF_KEY not in block:
            return block
        ref = block[_BLOB_REF_KEY]
        payload = payloads.get(ref) if isinstance(ref, str) else None
        if payload is None:
            return {"type": "text", "text": _MISSING_BLOB_TEXT}
        return {key: value for key, value in block.items() if key != _BLOB_REF_KEY} | {"base64": payload}

    result = []
    for message in messages:
        if _message_has_refs(message):
            message = message.model_copy(update={"content": [hydrate(block) for block in message.content]})  # noqa: PLW2901
        result.append(message)
    return result


def _hydrate_messages(messages: Sequence[AnyMessage], backend: BackendProtocol, prefix: str, cache: dict[str, str]) -> list[AnyMessage]:
    """Restore base64 payloads for blob references; unavailable blobs become a text notice."""
    if not any(_message_has_refs(message) for message in messages):
        return list(messages)
    payloads, missing = _cached_payloads(_referenced_digests(messages), cache)
    if missing:
        try:
            responses = backend.download_files([_blob_path(prefix, digest) for digest in missing])
        except Exception:  # noqa: BLE001 -- unavailable blobs degrade to a text notice
            logger.warning("Failed to load offloaded read_file content", exc_info=True)
            responses = []
        _accept_downloads(missing, responses, payloads, cache)
    return _restore_payloads(messages, payloads)


async def _ahydrate_messages(messages: Sequence[AnyMessage], backend: BackendProtocol, prefix: str, cache: dict[str, str]) -> list[AnyMessage]:
    """Async version of `_hydrate_messages`."""
    if not any(_message_has_refs(message) for message in messages):
        return list(messages)
    payloads, missing = _cached_payloads(_referenced_digests(messages), cache)
    if missing:
        try:
            responses = await backend.adownload_files([_blob_path(prefix, digest) for digest in missing])
        except Exception:  # noqa: BLE001 -- unavailable blobs degrade to a text notice
            logger.warning("Failed to load offloaded read_file content", exc_info=True)
            responses = []
        _accept_downloads(missing, responses, payloads, cache)
    return _restore_payloads(messages, payloads)
