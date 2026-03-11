"""Remote agent client using the LangGraph SDK.

Provides `RemoteAgent` which wraps the `langgraph-sdk` client to
communicate with a LangGraph server over HTTP+SSE, matching the
interface used by the CLI's streaming and state management code.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logger = logging.getLogger(__name__)

_STREAM_MODES = ["messages", "updates"]


def _to_uuid(short_id: str) -> str:
    """Convert a short hex thread ID to a valid UUID string.

    Args:
        short_id: Hex string (e.g., 8-char from `generate_thread_id`).

    Returns:
        Valid UUID string. Already-valid UUIDs are returned as-is.
    """
    try:
        return str(uuid.UUID(short_id))
    except ValueError:
        padded = short_id.ljust(32, "0")
        return str(uuid.UUID(padded[:32]))


class RemoteAgent:
    """Client that talks to a LangGraph server, mimicking the `Pregel` interface.

    Wraps `langgraph-sdk`'s async client to provide `astream()`,
    `aget_state()`, and `aupdate_state()` with signatures compatible
    with the CLI's existing streaming code.
    """

    def __init__(
        self,
        url: str,
        *,
        assistant_id: str | None = None,
        graph_name: str = "agent",
    ) -> None:
        """Initialize the remote agent client.

        Args:
            url: Base URL of the LangGraph server.
            assistant_id: CLI assistant/agent identifier (used in metadata).
            graph_name: Name of the graph on the server.
        """
        self._url = url
        self._assistant_id = assistant_id
        self._graph_name = graph_name
        self._client: Any = None

    def _get_client(self) -> Any:  # noqa: ANN401
        """Lazily create the langgraph-sdk async client.

        Returns:
            The langgraph-sdk async client instance.
        """
        if self._client is None:
            from langgraph_sdk import get_client

            self._client = get_client(url=self._url, api_key=None)
        return self._client

    async def astream(
        self,
        input: dict | Any,  # noqa: A002, ANN401
        *,
        stream_mode: list[str] | None = None,
        subgraphs: bool = False,
        config: dict[str, Any] | None = None,
        durability: str | None = None,  # noqa: ARG002
    ) -> AsyncIterator[tuple[tuple[str, ...], str, Any]]:
        """Stream agent execution, yielding tuples matching Pregel's format.

        Translates LangGraph server SSE events into the 3-tuple format
        `(namespace, stream_mode, data)` that `execute_task_textual` and
        `_stream_agent` expect.

        Args:
            input: The input to send (messages dict or Command).
            stream_mode: Stream modes to request.
            subgraphs: Whether to stream subgraph events.
            config: LangGraph config with `configurable.thread_id`, etc.
            durability: Ignored (server manages durability).

        Yields:
            3-tuples of `(namespace, stream_mode, data)`.

        Raises:
            ValueError: If ``thread_id`` is not present in *config*.
        """
        client = self._get_client()
        config = config or {}
        configurable = config.get("configurable", {})
        thread_id = configurable.get("thread_id")
        metadata = config.get("metadata", {})

        if thread_id is None:
            msg = "thread_id is required in config.configurable"
            raise ValueError(msg)

        try:
            thread = await self._ensure_thread(thread_id, metadata)
        except Exception:
            logger.exception("Failed to ensure thread %s", thread_id)
            raise
        actual_thread_id = thread["thread_id"]

        modes = stream_mode or _STREAM_MODES

        is_command = _is_command(input)
        kwargs: dict[str, Any] = {
            "thread_id": actual_thread_id,
            "assistant_id": self._graph_name,
            "stream_mode": modes,
            "stream_subgraphs": subgraphs,
            "metadata": metadata,
        }
        if is_command:
            kwargs["command"] = _command_to_dict(input)
        else:
            kwargs["input"] = input

        converter = _StreamConverter()
        async for chunk in client.runs.stream(**kwargs):
            for converted in converter.convert(chunk, modes):
                yield converted

    async def aget_state(
        self,
        config: dict[str, Any],
    ) -> Any:  # noqa: ANN401
        """Get the current state of a thread.

        Args:
            config: Config with `configurable.thread_id`.

        Returns:
            Thread state object with `values` and `next` attributes.
        """
        client = self._get_client()
        thread_id = config.get("configurable", {}).get("thread_id")
        if thread_id is None:
            return None

        try:
            thread_id = await self._resolve_thread_id(thread_id)
            if thread_id is None:
                return None
            state = await client.threads.get_state(thread_id)
            return _StateWrapper(state)
        except Exception:
            logger.debug("Failed to get state for thread %s", thread_id, exc_info=True)
            return None

    async def aupdate_state(
        self,
        config: dict[str, Any],
        values: dict[str, Any],
    ) -> None:
        """Update the state of a thread.

        Args:
            config: Config with `configurable.thread_id`.
            values: State values to update.
        """
        client = self._get_client()
        thread_id = config.get("configurable", {}).get("thread_id")
        if thread_id is None:
            return

        try:
            thread_id = await self._resolve_thread_id(thread_id)
            if thread_id is None:
                return
            await client.threads.update_state(thread_id, values)
        except Exception:
            logger.debug(
                "Failed to update state for thread %s", thread_id, exc_info=True
            )

    async def _ensure_thread(
        self,
        thread_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Ensure a thread exists on the server, creating it if needed.

        Args:
            thread_id: Desired thread ID.
            metadata: Optional metadata for the thread.

        Returns:
            Thread dict from the server.
        """
        client = self._get_client()

        server_tid = _to_uuid(thread_id)

        try:
            return await client.threads.get(server_tid)
        except Exception:  # noqa: BLE001
            logger.debug("Thread %s not found, creating new one", thread_id)

        thread_metadata = dict(metadata or {})
        if self._assistant_id:
            thread_metadata.setdefault("assistant_id", self._assistant_id)
            thread_metadata.setdefault("agent_name", self._assistant_id)
        thread_metadata.setdefault("updated_at", datetime.now(UTC).isoformat())

        try:
            cwd = str(Path.cwd())
            thread_metadata.setdefault("cwd", cwd)
        except OSError:
            pass

        return await client.threads.create(
            thread_id=server_tid,
            metadata=thread_metadata,
        )

    async def _resolve_thread_id(self, thread_id: str) -> str | None:
        """Resolve a thread ID, returning None if the thread doesn't exist.

        Args:
            thread_id: Thread ID to resolve.

        Returns:
            The thread ID if it exists, None otherwise.
        """
        client = self._get_client()
        try:
            thread = await client.threads.get(_to_uuid(thread_id))
            return thread["thread_id"]
        except Exception:  # noqa: BLE001
            return None

    def with_config(self, config: dict[str, Any]) -> RemoteAgent:  # noqa: ARG002
        """Return self (config is passed per-call, not stored).

        Args:
            config: Ignored.

        Returns:
            Self.
        """
        return self


class _StateWrapper:
    """Wraps a server thread state dict to provide attribute access.

    Makes the server response compatible with code that accesses
    `state.values`, `state.next`, etc.
    """

    def __init__(self, state: dict[str, Any]) -> None:
        self._state = state

    @property
    def values(self) -> dict[str, Any]:
        """State values dict."""
        return self._state.get("values", {})

    @property
    def next(self) -> list[str]:
        """Next nodes to execute."""
        return self._state.get("next", [])

    def __bool__(self) -> bool:
        return bool(self._state)


def _is_command(input: Any) -> bool:  # noqa: A002, ANN401
    """Check if input is a LangGraph Command object.

    Args:
        input: The input to check.

    Returns:
        True if the input is a Command.
    """
    try:
        from langgraph.types import Command

        return isinstance(input, Command)
    except ImportError:
        return False


def _command_to_dict(cmd: Any) -> dict[str, Any]:  # noqa: ANN401
    """Convert a LangGraph Command to a dict for the server API.

    Args:
        cmd: A `Command` instance.

    Returns:
        Dict with `resume`, `goto`, and/or `update` keys.
    """
    result: dict[str, Any] = {}
    if hasattr(cmd, "resume") and cmd.resume is not None:
        result["resume"] = cmd.resume
    if hasattr(cmd, "goto") and cmd.goto is not None:
        result["goto"] = cmd.goto
    if hasattr(cmd, "update") and cmd.update is not None:
        result["update"] = cmd.update
    return result


class _StreamConverter:
    """Stateful converter that turns server SSE events into Pregel 3-tuples.

    Tracks accumulated text per message ID so `messages/partial` events
    (which contain the full text so far) are converted to incremental deltas.
    """

    def __init__(self) -> None:
        self._seen_text: dict[str | None, str] = {}
        self._seen_tool_call_ids: set[str] = set()

    def convert(
        self,
        chunk: Any,  # noqa: ANN401
        modes: list[str],  # noqa: ARG002
    ) -> list[tuple[tuple[str, ...], str, Any]]:
        """Convert a server StreamPart into Pregel-compatible 3-tuples.

        Args:
            chunk: A `StreamPart` from `client.runs.stream()`.
            modes: The requested stream modes.

        Returns:
            List of converted 3-tuples (may be empty for non-matching events).

        Raises:
            RuntimeError: If the server sends an SSE error event.
        """
        event = chunk.event if hasattr(chunk, "event") else ""
        data = chunk.data if hasattr(chunk, "data") else chunk

        results: list[tuple[tuple[str, ...], str, Any]] = []
        namespace: tuple[str, ...] = ()

        if event in {"messages/partial", "messages/complete"}:
            items = (
                data
                if isinstance(data, list)
                else [data]
                if isinstance(data, dict)
                else []
            )
            for item in items:
                if event == "messages/complete":
                    msg_obj = _convert_message_data(item)
                    if msg_obj is not None:
                        results.append((namespace, "messages", (msg_obj, {})))
                else:
                    delta = self._to_delta(item)
                    if delta is not None:
                        results.append((namespace, "messages", (delta, {})))

        elif event == "messages/metadata":
            pass

        elif event == "updates":
            if isinstance(data, dict):
                results.append((namespace, "updates", data))

        elif event == "values":
            if isinstance(data, dict) and "__interrupt__" in data:
                results.append(
                    (namespace, "updates", {"__interrupt__": data["__interrupt__"]})
                )

        elif event in {"metadata", "end"}:
            pass

        elif event == "error":
            detail = data.get("message", data) if isinstance(data, dict) else data
            msg = f"Server stream error: {detail}"
            raise RuntimeError(msg)

        return results

    def _to_delta(self, data: dict[str, Any]) -> Any:  # noqa: ANN401
        """Convert an accumulated message dict to a delta AIMessageChunk.

        Computes the text delta by comparing against the previously seen
        text for this message ID.

        Returns:
            A delta message chunk, or None if there is no new content.
        """
        msg_id = data.get("id")
        content = data.get("content", "")

        full_text = _extract_text(content)
        prev_text = self._seen_text.get(msg_id, "")

        if full_text.startswith(prev_text):
            delta_text = full_text[len(prev_text) :]
        else:
            delta_text = full_text
        self._seen_text[msg_id] = full_text

        delta_data = dict(data)
        if isinstance(content, str):
            delta_data["content"] = delta_text
        elif isinstance(content, list):
            delta_data["content"] = (
                [{"type": "text", "text": delta_text}] if delta_text else []
            )
        else:
            delta_data["content"] = delta_text

        new_tool_calls = [
            tc
            for tc in data.get("tool_calls", [])
            if tc.get("id") and tc["id"] not in self._seen_tool_call_ids
        ]
        for tc in new_tool_calls:
            self._seen_tool_call_ids.add(tc["id"])
        delta_data["tool_calls"] = new_tool_calls

        if not delta_text and not new_tool_calls and not data.get("invalid_tool_calls"):
            return None

        return _convert_message_data(delta_data)


def _extract_text(content: str | list | Any) -> str:  # noqa: ANN401
    """Extract plain text from message content.

    Args:
        content: String content or list of content blocks.

    Returns:
        Concatenated text string.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return ""


def _convert_stream_chunk(
    chunk: Any,  # noqa: ANN401
    modes: list[str],
) -> list[tuple[tuple[str, ...], str, Any]]:
    """Stateless conversion for backward compatibility (used in tests).

    Args:
        chunk: A `StreamPart` from `client.runs.stream()`.
        modes: The requested stream modes.

    Returns:
        List of converted 3-tuples.
    """
    return _StreamConverter().convert(chunk, modes)


def _convert_message_data(data: dict[str, Any]) -> Any:  # noqa: ANN401
    """Convert a server message dict into a LangChain message object.

    Args:
        data: Message dict from the server.

    Returns:
        A LangChain message object, or None if conversion fails.
    """
    from langchain_core.messages import (
        AIMessageChunk,
        HumanMessage,
        ToolMessage,
    )

    msg_type = data.get("type", "")

    if msg_type in {"ai", "AIMessage", "AIMessageChunk"}:
        content = data.get("content", "")
        tool_calls = data.get("tool_calls", [])
        usage_metadata = data.get("usage_metadata")
        response_metadata = data.get("response_metadata", {})

        content_blocks = []
        if isinstance(content, str) and content:
            content_blocks.append({"type": "text", "text": content})
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    content_blocks.append(block)
                elif isinstance(block, str):
                    content_blocks.append({"type": "text", "text": block})

        content_blocks.extend(
            {
                "type": "tool_call",
                "id": tc.get("id"),
                "name": tc.get("name"),
                "args": tc.get("args", {}),
            }
            for tc in tool_calls
        )

        invalid_tool_calls = data.get("invalid_tool_calls", [])
        content_blocks.extend(
            {
                "type": "tool_call_chunk",
                "id": itc.get("id"),
                "name": itc.get("name"),
                "args": itc.get("args", ""),
                "index": itc.get("index"),
            }
            for itc in invalid_tool_calls
        )

        chunk = AIMessageChunk(
            content=content_blocks or content,
            tool_calls=tool_calls,
            id=data.get("id"),
            response_metadata=response_metadata,
        )
        if usage_metadata:
            chunk.usage_metadata = usage_metadata
        return chunk

    if msg_type in {"human", "HumanMessage"}:
        return HumanMessage(
            content=data.get("content", ""),
            id=data.get("id"),
        )

    if msg_type in {"tool", "ToolMessage"}:
        return ToolMessage(
            content=data.get("content", ""),
            tool_call_id=data.get("tool_call_id", ""),
            name=data.get("name", ""),
            id=data.get("id"),
            status=data.get("status", "success"),
        )

    logger.debug("Unknown message type in stream: %s", msg_type)
    return None
