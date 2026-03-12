"""Remote agent client using the LangGraph SDK.

Provides `RemoteAgent` which wraps the `langgraph-sdk` client to communicate
with a LangGraph server over HTTP+SSE, matching the interface used by the CLI's
streaming and state management code.
"""

from __future__ import annotations

import logging
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logger = logging.getLogger(__name__)

if os.environ.get("DEEPAGENTS_DEBUG"):
    _debug_path = Path(
        os.environ.get(
            "DEEPAGENTS_DEBUG_FILE",
            "/tmp/deepagents_debug.log",  # noqa: S108
        )
    )
    _fh = logging.FileHandler(str(_debug_path), mode="a")
    _fh.setLevel(logging.DEBUG)
    _fh.setFormatter(logging.Formatter("%(asctime)s %(name)s %(message)s"))
    logger.addHandler(_fh)
    logger.setLevel(logging.DEBUG)

_STREAM_MODES = ["messages", "updates"]


_THREAD_UUID_NAMESPACE = uuid.UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")


def _to_uuid(short_id: str) -> str:
    """Convert a thread ID string to a valid UUID string.

    Args:
        short_id: Thread ID (typically a UUID7 from `generate_thread_id`, but
            shorter non-UUID strings are also accepted and converted via uuid5).

    Returns:
        Valid UUID string. Already-valid UUIDs are returned as-is.
    """
    try:
        return str(uuid.UUID(short_id))
    except ValueError:
        return str(uuid.uuid5(_THREAD_UUID_NAMESPACE, short_id))


class RemoteAgent:
    """Client that talks to a LangGraph server over HTTP+SSE.

    Provides `astream()`, `aget_state()`, and `aupdate_state()` with signatures
    compatible with the compiled graph interface used by the CLI's streaming and
    state management code.
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
            ValueError: If `thread_id` is not present in `config`.
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
            event = chunk.event if hasattr(chunk, "event") else ""
            data = chunk.data if hasattr(chunk, "data") else chunk
            logger.debug(
                "SSE event=%s data_type=%s data=%s",
                event,
                type(data).__name__,
                repr(data)[:500],
            )
            converted_list = converter.convert(chunk, modes)
            logger.debug(
                "Converted %d tuples from event=%s", len(converted_list), event
            )
            for converted in converted_list:
                yield converted

    async def aget_state(
        self,
        config: dict[str, Any],
    ) -> Any:  # noqa: ANN401
        """Get the current state of a thread.

        Uses `_ensure_thread` so that threads persisted in the SQLite
        checkpointer (from a previous server session) are registered in the
        server's in-memory thread registry before reading state.

        Args:
            config: Config with `configurable.thread_id`.

        Returns:
            Thread state object with `values` and `next` attributes, or `None`
                if the thread does not exist.
        """
        client = self._get_client()
        thread_id = config.get("configurable", {}).get("thread_id")
        if thread_id is None:
            return None

        try:
            thread = await self._ensure_thread(thread_id)
            state = await client.threads.get_state(thread["thread_id"])
            return _StateWrapper(state)
        except Exception:
            logger.warning(
                "Failed to get state for thread %s", thread_id, exc_info=True
            )
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
            thread = await self._ensure_thread(thread_id)
            await client.threads.update_state(thread["thread_id"], values)
        except Exception:
            logger.warning(
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
        except Exception as exc:  # noqa: BLE001
            # Only create a new thread if the server says "not found".
            # For other errors (network, auth, 500) let the caller handle it.
            import httpx

            is_not_found = isinstance(exc, httpx.HTTPStatusError) and (
                exc.response.status_code == 404  # noqa: PLR2004
            )
            if not is_not_found:
                logger.warning(
                    "Unexpected error looking up thread %s: %s", thread_id, exc
                )
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
            logger.debug("Could not determine cwd for thread metadata")

        return await client.threads.create(
            thread_id=server_tid,
            metadata=thread_metadata,
        )

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

    Makes the server response compatible with code that accesses `state.values`,
    `state.next`, etc.
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
    """Check if input is a LangGraph `Command` object.

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
    """Convert a LangGraph `Command` to a dict for the server API.

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
        self._seen_msg_ids: set[str] = set()
        self._seen_tool_args: dict[str, str] = {}

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
                msg_id = item.get("id") if isinstance(item, dict) else None
                if msg_id:
                    self._seen_msg_ids.add(msg_id)
                if event == "messages/complete":
                    if msg_id and msg_id in self._seen_text:
                        # Text already streamed via partial — but the
                        # complete event often carries usage_metadata that
                        # was absent on partials.  Emit a content-free
                        # chunk so the adapter can record token counts.
                        if isinstance(item, dict) and item.get("usage_metadata"):
                            usage_stub = dict(item)
                            usage_stub["content"] = ""
                            usage_stub["tool_calls"] = []
                            msg_obj = _convert_message_data(usage_stub)
                            if msg_obj is not None:
                                results.append((namespace, "messages", (msg_obj, {})))
                        continue
                    msg_obj = _convert_message_data(item)
                    if msg_obj is not None:
                        results.append((namespace, "messages", (msg_obj, {})))
                else:
                    results.extend(
                        (namespace, "messages", (d, {})) for d in self._to_delta(item)
                    )

        elif event == "messages/metadata":
            pass

        elif event == "updates":
            if isinstance(data, dict):
                update_data = data
                if "__interrupt__" in data:
                    interrupts = _convert_interrupts(data["__interrupt__"])
                    update_data = {**data, "__interrupt__": interrupts}
                results.append((namespace, "updates", update_data))
                results.extend(
                    (namespace, "messages", (msg_obj, {}))
                    for msg_obj in self._extract_messages_from_update(data)
                )

        elif event == "values":
            if isinstance(data, dict) and "__interrupt__" in data:
                interrupts = _convert_interrupts(data["__interrupt__"])
                if interrupts:
                    results.append(
                        (namespace, "updates", {"__interrupt__": interrupts})
                    )

        elif event in {"metadata", "end"}:
            pass

        elif event == "error":
            detail = data.get("message", data) if isinstance(data, dict) else data
            msg = f"Server stream error: {detail}"
            raise RuntimeError(msg)

        return results

    def _to_delta(self, data: dict[str, Any]) -> list[Any]:
        """Convert an accumulated message dict to delta AIMessageChunks.

        Computes the text delta and tool call args deltas separately.
        Returns them as separate messages so `content_blocks` produces
        `tool_call_chunk` entries (which requires empty content).

        Returns:
            List of delta message chunks (may be empty).
        """
        msg_id = data.get("id")
        content = data.get("content", "")
        tool_calls = data.get("tool_calls", [])

        full_text = _extract_text(content)
        prev_text = self._seen_text.get(msg_id, "")

        if full_text.startswith(prev_text):
            delta_text = full_text[len(prev_text) :]
        else:
            delta_text = full_text
        self._seen_text[msg_id] = full_text

        delta_tool_calls = _compute_tool_call_deltas(
            tool_calls, content, self._seen_tool_args
        )

        if not delta_text and not delta_tool_calls:
            # The final partial often carries usage_metadata with no new text.
            # Emit a content-free stub so the adapter can record token counts.
            usage = data.get("usage_metadata")
            if usage:
                stub = dict(data)
                stub["content"] = ""
                stub["tool_calls"] = []
                msg = _convert_message_data(stub)
                return [msg] if msg is not None else []
            return []

        clean_meta = {
            k: v
            for k, v in data.get("response_metadata", {}).items()
            if k != "model_provider"
        }
        results: list[Any] = []

        if delta_text:
            text_data = dict(data)
            text_data["content"] = delta_text
            text_data["tool_calls"] = []
            text_data["response_metadata"] = clean_meta
            msg = _convert_message_data(text_data)
            if msg is not None:
                results.append(msg)

        if delta_tool_calls:
            tc_data = dict(data)
            tc_data["content"] = ""
            tc_data["tool_calls"] = delta_tool_calls
            tc_data["response_metadata"] = clean_meta
            msg = _convert_message_data(tc_data)
            if msg is not None:
                results.append(msg)

        return results

    def _extract_messages_from_update(self, data: dict[str, Any]) -> list[Any]:
        """Extract messages from an `updates` event as a fallback.

        The `updates` event contains node outputs like
        `{"agent": {"messages": [...]}}`. This extracts any messages that
        haven't already been seen via `messages/partial` or
        `messages/complete` events.

        Args:
            data: The `updates` event data dict.

        Returns:
            List of converted message objects.
        """
        results = []
        for node_output in data.values():
            if not isinstance(node_output, dict):
                continue
            messages = node_output.get("messages", [])
            if not isinstance(messages, list):
                continue
            for msg_data in messages:
                if not isinstance(msg_data, dict):
                    continue
                msg_id = msg_data.get("id")
                if msg_id and msg_id in self._seen_msg_ids:
                    continue
                if msg_id:
                    self._seen_msg_ids.add(msg_id)
                msg_obj = _convert_message_data(msg_data)
                if msg_obj is not None:
                    results.append(msg_obj)
        return results


def _convert_interrupts(raw: Any) -> list[Any]:  # noqa: ANN401
    """Convert interrupt dicts from the server into Interrupt objects.

    Args:
        raw: List of interrupt dicts or Interrupt objects from the server.

    Returns:
        List of Interrupt objects.
    """
    from langgraph.types import Interrupt

    if not isinstance(raw, list):
        return []
    results = []
    for item in raw:
        if isinstance(item, Interrupt):
            results.append(item)
        elif isinstance(item, dict) and "value" in item:
            results.append(Interrupt(value=item["value"], id=item.get("id", "")))
        else:
            results.append(item)
    return results


def _compute_tool_call_deltas(
    tool_calls: list[dict[str, Any]],
    content: str | list[Any] | Any,  # noqa: ANN401
    seen_args: dict[str, str],
) -> list[dict[str, Any]]:
    """Compute incremental tool call deltas from accumulated server data.

    Uses `partial_json` from provider-specific content blocks (e.g.,
    Anthropic `tool_use`) when available, falling back to serialized
    `tool_calls.args`. Computes string deltas so the textual adapter receives
    incremental fragments for `tool_call_chunk` processing.

    Args:
        tool_calls: Accumulated tool calls from the server.
        content: Message content (may contain `tool_use` blocks).
        seen_args: Mutable dict tracking accumulated args per tool ID.

    Returns:
        List of tool call dicts with delta args strings.
    """
    import json as _json

    pj_by_id: dict[str, str] = {}
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                tc_id = block.get("id", "")
                pj = block.get("partial_json")
                if tc_id and pj is not None:
                    pj_by_id[tc_id] = pj

    deltas = []
    for tc in tool_calls:
        tc_id = tc.get("id", "")
        if tc_id in pj_by_id:
            args_str = pj_by_id[tc_id]
        else:
            args = tc.get("args")
            args_str = _json.dumps(args, separators=(",", ":")) if args else ""
        first_seen = tc_id not in seen_args
        prev = seen_args.get(tc_id, "")
        if not first_seen and args_str == prev:
            continue
        delta_str = args_str.removeprefix(prev)
        seen_args[tc_id] = args_str
        delta_tc = dict(tc)
        delta_tc["args"] = delta_str
        deltas.append(delta_tc)
    return deltas


def _extract_text(content: str | list | Any) -> str:  # noqa: ANN401
    """Extract plain text from message content.

    Args:
        content: String content or list of content blocks.

    Returns:
        Concatenated text string.
    """
    if not isinstance(content, (str, list)):
        return ""

    from langchain_core.messages import AIMessageChunk

    return str(AIMessageChunk(content=content).text)


def _convert_stream_chunk(
    chunk: Any,  # noqa: ANN401
    modes: list[str],
) -> list[tuple[tuple[str, ...], str, Any]]:
    """Stateless convenience wrapper around `_StreamConverter.convert` for tests.

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

        has_str_args = any(isinstance(tc.get("args"), str) for tc in tool_calls)
        kwargs: dict[str, Any] = {
            "content": content,
            "id": data.get("id"),
            "response_metadata": response_metadata,
        }
        if has_str_args:
            kwargs["tool_call_chunks"] = [
                {
                    "name": tc.get("name"),
                    "args": tc.get("args", ""),
                    "id": tc.get("id"),
                    "index": i,
                }
                for i, tc in enumerate(tool_calls)
            ]
        else:
            kwargs["tool_calls"] = tool_calls

        chunk = AIMessageChunk(**kwargs)
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
