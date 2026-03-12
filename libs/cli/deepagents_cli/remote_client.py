"""Remote agent client — thin wrapper around LangGraph's `RemoteGraph`.

Delegates streaming, state management, and SSE handling to
`langgraph.pregel.remote.RemoteGraph`. The only added logic is converting raw
message dicts from the server into LangChain message objects that the CLI's
Textual adapter expects.
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logger = logging.getLogger(__name__)

if os.environ.get("DEEPAGENTS_DEBUG"):
    from pathlib import Path

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

    Wraps `langgraph.pregel.remote.RemoteGraph` which handles SSE parsing,
    stream-mode negotiation (`messages-tuple`), namespace extraction, and
    interrupt detection. This class adds only message-object conversion for the
    Textual adapter and thread-ID normalization.
    """

    def __init__(
        self,
        url: str,
        *,
        graph_name: str = "agent",
    ) -> None:
        """Initialize the remote agent client.

        Args:
            url: Base URL of the LangGraph server.
            graph_name: Name of the graph on the server.
        """
        self._url = url
        self._graph_name = graph_name
        self._graph: Any = None

    def _get_graph(self) -> Any:  # noqa: ANN401
        """Lazily create the `RemoteGraph` instance.

        Returns:
            A `RemoteGraph` connected to the server.
        """
        if self._graph is None:
            from langgraph.pregel.remote import RemoteGraph

            self._graph = RemoteGraph(
                self._graph_name,
                url=self._url,
            )
        return self._graph

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

        Delegates to `RemoteGraph.astream` (which handles `messages-tuple`
        negotiation, SSE routing, and namespace parsing) and converts the raw
        message dicts into LangChain message objects for the adapter.

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
        from langchain_core.messages import BaseMessage

        graph = self._get_graph()
        config = _prepare_config(config)

        if not config.get("configurable", {}).get("thread_id"):
            msg = "thread_id is required in config.configurable"
            raise ValueError(msg)

        async for ns, mode, data in graph.astream(
            input,
            stream_mode=stream_mode or ["messages", "updates"],
            subgraphs=subgraphs,
            config=config,
        ):
            logger.debug("RemoteGraph event mode=%s ns=%s", mode, ns)

            if mode == "messages":
                msg_dict, meta = data
                if isinstance(msg_dict, dict):
                    msg_obj = _convert_message_data(msg_dict)
                    if msg_obj is not None:
                        yield (ns, "messages", (msg_obj, meta or {}))
                elif isinstance(msg_dict, BaseMessage):
                    # Already a LangChain message object (pre-deserialized)
                    yield (ns, "messages", (msg_dict, meta or {}))
                else:
                    logger.warning(
                        "Unexpected message data type in stream: %s",
                        type(msg_dict).__name__,
                    )
                continue

            if mode == "updates" and isinstance(data, dict):
                update_data = data
                if "__interrupt__" in data:
                    update_data = {
                        **data,
                        "__interrupt__": _convert_interrupts(data["__interrupt__"]),
                    }
                yield (ns, "updates", update_data)
                continue

            yield (ns, mode, data)

    async def aget_state(
        self,
        config: dict[str, Any],
    ) -> Any:  # noqa: ANN401
        """Get the current state of a thread.

        Returns `None` for missing `thread_id` or when the thread does not
        exist on the server (404). All other errors (network, auth, 500) are
        logged at WARNING and re-raised so callers can handle them.

        Args:
            config: Config with `configurable.thread_id`.

        Returns:
            Thread state object with `values` and `next` attributes, or `None`
                if the thread is not found.
        """
        from langgraph_sdk.errors import NotFoundError

        thread_id = config.get("configurable", {}).get("thread_id")
        if thread_id is None:
            return None

        graph = self._get_graph()
        try:
            return await graph.aget_state(_prepare_config(config))
        except NotFoundError:
            logger.debug("Thread %s not found on server", thread_id)
            return None
        except Exception:
            logger.warning(
                "Failed to get state for thread %s", thread_id, exc_info=True
            )
            raise

    async def aupdate_state(
        self,
        config: dict[str, Any],
        values: dict[str, Any],
    ) -> None:
        """Update the state of a thread.

        Exceptions from the underlying graph (server/network errors) are logged
        at WARNING level and then re-raised so callers can handle them.

        Args:
            config: Config with `configurable.thread_id`.
            values: State values to update.
        """
        thread_id = config.get("configurable", {}).get("thread_id")
        if thread_id is None:
            return

        graph = self._get_graph()
        try:
            await graph.aupdate_state(_prepare_config(config), values)
        except Exception:
            logger.warning(
                "Failed to update state for thread %s", thread_id, exc_info=True
            )
            raise

    def with_config(self, config: dict[str, Any]) -> RemoteAgent:  # noqa: ARG002
        """Return self (config is passed per-call, not stored).

        Args:
            config: Ignored.

        Returns:
            Self.
        """
        return self


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prepare_config(config: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize config, converting `thread_id` to a valid UUID.

    Args:
        config: Raw config dict.

    Returns:
        A shallow copy with `configurable.thread_id` as a UUID string.
    """
    config = dict(config or {})
    configurable = dict(config.get("configurable", {}))
    thread_id = configurable.get("thread_id")
    if thread_id:
        configurable["thread_id"] = _to_uuid(thread_id)
    config["configurable"] = configurable
    return config


def _convert_interrupts(raw: Any) -> list[Any]:  # noqa: ANN401
    """Convert interrupt dicts from the server into Interrupt objects.

    Args:
        raw: List of interrupt dicts or Interrupt objects from the server.

    Returns:
        List of Interrupt objects.
    """
    from langgraph.types import Interrupt

    if not isinstance(raw, list):
        logger.warning(
            "Expected list for __interrupt__ data, got %s",
            type(raw).__name__,
        )
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


def _convert_message_data(data: dict[str, Any]) -> Any:  # noqa: ANN401
    """Convert a server message dict into a LangChain message object.

    Args:
        data: Message dict from the server.

    Returns:
        A LangChain message object, or `None` if conversion fails.
    """
    from langchain_core.messages import (
        AIMessageChunk,
        HumanMessage,
        ToolMessage,
    )

    msg_type = data.get("type", "")

    if msg_type in {"ai", "AIMessage", "AIMessageChunk"}:
        content = data.get("content", "")
        tool_call_chunks = data.get("tool_call_chunks", [])
        tool_calls = data.get("tool_calls", [])
        usage_metadata = data.get("usage_metadata")
        response_metadata = data.get("response_metadata", {})

        kwargs: dict[str, Any] = {
            "content": content,
            "id": data.get("id"),
            "response_metadata": response_metadata,
        }
        # messages-tuple sends tool_call_chunks (string args) directly.
        if tool_call_chunks:
            kwargs["tool_call_chunks"] = [
                {
                    "name": tc.get("name"),
                    "args": tc.get("args", ""),
                    "id": tc.get("id"),
                    "index": tc.get("index", i),
                }
                for i, tc in enumerate(tool_call_chunks)
            ]
        elif tool_calls:
            has_str_args = any(isinstance(tc.get("args"), str) for tc in tool_calls)
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

        try:
            chunk = AIMessageChunk(**kwargs)
        except Exception:
            logger.warning(
                "Failed to construct AIMessageChunk from server data (id=%s)",
                data.get("id"),
                exc_info=True,
            )
            return None
        if usage_metadata:
            chunk.usage_metadata = usage_metadata
        return chunk

    if msg_type in {"human", "HumanMessage"}:
        try:
            return HumanMessage(
                content=data.get("content", ""),
                id=data.get("id"),
            )
        except Exception:
            logger.warning(
                "Failed to construct HumanMessage from server data (id=%s)",
                data.get("id"),
                exc_info=True,
            )
            return None

    if msg_type in {"tool", "ToolMessage"}:
        try:
            return ToolMessage(
                content=data.get("content", ""),
                tool_call_id=data.get("tool_call_id", ""),
                name=data.get("name", ""),
                id=data.get("id"),
                status=data.get("status", "success"),
            )
        except Exception:
            logger.warning(
                "Failed to construct ToolMessage from server data (id=%s)",
                data.get("id"),
                exc_info=True,
            )
            return None

    logger.warning("Unknown message type in stream: %s", msg_type)
    return None
