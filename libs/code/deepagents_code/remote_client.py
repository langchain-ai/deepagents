"""Remote agent client — thin wrapper around LangGraph's `RemoteGraph`.

Delegates streaming, state management, and SSE handling to
`langgraph.pregel.remote.RemoteGraph`. This wrapper converts streamed message
dicts into LangChain message objects for the app's Textual adapter, but leaves
state snapshots in the server's serialized form.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Mapping

logger = logging.getLogger(__name__)

_RUN_CANCEL_WAIT_SECONDS = 10.0
"""Per-run cancel wait. Picked so a stuck server-side run can't hang the UI on
Esc for more than ~10s, while leaving room for an actually-cancelling run to
finish its in-flight tool call.

Concurrent cancels keep aggregate wall time bounded by this value regardless of
how many runs are active.
"""


def _require_thread_id(config: Mapping[str, Any] | None) -> str:
    """Extract and validate that `thread_id` is present in config.

    Args:
        config: Config dict with `configurable.thread_id`.

    Returns:
        The thread ID string.

    Raises:
        ValueError: If `thread_id` is missing.
    """
    thread_id = (config or {}).get("configurable", {}).get("thread_id")
    if not thread_id:
        msg = "thread_id is required in config.configurable"
        raise ValueError(msg)
    return thread_id


def agent_error_type(exc: BaseException) -> str:
    """Best-effort error-type name for an exception from `RemoteAgent.astream`.

    The LangGraph server serializes non-allowlisted exceptions as
    `{"error": <ExceptionType>, "message": ...}` wrapped in
    `RemoteException(payload)` (see `langgraph_api.serde`). The server-reported
    `"error"` type is the authoritative name when present; otherwise the
    exception's own class name is used. This is the single source of truth for
    "what error did the stream report" — both `format_agent_exception` (display
    string) and the UI's error-enrichment path (error-type dispatch) read it.

    Args:
        exc: The exception caught from the agent stream.

    Returns:
        The serialized error type from a `RemoteException` dict payload, else
        the exception's class name.
    """
    payload = exc.args[0] if exc.args else None
    if isinstance(payload, dict):
        err_type = payload.get("error")
        if isinstance(err_type, str) and err_type:
            return err_type
    return type(exc).__name__


def format_agent_exception(exc: BaseException) -> str:
    """Render an exception from `RemoteAgent.astream` for the UI.

    The LangGraph server serializes non-allowlisted exceptions as
    `{"error": <ExceptionType>, "message": <text or "An internal error occurred">}`
    (see `langgraph_api.serde`). `RemoteGraph` wraps that dict in
    `RemoteException(payload)`, so the default `str(exc)` renders as an ugly
    Python dict repr in the UI.

    Args:
        exc: The exception caught from the agent stream.

    Returns:
        `"<ErrorType>: <message>"` for `RemoteException` dict payloads,
        otherwise `str(exc)`, falling back to the exception's class name when
        the string form is empty.
    """
    payload = exc.args[0] if exc.args else None
    if isinstance(payload, dict):
        err_type = agent_error_type(exc)
        message = payload.get("message")
        if isinstance(message, str) and message:
            return f"{err_type}: {message}"
        return err_type
    text = str(exc)
    return text or type(exc).__name__


class RemoteAgent:
    """Client that talks to a LangGraph server over HTTP+SSE.

    Wraps `langgraph.pregel.remote.RemoteGraph` which handles SSE parsing,
    stream-mode negotiation (`messages-tuple`), namespace extraction, and
    interrupt detection. This class adds streamed message-object conversion for
    the Textual adapter and thread-ID normalization. State snapshots are
    returned as provided by the server.
    """

    def __init__(
        self,
        url: str,
        *,
        graph_name: str = "agent",
        api_key: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        """Initialize the remote agent client.

        Args:
            url: Base URL of the LangGraph server.
            graph_name: Name of the graph on the server.
            api_key: API key for authenticated deployments.

                When `None`, `RemoteGraph` auto-reads `LANGGRAPH_API_KEY`,
                `LANGSMITH_API_KEY`, or `LANGCHAIN_API_KEY` from
                the environment.
            headers: Extra HTTP headers to include in every request
                (e.g. bearer tokens, proxy headers).
        """
        self._url = url
        self._graph_name = graph_name
        self._api_key = api_key
        self._headers = headers
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
                api_key=self._api_key,
                headers=self._headers,
            )
        return self._graph

    async def astream(
        self,
        input: dict | Any,  # noqa: A002, ANN401
        *,
        stream_mode: list[str] | None = None,
        subgraphs: bool = False,
        config: dict[str, Any] | None = None,
        context: Any | None = None,  # noqa: ANN401
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
            context: Runtime context (e.g. `CLIContext`) forwarded to the
                server via the SDK's `context=` parameter.
            durability: Ignored (server manages durability).

        Yields:
            3-tuples of `(namespace, stream_mode, data)`.

        Raises:
            ValueError: If `thread_id` is not present in `config`.
        """  # noqa: DOC502 — raised by _require_thread_id
        from langchain_core.messages import BaseMessage

        _require_thread_id(config)

        graph = self._get_graph()
        config = _prepare_config(config)
        dropped_count = 0

        # Mirror this server-side run to an extra LangSmith project when
        # configured. The server (`langgraph_api`) reads the SDK's
        # `langsmith_tracing` field and replicates the run to that project plus
        # its primary project. This is the only channel that reaches the run:
        # it executes in the server process, and reserved `configurable` keys
        # like `__langsmith_project__` are stripped from client input
        # server-side, so they cannot be set directly here. (Server internals
        # verified against `langgraph-api` 0.10.0 and subject to change.)
        from deepagents_code.config import get_langsmith_replica_project

        extra_stream_kwargs: dict[str, Any] = {}
        replica_project = get_langsmith_replica_project()
        if replica_project:
            extra_stream_kwargs["langsmith_tracing"] = {"project_name": replica_project}

        async for ns, mode, data in graph.astream(
            input,
            stream_mode=stream_mode or ["messages", "updates"],
            subgraphs=subgraphs,
            config=config,
            context=context,
            **extra_stream_kwargs,
        ):
            logger.debug("RemoteGraph event mode=%s ns=%s", mode, ns)

            if mode == "messages":
                msg_dict, meta = data
                if isinstance(msg_dict, dict):
                    msg_obj = _convert_message_data(msg_dict)
                    if msg_obj is not None:
                        yield (ns, "messages", (msg_obj, meta or {}))
                    else:
                        dropped_count += 1
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

        if dropped_count:
            logger.warning(
                "Dropped %d message(s) during stream due to conversion failures",
                dropped_count,
            )

    async def aget_state(
        self,
        config: dict[str, Any],
    ) -> Any:  # noqa: ANN401
        """Get the current state of a thread.

        Returns `None` when the thread does not exist on the server (404) or
        when the thread exists but has no checkpoint yet (new/empty thread).
        All other errors (network, auth, 500) are logged at WARNING and
        re-raised so callers can handle them.

        Unlike `astream`, message values are not deserialized; callers may
        receive serialized message dicts in `values["messages"]` from the
        server.

        Args:
            config: Config with `configurable.thread_id`.

        Returns:
            Thread state object with `values` and `next` attributes, or `None`
                if the thread is not found or has no checkpoint.

        Raises:
            ValueError: If `thread_id` is not present in `config`.
            TypeError: If the server returns an unexpected state shape.
        """  # noqa: DOC502 — raised by _require_thread_id
        from langgraph_sdk.errors import NotFoundError

        thread_id = _require_thread_id(config)

        graph = self._get_graph()
        try:
            return await graph.aget_state(_prepare_config(config))
        except NotFoundError:
            logger.debug("Thread %s not found on server", thread_id)
            return None
        except TypeError as e:
            # langgraph SDK bug: _create_state_snapshot does
            # state["checkpoint"]["thread_id"], but the server returns
            # checkpoint=null for threads with no checkpoint yet (new threads,
            # or threads registered via aensure_thread before any run).
            if "subscriptable" in str(e).lower():
                logger.debug(
                    "Thread %s has no checkpoint yet; treating as empty", thread_id
                )
                return None
            logger.warning(
                "Failed to get state for thread %s", thread_id, exc_info=True
            )
            raise
        except Exception:
            logger.warning(
                "Failed to get state for thread %s", thread_id, exc_info=True
            )
            raise

    async def acancel_active_runs(self, config: dict[str, Any]) -> None:
        """Cancel pending/running runs on the configured thread.

        Best-effort: per-run cancellation failures are swallowed by
        `_cancel_active_runs`. Intended for proactive cancellation on
        interrupt, before recovery-state writes.

        Args:
            config: Config with `configurable.thread_id`.

        Raises:
            ValueError: If `thread_id` is not present in `config`.
        """  # noqa: DOC502 — raised by _require_thread_id
        thread_id = _require_thread_id(config)
        await _cancel_active_runs(self._get_graph(), thread_id)

    async def aupdate_state(
        self,
        config: Mapping[str, Any],
        values: dict[str, Any],
        *,
        as_node: str | None = None,
    ) -> None:
        """Update the state of a thread.

        On HTTP 409 (`ConflictError`) the server still considers the thread
        busy — typically because the client cancelled the SSE stream before
        the server finished the run. In that case, cancel any pending/running
        runs with `wait=True` and retry the state update once. Per-run cancel
        waits are bounded by `_RUN_CANCEL_WAIT_SECONDS` and run concurrently,
        so callers cannot block indefinitely regardless of how many runs were
        active.

        Other exceptions from the underlying graph (server/network errors) are
        logged at DEBUG level and re-raised so callers can decide how to
        surface them (callers typically log at WARNING with a friendlier
        message).

        Args:
            config: Config with `configurable.thread_id`.
            values: State values to update.
            as_node: Optional graph node to attribute the state update to.

        Raises:
            ValueError: If `thread_id` is not present in `config`.
        """  # noqa: DOC502 — raised by _require_thread_id
        from langgraph_sdk.errors import ConflictError

        thread_id = _require_thread_id(config)
        prepared = _prepare_config(config)
        graph = self._get_graph()

        try:
            await graph.aupdate_state(prepared, values, as_node=as_node)
        except ConflictError:
            pass
        except Exception:
            logger.debug(
                "Failed to update state for thread %s", thread_id, exc_info=True
            )
            raise
        else:
            return

        await _cancel_active_runs(graph, thread_id)

        try:
            await graph.aupdate_state(prepared, values, as_node=as_node)
        except Exception:
            logger.debug(
                "Retry of update_state still failed for thread %s",
                thread_id,
                exc_info=True,
            )
            raise

    async def aput_store_item(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
    ) -> None:
        """Write an item to the server-side LangGraph Store.

        Args:
            namespace: Store namespace.
            key: Item key within `namespace`.
            value: JSON-serializable item value.

        Notes:
            A failed write is logged at debug and re-raised. The re-raise is
            load-bearing: callers (`awrite_approval_mode` and its callers)
            depend on the failure propagating so they can fail closed — drop
            the live approval-mode key and interrupt rather than keep
            auto-approving. Removing the `raise` would turn the debug log into
            a silent-failure hole, so the higher-severity logging is left to
            those callers, which re-log at warning with `exc_info`.
        """
        graph = self._get_graph()
        try:
            client = graph._validate_client()
            await client.store.put_item(namespace, key, value, index=False)
        except Exception:
            logger.debug(
                "Failed to write store item %s/%s",
                ".".join(namespace),
                key,
                exc_info=True,
            )
            # Load-bearing: see Notes. Callers fail closed on this propagation.
            raise

    async def aensure_thread(self, config: dict[str, Any]) -> None:
        """Ensure the remote thread record exists before mutating state.

        In the LangGraph dev server, checkpoint persistence and HTTP thread
        registration are separate. After a server restart, a thread may still
        have checkpointed state on disk while `POST /threads/{id}/state`
        returns 404 because the server has not yet materialized that thread in
        its live store.

        This method performs the idempotent HTTP-side registration with
        `if_exists='do_nothing'` so callers that recovered state from
        persistence can safely follow up with `aupdate_state`.

        Args:
            config: Config with `configurable.thread_id` and optional metadata.

        Raises:
            ValueError: If `thread_id` is not present in `config`.
        """  # noqa: DOC502 — raised by _require_thread_id
        _require_thread_id(config)

        graph = self._get_graph()
        prepared = _prepare_config(config)
        thread_id = prepared["configurable"]["thread_id"]
        metadata = prepared.get("metadata")
        thread_metadata = metadata if isinstance(metadata, dict) else None

        try:
            client = graph._validate_client()
            await client.threads.create(
                thread_id=thread_id,
                if_exists="do_nothing",
                metadata=thread_metadata,
                graph_id=self._graph_name,
            )
        except Exception:
            logger.warning(
                "Failed to ensure thread %s exists on remote server",
                thread_id,
                exc_info=True,
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


async def _cancel_active_runs(graph: Any, thread_id: str) -> None:  # noqa: ANN401
    """Cancel pending/running runs on a thread and wait for them to settle.

    Best-effort: per-run cancellation failures are logged at DEBUG and
    swallowed. Conditions that imply the retry will likely still 409 — failing
    to obtain the SDK client, or failing to list runs in every status — are
    logged at WARNING so they show up in default logs.

    The SDK client is reached via `graph._validate_client()`, a private
    attribute on `langgraph.pregel.remote.RemoteGraph`. If upstream renames
    or removes it, this helper degrades to no-op and the caller's retry will
    re-raise the original `ConflictError`.

    Per-run cancels run concurrently and are bounded by
    `_RUN_CANCEL_WAIT_SECONDS`, so aggregate wall time stays near that bound
    regardless of how many runs are active.

    Args:
        graph: Underlying `RemoteGraph` instance.
        thread_id: Server-side thread identifier.
    """
    try:
        client = graph._validate_client()
    except Exception:
        logger.warning(
            "Could not obtain SDK client for thread %s; retry will likely "
            "still see the conflict",
            thread_id,
            exc_info=True,
        )
        return

    run_ids: list[str] = []
    listed_any = False
    for status in ("running", "pending"):
        try:
            runs = await client.runs.list(thread_id, status=status, limit=10)
        except Exception:
            logger.debug(
                "Failed to list %s runs for thread %s",
                status,
                thread_id,
                exc_info=True,
            )
            continue
        listed_any = True
        for run in runs:
            run_id = run.get("run_id") if isinstance(run, dict) else None
            if run_id:
                run_ids.append(run_id)

    if not listed_any:
        logger.warning(
            "Could not list active runs for thread %s; retry will likely "
            "still see the conflict",
            thread_id,
        )
        return

    if not run_ids:
        return

    async def _cancel_one(run_id: str) -> None:
        try:
            await asyncio.wait_for(
                client.runs.cancel(thread_id, run_id, wait=True, action="interrupt"),
                timeout=_RUN_CANCEL_WAIT_SECONDS,
            )
        except TimeoutError:
            logger.warning(
                "Timed out after %.1fs waiting for run %s on thread %s to "
                "cancel; retry may still see the conflict",
                _RUN_CANCEL_WAIT_SECONDS,
                run_id,
                thread_id,
            )
        except Exception:
            logger.debug(
                "Failed to cancel run %s on thread %s",
                run_id,
                thread_id,
                exc_info=True,
            )

    await asyncio.gather(*(_cancel_one(rid) for rid in run_ids))


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _prepare_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Shallow-copy config so callers' dicts are not mutated.

    Args:
        config: Raw config dict.

    Returns:
        A shallow copy of the config.
    """
    config = dict(config or {})
    configurable = dict(config.get("configurable", {}))
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
        return [raw] if raw is not None else []
    results = []
    for item in raw:
        if isinstance(item, Interrupt):
            results.append(item)
        elif isinstance(item, dict) and "value" in item:
            results.append(Interrupt(value=item["value"], id=item.get("id", "")))
        else:
            results.append(item)
    return results


# ---------------------------------------------------------------------------
# Message conversion — per-type converters with a dispatch table
# ---------------------------------------------------------------------------
#
# Each converter handles one LangChain message type.  The dispatch table
# maps type strings (both short and class-name forms) to the appropriate
# converter.  This keeps each converter focused and makes adding new
# message types a one-line addition to the table.
# ---------------------------------------------------------------------------


def _convert_ai_message(data: dict[str, Any]) -> Any:  # noqa: ANN401
    """Convert a server AI message dict to an `AIMessageChunk`.

    Handles the three tool-call representations the server may emit:

    - `tool_call_chunks`: streaming partial args (string `args`).
    - `tool_calls` with string `args`: legacy streaming format,
        normalized to `tool_call_chunks`.
    - `tool_calls` with dict `args`: fully parsed calls.

    Args:
        data: Raw message dict from the server.

    Returns:
        An `AIMessageChunk`, or `None` on construction failure.
    """
    from langchain_core.messages import AIMessageChunk

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
    except (TypeError, ValueError, KeyError):
        logger.warning(
            "Failed to construct AIMessageChunk from server data (id=%s)",
            data.get("id"),
            exc_info=True,
        )
        return None

    if usage_metadata:
        chunk.usage_metadata = usage_metadata
    return chunk


def _convert_human_message(data: dict[str, Any]) -> Any:  # noqa: ANN401
    """Convert a server human message dict to a `HumanMessage`.

    Args:
        data: Raw message dict from the server.

    Returns:
        A `HumanMessage`, or `None` on construction failure.
    """
    from langchain_core.messages import HumanMessage

    try:
        return HumanMessage(
            content=data.get("content", ""),
            id=data.get("id"),
        )
    except (TypeError, ValueError, KeyError):
        logger.warning(
            "Failed to construct HumanMessage from server data (id=%s)",
            data.get("id"),
            exc_info=True,
        )
        return None


def _convert_tool_message(data: dict[str, Any]) -> Any:  # noqa: ANN401
    """Convert a server tool message dict to a `ToolMessage`.

    Args:
        data: Raw message dict from the server.

    Returns:
        A `ToolMessage`, or `None` on construction failure.
    """
    from langchain_core.messages import ToolMessage

    try:
        return ToolMessage(
            content=data.get("content", ""),
            tool_call_id=data.get("tool_call_id", ""),
            name=data.get("name", ""),
            id=data.get("id"),
            status=data.get("status", "success"),
        )
    except (TypeError, ValueError, KeyError):
        logger.warning(
            "Failed to construct ToolMessage from server data (id=%s)",
            data.get("id"),
            exc_info=True,
        )
        return None


_MESSAGE_CONVERTERS: dict[str, Callable[[dict[str, Any]], Any]] = {
    "ai": _convert_ai_message,
    "AIMessage": _convert_ai_message,
    "AIMessageChunk": _convert_ai_message,
    "human": _convert_human_message,
    "HumanMessage": _convert_human_message,
    "tool": _convert_tool_message,
    "ToolMessage": _convert_tool_message,
}
"""Maps server message `type` strings to their converter functions.

Both short forms (`'ai'`, `'human'`, `'tool'`) and class-name forms
(`'AIMessage'`, `'HumanMessage'`, `'ToolMessage'`) are supported so
the converter works regardless of how the server serializes the type field.
"""


def _convert_message_data(data: dict[str, Any]) -> Any:  # noqa: ANN401
    """Convert a server message dict into a LangChain message object.

    Dispatches to a per-type converter via `_MESSAGE_CONVERTERS`. New message
    types can be supported by adding a converter function and a table entry —
    no changes to this dispatcher are needed.

    Args:
        data: Message dict from the server.

    Returns:
        A LangChain message object, or `None` if conversion fails.
    """
    msg_type = data.get("type", "")
    converter = _MESSAGE_CONVERTERS.get(msg_type)
    if converter is not None:
        return converter(data)
    logger.warning("Unknown message type in stream: %s", msg_type)
    return None
