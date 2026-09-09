"""Remote agent client — thin wrapper around LangGraph's `RemoteGraph`.

Delegates streaming, state management, and SSE handling to
`langgraph.pregel.remote.RemoteGraph`. This wrapper converts streamed message
dicts into LangChain message objects for the app's Textual adapter, but leaves
state snapshots in the server's serialized form.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable

    from deepagents_code.mcp_tools import MCPServerInfo
    from deepagents_code.offload_middleware import OffloadResult

logger = logging.getLogger(__name__)

_RUN_CANCEL_WAIT_SECONDS = 10.0
"""Per-run cancel wait. Picked so a stuck server-side run can't hang the UI on
Esc for more than ~10s, while leaving room for an actually-cancelling run to
finish its in-flight tool call.

Concurrent cancels keep aggregate wall time bounded by this value regardless of
how many runs are active.
"""

_OFFLOAD_CANCEL_WAIT_SECONDS = 10.0
"""Bound the Esc path while the server confirms offload termination."""

_OFFLOAD_MAX_RESUME_ROUNDS = 32
"""Bound hook transport rounds for a single server operation.

Counts *hook fulfillments*, not POSTs: the loop runs one extra iteration so the
round that follows the last fulfillment can observe completion. Exceeding this
many distinct invocations means either genuinely many hooks or an unstable
invocation-id derivation server-side.
"""

_OFFLOAD_RESULT_INT_FIELDS = (
    "messages_offloaded",
    "messages_kept",
    "tokens_before",
    "tokens_after",
)


async def _join_task_deferring_cancellation[T](task: asyncio.Task[T]) -> None:
    """Join a task despite repeated cancellation of the waiting caller."""
    while not task.done():
        try:
            await asyncio.wait((task,))
        except asyncio.CancelledError:
            continue


async def _cancel_server_offload(
    graph: Any,  # noqa: ANN401  # untyped RemoteGraph client
    thread_id: str,
    operation_id: str,
) -> str:
    """Request cancellation and wait for the server's terminal acknowledgement.

    Returns:
        Server terminal status (`cancelled` or `finished`).

    Raises:
        RuntimeError: If the server returns an invalid acknowledgement.
    """
    response = await asyncio.wait_for(
        graph.client.http.post(
            f"/dcode/threads/{thread_id}/offload/{operation_id}/cancel",
            json={},
        ),
        timeout=_OFFLOAD_CANCEL_WAIT_SECONDS,
    )
    status = response.get("status") if isinstance(response, dict) else None
    if status not in {"cancelled", "finished"}:
        msg = "Offload server returned an invalid cancellation acknowledgement."
        raise RuntimeError(msg)
    return status


async def _await_offload_step[T](
    awaitable: Awaitable[T],
    *,
    graph: Any,  # noqa: ANN401  # untyped RemoteGraph client
    thread_id: str,
    operation_id: str,
) -> T:
    """Await one offload step and confirm server termination if cancelled.

    Returns:
        The awaited step's result.

    Raises:
        asyncio.CancelledError: After the server confirms the operation ended.
    """
    try:
        return await awaitable
    except asyncio.CancelledError:
        cancellation = asyncio.create_task(
            _cancel_server_offload(graph, thread_id, operation_id)
        )
        await _join_task_deferring_cancellation(cancellation)
        cancellation.result()
        raise


def _validated_offload_result(result: object) -> OffloadResult:
    """Check a server offload result before any caller indexes it.

    The renderer subscripts these fields by key and unguarded. Validating here
    means a protocol skew fails with a message naming the problem, instead of a
    `KeyError` reported as a generic reporting failure for an offload the server
    already committed.

    Args:
        result: The `result` object from a `complete` operation response.

    Returns:
        The same mapping, once its required fields are known to be present.

    Raises:
        RuntimeError: If a required field is missing or has the wrong type.
    """
    if not isinstance(result, dict):
        msg = "Offload server completed without a typed result."
        raise RuntimeError(msg)  # noqa: TRY004  # protocol fault, not a type misuse
    status = result.get("status")
    if not isinstance(status, str) or not status:
        msg = "Offload result has no status."
        raise RuntimeError(msg)
    # Statistics are only meaningful (and only read) for a committed compaction.
    if status == "compacted":
        for field in _OFFLOAD_RESULT_INT_FIELDS:
            value = result.get(field)
            if not isinstance(value, int) or isinstance(value, bool):
                msg = (
                    f"Offload result field {field!r} must be an integer, got "
                    f"{type(value).__name__}."
                )
                raise RuntimeError(msg)  # noqa: TRY004  # protocol fault
    return cast("OffloadResult", result)


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


def state_has_pending_work(state: object) -> bool:
    """Return whether a checkpoint snapshot still holds unfinished graph work.

    Single definition of "pending" shared by the app-side detector and the
    post-recovery verification, so the two cannot drift apart.

    Args:
        state: A `StateSnapshot`-shaped object, or `None`.

    Returns:
        Whether the snapshot has a queued node, task, or interrupt.
    """
    if state is None:
        return False
    return bool(
        getattr(state, "next", None)
        or getattr(state, "tasks", None)
        or getattr(state, "interrupts", None)
    )


def _cancelled_tool_messages(values: object) -> list[Any]:
    """Build terminal results for tool calls left unanswered by a lost run.

    Only the trailing turn is considered. The queued `tools` step is triggered
    by the final `AIMessage`, and a provider requires every `tool_result` to
    sit immediately after the `tool_use` it answers. Earlier unanswered calls
    are left alone on purpose: interrupt recovery writes a partial `AIMessage`
    with its in-flight `tool_calls` and then closes the turn with a following
    message (`_build_interrupted_ai_message` in the TUI adapter), so those
    calls stay dangling by design. Answering them here would append their
    results at the tail, far from their `tool_use`, and the next model call --
    the compaction summarizer that runs right after recovery -- would be
    rejected.

    Blocking and CPU-bound over the checkpoint history; async callers must
    offload it with `asyncio.to_thread`.

    Returns:
        Error results for the trailing turn's unanswered tool calls.
    """
    if not isinstance(values, dict):
        return []
    from langchain_core.messages import AIMessage, ToolMessage, convert_to_messages

    raw_messages = values.get("messages")
    if not isinstance(raw_messages, list):
        return []
    messages = convert_to_messages(cast("list[Any]", raw_messages))
    # Walk back over the results the lost run did manage to write, then stop at
    # the message that owns them. Anything but an `AIMessage` there means the
    # turn was already closed out and nothing is dangling.
    answered: set[str] = set()
    for message in reversed(messages):
        if isinstance(message, ToolMessage):
            answered.add(message.tool_call_id)
            continue
        if not isinstance(message, AIMessage):
            return []
        return [
            ToolMessage(
                content="Tool call cancelled because the previous session ended.",
                name=call["name"],
                tool_call_id=call["id"],
                status="error",
            )
            for call in message.tool_calls
            if call["id"] not in answered
        ]
    return []


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
    """Render an exception from any `RemoteAgent` call for the UI.

    The LangGraph server serializes non-allowlisted exceptions as
    `{"error": <ExceptionType>, "message": <text or "An internal error occurred">}`
    (see `langgraph_api.serde`). `RemoteGraph` wraps that dict in
    `RemoteException(payload)`, so the default `str(exc)` renders as an ugly
    Python dict repr in the UI.

    Args:
        exc: The exception caught from an agent call -- the SSE stream, or an
            HTTP operation such as the offload route.

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
        self._workspaces: dict[str, dict[str, Any]] = {}
        self._workspace_config: dict[str, Any] | None = None
        self._workspace_config_fingerprint: str | None = None
        self._workspace_cwd: str | None = None

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

    async def aoffload(
        self,
        *,
        config: Mapping[str, Any],
        context: Mapping[str, Any],
        fulfill_hook: Callable[[object], Awaitable[dict[str, object]]],
    ) -> OffloadResult:
        """Request server-owned offload and fulfill its hook callbacks.

        Args:
            config: Runnable config identifying the thread.
            context: Runtime model and Hooks v2 context.
            fulfill_hook: Client hook executor for server requests.

        Returns:
            Typed offload result from the server operation.

        Raises:
            TypeError: If the server response is not a JSON object.
            RuntimeError: If the server does not provide the offload route,
                returns an invalid protocol response, or exceeds the hook round
                limit.
            APIStatusError: From the server operation. A 409 (conflict) or 422
                (malformed request) means no state was committed; a 500 raised
                as indeterminate means a commit may have landed and carries
                user-actionable text the caller must surface rather than
                swallow.
        """  # noqa: DOC502 -- APIStatusError is raised by the SDK transport
        from uuid import uuid4

        from langgraph_sdk.errors import NotFoundError

        from deepagents_code.hooks.interrupt import is_hook_interrupt_payload

        thread_id = _require_thread_id(config)
        workspace = await self._workspace_for_thread(config)
        operation_context = dict(context)
        operation_context["workspace"] = workspace
        # The operation reads and writes thread state over HTTP, so the thread's
        # live row must exist first. Checkpoint persistence and registration are
        # separate on the dev server (see `aensure_thread`), so a resumed thread
        # -- or one whose server restarted mid-session -- has state on disk and
        # no row, and every request below would 404.
        await self.aensure_thread({"configurable": {"thread_id": thread_id}})
        operation_id = str(uuid4())
        hook_responses: dict[str, object] = {}
        graph = self._get_graph()
        for round_index in range(_OFFLOAD_MAX_RESUME_ROUNDS + 1):
            try:
                response = await _await_offload_step(
                    graph.client.http.post(
                        f"/dcode/threads/{thread_id}/offload",
                        json={
                            "operation_id": operation_id,
                            "context": operation_context,
                            "hook_responses": hook_responses,
                        },
                    ),
                    graph=graph,
                    thread_id=thread_id,
                    operation_id=operation_id,
                )
            except NotFoundError as exc:
                # The route itself is missing. An unregistered thread cannot
                # reach here as a 404: the server catches that and answers 409
                # with its own message. A custom `graph_ref` server, or one
                # older than this operation, never registers the dcode HTTP app,
                # and the SDK's bare "404 Not Found" names neither the cause nor
                # a fix.
                msg = (
                    "This server does not provide dcode's /offload operation. "
                    "Use the built-in dcode server, or upgrade the server to a "
                    "version that registers it."
                )
                raise RuntimeError(msg) from exc
            if not isinstance(response, dict):
                msg = "Offload server returned a non-object response."
                raise TypeError(msg)
            status = response.get("status")
            if status == "complete":
                return _validated_offload_result(response.get("result"))
            request = response.get("request")
            if status != "interrupt" or not is_hook_interrupt_payload(request):
                msg = "Offload server returned an invalid operation response."
                raise RuntimeError(msg)
            invocation = request.get("request")
            invocation_id = (
                invocation.get("invocation_id")
                if isinstance(invocation, dict)
                else None
            )
            if not isinstance(invocation_id, str) or not invocation_id:
                msg = "Offload hook request has no invocation id."
                raise RuntimeError(msg)
            logger.debug(
                "Offload round %d fulfilling hook invocation %s",
                round_index,
                invocation_id,
            )
            if round_index == _OFFLOAD_MAX_RESUME_ROUNDS:
                # The extra iteration exists to POST the last fulfillment and
                # read the result, not to answer one more hook. Without this the
                # loop fulfills 33 hooks and then reports "after 32 rounds".
                break
            hook_responses[invocation_id] = await _await_offload_step(
                fulfill_hook(request),
                graph=graph,
                thread_id=thread_id,
                operation_id=operation_id,
            )
        # The server only re-requests an invocation id it has not been given, so
        # exhaustion means this many *distinct* ids. That is either genuinely
        # many hooks or an unstable invocation-id derivation server-side; log the
        # ids so the two are distinguishable, and do not assert a cause in the
        # user-facing message.
        logger.warning(
            "Offload exceeded %d hook rounds; fulfilled invocation ids: %s",
            _OFFLOAD_MAX_RESUME_ROUNDS,
            sorted(hook_responses),
        )
        msg = (
            f"Offload did not complete after {_OFFLOAD_MAX_RESUME_ROUNDS} hook "
            "rounds. Check the server log for the hook invocations it requested."
        )
        raise RuntimeError(msg)

    async def astream(
        self,
        input: dict | Any,  # noqa: A002, ANN401
        *,
        stream_mode: list[str] | None = None,
        subgraphs: bool = False,
        config: Mapping[str, Any] | None = None,
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

        payload = dict(context) if isinstance(context, Mapping) else {}
        payload["workspace"] = await self._workspace_for_thread(config)

        extra_stream_kwargs: dict[str, Any] = {}
        replica_project = get_langsmith_replica_project()
        if replica_project:
            extra_stream_kwargs["langsmith_tracing"] = {"project_name": replica_project}

        async for ns, mode, data in graph.astream(
            input,
            stream_mode=stream_mode or ["messages", "updates"],
            subgraphs=subgraphs,
            config=config,
            context=payload,
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
        values: dict[str, Any] | None,
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
            logger.debug(
                "update_state conflict for thread %s; cancelling active runs "
                "and retrying",
                thread_id,
            )
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

    async def aabandon_pending_work(self, config: Mapping[str, Any]) -> None:
        """Cancel active runs and discard checkpointed work without replaying it.

        The state writes go through `aupdate_state` so they inherit its HTTP
        409 recovery: this method runs on exactly the threads whose run was
        cancelled out from under the client, which is when 409s occur.

        Args:
            config: Config with `configurable.thread_id`.

        Raises:
            RuntimeError: If pending work remains after the state update.
        """
        thread_id = _require_thread_id(config)
        prepared = _prepare_config(config)
        await _cancel_active_runs(self._get_graph(), thread_id)
        state = await self.aget_state(prepared)
        cancelled = await asyncio.to_thread(
            _cancelled_tool_messages, getattr(state, "values", None)
        )
        if cancelled:
            # `create_agent` names the tool step "tools", but only adds the node
            # when the agent has tools. A toolless graph therefore rejects this
            # update with `InvalidUpdateError` -- and could not have produced a
            # dangling tool call in the first place, so the branch is dead
            # there. The caller reports the failure rather than compacting.
            await self.aupdate_state(prepared, {"messages": cancelled}, as_node="tools")
        await self.aupdate_state(prepared, None, as_node="__end__")
        if state_has_pending_work(await self.aget_state(prepared)):
            msg = (
                f"Pending graph work remained on thread {thread_id} after "
                "clearing checkpoint state"
            )
            raise RuntimeError(msg)

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

    async def _workspace_for_thread(
        self,
        config: Mapping[str, Any],
    ) -> dict[str, Any]:
        thread_id = _require_thread_id(config)
        workspace = self._workspaces.get(thread_id)
        if workspace is not None:
            return workspace
        if self._workspace_cwd is None:
            msg = "RemoteAgent workspace is not configured."
            raise RuntimeError(msg)
        return await self.abind_workspace(config, self._workspace_cwd)

    async def _request_workspace(
        self, config: Mapping[str, Any], cwd: str
    ) -> tuple[dict[str, Any], list[MCPServerInfo] | None]:
        thread_id = _require_thread_id(config)
        payload: dict[str, Any] = {"cwd": cwd}
        if self._workspace_config is not None:
            payload["workspace_config"] = self._workspace_config
            payload["config_fingerprint"] = self._workspace_config_fingerprint
        response = await self._get_graph().client.http.post(
            f"/dcode/threads/{thread_id}/workspace",
            json=payload,
        )
        if not isinstance(response, dict) or not isinstance(
            response.get("workspace"), dict
        ):
            msg = "Workspace server returned an invalid binding response."
            raise TypeError(msg)
        raw_mcp = response.get("mcp_server_info")
        if raw_mcp is None:
            mcp_server_info = None
        elif isinstance(raw_mcp, list) and all(
            isinstance(item, dict) for item in raw_mcp
        ):
            from deepagents_code.mcp_tools import MCPServerInfo, MCPToolInfo

            try:
                mcp_server_info = [
                    MCPServerInfo(
                        name=item["name"],
                        transport=item["transport"],
                        tools=tuple(
                            MCPToolInfo(**tool) for tool in item.get("tools", [])
                        ),
                        status=item.get("status", "ok"),
                        error=item.get("error"),
                        pending_reconnect=item.get("pending_reconnect", False),
                        uses_oauth=item.get("uses_oauth", False),
                    )
                    for item in raw_mcp
                ]
            except (KeyError, TypeError, ValueError) as exc:
                msg = "Workspace server returned invalid MCP metadata."
                raise TypeError(msg) from exc
        else:
            msg = "Workspace server returned invalid MCP metadata."
            raise TypeError(msg)
        return cast("dict[str, Any]", response["workspace"]), mcp_server_info

    async def abind_workspace(
        self, config: Mapping[str, Any], cwd: str
    ) -> dict[str, Any]:
        """Create or verify the remote thread's durable workspace binding.

        Returns:
            The server-validated workspace descriptor.
        """
        thread_id = _require_thread_id(config)
        workspace, _ = await self._request_workspace(config, cwd)
        self._workspaces[thread_id] = workspace
        return workspace

    def _snapshot_workspace(self) -> tuple[str | None, dict[str, dict[str, Any]]]:
        return self._workspace_cwd, dict(self._workspaces)

    def _restore_workspace(
        self, snapshot: tuple[str | None, dict[str, dict[str, Any]]]
    ) -> None:
        self._workspace_cwd, workspaces = snapshot
        self._workspaces.clear()
        self._workspaces.update(workspaces)

    async def aswitch_workspace(
        self, config: Mapping[str, Any], cwd: str
    ) -> list[MCPServerInfo] | None:
        """Bind one thread to a workspace and make it the default for new threads.

        Returns:
            MCP metadata for the server runtime selected by the binding.
        """
        thread_id = _require_thread_id(config)
        workspace, mcp_server_info = await self._request_workspace(config, cwd)
        self._workspace_cwd = cwd
        self._workspaces.clear()
        self._workspaces[thread_id] = workspace
        return mcp_server_info

    def set_workspace(
        self,
        cwd: str,
        config: Mapping[str, Any] | None = None,
        *,
        config_fingerprint: str | None = None,
    ) -> None:
        """Configure the explicit workspace used when binding threads.

        Raises:
            ValueError: If only one policy field is provided.
        """
        if (config is None) != (config_fingerprint is None):
            msg = "Workspace policy and fingerprint must be configured together."
            raise ValueError(msg)
        self._workspace_cwd = cwd
        self._workspace_config = dict(config) if config is not None else None
        self._workspace_config_fingerprint = config_fingerprint
        self._workspaces.clear()

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

    `additional_kwargs` is forwarded. The TUI reads markers from it, such as
    `AUTO_DENIED_METADATA_KEY` on a synthetic auto-mode denial. The TUI always
    runs against a server, so a marker dropped here never reaches the client.
    Guarded by `isinstance`: the server payload is JSON, and a non-dict value
    would fail `ToolMessage` validation and discard the whole message.

    Args:
        data: Raw message dict from the server.

    Returns:
        A `ToolMessage`, or `None` on construction failure.
    """
    from langchain_core.messages import ToolMessage

    additional_kwargs = data.get("additional_kwargs")
    try:
        return ToolMessage(
            content=data.get("content", ""),
            tool_call_id=data.get("tool_call_id", ""),
            name=data.get("name", ""),
            id=data.get("id"),
            status=data.get("status", "success"),
            additional_kwargs=(
                additional_kwargs if isinstance(additional_kwargs, dict) else {}
            ),
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
