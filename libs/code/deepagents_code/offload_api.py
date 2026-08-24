"""Dcode-owned HTTP boundary for server-side thread offload."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast
from weakref import WeakValueDictionary

from langchain_core.messages import convert_to_messages
from langchain_core.runnables.config import var_child_runnable_config
from langgraph.runtime import ExecutionInfo, Runtime
from langgraph_sdk import get_client
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from deepagents_code._cli_context import CLIContextSchema
from deepagents_code.cost_tracking import prepare_operation_cost
from deepagents_code.hooks.interrupt import build_hook_interrupt_payload
from deepagents_code.hooks.server_middleware import (
    HookTransportInterruptError,
    operation_hook_responses,
)
from deepagents_code.offload_middleware import unchanged_offload_result
from deepagents_code.server_graph import get_server_runtime

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig
    from starlette.requests import Request

    from deepagents_code.offload_middleware import (
        OffloadResponse,
        _OffloadState,
    )

logger = logging.getLogger(__name__)

_OFFLOAD_API_VERSION = 1
"""Protocol version this route speaks.

Must equal `client.remote_client._OFFLOAD_PROTOCOL_VERSION`; the duplication is
deliberate so neither side imports the other, and
`test_remote_client.test_client_and_server_protocol_versions_agree` pins the two
together. Bump both, or that test fails.
"""
_thread_locks: WeakValueDictionary[str, asyncio.Lock] = WeakValueDictionary()

# One client for the process. `get_client` builds a fresh `httpx.AsyncClient`
# (with its own connection pool) per call and exposes no close hook we own, so
# calling it per request -- and this route runs once per hook resume round --
# would leak a pool for the lifetime of the server.
_client: Any = None


def _thread_client() -> Any:  # noqa: ANN401  # untyped LangGraph SDK client
    """Return the process-wide in-process LangGraph SDK client."""
    global _client  # noqa: PLW0603  # module-level singleton by design
    if _client is None:
        _client = get_client(url=None, api_key=None)
    return _client


def _thread_lock(thread_id: str) -> asyncio.Lock:
    """Return one live lock per thread without retaining inactive threads."""
    lock = _thread_locks.get(thread_id)
    if lock is None:
        lock = asyncio.Lock()
        _thread_locks[thread_id] = lock
    return lock


class _OffloadConflictError(RuntimeError):
    """The thread changed or became active during an offload attempt."""


class _OffloadUnavailableError(RuntimeError):
    """The server runtime could not be built, so no operation can run.

    `get_server_runtime` converts a construction failure into a startup-error
    marker and `sys.exit(1)`. That barrier was written for the `langgraph.json`
    graph factory, where exiting is right; reached from a request handler it
    would kill the server process mid-request, and `SystemExit` is a
    `BaseException`, so the route's own handler could not turn it into a
    response. Server-owned offload cannot run without that runtime, so report
    the condition instead.
    """


class _OffloadIndeterminateError(RuntimeError):
    """The state write may or may not have landed; the outcome is unknown.

    Raised only when the checkpoint write itself failed *and* a follow-up read
    shows the thread advanced anyway, so the operation cannot honestly claim
    either that it committed or that it did not.
    """


# Context fields the offload operation reads (everything else in
# `CLIContextSchema` — approval mode and turn ids, for example — drives
# interactive-run machinery this operation never touches). Validated at
# the HTTP boundary so a malformed client request fails with a 422 naming the
# field instead of a 500 deep in model resolution or hook dispatch.
_CONTEXT_STR_OR_NONE_FIELDS = (
    "model",
    "classifier_model",
    "approval_mode",
    "thread_id",
    "hooks_snapshot_id",
    "prompt_id",
)
_CONTEXT_DICT_FIELDS = ("model_params", "profile_overrides")


def _validate_context(context: dict[str, Any]) -> None:
    """Check the context fields the offload operation consumes.

    Only the listed keys are type-checked; unknown keys pass through so a
    newer client can keep talking to this server version.

    Args:
        context: The request's `context` object.

    Raises:
        TypeError: If a consumed field has the wrong type, naming the field.
    """
    for key in _CONTEXT_STR_OR_NONE_FIELDS:
        value = context.get(key)
        if value is not None and not isinstance(value, str):
            msg = f"context.{key} must be a string or null, got {type(value).__name__}."
            raise TypeError(msg)
    for key in _CONTEXT_DICT_FIELDS:
        value = context.get(key)
        if value is not None and not isinstance(value, dict):
            msg = f"context.{key} must be an object, got {type(value).__name__}."
            raise TypeError(msg)
    limit = context.get("model_context_limit")
    # bool is an int subclass, so exclude it explicitly: JSON `true` is not a
    # token limit.
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int)):
        msg = (
            "context.model_context_limit must be an integer or null, "
            f"got {type(limit).__name__}."
        )
        raise TypeError(msg)
    auto_approve = context.get("auto_approve")
    if auto_approve is not None and not isinstance(auto_approve, bool):
        msg = (
            f"context.auto_approve must be a boolean or null, "
            f"got {type(auto_approve).__name__}."
        )
        raise TypeError(msg)
    events = context.get("hooks_server_events")
    if events is not None and (
        not isinstance(events, list)
        or any(not isinstance(event, str) for event in events)
    ):
        msg = "context.hooks_server_events must be a list of strings or null."
        raise TypeError(msg)


def _checkpoint_id(state: Mapping[str, object]) -> str:
    checkpoint = state.get("checkpoint")
    value = checkpoint.get("checkpoint_id") if isinstance(checkpoint, Mapping) else None
    if not isinstance(value, str) or not value:
        msg = "The thread has no checkpoint to offload."
        raise _OffloadConflictError(msg)
    return value


def _operation_payload(
    payload: object,
) -> tuple[str, dict[str, Any], dict[str, object]]:
    """Validate the narrow client-to-operation request shape.

    Args:
        payload: Decoded request JSON.

    Returns:
        Operation id, runtime context, and accumulated hook responses.

    Raises:
        TypeError: If the payload or a structured field has the wrong shape.
    """
    if not isinstance(payload, dict):
        msg = "Offload request must be a JSON object."
        raise TypeError(msg)
    operation_id = payload.get("operation_id")
    context = payload.get("context")
    responses = payload.get("hook_responses", {})
    if not isinstance(operation_id, str) or not operation_id:
        msg = "operation_id must be a non-empty string."
        raise TypeError(msg)
    if not isinstance(context, dict):
        msg = "context must be a JSON object."
        raise TypeError(msg)
    if not isinstance(responses, dict):
        msg = "hook_responses must be a JSON object."
        raise TypeError(msg)
    validated_context = {str(key): value for key, value in context.items()}
    _validate_context(validated_context)
    return (
        operation_id,
        validated_context,
        {str(key): value for key, value in responses.items()},
    )


def _hydrate_state(values: object) -> _OffloadState:
    """Hydrate serialized checkpoint messages for the compaction service.

    Args:
        values: State values returned by LangGraph Server.

    Returns:
        A shallow state copy containing LangChain message objects.

    Raises:
        TypeError: If the server returns an unexpected state shape.
    """
    if not isinstance(values, dict):
        msg = "LangGraph returned non-object thread state."
        raise TypeError(msg)
    state = dict(values)
    messages = state.get("messages", [])
    if not isinstance(messages, list):
        msg = "LangGraph returned a non-list messages channel."
        raise TypeError(msg)
    state["messages"] = convert_to_messages(messages)

    # LangGraph serializes the summary stored inside the private event channel
    # independently of the top-level `messages` channel. The summarization SDK
    # prepends it to the effective conversation, so it must be a message object
    # too rather than the serialized dict returned by the thread API.
    event = state.get("_summarization_event")
    if isinstance(event, Mapping) and "summary_message" in event:
        hydrated_event = dict(event)
        summary_message = hydrated_event["summary_message"]
        hydrated_event["summary_message"] = convert_to_messages([summary_message])[0]
        state["_summarization_event"] = hydrated_event
    return cast("_OffloadState", state)


async def _require_idle_thread(client: Any, thread_id: str) -> None:  # noqa: ANN401
    """Reject offload while LangGraph reports an active thread.

    Args:
        client: In-process LangGraph SDK client.
        thread_id: Thread being compacted.

    Raises:
        _OffloadConflictError: If the thread is not idle, or is not registered
            on the server at all.
    """
    from langgraph_sdk.errors import NotFoundError

    try:
        thread = await client.threads.get(thread_id)
    except NotFoundError as exc:
        # Checkpoint persistence and HTTP thread registration are separate on
        # the dev server, so a thread can hold on-disk state while its live row
        # is absent (see `RemoteAgent.aensure_thread`). The client registers
        # before requesting the operation; reaching here means it could not, so
        # name the condition instead of letting a 404 become an opaque 500.
        msg = (
            "This thread is not registered on the server; send a message "
            "before offloading."
        )
        raise _OffloadConflictError(msg) from exc
    if thread.get("status") != "idle":
        msg = "Cannot offload while the thread has an active or interrupted run."
        raise _OffloadConflictError(msg)


async def _write_landed(
    client: Any,  # noqa: ANN401  # untyped LangGraph SDK client
    thread_id: str,
    checkpoint_id: str,
) -> bool:
    """Report whether the thread advanced past the checkpoint we read.

    Used only to disambiguate a failed `update_state`: a new checkpoint means
    the write most likely applied despite the error. A concurrent run could
    also have advanced the thread, so this is a bias, not a proof -- it biases
    toward keeping cost records claimed (understating spend at worst) over
    restoring them (which would double-charge).

    Args:
        client: In-process LangGraph SDK client.
        thread_id: Thread that was being compacted.
        checkpoint_id: Checkpoint the operation read and validated against.

    Returns:
        `True` if the thread's checkpoint changed or could not be read.
    """
    try:
        current = await client.threads.get_state(thread_id)
        return _checkpoint_id(current) != checkpoint_id
    except BaseException:
        # `BaseException`, not `Exception`: this runs inside the caller's
        # settlement handler, so an escape here -- a `CancelledError` from a
        # disconnect or a shutdown re-delivering cancellation while that handler
        # unwinds -- would skip the rollback entirely and delete the drained
        # cost records from the thread's lifetime total with no trace.
        #
        # An unreadable thread cannot rule the write out, so stay on the
        # conservative side and treat the outcome as indeterminate.
        logger.exception(
            "Could not read thread %s back to classify a failed offload write",
            thread_id,
        )
        return True


async def _execute_offload(
    thread_id: str,
    *,
    operation_id: str,
    context: dict[str, Any],
    hook_responses: dict[str, object],
) -> OffloadResponse:
    """Execute and commit one server-owned offload attempt.

    Args:
        thread_id: LangGraph thread to compact.
        operation_id: Opaque client-generated attempt identity.
        context: Runtime model and hooks context.
        hook_responses: Accumulated hook replies keyed by invocation id.

    Returns:
        A complete result or a hook request that must be answered.

    Raises:
        TypeError: If `thread_id` is empty.
        _OffloadConflictError: If the thread is active or changes before commit.
        _OffloadUnavailableError: If the server runtime cannot be built, so no
            operation can run.
        _OffloadIndeterminateError: If the state write failed but the thread
            advanced anyway, so the outcome cannot be determined.
        RuntimeError: If the operation attempts to write conversation messages.
    """
    if not thread_id:
        msg = "thread_id path parameter must be non-empty."
        raise TypeError(msg)
    client = _thread_client()
    async with _thread_lock(thread_id):
        await _require_idle_thread(client, thread_id)
        before = await client.threads.get_state(thread_id)
        if before.get("next") or before.get("tasks") or before.get("interrupts"):
            msg = "Cannot offload a thread with pending graph work."
            raise _OffloadConflictError(msg)

        state = _hydrate_state(before.get("values"))
        if not state.get("messages"):
            # An empty thread is "nothing to offload", not a failure. Answer it
            # here: `_checkpoint_id` below rejects a thread with no checkpoint,
            # so without this the graceful `empty` branch in
            # `OffloadOperation.execute` is unreachable over HTTP and the user
            # is told the operation failed.
            return {
                "status": "complete",
                "result": unchanged_offload_result("empty", messages=0, tokens=0),
            }

        checkpoint_id = _checkpoint_id(before)
        context["thread_id"] = thread_id
        namespace = f"dcode_offload:{operation_id}"
        info = ExecutionInfo(
            checkpoint_id=checkpoint_id,
            checkpoint_ns=namespace,
            task_id=operation_id,
            thread_id=thread_id,
            run_id=operation_id,
        )
        try:
            server = await get_server_runtime()
        except SystemExit as exc:
            msg = (
                "The server could not build its agent runtime, so /offload is "
                "unavailable. Check the server log for the startup failure."
            )
            raise _OffloadUnavailableError(msg) from exc
        runtime = Runtime[CLIContextSchema](
            context=cast("CLIContextSchema", context),
            store=getattr(server.agent, "store", None),
            execution_info=info,
        )
        config = cast(
            "RunnableConfig",
            {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_id": checkpoint_id,
                    "checkpoint_ns": namespace,
                    "run_id": operation_id,
                }
            },
        )
        token = var_child_runnable_config.set(config)
        try:
            with operation_hook_responses(hook_responses):
                execution = await server.offload.execute(state, runtime)
        except HookTransportInterruptError as interrupt:
            return {
                "status": "interrupt",
                "request": build_hook_interrupt_payload(interrupt.request),
            }
        finally:
            var_child_runnable_config.reset(token)

        await _require_idle_thread(client, thread_id)
        current = await client.threads.get_state(thread_id)
        if _checkpoint_id(current) != checkpoint_id:
            # Compaction already ran, so the summarizer model call has been made
            # and paid for. Name the discarded work: the records stay in the
            # recorder and would otherwise be swept into an unrelated later turn
            # with no trace of where they came from.
            logger.warning(
                "Discarding a completed offload for thread %s: the thread "
                "advanced past checkpoint %s while compaction was running, so "
                "the summary (and its model spend) cannot be committed",
                thread_id,
                checkpoint_id,
            )
            msg = (
                "The thread changed while offload was running; no state was committed."
            )
            raise _OffloadConflictError(msg)

        prepared = prepare_operation_cost(state, thread_id)
        update: dict[str, Any] = {**execution.update, **prepared.update}
        if "messages" in update:
            # A security boundary, not a defensive assertion: this route commits
            # to the latest checkpoint rather than the one it read, so a
            # `messages` write here would be unattributed to any run and could
            # clobber messages a concurrent run appended in that window. See
            # THREAT_MODEL.md (TB10/DF27) before relaxing this.
            msg = "Server offload operations may not write the messages channel."
            prepared.rollback()
            raise RuntimeError(msg)
        if not update:
            # Nothing to persist, but `prepare_operation_cost` already drained
            # the recorder. Returning without rolling back would delete that
            # spend from the thread's lifetime total (the drain is destructive).
            prepared.rollback()
            return {"status": "complete", "result": execution.result}
        try:
            # Apply the state-only delta to the latest checkpoint. The
            # thread API rejects in-flight runs itself; pinning this write
            # to `before` would instead branch from a stale checkpoint if
            # a run completed in the narrow window after our final check,
            # potentially hiding its newly appended messages.
            #
            # `as_node` is left unpinned, unlike every other write in dcode.
            # LangGraph then infers the node from `checkpoint["versions_seen"]`,
            # which raises `InvalidUpdateError("Ambiguous update, specify
            # as_node")` if the last superstep left two nodes at the same
            # version -- and it would raise *after* the summarizer has been
            # billed. Today's middleware layout resolves unambiguously (the real
            # server integration test exercises it), so this is an unguarded
            # dependency on that layout rather than a known bug.
            await client.threads.update_state(
                thread_id,
                update,
            )
        except BaseException as exc:
            # The write may have applied server-side before the failure reached
            # us (response decode, timeout, cancellation). Restoring the cost
            # records in that case would let the next drain price this same
            # spend a second time, so only roll back once the thread is known
            # not to have advanced.
            if await _write_landed(client, thread_id, checkpoint_id):
                logger.exception(
                    "Offload state write for thread %s failed, but the thread "
                    "advanced past checkpoint %s; keeping %d cost record(s) "
                    "claimed to avoid double-charging",
                    thread_id,
                    checkpoint_id,
                    len(prepared.records),
                )
                if isinstance(exc, asyncio.CancelledError):
                    # Cost records stay claimed either way, but suppressing a
                    # cancellation would leave the task never observing it.
                    raise
                msg = (
                    "Offload compacted the conversation but could not confirm "
                    "the state write. Run /context to check whether the "
                    "conversation was compacted before offloading again."
                )
                raise _OffloadIndeterminateError(msg) from None
            logger.warning(
                "Offload state write for thread %s failed with no thread "
                "advance; restoring %d cost record(s)",
                thread_id,
                len(prepared.records),
            )
            prepared.rollback()
            raise
        return {"status": "complete", "result": execution.result}


def capability(_request: Request) -> JSONResponse:
    """Report the dcode server-operation protocol version.

    Answers `{"offload": true, "version": <int>}` for protocol diagnostics and
    client/server version pinning.

    Deliberately cheap: it does not resolve the server runtime, so a probe never
    pays for building the agent. An unbuildable runtime surfaces as a 503 from
    the operation itself.

    Returns:
        JSON capability response.
    """
    return JSONResponse({"offload": True, "version": _OFFLOAD_API_VERSION})


async def offload(request: Request) -> JSONResponse:
    """Handle one thread offload or hook-resume round.

    Request body: `operation_id` (non-empty string, stable across the rounds of
    one attempt), `context` (runtime model and Hooks v2 context), and
    `hook_responses` (replies accumulated so far, keyed by invocation id).
    The thread comes from the path, never the body.

    A round either completes or returns a hook request to answer. There is no
    suspended coroutine server-side: a resume round **re-executes the operation
    from the top**, and `_invoke_hook` replays already-answered invocations from
    `hook_responses` instead of raising again. That is what makes the loop
    terminate, and it is why the dispatched call's id must be stable across
    rounds (`_forced_offload_call_id`).

    Status codes, and what each means for whether state committed:

    - 200 -- completed, or a resumable hook request; no state written in the
      latter case.
    - 422 -- malformed request, named by field. Nothing ran.
    - 409 -- thread conflict: active, interrupted, holding pending graph work,
      unregistered, or advanced past the checkpoint read. Nothing committed.
    - 503 -- the server runtime could not be built. Nothing ran.
    - 500 -- either an indeterminate write (compaction happened and the commit
      cannot be confirmed; the detail says so and is user-actionable) or an
      unexpected server fault.

    Invariants this boundary owns: it reads and hydrates checkpoint state itself,
    it commits only the channels `OffloadStateUpdate` permits and refuses any
    `messages` write outright, and it settles the drained cost records on every
    exit path. See THREAT_MODEL.md C18 (TB10/DF27).

    Returns:
        JSON operation response.
    """
    # Request validation is scoped to its own block so that a `TypeError` or
    # `ValueError` raised *inside* the operation (a server-side fault) is not
    # misreported to the client as a 4xx and, worse, swallowed without a log.
    try:
        thread_id = request.path_params["thread_id"]
        operation_id, context, hook_responses = _operation_payload(await request.json())
    except (TypeError, ValueError) as exc:
        return JSONResponse({"detail": str(exc)}, status_code=422)

    try:
        response = await _execute_offload(
            thread_id,
            operation_id=operation_id,
            context=context,
            hook_responses=hook_responses,
        )
    except _OffloadConflictError as exc:
        return JSONResponse({"detail": str(exc)}, status_code=409)
    except _OffloadUnavailableError as exc:
        # 503: the server cannot serve this operation at all, as distinct from a
        # thread-state conflict (409) or an ambiguous write (500).
        logger.exception("Offload unavailable: the server runtime failed to build")
        return JSONResponse({"detail": str(exc)}, status_code=503)
    except _OffloadIndeterminateError as exc:
        # 500: the operation did not complete cleanly. The detail is honest
        # about the uncertainty rather than asserting nothing was written.
        return JSONResponse({"detail": str(exc)}, status_code=500)
    except Exception:
        logger.exception("Server-owned /offload failed")
        return JSONResponse(
            {"detail": "Offload failed on the server; see the server log for details."},
            status_code=500,
        )
    return JSONResponse(response)


app = Starlette(
    routes=[
        Route("/dcode/offload", capability, methods=["GET"]),
        Route(
            "/dcode/threads/{thread_id:str}/offload",
            offload,
            methods=["POST"],
        ),
    ]
)
