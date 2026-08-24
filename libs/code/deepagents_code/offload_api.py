"""dcode-owned HTTP boundary for server-side thread offload."""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal, cast
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
from deepagents_code.offload_middleware import (
    OffloadStateUpdate,
    _archive_lock,
    unchanged_offload_result,
)
from deepagents_code.server_graph import get_server_runtime

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig
    from starlette.requests import Request

    from deepagents_code.cost_tracking import PreparedOperationCost
    from deepagents_code.offload_middleware import (
        OffloadExecution,
        OffloadResponse,
        _OffloadState,
    )

logger = logging.getLogger(__name__)

_WRITABLE_STATE_CHANNELS = frozenset(OffloadStateUpdate.__annotations__)
"""Checkpoint channels a server-owned offload may write.

Derived from `OffloadStateUpdate` so the runtime guard and the type cannot
drift: adding a channel to the type is the only way to permit writing it.
"""
_OFFLOADABLE_THREAD_STATUSES = frozenset({"idle", "error"})
"""Thread statuses that hold no in-flight work, so offload may proceed.

`error` is included deliberately. A run that raises anything other than an
interrupt or rollback leaves the thread row on `error` until the *next* run
completes, and `RemoteAgent.aensure_thread` uses `if_exists="do_nothing"`, so
it does not clear it. Excluding `error` would refuse `/offload` for the whole
window after a failed turn -- exactly when a user reaches for it to recover
from a context overflow. Quiescence is checked separately against the
checkpoint's `next`/`tasks`/`interrupts`, which still catches an errored run
that left a pending node.
"""
_thread_locks: WeakValueDictionary[str, asyncio.Lock] = WeakValueDictionary()
type _OperationKey = tuple[str, str]
type _OperationOutcome = Literal["cancelled", "finished"]
_active_operations: dict[_OperationKey, asyncio.Task[object]] = {}
_operation_outcomes: OrderedDict[_OperationKey, _OperationOutcome] = OrderedDict()
_MAX_OPERATION_OUTCOMES = 1024
"""Bound completed/cancelled ids retained to close request/cancel races."""

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


def _remember_operation(key: _OperationKey, outcome: _OperationOutcome) -> None:
    """Retain a bounded terminal outcome for late or reordered cancellation."""
    _operation_outcomes[key] = outcome
    _operation_outcomes.move_to_end(key)
    while len(_operation_outcomes) > _MAX_OPERATION_OUTCOMES:
        _operation_outcomes.popitem(last=False)


def _register_operation(key: _OperationKey) -> str | None:
    """Register the current request task.

    Returns:
        A refusal reason, or `None` when registration succeeds.
    """
    outcome = _operation_outcomes.get(key)
    if outcome == "cancelled":
        return "The offload operation was cancelled."
    if outcome == "finished":
        return "The offload operation already finished."
    if key in _active_operations:
        return "This offload operation already has an active request."
    task = asyncio.current_task()
    if task is None:
        return "The server could not register the offload operation."
    _active_operations[key] = cast("asyncio.Task[object]", task)
    return None


def _finish_operation(key: _OperationKey, outcome: _OperationOutcome | None) -> None:
    """Release an active round and optionally retain its terminal outcome."""
    task = asyncio.current_task()
    if _active_operations.get(key) is task:
        _active_operations.pop(key, None)
    if outcome is not None:
        _remember_operation(key, outcome)


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


# Context fields validated at this boundary (everything else in
# `CLIContextSchema` — turn ids and the approval-mode key, for example — drives
# interactive-run machinery this operation never touches). Validated here so a
# malformed client request fails with a 422 naming the field instead of a 500
# deep in model resolution or hook dispatch. Validated is not the same as read:
# `classifier_model` is checked for shape but feeds only auto mode, which this
# operation never enters.
_CONTEXT_STR_OR_NONE_FIELDS = (
    "model",
    "classifier_model",
    "approval_mode",
    "thread_id",
    "hooks_snapshot_id",
    "prompt_id",
)
_CONTEXT_DICT_FIELDS = ("model_params", "profile_overrides")

_TRANSPORT_MODEL_PARAM_KEYS = frozenset(
    {
        # Endpoint selection.
        "base_url",
        "api_base",
        "openai_api_base",
        "anthropic_api_url",
        "azure_endpoint",
        "azure_openai_api_base",
        "api_endpoint",
        # Proxy routing.
        "openai_proxy",
        "anthropic_proxy",
        "proxy",
        "proxies",
        # Outbound transport injection: these keys hand whole HTTP clients,
        # transports, or header maps to the model constructor.
        "http_client",
        "http_async_client",
        "transport",
        "default_headers",
        "custom_headers",
    }
)
"""`model_params` keys stripped from client-supplied offload context.

`create_model` merges these params verbatim into the model constructor
(`offload_middleware._summarization_for_runtime`), and the summarizer's
outbound provider calls carry the server's credentials. A client that sets an
endpoint/proxy/transport key therefore chooses where those credentials are
sent. The in-process paths trust `model_params` (the user supplied them
through their own flags and config); the HTTP boundary does not -- the dev
server accepts connections from any local process, so the request's model
selection must not extend to its network plumbing.

A backstop, not the primary control. `_checkpoint_model_context` discards the
request's `model` and `model_params` outright and substitutes the checkpointed
values, so a client-supplied endpoint cannot reach `create_model` even without
this filter. It is kept for the case that control cannot cover: a future path
that resolves a model before, or instead of, reading the checkpoint. Treat a
warning from here as a client sending params it should not, not as a breach.

Denylist rather than allowlist: `create_model` serves arbitrary providers, so a
fixed allowlist would silently drop legitimate provider-specific params.
"""


def _strip_transport_model_params(context: dict[str, Any]) -> dict[str, Any]:
    """Return a context copy with endpoint/transport model params removed.

    Args:
        context: The request's already type-checked `context` object.

    Returns:
        The same dict when `model_params` holds no stripped keys, otherwise a
        shallow copy whose `model_params` omits them.
    """
    params = context.get("model_params")
    if not isinstance(params, dict):
        return context
    stripped = {
        key: value
        for key, value in params.items()
        if key not in _TRANSPORT_MODEL_PARAM_KEYS
    }
    if len(stripped) == len(params):
        return context
    # Logged, not silent: dropping these changes where the summarizer's
    # credentialed calls go, so a user whose gateway config is being ignored has
    # something to find. Key names only -- the values are endpoints and headers.
    logger.warning(
        "Dropped transport key(s) %s from offload model_params; a server-owned "
        "operation does not accept a client-chosen endpoint",
        sorted(set(params) - set(stripped)),
    )
    return {**context, "model_params": stripped}


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
        _strip_transport_model_params(validated_context),
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


def _checkpoint_model_context(
    context: dict[str, Any], state: Mapping[str, object]
) -> dict[str, Any]:
    """Replace request model selection with server-checkpointed values.

    The client still supplies hook and profile context, but it cannot choose
    the model's outbound transport for this server-owned operation. Successful
    agent turns checkpoint the resolved model spec and the runtime overrides
    they actually used, so those values preserve trusted launch/model-switch
    settings such as a private `base_url` without accepting an arbitrary
    offload request's endpoint override.

    Args:
        context: Validated request context.
        state: Server-read checkpoint values for the target thread.

    Returns:
        Context using checkpointed model settings, or no model override when
            the thread predates model checkpointing so the startup summarizer
            is reused.
    """
    trusted = dict(context)
    trusted.pop("model", None)
    trusted.pop("model_params", None)
    model = state.get("_model_spec")
    params = state.get("_model_params")
    if isinstance(model, str) and model:
        trusted["model"] = model
        if isinstance(params, dict):
            trusted["model_params"] = dict(params)
    return trusted


async def _require_idle_thread(client: Any, thread_id: str) -> None:  # noqa: ANN401
    """Reject offload while LangGraph reports an active thread.

    Args:
        client: In-process LangGraph SDK client.
        thread_id: Thread being compacted.

    Raises:
        _OffloadConflictError: If the thread has work in flight, or is not
            registered on the server at all.
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
    if thread.get("status") not in _OFFLOADABLE_THREAD_STATUSES:
        msg = "Cannot offload while the thread has an active or interrupted run."
        raise _OffloadConflictError(msg)


async def _write_landed(
    client: Any,  # noqa: ANN401  # untyped LangGraph SDK client
    thread_id: str,
    checkpoint_id: str,
) -> Literal["advanced", "unchanged", "unreadable"]:
    """Classify a failed `update_state` against the checkpoint we read.

    A new checkpoint means the write most likely applied despite the error. A
    concurrent run could also have advanced the thread, so this is a bias, not a
    proof -- it biases toward keeping cost records claimed (understating spend at
    worst) over restoring them (which would double-charge).

    `unreadable` is reported separately from `advanced` so the caller can say
    which one happened. Both keep the records claimed, but only `advanced` has
    evidence the write landed; conflating them would log a thread advance that
    was never observed.

    Args:
        client: In-process LangGraph SDK client.
        thread_id: Thread that was being compacted.
        checkpoint_id: Checkpoint the operation read and validated against.

    Returns:
        `advanced` if the checkpoint changed, `unchanged` if it did not, or
            `unreadable` if the thread could not be read back.
    """
    try:
        current = await client.threads.get_state(thread_id)
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
        return "unreadable"
    return "advanced" if _checkpoint_id(current) != checkpoint_id else "unchanged"


async def _commit_state_update(
    client: Any,  # noqa: ANN401  # untyped LangGraph SDK client
    thread_id: str,
    checkpoint_id: str,
    update: dict[str, Any],
    prepared: PreparedOperationCost,
) -> None:
    """Persist the summary reservation and settle its claimed model cost.

    Raises:
        _OffloadIndeterminateError: If the write failed after the thread
            advanced and its outcome cannot be determined.
    """
    try:
        await client.threads.update_state(thread_id, update)
    except BaseException as exc:
        outcome = await _write_landed(client, thread_id, checkpoint_id)
        if outcome != "unchanged":
            if outcome == "advanced":
                logger.exception(
                    "Offload state write for thread %s failed after the thread "
                    "advanced past checkpoint %s; keeping %d cost record(s) "
                    "claimed",
                    thread_id,
                    checkpoint_id,
                    len(prepared.records),
                )
            else:
                # Distinct from `advanced`: no thread advance was observed, so
                # the write may never have landed. Naming the amount makes an
                # otherwise undetectable loss auditable.
                logger.exception(
                    "Offload state write for thread %s failed and the thread "
                    "could not be read back; keeping %d cost record(s) claimed "
                    "to avoid double-charging, so $%.6f may be lost from the "
                    "thread total",
                    thread_id,
                    len(prepared.records),
                    prepared.delta_usd,
                )
            # Deliberately settled rather than rolled back: the delta is
            # treated as persisted, so restoring the records would double-charge
            # the next drain.
            prepared.commit()
            if isinstance(exc, asyncio.CancelledError):
                raise
            msg = (
                "Offload compacted the conversation but could not confirm "
                "the state write. Run /context to check whether the "
                "conversation was compacted before offloading again."
            )
            raise _OffloadIndeterminateError(msg) from None
        logger.warning(
            "Offload state write for thread %s failed with no thread advance; "
            "restoring %d cost record(s)",
            thread_id,
            len(prepared.records),
        )
        prepared.rollback()
        raise
    prepared.commit()


async def _archive_path_landed(
    client: Any,  # noqa: ANN401  # untyped LangGraph SDK client
    thread_id: str,
    path: str,
) -> bool | None:
    """Check whether the follow-up checkpoint links the completed archive.

    Returns:
        `True` when linked, `False` when confirmed absent, or `None` when the
            checkpoint could not be read.
    """
    try:
        current = await client.threads.get_state(thread_id)
    except BaseException:
        logger.exception(
            "Could not verify archive-path update for thread %s", thread_id
        )
        return None
    values = current.get("values")
    event = values.get("_summarization_event") if isinstance(values, Mapping) else None
    return isinstance(event, Mapping) and event.get("file_path") == path


async def _commit_deferred_archive(
    client: Any,  # noqa: ANN401  # untyped LangGraph SDK client
    thread_id: str,
    checkpoint_id: str,
    execution: OffloadExecution,
    update: dict[str, Any],
    prepared: PreparedOperationCost,
) -> None:
    """Reserve summary state, then append and link its archive transactionally.

    Raises:
        _OffloadIndeterminateError: If the archive was written but its
            checkpoint link cannot be read back.
    """
    archive = execution.archive
    if archive is None:
        await _commit_state_update(client, thread_id, checkpoint_id, update, prepared)
        return
    async with _archive_lock(archive.session_id):
        await _commit_state_update(client, thread_id, checkpoint_id, update, prepared)
        try:
            append = await archive.write()
        except Exception:
            logger.exception(
                "/offload reserved its summary but the archive append failed"
            )
            return
        if append is None:
            logger.error("/offload reserved its summary but the archive append failed")
            return
        event = archive.update(append.path)["_summarization_event"]
        try:
            await client.threads.update_state(
                thread_id, {"_summarization_event": event}
            )
        except BaseException as exc:
            landed = await _archive_path_landed(client, thread_id, append.path)
            if landed is True:
                execution.result["archive_path"] = append.path
                if isinstance(exc, asyncio.CancelledError):
                    raise
                return
            if landed is False:
                await append.rollback()
                if isinstance(exc, asyncio.CancelledError):
                    raise
                logger.exception(
                    "Archive link failed for thread %s; restored prior archive",
                    thread_id,
                )
                return
            msg = "Offload wrote its archive but could not confirm the archive link."
            raise _OffloadIndeterminateError(msg) from None
        execution.result["archive_path"] = append.path


async def _join_task_deferring_cancellation[T](
    task: asyncio.Task[T],
) -> asyncio.CancelledError | None:
    """Join a settlement task while retaining the first cancellation edge.

    Returns:
        The cancellation to re-raise after settlement, or `None`.
    """
    cancellation: asyncio.CancelledError | None = None
    while not task.done():
        try:
            await asyncio.wait((task,))
        except asyncio.CancelledError as exc:
            cancellation = cancellation or exc
    return cancellation


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
        context = _checkpoint_model_context(context, state)
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
        if forbidden := set(update) - _WRITABLE_STATE_CHANNELS:
            # A security boundary, not a defensive assertion: this route commits
            # to the latest checkpoint rather than the one it read, so a
            # `messages` write here would be unattributed to any run and could
            # clobber messages a concurrent run appended in that window. See
            # THREAT_MODEL.md (TB10/DF27) before relaxing this.
            #
            # Checked as an allowlist against `OffloadStateUpdate` rather than
            # for `messages` alone, so the runtime guard enforces the same
            # invariant the type states instead of a subset of it: a future
            # merge that adds any other channel is refused here too.
            msg = (
                "Server offload operations may not write "
                f"{sorted(forbidden)} to the checkpoint."
            )
            prepared.rollback()
            raise RuntimeError(msg)
        if not update:
            # Nothing to persist, but `prepare_operation_cost` already drained
            # the recorder. Returning without rolling back would delete that
            # spend from the thread's lifetime total (the drain is destructive).
            prepared.rollback()
            return {"status": "complete", "result": execution.result}
        commit = asyncio.create_task(
            _commit_deferred_archive(
                client,
                thread_id,
                checkpoint_id,
                execution,
                update,
                prepared,
            )
        )
        cancellation = await _join_task_deferring_cancellation(commit)
        commit.result()
        if cancellation is not None:
            raise cancellation
        return {"status": "complete", "result": execution.result}


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
      unregistered, carrying no checkpoint to offload, or advanced past the
      checkpoint read. Nothing committed.
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

    Raises:
        asyncio.CancelledError: When the cancellation route stops this operation.
    """
    # Request validation is scoped to its own block so that a `TypeError` or
    # `ValueError` raised *inside* the operation (a server-side fault) is not
    # misreported to the client as a 4xx and, worse, swallowed without a log.
    try:
        thread_id = request.path_params["thread_id"]
        operation_id, context, hook_responses = _operation_payload(await request.json())
    except (TypeError, ValueError) as exc:
        return JSONResponse({"detail": str(exc)}, status_code=422)

    key = (thread_id, operation_id)
    refusal = _register_operation(key)
    if refusal is not None:
        return JSONResponse({"detail": refusal}, status_code=409)
    outcome: _OperationOutcome | None = "finished"
    try:
        try:
            response = await _execute_offload(
                thread_id,
                operation_id=operation_id,
                context=context,
                hook_responses=hook_responses,
            )
        except asyncio.CancelledError:
            outcome = "cancelled"
            raise
        except _OffloadConflictError as exc:
            return JSONResponse({"detail": str(exc)}, status_code=409)
        except _OffloadUnavailableError as exc:
            logger.exception("Offload unavailable: server runtime build failed")
            return JSONResponse({"detail": str(exc)}, status_code=503)
        except _OffloadIndeterminateError as exc:
            return JSONResponse({"detail": str(exc)}, status_code=500)
        except Exception:
            logger.exception("Server-owned /offload failed")
            return JSONResponse(
                {
                    "detail": (
                        "Offload failed on the server; see the server log for details."
                    )
                },
                status_code=500,
            )
        if response["status"] == "interrupt":
            outcome = None
        return JSONResponse(response)
    finally:
        _finish_operation(key, outcome)


async def cancel_offload(request: Request) -> JSONResponse:
    """Cancel one operation id and wait until its server task is terminal.

    Returns:
        JSON containing `cancelled` when cancellation won, or `finished` when
            the operation had already reached a terminal result.
    """
    key = (
        request.path_params["thread_id"],
        request.path_params["operation_id"],
    )
    outcome = _operation_outcomes.get(key)
    if outcome is not None:
        return JSONResponse({"status": outcome})
    task = _active_operations.get(key)
    if task is None:
        _remember_operation(key, "cancelled")
        return JSONResponse({"status": "cancelled"})
    task.cancel()
    await asyncio.wait((task,))
    return JSONResponse({"status": _operation_outcomes.get(key, "finished")})


app = Starlette(
    routes=[
        Route(
            "/dcode/threads/{thread_id:str}/offload",
            offload,
            methods=["POST"],
        ),
        Route(
            "/dcode/threads/{thread_id:str}/offload/{operation_id:str}/cancel",
            cancel_offload,
            methods=["POST"],
        ),
    ]
)
