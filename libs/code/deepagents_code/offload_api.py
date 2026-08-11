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
from deepagents_code.cost_tracking import CostState, prepare_operation_cost
from deepagents_code.hooks.interrupt import build_hook_interrupt_payload
from deepagents_code.hooks.server_middleware import (
    HookTransportInterruptError,
    operation_hook_responses,
)
from deepagents_code.server_graph import get_server_runtime

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig
    from starlette.requests import Request

    from deepagents_code.offload_middleware import _OffloadState

logger = logging.getLogger(__name__)

_OFFLOAD_API_VERSION = 1
_thread_locks: WeakValueDictionary[str, asyncio.Lock] = WeakValueDictionary()


def _thread_lock(thread_id: str) -> asyncio.Lock:
    """Return one live lock per thread without retaining inactive threads."""
    lock = _thread_locks.get(thread_id)
    if lock is None:
        lock = asyncio.Lock()
        _thread_locks[thread_id] = lock
    return lock


class _OffloadConflictError(RuntimeError):
    """The thread changed or became active during an offload attempt."""


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
    return (
        operation_id,
        {str(key): value for key, value in context.items()},
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
    return cast("_OffloadState", state)


async def _require_idle_thread(client: Any, thread_id: str) -> None:  # noqa: ANN401
    """Reject offload while LangGraph reports an active thread.

    Args:
        client: In-process LangGraph SDK client.
        thread_id: Thread being compacted.

    Raises:
        _OffloadConflictError: If the thread is not idle.
    """
    thread = await client.threads.get(thread_id)
    if thread.get("status") != "idle":
        msg = "Cannot offload while the thread has an active or interrupted run."
        raise _OffloadConflictError(msg)


async def _execute_offload(
    thread_id: str,
    *,
    operation_id: str,
    context: dict[str, Any],
    hook_responses: dict[str, object],
) -> dict[str, object]:
    """Execute and commit one server-owned offload attempt.

    Args:
        thread_id: LangGraph thread to compact.
        operation_id: Opaque client-generated attempt identity.
        context: Runtime model and hooks context.
        hook_responses: Accumulated hook replies keyed by invocation id.

    Returns:
        A complete result or a hook request that must be answered.

    Raises:
        _OffloadConflictError: If the thread is active or changes before commit.
        RuntimeError: If the operation attempts to write conversation messages.
    """
    client = get_client(url=None, api_key=None)
    async with _thread_lock(thread_id):
        await _require_idle_thread(client, thread_id)
        before = await client.threads.get_state(thread_id)
        if before.get("next") or before.get("tasks") or before.get("interrupts"):
            msg = "Cannot offload a thread with pending graph work."
            raise _OffloadConflictError(msg)

        checkpoint_id = _checkpoint_id(before)
        state = _hydrate_state(before.get("values"))
        context["thread_id"] = thread_id
        namespace = f"dcode_offload:{operation_id}"
        info = ExecutionInfo(
            checkpoint_id=checkpoint_id,
            checkpoint_ns=namespace,
            task_id=operation_id,
            thread_id=thread_id,
            run_id=operation_id,
        )
        server = await get_server_runtime()
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
            msg = (
                "The thread changed while offload was running; no state was committed."
            )
            raise _OffloadConflictError(msg)

        prepared = prepare_operation_cost(cast("CostState", state), thread_id)
        update = {**execution.update, **prepared.update}
        if "messages" in update:
            msg = "Server offload operations may not write the messages channel."
            prepared.rollback()
            raise RuntimeError(msg)
        if update:
            try:
                # Apply the state-only delta to the latest checkpoint. The
                # thread API rejects in-flight runs itself; pinning this write
                # to `before` would instead branch from a stale checkpoint if
                # a run completed in the narrow window after our final check,
                # potentially hiding its newly appended messages.
                await client.threads.update_state(
                    thread_id,
                    update,
                )
            except BaseException:
                prepared.rollback()
                raise
        return {"status": "complete", "result": execution.result}


def capability(_request: Request) -> JSONResponse:
    """Report the dcode server-operation protocol version.

    Returns:
        JSON capability response.
    """
    return JSONResponse({"offload": True, "version": _OFFLOAD_API_VERSION})


async def offload(request: Request) -> JSONResponse:
    """Handle one thread offload or hook-resume round.

    Returns:
        JSON operation response.
    """
    try:
        operation_id, context, hook_responses = _operation_payload(await request.json())
        response = await _execute_offload(
            request.path_params["thread_id"],
            operation_id=operation_id,
            context=context,
            hook_responses=hook_responses,
        )
    except (TypeError, ValueError) as exc:
        return JSONResponse({"detail": str(exc)}, status_code=422)
    except _OffloadConflictError as exc:
        return JSONResponse({"detail": str(exc)}, status_code=409)
    except Exception:
        logger.exception("Server-owned /offload failed before checkpoint commit")
        return JSONResponse(
            {"detail": "Offload failed before checkpoint commit."},
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
