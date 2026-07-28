"""Client-side fulfillment for server-owned Hooks v2 interrupts."""

from __future__ import annotations

import asyncio
import logging
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from uuid import UUID

from deepagents_code.hooks.interrupt import (
    build_hook_resume_value,
    parse_hook_interrupt_payload,
)
from deepagents_code.hooks.models.transport import HookInvocationResponse

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping

    from deepagents_code.hooks.models.domain import HookDecision
    from deepagents_code.hooks.models.transport import HookInvocationRequest
    from deepagents_code.hooks.runtime import HooksRuntime
    from deepagents_code.json_types import JsonObject

logger = logging.getLogger(__name__)

_FulfillmentKey = tuple[str, UUID]


class HooksSnapshotChangedError(RuntimeError):
    """A hook resume was attempted against a stale configuration snapshot.

    Raised when a resumed interrupt carries a different `snapshot_id` than the
    session's live runtime, which happens when a checkpoint is replayed after
    the hooks configuration changed. The turn cannot be resumed safely; the
    caller should surface this and restart rather than apply a decision made
    against stale hooks.
    """


@dataclass(slots=True)
class HookFulfillmentLedger:
    """Deduplicate hook fulfillment for one client session."""

    _in_flight: dict[_FulfillmentKey, asyncio.Task[HookInvocationResponse]] = field(
        default_factory=dict
    )
    _completed: dict[_FulfillmentKey, HookInvocationResponse] = field(
        default_factory=dict
    )
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def fulfill(
        self,
        key: _FulfillmentKey,
        operation: Callable[[], Awaitable[HookInvocationResponse]],
    ) -> HookInvocationResponse:
        """Return one shared result for concurrent and repeated delivery."""
        async with self._lock:
            completed = self._completed.get(key)
            if completed is not None:
                return completed
            task = self._in_flight.get(key)
            if task is None:
                task = asyncio.create_task(self._run(key, operation))
                self._in_flight[key] = task
        return await asyncio.shield(task)

    async def _run(
        self,
        key: _FulfillmentKey,
        operation: Callable[[], Awaitable[HookInvocationResponse]],
    ) -> HookInvocationResponse:
        try:
            result = await operation()
        except BaseException:
            async with self._lock:
                self._in_flight.pop(key, None)
            raise
        async with self._lock:
            self._completed[key] = result
            self._in_flight.pop(key, None)
        return result


async def fulfill_hook_invocation(
    runtime: HooksRuntime,
    request: HookInvocationRequest,
) -> JsonObject:
    """Execute a server-owned hook request and return a resume payload.

    Args:
        runtime: Session-scoped client Hooks runtime.
        request: Validated invocation request from the server.

    Returns:
        JSON-compatible resume value for `Command(resume=...)`.

    Raises:
        HooksSnapshotChangedError: If the request snapshot does not match this
            session.
    """
    if request.snapshot_id != runtime.snapshot_id:
        msg = (
            f"Hook snapshot mismatch: request {request.snapshot_id} != "
            f"runtime {runtime.snapshot_id}"
        )
        raise HooksSnapshotChangedError(msg)

    async def execute() -> HookInvocationResponse:
        decision = await runtime.invoke(request.invocation)
        _apply_client_side_effects(decision)
        return HookInvocationResponse(
            protocol_version=1,
            invocation_id=request.invocation_id,
            snapshot_id=request.snapshot_id,
            decision=decision,
        )

    response = await runtime.fulfillments.fulfill(
        (request.snapshot_id, request.invocation_id),
        execute,
    )
    return build_hook_resume_value(response)


async def fulfill_hook_interrupt(
    runtime: HooksRuntime,
    interrupt_value: object,
) -> JsonObject | None:
    """Fulfill a raw interrupt value when it is a hook invocation.

    Args:
        runtime: Session-scoped client Hooks runtime.
        interrupt_value: Raw LangGraph interrupt payload.

    Returns:
        Resume value for hook interrupts, otherwise `None`.
    """
    request = parse_hook_interrupt_payload(interrupt_value)
    if request is None:
        return None
    return await fulfill_hook_invocation(runtime, request)


async def fulfill_pending_hook_interrupts(
    runtime: HooksRuntime,
    pending: Mapping[str, object],
) -> dict[str, JsonObject]:
    """Fulfill pending hook interrupts into a resume map keyed by interrupt id.

    Args:
        runtime: Session-scoped client Hooks runtime.
        pending: Mapping of LangGraph interrupt id to raw interrupt payload.

    Returns:
        Resume values ready for `Command(resume=...)`.

    Raises:
        RuntimeError: If a payload is not a valid hook interrupt.
    """
    resumes: dict[str, JsonObject] = {}
    for interrupt_id, payload in pending.items():
        resume_value = await fulfill_hook_interrupt(runtime, payload)
        if resume_value is None:
            msg = f"Failed to parse hook interrupt {interrupt_id}"
            raise RuntimeError(msg)
        resumes[interrupt_id] = resume_value
    return resumes


def _apply_client_side_effects(decision: HookDecision) -> None:
    """Surface user notices and emit validated terminal sequences.

    `systemMessage` must never become model context; notices are logged for the
    operator. Terminal sequences were allowlisted in the reducer.
    """
    for notice in decision.user_notices:
        logger.warning("Hook user notice: %s", notice)
    for sequence in decision.terminal_sequences:
        sys.stdout.write(sequence)
    if decision.terminal_sequences:
        sys.stdout.flush()
