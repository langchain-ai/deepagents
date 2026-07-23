"""Client-side fulfillment for server-owned Hooks v2 interrupts."""

from __future__ import annotations

from typing import TYPE_CHECKING

from deepagents_code.hooks.interrupt import (
    build_hook_resume_value,
    parse_hook_interrupt_payload,
)
from deepagents_code.hooks.models.transport import HookInvocationResponse

if TYPE_CHECKING:
    from deepagents_code.hooks.models.transport import HookInvocationRequest
    from deepagents_code.hooks.runtime import HooksRuntime


async def fulfill_hook_invocation(
    runtime: HooksRuntime,
    request: HookInvocationRequest,
) -> dict[str, object]:
    """Execute a server-owned hook request and return a resume payload.

    Args:
        runtime: Session-scoped client Hooks runtime.
        request: Validated invocation request from the server.

    Returns:
        JSON-compatible resume value for `Command(resume=...)`.

    Raises:
        ValueError: If the request snapshot does not match this session.
    """
    if request.snapshot_id != runtime.snapshot_id:
        msg = (
            f"Hook snapshot mismatch: request {request.snapshot_id} != "
            f"runtime {runtime.snapshot_id}"
        )
        raise ValueError(msg)

    decision = await runtime.invoke(request.invocation)
    response = HookInvocationResponse(
        protocol_version=1,
        invocation_id=request.invocation_id,
        snapshot_id=request.snapshot_id,
        decision=decision,
    )
    return build_hook_resume_value(response)


async def fulfill_hook_interrupt(
    runtime: HooksRuntime,
    interrupt_value: object,
) -> dict[str, object] | None:
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
