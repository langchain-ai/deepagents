"""Middleware that delivers steered messages into a running agent turn.

Runs in the agent server process. See `steering` for the inbox layout and why the
Store is the delivery channel. The inbox is drained in `abefore_model`, which is
the only boundary where appending a `HumanMessage` is both timely (the model sees
it on its very next call) and structurally valid (tool results for the previous
`AIMessage` are already complete).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Annotated, Any, NotRequired

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    PrivateStateAttr,
)
from langchain_core.messages import HumanMessage

from deepagents_code.auto_mode import USER_PROMPT_METADATA_KEY, user_prompt_metadata
from deepagents_code.steering import (
    SteerItem,
    adelete_steers,
    aread_pending_steers,
    coerce_consumed_seq,
    steering_enabled,
)

if TYPE_CHECKING:
    from langgraph.runtime import Runtime

logger = logging.getLogger(__name__)


class SteerState(AgentState):
    """Agent state extended with the steering consumption watermark."""

    _steer_consumed_seq: Annotated[NotRequired[int], PrivateStateAttr]
    """Highest steering `seq` already delivered on this thread."""


def _context_value(context: object, name: str) -> object:
    """Read a context field from either a mapping or an attribute holder.

    Args:
        context: Run context supplied by the graph server.
        name: Field to read.

    Returns:
        The field value, or `None` when the context does not carry it.
    """
    if isinstance(context, Mapping):
        return context.get(name)
    return getattr(context, name, None)


def _resolve_thread_id(runtime: object) -> str | None:
    """Return the thread whose inbox this run may drain.

    The run's own `execution_info.thread_id` is authoritative; the context copy
    is cross-checked so a mismatched context can never point the drain at
    another thread's namespace.

    Args:
        runtime: LangGraph runtime for the current step.

    Returns:
        The thread id, or `None` when it is missing or inconsistent.
    """
    execution = getattr(runtime, "execution_info", None)
    execution_id = getattr(execution, "thread_id", None)
    context_id = _context_value(getattr(runtime, "context", None), "thread_id")
    candidates = [
        value
        for value in (execution_id, context_id)
        if isinstance(value, str) and value
    ]
    if not candidates:
        return None
    if len(candidates) == 2 and candidates[0] != candidates[1]:  # noqa: PLR2004  # both ids present
        logger.warning("Steering thread id mismatch; skipping inbox drain")
        return None
    return candidates[0]


def _steer_message(item: SteerItem) -> HumanMessage:
    """Build the injected user message for one inbox item.

    The trusted prompt metadata is rebuilt here from validated fields only, and
    never carries referenced paths: steered text is delivered verbatim (no `@`
    file expansion), so an inbox item cannot assert that the user referenced a
    path they never referenced.

    Args:
        item: Validated inbox item.

    Returns:
        The `HumanMessage` appended to agent state.
    """
    return HumanMessage(
        content=item.text,
        additional_kwargs={
            USER_PROMPT_METADATA_KEY: user_prompt_metadata(
                item.literal_user_text,
                [],
                turn_id=item.turn_id,
            )
        },
    )


class SteeringMiddleware(AgentMiddleware[SteerState, ContextT]):
    """Injects client-steered messages at the next model call.

    Only the async hook is implemented: the agent server always drives the graph
    asynchronously, and the Store it supplies rejects synchronous reads from the
    event-loop thread. A synchronous run simply does not steer, which degrades to
    today's behavior (queued messages run after the turn).
    """

    state_schema = SteerState

    async def abefore_model(  # noqa: PLR6301  # AgentMiddleware hook must be an instance method.
        self,
        state: SteerState,
        runtime: Runtime[ContextT],
    ) -> dict[str, Any] | None:
        """Append any undelivered steering messages before the model runs.

        Args:
            state: Current agent state, read for the consumption watermark.
            runtime: Runtime carrying the run context and Store.

        Returns:
            State update with the injected messages and the advanced watermark,
            or `None` when there is nothing to deliver.
        """
        if not steering_enabled():
            return None
        thread_id = _resolve_thread_id(runtime)
        if thread_id is None:
            return None

        consumed = coerce_consumed_seq(state.get("_steer_consumed_seq"))
        store = getattr(runtime, "store", None)
        items = await aread_pending_steers(store, thread_id, after_seq=consumed)
        if not items:
            return None

        logger.info("Delivering %d steering message(s) mid-turn", len(items))
        await adelete_steers(store, thread_id, [item.key for item in items])
        return {
            "messages": [_steer_message(item) for item in items],
            "_steer_consumed_seq": items[-1].seq,
        }
