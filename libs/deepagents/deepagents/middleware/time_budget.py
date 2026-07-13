"""Time budget middleware for Deep Agents.

`TimeBudgetMiddleware` gives a run a wall-clock budget for *agent-active
time* -- the time spent inside model calls and tool calls -- and enforces
it when the budget is exhausted.

Why "agent-active time" and not a fixed deadline? Metering the summed
duration of model and tool calls (rather than an absolute wall-clock
deadline) makes the budget robust to two things a naive deadline gets
wrong:

- **Human-in-the-loop pauses.** Interrupts happen *between* actions, so a
    human taking minutes to approve a tool call never counts against the
    budget.
- **Checkpoint / resume.** Consumed time is persisted in graph state as an
    accumulated number of seconds, not as a deadline timestamp, so a run
    that pauses overnight and resumes tomorrow is not instantly "over
    budget".

Enforcement happens at action boundaries (middleware can gate *between*
actions but cannot preempt one already running):

- ``on_exceed="wind_down"`` (default): once the budget is spent, the next
    model call has its tools stripped and a "finalize now" instruction
    injected, forcing the model to produce a final answer from what it
    already has.
- ``on_exceed="hard_stop"``: once the budget is spent, the run ends
    immediately with a short notice message instead of calling the model
    again.

While under budget, the middleware can also inject a short awareness note
("~120s of 360s remaining") so the model can pace itself.

The middleware is a no-op to *include* only in the sense that it must be
given a positive ``total_seconds``; to run without a budget, simply do not
add it.
"""

from __future__ import annotations

import logging
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Literal,
    NotRequired,
)

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ModelResponse,
    PrivateStateAttr,
    ResponseT,
    hook_config,
)
from langchain_core._api import beta
from langchain_core.messages import AIMessage
from langgraph.types import Command

from deepagents.middleware._utils import append_to_system_message

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain.agents.middleware.types import ModelRequest
    from langchain_core.messages import ToolMessage
    from langgraph.runtime import Runtime

logger = logging.getLogger(__name__)


OnExceed = Literal["wind_down", "hard_stop"]
"""What the middleware does once the time budget is exhausted.

- ``"wind_down"``: strip tools and inject a "finalize now" instruction on
    the next model call so the model returns a final answer.
- ``"hard_stop"``: end the run immediately with a notice message.
"""

TimeBudgetStatus = Literal["ok", "warning", "wind_down", "hard_stop"]
"""Per-model-call budget status, recorded on state and passed to ``on_event``.

- ``"ok"``: comfortably within budget.
- ``"warning"``: past the warn threshold but not yet exhausted.
- ``"wind_down"``: budget exhausted; the run is being wound down.
- ``"hard_stop"``: budget exhausted; the run was hard-stopped.
"""

TIME_BUDGET_EVENT_SOURCE = "time_budget"
"""``lc_source`` tag stamped on the synthetic message injected on a hard stop."""


def _add_seconds(left: float | None, right: float | None) -> float:
    """Reducer that accumulates consumed seconds across (possibly parallel) actions."""
    return (left or 0.0) + (right or 0.0)


class TimeBudgetState(AgentState):
    """State schema for [`TimeBudgetMiddleware`][deepagents.middleware.time_budget.TimeBudgetMiddleware].

    All fields are bookkeeping and annotated with
    [`PrivateStateAttr`][langchain.agents.middleware.types.PrivateStateAttr]
    so they are omitted from the agent's input/output schemas. They remain
    readable via ``agent.get_state(config).values`` on a checkpointed thread,
    and the same numbers are delivered to the ``on_event`` callback.
    """

    _time_budget_consumed: NotRequired[Annotated[float, PrivateStateAttr, _add_seconds]]
    """Accumulated agent-active seconds (sum of model- and tool-call durations)."""

    _time_budget_model_start: NotRequired[Annotated[float | None, PrivateStateAttr]]
    """Clock reading captured in ``before_model``; consumed by ``after_model``."""

    _time_budget_status: NotRequired[Annotated[TimeBudgetStatus | None, PrivateStateAttr]]
    """Status recorded for the most recent model step."""


@beta(obj_type="middleware")
class TimeBudgetMiddleware(AgentMiddleware[TimeBudgetState, ContextT, ResponseT]):
    """Meter agent-active time and enforce a total budget.

    Add it to a deep agent via the ``middleware`` parameter::

        from deepagents import create_deep_agent
        from deepagents.middleware import TimeBudgetMiddleware

        agent = create_deep_agent(
            model="anthropic:claude-sonnet-4-6",
            middleware=[TimeBudgetMiddleware(total_seconds=360)],
        )

    Args:
        total_seconds: The agent-active time budget, in seconds. Must be
            positive. To run without a budget, do not add the middleware.
        warn_fraction: Fraction of the budget consumed at which the status
            becomes ``"warning"`` and (if ``inject_awareness``) the awareness
            note starts telling the model to wrap up. Must be in ``(0, 1]``.
            Defaults to ``0.8``.
        on_exceed: Behavior once the budget is exhausted. ``"wind_down"``
            (default) strips tools and forces a final answer; ``"hard_stop"``
            ends the run with a notice message.
        inject_awareness: When ``True`` (default), a short remaining-budget
            note is appended to the system prompt on each model call so the
            model can pace itself.
        clock: Zero-argument callable returning a monotonically increasing
            time in seconds. Defaults to ``time.monotonic``. Injectable so
            tests can advance time deterministically.
        on_event: Optional callback invoked once per model step with a dict
            describing the budget state (``status``, ``consumed``,
            ``remaining``, ``total``). Exceptions raised by the callback are
            logged and swallowed.

    Raises:
        ValueError: If ``total_seconds <= 0`` or ``warn_fraction`` is outside
            ``(0, 1]``.
    """

    state_schema = TimeBudgetState

    def __init__(  # noqa: D107
        self,
        total_seconds: float,
        *,
        warn_fraction: float = 0.8,
        on_exceed: OnExceed = "wind_down",
        inject_awareness: bool = True,
        clock: Callable[[], float] | None = None,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        super().__init__()
        if total_seconds <= 0:
            msg = f"total_seconds must be positive, got {total_seconds}"
            raise ValueError(msg)
        if not 0 < warn_fraction <= 1:
            msg = f"warn_fraction must be in (0, 1], got {warn_fraction}"
            raise ValueError(msg)
        self.total_seconds = float(total_seconds)
        self.warn_fraction = float(warn_fraction)
        self.on_exceed: OnExceed = on_exceed
        self.inject_awareness = inject_awareness
        self._on_event = on_event
        if clock is None:
            import time  # noqa: PLC0415

            clock = time.monotonic
        self._clock = clock

    @property
    def _warn_threshold(self) -> float:
        return self.total_seconds * self.warn_fraction

    @staticmethod
    def _consumed(state: AgentState[Any]) -> float:
        return state.get("_time_budget_consumed", 0.0) or 0.0

    def _remaining(self, consumed: float) -> float:
        return self.total_seconds - consumed

    def _status_for(self, consumed: float) -> TimeBudgetStatus:
        """Status implied by the time already consumed *before* a model call.

        Uses the consumed-at-start value so the reported status matches the
        enforcement decision taken for that same call.
        """
        if self._remaining(consumed) <= 0:
            return "wind_down" if self.on_exceed == "wind_down" else "hard_stop"
        if consumed >= self._warn_threshold:
            return "warning"
        return "ok"

    def _emit(self, status: TimeBudgetStatus, consumed: float) -> None:
        if self._on_event is None:
            return
        event = {
            "source": TIME_BUDGET_EVENT_SOURCE,
            "status": status,
            "consumed": consumed,
            "remaining": max(0.0, self._remaining(consumed)),
            "total": self.total_seconds,
        }
        try:
            self._on_event(event)
        except Exception:
            logger.exception("TimeBudgetMiddleware on_event callback raised")

    def _awareness_text(self, remaining: float) -> str:
        if remaining <= self.total_seconds * (1 - self.warn_fraction):
            return (
                f"[time budget] Running low: about {remaining:.0f}s of "
                f"{self.total_seconds:.0f}s remaining. Start finalizing your answer "
                "and avoid starting long new tasks."
            )
        return f"[time budget] About {remaining:.0f}s of {self.total_seconds:.0f}s remaining. Pace your work accordingly."

    def _wind_down_text(self) -> str:
        return (
            f"[time budget] Time is up (budget was {self.total_seconds:.0f}s). Provide your "
            "final answer now using only the information you already have. Do not call any more tools."
        )

    def _hard_stop_notice(self) -> str:
        return f"Stopped: the agent's time budget of {self.total_seconds:.0f}s was exhausted before a final answer was produced."

    @hook_config(can_jump_to=["end"])
    def before_model(self, state: TimeBudgetState, runtime: Runtime[ContextT]) -> dict[str, Any] | None:  # noqa: ARG002
        """Hard-stop the run if over budget; otherwise stamp the model start time."""
        return self._before_model(state)

    @hook_config(can_jump_to=["end"])
    async def abefore_model(self, state: TimeBudgetState, runtime: Runtime[ContextT]) -> dict[str, Any] | None:  # noqa: ARG002
        """Async variant of `before_model`."""
        return self._before_model(state)

    def _before_model(self, state: TimeBudgetState) -> dict[str, Any] | None:
        consumed = self._consumed(state)
        if self.on_exceed == "hard_stop" and self._remaining(consumed) <= 0:
            self._emit("hard_stop", consumed)
            return {
                "jump_to": "end",
                "_time_budget_status": "hard_stop",
                "messages": [
                    AIMessage(
                        content=self._hard_stop_notice(),
                        additional_kwargs={"lc_source": TIME_BUDGET_EVENT_SOURCE},
                    )
                ],
            }
        return {"_time_budget_model_start": self._clock()}

    def after_model(self, state: TimeBudgetState, runtime: Runtime[ContextT]) -> dict[str, Any] | None:  # noqa: ARG002
        """Record the model call's duration and the resulting budget status."""
        return self._after_model(state)

    async def aafter_model(self, state: TimeBudgetState, runtime: Runtime[ContextT]) -> dict[str, Any] | None:  # noqa: ARG002
        """Async variant of `after_model`."""
        return self._after_model(state)

    def _after_model(self, state: TimeBudgetState) -> dict[str, Any] | None:
        start = state.get("_time_budget_model_start")
        if start is None:
            return None
        consumed_at_start = self._consumed(state)
        delta = max(0.0, self._clock() - start)
        status = self._status_for(consumed_at_start)
        self._emit(status, consumed_at_start + delta)
        return {
            "_time_budget_consumed": delta,
            "_time_budget_status": status,
            "_time_budget_model_start": None,
        }

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT]:
        """Inject awareness / wind-down instructions before the model runs."""
        return handler(self._modify_request(request))

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT]:
        """Async variant of `wrap_model_call`."""
        return await handler(self._modify_request(request))

    def _modify_request(self, request: ModelRequest[ContextT]) -> ModelRequest[ContextT]:
        consumed = self._consumed(request.state)
        remaining = self._remaining(consumed)

        if self.on_exceed == "wind_down" and remaining <= 0:
            no_tools = request.override(tools=[])
            new_system = append_to_system_message(request.system_message, self._wind_down_text())
            return no_tools.override(system_message=new_system)
        if self.inject_awareness:
            new_system = append_to_system_message(request.system_message, self._awareness_text(remaining))
            return request.override(system_message=new_system)
        return request

    def wrap_tool_call(
        self,
        request: Any,  # noqa: ANN401
        handler: Callable[[Any], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Time the tool call and fold its duration into the budget."""
        start = self._clock()
        result = handler(request)
        return self._attach_tool_delta(result, max(0.0, self._clock() - start))

    async def awrap_tool_call(
        self,
        request: Any,  # noqa: ANN401
        handler: Callable[[Any], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Async variant of `wrap_tool_call`."""
        start = self._clock()
        result = await handler(request)
        return self._attach_tool_delta(result, max(0.0, self._clock() - start))

    @staticmethod
    def _attach_tool_delta(result: ToolMessage | Command[Any], delta: float) -> ToolMessage | Command[Any]:
        """Fold ``delta`` seconds into ``_time_budget_consumed`` alongside the tool result.

        A tool may return a plain ``ToolMessage`` or a ``Command`` (e.g.
        filesystem tools that also update state). In both cases the duration
        is merged into the ``_time_budget_consumed`` channel via its additive
        reducer.
        """
        if isinstance(result, Command):
            if isinstance(result.update, dict):
                result.update["_time_budget_consumed"] = _add_seconds(result.update.get("_time_budget_consumed"), delta)

            return result
        return Command(update={"messages": [result], "_time_budget_consumed": delta})
