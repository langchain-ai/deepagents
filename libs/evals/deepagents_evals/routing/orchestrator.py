"""Hand control between a planner and an executor model within one trajectory.

The planner starts each thread: it inspects the environment and calls
`handoff_to_executor` with a plan. The executor does the work and calls
`escalate_to_planner` when the plan breaks; the planner repairs the plan and hands
back. A run of failed tool results also forces an escalation, so a stuck executor
cannot keep guessing. Escalations are capped to stop the models trading control
forever. Both models share one message history, so neither re-reads what the other
already explored.

Standalone: this module shares no code with the difficulty ladder in `middleware.py`.
Read handoffs from the agent output's `decision_log`.
"""

from __future__ import annotations

import json
import operator
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, NotRequired, cast

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
    OmitFromInput,
    PrivateStateAttr,
    ResponseT,
    TracePolicy,
    omit_payload,
)
from langchain.chat_models import init_chat_model
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.types import Command

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping, Sequence

    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import BaseMessage
    from langchain_core.tools import BaseTool
    from langgraph.runtime import Runtime

Phase = Literal["plan", "execute"]

GATEWAY_ANTHROPIC_URL = "https://gateway.smith.langchain.com/anthropic"
GATEWAY_OPENAI_URL = "https://gateway.smith.langchain.com/openai/v1"
GATEWAY_URLS = {"anthropic": GATEWAY_ANTHROPIC_URL, "openai": GATEWAY_OPENAI_URL}

DEFAULT_CONFIG = Path(__file__).with_name("default_orchestrator.json")
CONFIG_ENV = "ROUTING_CONFIG"
"""Env var holding JSON whose top-level keys override the packaged default, such as `planner`."""

HANDOFF = "handoff_to_executor"
ESCALATE = "escalate_to_planner"
_FAILED_COMMAND = "[Command failed with exit code"
# ponytail: blocklist, not allowlist, so MCP and shell tools stay usable while
# planning; the planner prompt asks for inspection only, which a shell cannot enforce.
_PLANNER_BLOCKED = frozenset({"write_file", "edit_file", "delete", ESCALATE})

_PLANNER_PROMPT = """## Orchestrator: planning phase

You are the planner. Another, faster model executes your plan in this same conversation.
Inspect only: read files, search, and run commands that do not change state (listing,
versions, `--help`, existing tests). Do not edit files, install packages, or change the system.
When you understand the task, call `handoff_to_executor` with a concrete plan: numbered steps,
how to verify each step, and explicit re-planning triggers, the observations that mean the
plan is wrong and the executor should escalate back to you. If the task needs no changes,
answer directly."""

_REPLAN_PROMPT = """The executor returned control to you: {reason}
Inspect the current state, repair the plan, and hand off again."""

_EXECUTOR_PROMPT = """## Orchestrator: execution phase

You are the executor. Follow the plan from the latest `handoff_to_executor` call: edit files,
install packages, and run tests until the task is verifiably done."""

_ESCALATE_PROMPT = """If a re-planning trigger fires, the plan proves wrong, or repeated attempts
fail, call `escalate_to_planner` with what you tried and observed instead of guessing."""

_NO_ESCALATIONS_PROMPT = "No escalations remain: finish the task yourself."


def _latest[T](_current: T, update: T) -> T:
    """Let parallel handoff calls in one step agree instead of conflicting."""
    return update


class OrchestratorState(AgentState):
    """Expose handoffs in results while keeping control bookkeeping private."""

    phase: NotRequired[Annotated[Phase, PrivateStateAttr, _latest]]
    """Which model serves calls now; absent means planning."""

    escalations: NotRequired[Annotated[int, PrivateStateAttr, _latest]]
    """Returns to the planner so far, by tool or by error streak."""

    error_streak: NotRequired[Annotated[int, PrivateStateAttr, _latest]]
    """Consecutive failed tool results while executing."""

    replan_reason: NotRequired[Annotated[str | None, PrivateStateAttr, _latest]]
    """Why the executor returned control, shown to the planner until it hands off."""

    decision_log: Annotated[list[dict[str, Any]], OmitFromInput, operator.add]
    """Output-only records of every change of control."""


def _record(src: Phase, dst: Phase, trigger: str, reason: str, escalations: int) -> dict[str, Any]:
    return {
        "kind": "orchestrator",
        "from_phase": src,
        "to_phase": dst,
        "trigger": trigger,
        "reason": reason,
        "escalations": escalations,
    }


@tool(HANDOFF)
def handoff_to_executor(plan: str, runtime: ToolRuntime) -> Command:
    """Hand control to the executor with the plan it should follow.

    Args:
        plan: Numbered steps, how to verify each, and the re-planning triggers.
        runtime: Injected agent state and tool call ID.
    """
    escalations = runtime.state.get("escalations", 0)
    return Command(
        update={
            "phase": "execute",
            "error_streak": 0,
            "replan_reason": None,
            "decision_log": [_record("plan", "execute", "tool", plan, escalations)],
            "messages": [
                ToolMessage("Handed off to the executor.", tool_call_id=runtime.tool_call_id)
            ],
        }
    )


@tool(ESCALATE)
def escalate_to_planner(reason: str, runtime: ToolRuntime) -> Command:
    """Return control to the planner when the plan breaks or progress stalls.

    Args:
        reason: What you tried, what you observed, and which assumption failed.
        runtime: Injected agent state and tool call ID.
    """
    escalations = runtime.state.get("escalations", 0) + 1
    return Command(
        update={
            "phase": "plan",
            "escalations": escalations,
            "error_streak": 0,
            "replan_reason": reason,
            "decision_log": [_record("execute", "plan", "tool", reason, escalations)],
            "messages": [
                ToolMessage("Escalated to the planner.", tool_call_id=runtime.tool_call_id)
            ],
        }
    )


def load_gateway_key() -> str:
    """Remove stray characters from `LC_GATEWAY_KEY` and check its format.

    Returns:
        A personal or service key containing only ASCII letters, digits, underscores, and hyphens.

    Raises:
        KeyError: If `LC_GATEWAY_KEY` is not set.
        ValueError: If the cleaned value does not match a LangSmith personal or service key.
    """
    key = re.sub(r"[^A-Za-z0-9_-]", "", os.environ["LC_GATEWAY_KEY"])
    if not re.fullmatch(r"lsv2_(?:pt|sk)_[A-Za-z0-9_-]+", key):
        msg = "LC_GATEWAY_KEY must match a LangSmith personal or service key after cleanup (lsv2_pt_... or lsv2_sk_...)."
        raise ValueError(msg)
    return key


def _gateway_model(spec: Mapping[str, Any]) -> BaseChatModel:
    """A config entry's `model` (`provider:model`) reached through the gateway, built with its `model_kwargs`.

    Raises:
        ValueError: If the provider has no gateway route.
    """
    model = spec["model"]
    provider = model.partition(":")[0]
    if provider not in GATEWAY_URLS:
        msg = f"no gateway route for provider {provider!r} in {model!r}; expected one of {sorted(GATEWAY_URLS)}"
        raise ValueError(msg)
    return init_chat_model(
        model,
        base_url=GATEWAY_URLS[provider],
        api_key=load_gateway_key(),
        **spec.get("model_kwargs", {}),
        **({"profile": spec["capabilities"]} if "capabilities" in spec else {}),
    )


@dataclass(frozen=True)
class OrchestratorConfig:
    """The models and limits one orchestrated run uses.

    Args:
        planner: Strong model that inspects, plans, and repairs plans.
        executor: Cheaper model that acts; also the anchor for summaries and subagents.
        max_escalations: Returns to the planner allowed per thread.
        error_streak: Consecutive failed tool results that force an escalation.
    """

    planner: BaseChatModel
    executor: BaseChatModel
    max_escalations: int = 4
    error_streak: int = 3


def requested() -> bool:
    """Whether `ROUTING_CONFIG` selects the orchestrator over the difficulty ladder."""
    return json.loads(os.environ.get(CONFIG_ENV) or "{}").get("mode") == "orchestrator"


def load_config() -> OrchestratorConfig:
    """Read the packaged default, with top-level keys from `ROUTING_CONFIG` overriding it."""
    raw = json.loads(DEFAULT_CONFIG.read_text()) | json.loads(os.environ.get(CONFIG_ENV) or "{}")
    return OrchestratorConfig(
        planner=_gateway_model(raw["planner"]),
        executor=_gateway_model(raw["executor"]),
        max_escalations=raw.get("max_escalations", 4),
        error_streak=raw.get("error_streak", 3),
    )


def _failed(message: ToolMessage) -> bool:
    return message.status == "error" or _FAILED_COMMAND in message.text


def _latest_results(messages: Sequence[BaseMessage]) -> list[ToolMessage]:
    """Tool results since the last model call."""
    results = []
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            break
        if isinstance(message, ToolMessage):
            results.append(message)
    return results


class OrchestratorMiddleware(AgentMiddleware[OrchestratorState, None, ResponseT]):
    """Serve planning calls on the planner and execution calls on the executor."""

    state_schema = OrchestratorState
    trace_policy = TracePolicy(process_inputs=omit_payload)
    tools: Sequence[BaseTool] = (handoff_to_executor, escalate_to_planner)

    def __init__(
        self,
        planner: BaseChatModel,
        executor: BaseChatModel,
        *,
        max_escalations: int = 4,
        error_streak: int = 3,
    ) -> None:
        """Initialize the orchestrator.

        Args:
            planner: Model for inspection and planning.
            executor: Model for edits, installs, and tests.
            max_escalations: Returns to the planner allowed per thread; after that the
                executor loses `escalate_to_planner` and is never forced back.
            error_streak: Consecutive failed tool results that force an escalation.
        """
        super().__init__()
        self.planner = planner
        self.executor = executor
        self.max_escalations = max_escalations
        self.error_streak = error_streak

    def before_model(
        self,
        state: OrchestratorState,
        runtime: Runtime[None],  # noqa: ARG002  # Required middleware hook signature.
    ) -> dict[str, Any] | None:
        """Count failed tool results and force an escalation after a streak."""
        results = _latest_results(state["messages"])
        if state.get("phase") != "execute" or not results:
            return None
        if not all(_failed(m) for m in results):
            return {"error_streak": 0}
        streak = state.get("error_streak", 0) + len(results)
        escalations = state.get("escalations", 0)
        if streak < self.error_streak or escalations >= self.max_escalations:
            return {"error_streak": streak}
        reason = f"{streak} consecutive tool calls failed."
        return {
            "phase": "plan",
            "escalations": escalations + 1,
            "error_streak": 0,
            "replan_reason": reason,
            "decision_log": [_record("execute", "plan", "errors", reason, escalations + 1)],
        }

    async def abefore_model(
        self,
        state: OrchestratorState,
        runtime: Runtime[None],
    ) -> dict[str, Any] | None:
        """Count failed tool results and force an escalation after a streak (async version)."""
        return self.before_model(state, runtime)

    def wrap_model_call(
        self,
        request: ModelRequest[None],
        handler: Callable[[ModelRequest[None]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT]:
        """Run the call on the model, tools, and instructions of the current phase."""
        return handler(self._phased(request))

    async def awrap_model_call(
        self,
        request: ModelRequest[None],
        handler: Callable[[ModelRequest[None]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT]:
        """Run the call on the model, tools, and instructions of the current phase (async version)."""
        return await handler(self._phased(request))

    def _phased(self, request: ModelRequest[None]) -> ModelRequest[None]:
        state = cast("OrchestratorState", request.state)
        phase = state.get("phase", "plan")
        if phase == "plan":
            model, blocked, prompt = self.planner, _PLANNER_BLOCKED, self._planner_prompt(state)
        else:
            model, blocked, prompt = self.executor, *self._executor_setup(state)
        metadata = {**(model.metadata or {}), "orchestrator_phase": phase}
        return request.override(
            model=model.model_copy(update={"metadata": metadata}),
            tools=[t for t in request.tools if _tool_name(t) not in blocked],
            system_message=_appended(request.system_message, prompt),
        )

    @staticmethod
    def _planner_prompt(state: OrchestratorState) -> str:
        reason = state.get("replan_reason")
        return (
            f"{_PLANNER_PROMPT}\n\n{_REPLAN_PROMPT.format(reason=reason)}"
            if reason
            else _PLANNER_PROMPT
        )

    def _executor_setup(self, state: OrchestratorState) -> tuple[frozenset[str], str]:
        if state.get("escalations", 0) >= self.max_escalations:
            return frozenset({HANDOFF, ESCALATE}), f"{_EXECUTOR_PROMPT}\n\n{_NO_ESCALATIONS_PROMPT}"
        return frozenset({HANDOFF}), f"{_EXECUTOR_PROMPT}\n\n{_ESCALATE_PROMPT}"


def _tool_name(item: BaseTool | dict[str, Any]) -> str | None:
    return item.get("name") if isinstance(item, dict) else item.name


def _appended(system: SystemMessage | None, text: str) -> SystemMessage:
    if system is None:
        return SystemMessage(content=text)
    return SystemMessage(content_blocks=[*system.content_blocks, {"type": "text", "text": text}])
