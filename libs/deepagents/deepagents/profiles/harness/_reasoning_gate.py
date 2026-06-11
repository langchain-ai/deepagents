"""Conditional reasoning-verification controller for MiniMax.

A single middleware that fires every turn and decides — from the turn's tool
activity alone, no model call — whether the agent took actions that warrant a
verification pass. Only then does it invoke an internal `RubricMiddleware` to
grade the agent's work against a fixed, domain-agnostic process rubric and, on a
real violation, inject the specific gap and re-run the actor once.

Design points:
- The gate is deterministic: a turn is verified when the agent used a tool this
  turn (state may have changed) or in an earlier turn (this final message must
  faithfully report what was done). Pure chat/acknowledgement turns are skipped.
  An LLM classifier was tried here and removed: it labelled ~14% of
  state-changing turns SIMPLE — skipping verification exactly where it mattered —
  while filtering out only ~18% of turns overall, so it cost a model call per
  turn for a coverage hole.
- The internal `RubricMiddleware` is NOT registered in the agent's middleware
  stack; it is owned and driven by this controller, so grading happens only when
  the gate says so.
- The grader is a separate instance of the *same kind* of model (e.g. MiniMax
  M3). The rubric criteria are reasoning-validity checks (not outcome templates),
  so they reward a correct refusal of a non-permitted request rather than pushing
  the agent to fulfil it. The agent's system prompt (the policy) is embedded in
  the rubric so the grader can judge permission/feasibility.
"""

from __future__ import annotations

# ruff: noqa: E501
# Rubric/classify prompts are written as whole-sentence lines by design.
import uuid
from typing import (  # Annotated must stay importable at runtime: get_type_hints evaluates the state-schema annotation when create_deep_agent resolves schemas.
    TYPE_CHECKING,
    Annotated,
    Any,
    NotRequired,
)

from langchain.agents.middleware.types import (
    AgentMiddleware,
    PrivateStateAttr,
    hook_config,
)
from langchain_core.messages import AIMessage, SystemMessage

from deepagents.middleware.rubric import RubricMiddleware, RubricState

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langgraph.runtime import Runtime

_FIXED_RUBRIC = """\
Evaluate whether the agent's work satisfies every criterion. Judge the agent's reasoning against the operating rules and the facts available to it; reward a correct refusal of a request the rules don't permit, and never require the agent to do something the rules forbid.

1. Requirement coverage — every distinct thing the user asked for is either done or explicitly addressed/declined with a reason; nothing the user still needs is silently dropped.
2. Feasibility & permission — before acting, the agent determined whether each requested action is allowed and possible under the operating rules and available facts.
3. Reasoning validity — the agent's conclusions follow from the available facts and rules; it did not miss a combination of known facts that changes the outcome (e.g. declaring something impossible when a permitted sequence of actions achieves the user's goal), and it did not apply a rule it never actually checked.
4. Verified state changes — for EVERY action that changed state, the transcript must show the agent confirming the result by RE-READING or RE-QUERYING the affected state afterward, and that re-read must match exactly what the user asked for (correct target, correct values, nothing missing or extra). A tool call that returned "success" is NOT sufficient evidence on its own: if there is no subsequent re-read confirming the new state, this criterion FAILS; if a re-read shows the state does not match the request, this criterion FAILS. (If the agent made no state changes, this criterion is trivially satisfied.)
5. Faithful reporting — the agent's final message reports only what it actually verified (consistent with criterion 4) and gives the user the specific information they asked for; it must not claim or imply an outcome it did not confirm.
6. Transparency — any rule or limit that changed what the agent did was stated to the user, not applied silently."""

_OPERATING_RULES_TEMPLATE = "{rubric}\n\n<operating_rules>\n{policy}\n</operating_rules>"


class _ReasoningGateState(RubricState):
    """RubricState plus markers for cheap tool-use detection."""

    _gate_baseline: NotRequired[Annotated[int, PrivateStateAttr]]
    """Message count at turn start; isolates this turn's messages."""

    _tool_seen: NotRequired[Annotated[bool, PrivateStateAttr]]
    """Sticky: set once any turn uses a tool, never cleared.

    Lets `_should_verify` skip re-walking prior turns — it only scans this
    invocation's new messages and ORs in this flag, so the per-turn cost is
    O(new messages) instead of O(whole conversation).
    """


def _message_text(message: object) -> str:
    """Best-effort plain text for a message."""
    text = getattr(message, "text", None)
    if isinstance(text, str):
        return text
    content = getattr(message, "content", "")
    return content if isinstance(content, str) else str(content)


class ReasoningGateMiddleware(AgentMiddleware):
    """Grade every turn where the agent used a tool (now or earlier) via RubricMiddleware."""

    state_schema = _ReasoningGateState

    def __init__(self, grader_model: str | BaseChatModel, *, max_iterations: int = 2) -> None:
        """Initialize the controller.

        Args:
            grader_model: Model (or spec string) used for rubric grading — a
                separate instance of the same kind of model as the actor.
            max_iterations: Rubric grade iterations. 2 = one real revision pass
                (grade -> revise -> re-grade).
        """
        self._grader_model = grader_model
        self._max_iterations = max_iterations
        self._rubric: RubricMiddleware | None = None

    def _ensure_ready(self) -> None:
        """Lazily resolve the model and build the internal RubricMiddleware."""
        if self._rubric is not None:
            return
        model = self._grader_model
        if isinstance(model, str):
            from deepagents._models import resolve_model  # noqa: PLC0415

            model = resolve_model(model)
        self._rubric = RubricMiddleware(model=model, max_iterations=self._max_iterations)

    # -- turn boundary -------------------------------------------------------

    def before_agent(self, state: _ReasoningGateState, runtime: Runtime) -> dict[str, Any]:  # noqa: ARG002
        """Record the turn's start index for tool-use detection."""
        return {"_gate_baseline": len(state["messages"])}

    async def abefore_agent(self, state: _ReasoningGateState, runtime: Runtime) -> dict[str, Any]:  # noqa: ARG002
        """Async variant of `before_agent`."""
        return {"_gate_baseline": len(state["messages"])}

    @hook_config(can_jump_to=["model"])
    def after_agent(self, state: _ReasoningGateState, runtime: Runtime) -> dict[str, Any] | None:
        """Continue an active grade loop, else gate-then-grade this turn."""
        self._ensure_ready()
        if state.get("_rubric_status") == "needs_revision":
            return self._rubric.after_agent(state, runtime)  # type: ignore[union-attr]
        if not self._should_verify(state):
            return None
        fresh = self._fresh_rubric_run(state)
        result = self._rubric.after_agent({**state, **fresh}, runtime) or {}  # type: ignore[union-attr]
        return {**fresh, **result, "_tool_seen": True}

    @hook_config(can_jump_to=["model"])
    async def aafter_agent(self, state: _ReasoningGateState, runtime: Runtime) -> dict[str, Any] | None:
        """Async variant of `after_agent`."""
        self._ensure_ready()
        if state.get("_rubric_status") == "needs_revision":
            return await self._rubric.aafter_agent(state, runtime)  # type: ignore[union-attr]
        if not self._should_verify(state):
            return None
        fresh = self._fresh_rubric_run(state)
        result = await self._rubric.aafter_agent({**state, **fresh}, runtime) or {}  # type: ignore[union-attr]
        return {**fresh, **result, "_tool_seen": True}

    # -- gate ----------------------------------------------------------------

    @staticmethod
    def _used_tools(messages: list[Any]) -> bool:
        """Whether any message in the slice is an AI turn that called a tool."""
        return any(isinstance(m, AIMessage) and m.tool_calls for m in messages)

    def _should_verify(self, state: _ReasoningGateState) -> bool:
        """Verify when the agent acted — this invocation or earlier.

        A tool call this invocation means state may have changed; a tool call in
        an earlier turn means this (possibly tool-free) final message must report
        those results faithfully. A conversation with no tool calls at all is
        plain chat and is skipped.

        Prior turns are captured by the sticky `_tool_seen` flag (set the first
        time any turn grades), so we only scan this invocation's new messages
        (``messages[baseline:]``) rather than re-walking the whole history each
        turn — O(new messages) per turn instead of O(whole conversation).
        """
        if state.get("_tool_seen"):
            return True
        messages = state["messages"]
        baseline = state.get("_gate_baseline", 0)
        return self._used_tools(messages[baseline:])

    # -- rubric --------------------------------------------------------------

    def _fresh_rubric_run(self, state: _ReasoningGateState) -> dict[str, Any]:
        """Mint a fresh rubric-grading run (mirrors RubricMiddleware reset)."""
        rubric = self._build_rubric(state)
        return {
            "rubric": rubric,
            "_active_rubric": rubric,
            "_rubric_status": None,
            "_rubric_iterations": 0,
            "_current_grading_run_id": str(uuid.uuid4()),
        }

    def _build_rubric(self, state: _ReasoningGateState) -> str:
        """Fixed criteria + the agent's system prompt as operating rules."""
        policy = next(
            (_message_text(m) for m in state["messages"] if isinstance(m, SystemMessage)),
            "",
        )
        if not policy.strip():
            return _FIXED_RUBRIC
        return _OPERATING_RULES_TEMPLATE.format(rubric=_FIXED_RUBRIC, policy=policy)
