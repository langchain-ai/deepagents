"""HITL middleware that adds a model-generated reason to approval requests.

Subclasses the SDK's `HumanInTheLoopMiddleware` so the approval interrupt
carries a short, model-generated justification (`reason`) for high-risk tool
calls. The CLI's approval widget renders this as `Reason: <italic>` between the
title and the tool-specific content.

The reason is produced by a single extra model call per interrupting batch
(approach "b" in langchain-ai/deepagents#1374): every reason-flagged tool call
in a batch shares one call, no upstream tool-schema or `ActionRequest` changes
are required, and the justification stays isolated to the CLI. The call is
tagged with `lc_source="reason"` so the Textual adapter filters it out of the
streamed transcript.

The model call is synchronous and runs inside `after_model`, so it briefly
blocks before the approval interrupt is raised. It is deliberately not offloaded
to a thread: `interrupt()` and its surrounding control flow must run on the
graph's own task, and the user is about to be blocked on the approval prompt
regardless, so the short delay is acceptable.

Which tools receive a reason is configured per tool via a `reason: True` flag
on the `InterruptOnConfig`; tools without it behave exactly as before.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, NotRequired

from langchain.agents.middleware.human_in_the_loop import (
    HITLRequest,
    HumanInTheLoopMiddleware,
    InterruptOnConfig,
)
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import interrupt

if TYPE_CHECKING:
    from langchain.agents.middleware.human_in_the_loop import (
        ActionRequest,
        ReviewConfig,
    )
    from langchain.agents.middleware.types import AgentState
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import ToolCall
    from langgraph.runtime import Runtime

logger = logging.getLogger(__name__)


class ReasonInterruptOnConfig(InterruptOnConfig):
    """`InterruptOnConfig` with an opt-in flag for model-generated reasons."""

    reason: NotRequired[bool]
    """When `True`, generate a model reason for this tool's approval request."""


REASON_LC_SOURCE = "reason"
"""`config.metadata.lc_source` value tagging the reason-generation call.

The Textual adapter uses this to filter the call's output from the transcript.
"""

_MAX_REASON_CHARS = 280
"""Reasons longer than this are truncated for display."""

_REASON_SYSTEM_PROMPT = (
    "You are about to run one or more tools that require the user's approval. "
    "In one short sentence, written in the first person and addressed to the "
    "user, explain why you want to run them and what you expect them to "
    "accomplish. Do not greet, apologize, or restate the arguments verbatim. "
    "Reply with the sentence only."
)


class ReasonInterruptMiddleware(HumanInTheLoopMiddleware):
    """HITL middleware that attaches a model-generated `reason` to requests.

    Behaves like `HumanInTheLoopMiddleware` but, for tools whose
    `InterruptOnConfig` sets `reason: True`, calls `model` once per interrupt
    batch to generate a short justification that is stored on each
    `ActionRequest` under the `reason` key.
    """

    def __init__(
        self,
        interrupt_on: dict[str, bool | InterruptOnConfig],
        *,
        model: BaseChatModel,
        description_prefix: str = "Tool execution requires approval",
    ) -> None:
        """Initialize the middleware.

        Args:
            interrupt_on: Mapping of tool name to interrupt config, forwarded to
                `HumanInTheLoopMiddleware`. Entries may set `reason: True` to
                request a model-generated justification for that tool.
            model: Chat model used to generate reasons.
            description_prefix: Forwarded to `HumanInTheLoopMiddleware`.
        """
        self._reason_tools: frozenset[str] = frozenset(
            name
            for name, cfg in interrupt_on.items()
            if isinstance(cfg, dict) and cfg.get("reason")
        )
        super().__init__(
            interrupt_on=interrupt_on,
            description_prefix=description_prefix,
        )
        self._reason_model = model

    def after_model(
        self, state: AgentState[Any], runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        """Trigger interrupt flows and attach one generated reason per batch.

        Returns:
            Updated messages with revised tool calls, or `None` when no
            interrupt is needed.

        Raises:
            ValueError: If the number of human decisions does not match the
                number of interrupted tool calls.
        """
        messages = state["messages"]
        if not messages:
            return None

        last_ai_msg = next(
            (msg for msg in reversed(messages) if isinstance(msg, AIMessage)), None
        )
        if not last_ai_msg or not last_ai_msg.tool_calls:
            return None

        action_requests: list[ActionRequest] = []
        review_configs: list[ReviewConfig] = []
        interrupt_indices: list[int] = []
        interrupt_tool_calls: list[ToolCall] = []

        for idx, tool_call in enumerate(last_ai_msg.tool_calls):
            if (config := self.interrupt_on.get(tool_call["name"])) is not None:
                if not self._should_interrupt(tool_call, config, state, runtime):
                    continue
                action_request, review_config = self._create_action_and_config(
                    tool_call, config, state, runtime
                )
                action_requests.append(action_request)
                review_configs.append(review_config)
                interrupt_indices.append(idx)
                interrupt_tool_calls.append(tool_call)

        if not action_requests:
            return None

        self._attach_batch_reason(action_requests, interrupt_tool_calls, state)

        hitl_request = HITLRequest(
            action_requests=action_requests,
            review_configs=review_configs,
        )

        decisions = interrupt(hitl_request)["decisions"]

        if (decisions_len := len(decisions)) != (
            interrupt_count := len(interrupt_indices)
        ):
            msg = (
                f"Number of human decisions ({decisions_len}) does not match "
                f"number of hanging tool calls ({interrupt_count})."
            )
            raise ValueError(msg)

        revised_tool_calls: list[ToolCall] = []
        artificial_tool_messages = []
        decision_idx = 0

        for idx, tool_call in enumerate(last_ai_msg.tool_calls):
            if idx in interrupt_indices:
                config = self.interrupt_on[tool_call["name"]]
                decision = decisions[decision_idx]
                decision_idx += 1

                revised_tool_call, tool_message = self._process_decision(
                    decision, tool_call, config
                )
                if revised_tool_call is not None:
                    revised_tool_calls.append(revised_tool_call)
                if tool_message:
                    artificial_tool_messages.append(tool_message)
            else:
                revised_tool_calls.append(tool_call)

        last_ai_msg.tool_calls = revised_tool_calls

        return {"messages": [last_ai_msg, *artificial_tool_messages]}

    def _attach_batch_reason(
        self,
        action_requests: list[ActionRequest],
        tool_calls: list[ToolCall],
        state: AgentState[Any],
    ) -> None:
        """Attach one generated reason to all configured requests in a batch."""
        reason_action_requests = [
            action_request
            for action_request, tool_call in zip(
                action_requests, tool_calls, strict=True
            )
            if tool_call["name"] in self._reason_tools
        ]
        if not reason_action_requests:
            return
        reason_tool_calls = [
            tool_call
            for tool_call in tool_calls
            if tool_call["name"] in self._reason_tools
        ]
        reason = self._generate_reason(reason_tool_calls, state)
        if not reason:
            return
        for action_request in reason_action_requests:
            action_request["reason"] = reason  # ty: ignore[invalid-key]  # reason is a CLI-only extension to ActionRequest

    @staticmethod
    def _build_prompt(tool_calls: list[ToolCall], state: AgentState[Any]) -> list[Any]:
        """Build the message list for the reason-generation model call.

        Returns:
            A system prompt, the most recent user request (when available), and
            a final instruction naming the tool calls and their arguments.
        """
        messages: list[Any] = [SystemMessage(content=_REASON_SYSTEM_PROMPT)]
        last_human = next(
            (
                m
                for m in reversed(state.get("messages") or [])
                if isinstance(m, HumanMessage)
            ),
            None,
        )
        if last_human is not None and last_human.text:
            messages.append(HumanMessage(content=f"User's request:\n{last_human.text}"))
        tool_lines = "\n".join(
            f"{idx}. Tool: {tool_call['name']}\n   Arguments: {tool_call['args']}"
            for idx, tool_call in enumerate(tool_calls, start=1)
        )
        messages.append(
            HumanMessage(
                content=(
                    f"Tool calls requiring approval:\n{tool_lines}\n\n"
                    "Why do you want to run these tool calls?"
                )
            )
        )
        return messages

    def _generate_reason(
        self, tool_calls: list[ToolCall], state: AgentState[Any]
    ) -> str | None:
        """Generate a reason synchronously, returning `None` on any failure.

        Returns:
            A short justification, or `None` when generation fails or is empty.
        """
        try:
            response = self._reason_model.invoke(
                self._build_prompt(tool_calls, state),
                config={"metadata": {"lc_source": REASON_LC_SOURCE}},
            )
        except Exception:  # reason is best-effort; never block approval
            # Broad catch: provider exceptions (auth, network, rate-limit,
            # timeout) share no common base. WARNING, not DEBUG, so a persistent
            # failure (e.g. a rejected API key) is diagnosable rather than silent.
            logger.warning("Reason generation failed", exc_info=True)
            return None
        return _clean_reason(response)


def _clean_reason(response: AIMessage) -> str | None:
    """Normalize a model response into a single trimmed line, or `None`.

    Args:
        response: The model response from a reason-generation call.

    Returns:
        A single-line, length-capped reason, or `None` when the response has no
        usable text.
    """
    text = (response.text or "").strip()
    if not text:
        return None
    text = " ".join(text.split())
    if len(text) > _MAX_REASON_CHARS:
        text = text[: _MAX_REASON_CHARS - 1].rstrip() + "\u2026"
    return text
