"""HITL middleware that adds a model-generated reason to approval requests.

Subclasses the SDK's `HumanInTheLoopMiddleware` so the approval interrupt
carries a short, model-generated justification (`reason`) for high-risk tool
calls. The CLI's approval widget renders this as `Reason: <italic>` between the
title and the tool-specific content, mirroring Codex's treatment.

The reason is produced by a single extra model call per interrupting batch
(approach "b" in langchain-ai/deepagents#1374): no upstream tool-schema or
`ActionRequest` changes are required, and the justification stays isolated to
the CLI. The call is tagged with `lc_source="reason"` so the Textual adapter
filters it out of the streamed transcript.

Which tools receive a reason is configured per tool via a `reason: True` flag
on the `InterruptOnConfig`; tools without it behave exactly as before.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, NotRequired

from langchain.agents.middleware.human_in_the_loop import (
    HumanInTheLoopMiddleware,
    InterruptOnConfig,
)
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

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
    "You are about to run a tool that requires the user's approval. In one "
    "short sentence, written in the first person and addressed to the user, "
    "explain why you want to run it and what you expect it to accomplish. "
    "Do not greet, apologize, or restate the arguments verbatim. Reply with "
    "the sentence only."
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
        self._reason_tools = {
            name
            for name, cfg in interrupt_on.items()
            if isinstance(cfg, dict) and cfg.get("reason")
        }
        super().__init__(
            interrupt_on=interrupt_on,
            description_prefix=description_prefix,
        )
        self._reason_model = model

    def _create_action_and_config(
        self,
        tool_call: ToolCall,
        config: InterruptOnConfig,
        state: AgentState[Any],
        runtime: Runtime[Any],
    ) -> tuple[ActionRequest, ReviewConfig]:
        """Attach a synchronous model-generated `reason` when configured.

        Returns:
            The `(ActionRequest, ReviewConfig)` pair, with `reason` populated on
            the request when the tool opts in and the model returns text.
        """
        action_request, review_config = super()._create_action_and_config(
            tool_call, config, state, runtime
        )
        if tool_call["name"] in self._reason_tools:
            reason = self._generate_reason(tool_call, state)
            if reason:
                action_request["reason"] = reason  # ty: ignore[invalid-key]  # reason is a CLI-only extension to ActionRequest
        return action_request, review_config

    @staticmethod
    def _build_prompt(tool_call: ToolCall, state: AgentState[Any]) -> list[Any]:
        """Build the message list for the reason-generation model call.

        Returns:
            A system prompt, the most recent user request (when available), and
            a final instruction naming the tool and its arguments.
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
        messages.append(
            HumanMessage(
                content=(
                    f"Tool: {tool_call['name']}\n"
                    f"Arguments: {tool_call['args']}\n\n"
                    "Why do you want to run this tool?"
                )
            )
        )
        return messages

    def _generate_reason(
        self, tool_call: ToolCall, state: AgentState[Any]
    ) -> str | None:
        """Generate a reason synchronously, returning `None` on any failure.

        Returns:
            A short justification, or `None` when generation fails or is empty.
        """
        try:
            response = self._reason_model.invoke(
                self._build_prompt(tool_call, state),
                config={"metadata": {"lc_source": REASON_LC_SOURCE}},
            )
        except Exception:  # reason is best-effort; never block approval
            logger.debug("Reason generation failed", exc_info=True)
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
