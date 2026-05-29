"""Profile-driven `rejection_response` for `HumanInTheLoopMiddleware`.

Two patterns. Both demonstrate how Deep Agents profiles can ship a tuned
`rejection_response` so individual `create_deep_agent` callers don't have
to remember the wiring.

Pattern 1: a `HarnessProfile` ships a fully-configured HITL middleware via
    `extra_middleware`. The profile owns the policy (which tools, which
    decisions, which rejection wording), so calling code stays clean.

Pattern 2: a small helper exports the tuned factory only. Callers continue
    to construct their own `HumanInTheLoopMiddleware`, choosing their tool
    set per-app, but pull the rejection wording from the profile-aligned
    helper so it stays consistent across the org.

Background: `langchain-ai/deepagents#2947`. The upstream default is
`ToolMessage(status="error")`, which several models (gpt-4o, current
Anthropic tiers) treat as a transient tool failure and immediately
re-emit the same call. Setting `status="success"` plus retry-discouraging
content via `rejection_response` breaks the loop. The knob is alpha.
"""

from __future__ import annotations

from typing import Final

from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    InterruptOnConfig,
)
from langchain.agents.middleware.human_in_the_loop import (
    RejectDecision,
    RejectionResponseFactory,
)
from langchain_core.messages import ToolCall, ToolMessage

from deepagents import (
    HarnessProfile,
    register_harness_profile,
)

# ---------------------------------------------------------------------------
# Shared building block: a tuned rejection-response factory.
#
# `status="success"` is the load-bearing change: it tells the provider this
# was a deliberate outcome, not a transient tool failure. The content also
# carries explicit "do not retry" copy as a belt-and-suspenders measure.
# ---------------------------------------------------------------------------


def _no_retry_rejection(tool_call: ToolCall, decision: RejectDecision) -> ToolMessage:
    reason = decision.get("message") or "no reason given"
    return ToolMessage(
        content=(
            f"User declined to run `{tool_call['name']}`. {reason} "
            "Do not retry; respond to the user instead."
        ),
        name=tool_call["name"],
        tool_call_id=tool_call["id"],
        status="success",
        additional_kwargs={"hitl_decision": "reject"},
    )


# ---------------------------------------------------------------------------
# Pattern 1: HarnessProfile ships a fully-configured HITL middleware.
#
# Use when the policy is uniform across an org for a given model — every
# agent on this model must require approval for the same dangerous tools,
# with the same rejection wording. Calling code does not have to register
# the HITL middleware itself.
#
# Caveat: `extra_middleware` *appends*, so callers who also pass their own
# `HumanInTheLoopMiddleware` via `create_deep_agent(middleware=[...])` will
# end up with two HITL middlewares. If that's a concern, prefer Pattern 2.
# ---------------------------------------------------------------------------


_DANGEROUS_TOOLS: Final[dict[str, InterruptOnConfig]] = {
    "payments.transfer": {
        "allowed_decisions": ["approve", "reject"],
        "rejection_response": _no_retry_rejection,
    },
    "fs.delete": {
        "allowed_decisions": ["approve", "edit", "reject"],
        "rejection_response": _no_retry_rejection,
    },
}


def _build_hitl_middleware() -> tuple[HumanInTheLoopMiddleware, ...]:
    """Factory form so the main agent and subagents each get their own instance."""
    return (HumanInTheLoopMiddleware(interrupt_on=_DANGEROUS_TOOLS),)


register_harness_profile(
    "anthropic:claude-opus-4-7",
    HarnessProfile(extra_middleware=_build_hitl_middleware),
)


# ---------------------------------------------------------------------------
# Pattern 2: helper-only — caller composes the HITL middleware themselves.
#
# Use when the *tool list* requiring HITL is app-specific (varies between
# agents) but the *rejection wording* should be consistent across the org.
# This is also the right choice when calling code already supplies its own
# `HumanInTheLoopMiddleware` and you do not want a second one appended.
# ---------------------------------------------------------------------------


def rejection_response_for(provider: str) -> RejectionResponseFactory | None:
    """Return the org-standard `rejection_response` factory for a provider.

    `None` is returned for providers without a tuned default; the caller's
    `HumanInTheLoopMiddleware` then falls back to the upstream default
    (`status="error"`, generic content).
    """
    if provider in {"anthropic", "openai"}:
        return _no_retry_rejection
    return None


# Caller-side composition — kept here as illustration only; in real use
# this lives wherever the app builds its agent.

_caller_facing_hitl = HumanInTheLoopMiddleware(
    interrupt_on={
        "payments.transfer": {
            "allowed_decisions": ["approve", "reject"],
            "rejection_response": rejection_response_for("anthropic"),
        },
    },
)


# ---------------------------------------------------------------------------
# Demonstration entry point.
#
# Skipped when run without an API key set; the goal of this file is to
# illustrate the wiring, not to act as an integration test.
# ---------------------------------------------------------------------------


def main() -> None:
    import os

    if not os.getenv("ANTHROPIC_API_KEY"):
        print(
            "ANTHROPIC_API_KEY not set — registration only. The HarnessProfile "
            "registered for 'anthropic:claude-opus-4-7' will contribute "
            "`HumanInTheLoopMiddleware(interrupt_on=_DANGEROUS_TOOLS)` to every "
            "create_deep_agent call against that model."
        )
        return

    from deepagents import create_deep_agent

    # Pattern 1 in action: no explicit HITL wiring, profile contributes it.
    agent = create_deep_agent(
        model="anthropic:claude-opus-4-7",
        tools=[],
        system_prompt="You are a careful assistant.",
    )
    print("Built agent with profile-contributed HITL middleware:", agent)


if __name__ == "__main__":
    main()
