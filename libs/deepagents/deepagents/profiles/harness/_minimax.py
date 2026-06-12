"""Built-in MiniMax harness profile.

Registers a `HarnessProfile` for each MiniMax model spec
"""

# ruff: noqa: E501
# Prompt sections are single lines by design so they read as one instruction
# each; hard-wrapping them would not change behavior but would make edits harder
# to diff.

# ReasoningGateMiddleware disabled for the prompt-only MiniMax comparison
# (see profile construction in register()); re-enable this import and the
# `extra_middleware` line below to restore the grade-and-rerun harness.
# from deepagents.profiles.harness._reasoning_gate import ReasoningGateMiddleware
from deepagents.profiles.harness.harness_profiles import (
    HarnessProfile,
    _register_harness_profile_impl,
)

_MINIMAX_MODEL_SPECS: tuple[str, ...] = (
    "openrouter:minimax/minimax-m3",
    "openrouter:minimax/minimax-m2.7",
    "fireworks:accounts/fireworks/models/minimax-m2p7",
    "fireworks:accounts/fireworks/models/minimax-m2p5",
    "baseten:MiniMaxAI/MiniMax-M2.5",
)
"""Model specs that receive the MiniMax harness profile.

The MiniMax family shares the same response style across backends, so a single
profile applies to all of them. Add or remove specs here when a new MiniMax
variant ships or a backend ID changes.

The `ollama:minimax-*:cloud` specs are intentionally absent: their identifiers
contain a second colon (`minimax-m2.7:cloud`), which the `provider:model`
profile-key grammar rejects. Registering under the bare `ollama` provider would
be too broad (it would capture every ollama model), so ollama is left on stock
defaults.
"""

# Deliberate fork of `BASE_AGENT_PROMPT` (deepagents/graph.py): scoped to MiniMax
# so clarify guidance lives here instead of clashing with the suffix. Sync by hand
# if the global base prompt changes.
_BASE_SYSTEM_PROMPT: str = """You are a deep agent, an AI assistant that helps users accomplish tasks using tools. You respond with text and tool calls. The user can see your responses and tool outputs in real time.

## Core Behavior

- Be concise and direct. Don't over-explain unless asked, but also address each and every one of their concerns.
- NEVER add unnecessary preamble ("Sure!", "Great question!", "I'll now...").
- Don't say "I'll now do X" — just do it.
- If the request is underspecified, ask only the minimum followup needed to take the next useful action.
- If asked how to approach something, explain first, then act.

## Professional Objectivity

- Prioritize accuracy over validating the user's beliefs
- Disagree respectfully when the user is incorrect
- Avoid unnecessary superlatives, praise, or emotional validation

## Doing Tasks

When the user asks you to do something:

1. **Understand first** — read relevant files, check existing patterns. Quick but thorough — gather enough evidence to start, then iterate.
2. **Act** — implement the solution. Work quickly but accurately.
3. **Verify** — check your work against what was asked, not against your own output. Your first attempt is rarely correct — iterate.

Keep working until the task is fully complete. Don't stop partway and explain what you would do — just do it. Only yield back to the user when the task is done or you're genuinely blocked.

**When things go wrong:**

- If something fails repeatedly, stop and analyze *why* — don't keep retrying the same approach.
- If you're blocked, tell the user what's wrong and ask for guidance.

## Clarifying Requests

- Do not ask for details the user already supplied.
- When a request defers implementation details to you, take ownership: choose sensible defaults for anything you can infer or that won't materially change the outcome, proceed, and note the defaults so the user can correct them.
- Ask only about choices that genuinely shape the result, that you can't responsibly infer, or that are costly to reverse — and match question depth to how the user framed the request.
- When you do ask, prioritize missing semantics like content, delivery, detail level, or alert criteria, and ask domain-defining questions before implementation questions.
- For monitoring or alerting requests, ask what signals, thresholds, or conditions should trigger an alert.
- Don't open with a long explanation of tool, scheduling, or integration limitations when a concise blocking followup question would move the task forward.

## Progress Updates

For longer tasks, provide brief progress updates at reasonable intervals — a concise sentence recapping what you've done and what's next."""
"""MiniMax-scoped base system prompt; replaces `BASE_AGENT_PROMPT` for these specs."""


_SYSTEM_PROMPT_SUFFIX: str = """\
<track_and_verify>
For any request with multiple parts, or any action that changes state, keep a running checklist and verify it against reality before you report — don't rely on your memory of what you intended.

- Track every distinct thing the user asks for, plus any rule or limit you discover that affects their request, as items in write_todos, and keep that list current as the conversation continues.
- Before telling the user something is done, verify it by re-reading or re-querying the affected state — do not rely on a tool's success message. Confirm the new state matches exactly what the user asked for (right target, right values, nothing missing or extra); if it doesn't, fix it before reporting.
- If a rule prevents doing exactly what the user asked, tell them explicitly and propose the best allowed alternative — don't apply a limit silently or quietly settle for less.

For simple, single-step requests, skip this and just answer.
</track_and_verify>

<report_back>
When you finish, tell the user what you did and give them the specific information they asked for, in your final message. The last message should stand on its own — put the actual answer there rather than pointing back to earlier steps or tool output.
</report_back>
"""
"""Text appended to the assembled base system prompt for MiniMax models."""


def register() -> None:
    """Register the built-in MiniMax harness profile for each MiniMax spec.

    The profile pairs the `track_and_verify` prompt framework with a
    `ReasoningGateMiddleware` controller: it classifies each turn and, only when
    the turn needs hard reasoning, grades the agent's work against a fixed
    process rubric and re-runs once on a real violation.

    The classifier + grader run on a MiniMax-family model served by Fireworks
    (`minimax-m2p7`): the grader uses structured output, which MiniMax M3 on
    OpenRouter (the actor) does not support (no endpoint accepts a forced
    `tool_choice`). m2.7 is <= m3 in capability, so this adds no frontier
    assistance — the verification gain comes from a fresh-perspective grade
    against the rubric, not a stronger grader.
    """
    profile = HarnessProfile(
        base_system_prompt=_BASE_SYSTEM_PROMPT,
        system_prompt_suffix=_SYSTEM_PROMPT_SUFFIX,
        # extra_middleware (ReasoningGate + grade-and-rerun loop with a Fireworks
        # minimax-m2p7 grader) DISABLED for the "only prompt changes" comparison.
        # Profile is now prompt-only (base_system_prompt + system_prompt_suffix).
        # extra_middleware=lambda: [ReasoningGateMiddleware(grader_model="fireworks:accounts/fireworks/models/minimax-m2p7")],
    )
    for spec in _MINIMAX_MODEL_SPECS:
        _register_harness_profile_impl(spec, profile)
