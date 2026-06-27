"""Built-in OpenAI Codex harness profile.

Registers a `HarnessProfile` for each OpenAI Codex model spec with a
behavior-shaping `system_prompt_suffix` that aligns Deep Agents'
runtime defaults with how Codex was trained to operate — autonomous
senior engineer demeanor, bias to action, parallel tool use, and TODO
hygiene when `TodoListMiddleware` is present.

The suffix is appended to whatever `base_system_prompt` is ultimately
assembled for the agent, so it layers cleanly on top of user- or
SDK-provided base prompts without fighting them.

Per-model keys (not the `"openai"` prefix) keep the default behavior of
non-Codex OpenAI models unchanged.
"""

from langchain.agents.middleware import TodoListMiddleware

from deepagents.profiles.harness.harness_profiles import (
    HarnessProfile,
    ProfileSuffixContext,
    _register_harness_profile_impl,
)

_CODEX_MODEL_SPECS: tuple[str, ...] = (
    "openai:gpt-5.1-codex",
    "openai:gpt-5.2-codex",
    "openai:gpt-5.3-codex",
)
"""Model specs that receive the Codex harness profile.

All three variants share the same trained response style, so a single
suffix works across the family. Add or remove specs here only when a
new Codex variant ships with divergent training expectations that
warrant a different suffix.
"""

_BASE_SUFFIX: str = """\
## Codex-Specific Behavior

- You are an autonomous senior engineer. Once given a direction, proactively \
gather context, plan, implement, and verify without waiting for additional \
prompts at each step.
- Persist until the task is fully handled end-to-end within the current turn \
whenever feasible. Do not stop at analysis or partial fixes; carry changes \
through implementation, verification, and a clear explanation of outcomes.
- Bias to action: default to implementing with reasonable assumptions. Do not \
end your turn with clarifications unless truly blocked.
- Do not communicate an upfront plan or status preamble before acting. Just act.

## Parallel Tool Use

- Before any tool call, decide ALL files and resources you will need.
- Batch reads, searches, and other independent operations into parallel tool \
calls instead of issuing them one at a time.
- Only make sequential calls when you truly cannot determine the next step \
without seeing a prior result."""
"""Codex suffix text that does not depend on optional middleware."""


_PLAN_HYGIENE: str = """\
## Plan Hygiene

- Before finishing, reconcile every TODO or plan item created via write_todos. \
Mark each as done, blocked (with a one-sentence reason), or cancelled. Do not \
finish with pending items."""
"""Codex suffix segment that requires `TodoListMiddleware` / `write_todos`."""


def _codex_suffix(ctx: ProfileSuffixContext) -> str:
    """Append Plan Hygiene only when `TodoListMiddleware` is in the stack.

    Plan Hygiene tells the model to use `write_todos`. A user (or layered
    profile) listing `TodoListMiddleware` in `HarnessProfile.excluded_middleware`
    strips that tool from the agent — keeping the guidance would point the
    model at something that no longer exists.
    """
    if TodoListMiddleware in ctx:
        return _BASE_SUFFIX + "\n\n" + _PLAN_HYGIENE
    return _BASE_SUFFIX


def register() -> None:
    """Register the built-in Codex harness profile for each Codex spec."""
    profile = HarnessProfile(system_prompt_suffix=_codex_suffix)
    for spec in _CODEX_MODEL_SPECS:
        _register_harness_profile_impl(spec, profile)
