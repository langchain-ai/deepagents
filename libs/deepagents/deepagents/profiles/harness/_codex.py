"""Built-in Codex harness profile.

Registers a `HarnessProfile` for each OpenAI Codex model spec with:

* A behavior-shaping `system_prompt_suffix` that aligns Deep Agents'
    runtime defaults with how Codex was trained to operate — autonomous
    senior engineer demeanor, bias to action, parallel tool use, and TODO
    hygiene; and
* A backend-aware `extra_middleware` factory that contributes an
    `_ApplyPatchMiddleware` so the agent can apply V4A diffs through the
    same filesystem backend `FilesystemMiddleware` uses. Codex models are
    trained to emit V4A `apply_patch` invocations; without the middleware
    they underperform on file-editing tasks.

The suffix is appended to whatever `base_system_prompt` is ultimately
assembled for the agent, so it layers cleanly on top of user- or
SDK-provided base prompts without fighting them.

Per-model keys (not the `"openai"` prefix) keep the default behavior of
non-Codex OpenAI models unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from deepagents.middleware._apply_patch import _ApplyPatchMiddleware
from deepagents.profiles.harness_profiles import (
    HarnessProfile,
    register_harness_profile,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from langchain.agents.middleware.types import AgentMiddleware

    from deepagents.backends.protocol import BACKEND_TYPES

_CODEX_MODEL_SPECS: tuple[str, ...] = (
    "openai:gpt-5.1-codex",
    "openai:gpt-5.2-codex",
    "openai:gpt-5.3-codex",
)
"""Model specs that receive the Codex harness profile.

All three variants share the same trained tool vocabulary and response
style, so a single suffix works across the family. Add or remove specs
here only when a new Codex variant ships with divergent training
expectations that warrant a different suffix.
"""

_CODEX_SYSTEM_PROMPT_SUFFIX: str = """\
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
- Avoid excessive looping or repetition. If you have already called a tool and \
received a valid response, use that information — do not re-invoke the same \
tool with identical arguments. If you find yourself re-reading, re-querying, \
or re-editing without clear progress, stop and summarize what is blocking you.

## Parallel Tool Use

- Before any tool call, decide ALL files and resources you will need.
- Batch reads, searches, and other independent operations into parallel tool \
calls instead of issuing them one at a time.
- Only make sequential calls when you truly cannot determine the next step \
without seeing a prior result.

## Plan Hygiene

- Before finishing, reconcile every TODO or plan item created via write_todos. \
Mark each as done, blocked (with a one-sentence reason), or cancelled. Do not \
finish with pending items."""
"""Runtime behavior guidance appended to every Codex agent's system prompt.

Scope is intentionally limited to runtime demeanor: autonomy, bias to
action, parallel tool use, and plan hygiene. Tool-specific guidance
(e.g. `apply_patch` file editing or `shell_command` vs. `execute`
aliasing) lives with its corresponding capability and is added in
follow-up work that introduces those capabilities to the harness.
"""
"""Text appended to the assembled base system prompt."""


def _apply_patch_factory(backend: BACKEND_TYPES) -> Sequence[AgentMiddleware[Any, Any, Any]]:
    """Contribute `_ApplyPatchMiddleware` bound to the agent's backend.

    Registered through `HarnessProfile.extra_middleware` as a
    backend-aware factory so the middleware uses the same backend as
    `FilesystemMiddleware` — `create_deep_agent` threads a single
    backend into every `extra_middleware` factory at assembly time.
    Sharing the backend is essential: `apply_patch` must read and
    write the same files the rest of the filesystem tools operate on.
    """
    return [_ApplyPatchMiddleware(backend=backend)]


def register() -> None:
    """Register the built-in Codex harness profile for each Codex spec."""
    profile = HarnessProfile(
        system_prompt_suffix=_CODEX_SYSTEM_PROMPT_SUFFIX,
        extra_middleware=_apply_patch_factory,
    )
    for spec in _CODEX_MODEL_SPECS:
        register_harness_profile(spec, profile)
