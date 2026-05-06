"""Built-in Fireworks GLM-5p1 harness profile.

Registers a `HarnessProfile` for `fireworks:accounts/fireworks/models/glm-5p1`
that targets three model-fault clusters surfaced by the deep-agents
eval suite on this model:

- *Output channel routing* — when `tools` are bound and the model
  produces a very short final answer without calling a tool, GLM-5p1
  served via Fireworks routes the answer into the `reasoning_content`
  field with empty `content`. Reproduces deterministically against
  the raw chat-completions endpoint, independent of the SDK adapter.
  An imperative rule at the top of the suffix moves direct API probes
  from 1/8 to 8/8; the rule has a ceiling inside deepagents' larger
  bound-tool prompt and a `reasoning_content` → `content` middleware
  is the deterministic close (deferred to v3 if needed).
- *Argument fidelity on mutating tools* — the model selects the right
  tool but routes a wrong target ID, an inverted state value, or an
  empty/defaulted string parameter. Most visible on tau2 db-state
  mismatches, BFCL state mismatches, and HITL tests asserting on
  specific tool-call arguments.
- *Plan / stop discipline* — the model loops on read-only calls,
  re-runs successful mutations, or drops a required final mutation.
  Most visible on tau2-airline tasks where actions match the expected
  trajectory but the agent fails to converge.

The suffix is appended to whatever `base_system_prompt` is ultimately
assembled for the agent, so it layers cleanly on top of user- or
SDK-provided base prompts without fighting them.

This module exists as the audit anchor for the model: its presence
documents that GLM-5p1 has been profiled against eval data. If a
future GLM revision changes its training or output discipline, add
the new key here (mirroring the Codex `_CODEX_MODEL_SPECS` pattern)
rather than reusing this exact key.

See `evals/fireworks_glm_5p1_profile_findings.md` at the repo root
for the v1 → v2 evaluation diff and rationale for each section.
"""

from deepagents.profiles.harness.harness_profiles import (
    HarnessProfile,
    _register_harness_profile_impl,
)

_GLM_5P1_MODEL_SPEC: str = "fireworks:accounts/fireworks/models/glm-5p1"
"""Exact `init_chat_model` spec for the Fireworks GLM-5p1 hosted model.

Matches both registry lookup paths in `_harness_profile_for_model`: the
caller-provided spec string and the `provider:identifier` key derived
from a pre-built `ChatFireworks` instance whose `model_name` is
`accounts/fireworks/models/glm-5p1`.
"""

_SYSTEM_PROMPT_SUFFIX: str = """\
## Output Channel

- Always begin your reply with the user-visible answer in plain text \
in the content field. Never leave content empty when you are not \
issuing a tool call. The reasoning channel is for internal thoughts \
only and must not carry the final answer.
- If the final answer is short (e.g. "4", "Paris", "yes"), wrap it \
in a brief sentence so content is non-empty — for example: \
"The answer is 4.".

## Tool Execution Discipline

- Before issuing a tool call that mutates state (creates, updates, \
cancels, sends, books, sets), restate in one sentence the target \
object and the intended change. This forces a deliberate check of \
the parameters before they go on the wire.
- Tool arguments must come directly from the conversation. Never \
default a string parameter to "" (empty), "latest", or a value from \
an earlier unrelated turn that the user did not explicitly carry \
forward. When a required argument is ambiguous, ask one targeted \
question instead of guessing.
- Do not re-issue a read-only tool call whose result is already \
visible above. Refer back to the prior result instead.
- Do not repeat a successful mutation to "double-check" — repeated \
calls accumulate side effects on the underlying system.

## Parallel Tool Use

- When tool calls do not depend on each other's outputs, batch them \
into a single response (for example, reading multiple reference \
files at once).
- Do not parallelize tool calls when one depends on the result of \
another. Never use placeholders or guess missing parameters."""
"""Text appended to the assembled base system prompt.

The order is deliberate: `Output Channel` comes first because GLM-5p1
defaults to routing single-token answers into `reasoning_content` when
tools are bound, and a late suffix rule loses to that prior in long
prompts. `Tool Execution Discipline` and `Parallel Tool Use` follow.

The closure-confirmation bullet that lived under a former
`Stop Conditions` section was removed in v2: it traded a small
multi-step win for a >0.10 conversation-category regression on
followup-quality evals (the agent emitted long numbered-question
lists). The remaining "do not double-execute" line was folded into
`Tool Execution Discipline`. The TODO-resolution rule already lives
inside the SDK base prompt's `write_todos` documentation, so it does
not need to be restated here."""


def register() -> None:
    """Register the built-in Fireworks GLM-5p1 harness profile."""
    _register_harness_profile_impl(
        _GLM_5P1_MODEL_SPEC,
        HarnessProfile(system_prompt_suffix=_SYSTEM_PROMPT_SUFFIX),
    )
