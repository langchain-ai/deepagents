"""Built-in Fireworks GLM-5p1 harness profile.

Registers a `HarnessProfile` for `fireworks:accounts/fireworks/models/glm-5p1`
that targets two model-fault clusters surfaced by the deep-agents
eval suite on this model:

- *Plan / stop discipline* — the model loops on read-only calls, drops
  required final mutations, or repeats a successful mutation. Most
  visible on tau2-airline tasks where actions match the expected
  trajectory but the agent fails to converge.
- *Argument fidelity on mutating tools* — the model selects the right
  tool but routes a wrong target ID, an inverted state value, or an
  empty/defaulted string parameter. Most visible on tau2 db-state
  mismatches, BFCL state mismatches, and HITL tests asserting on
  specific tool-call arguments.

A third cluster — *output channel routing*, where GLM-5p1 routes
short final answers into `additional_kwargs.reasoning_content` with
empty `content` — is **not** addressed in this suffix. An earlier
revision included an "Output Channel" section that explicitly told the
model never to leave `content` empty. A local A/B / ablation study
(`/tmp/ablation_results.tsv`, N=5 per cell, 7 variants, 2 tests, 70
runs) showed that section was the *sole* cause of stable regressions on
`test_single_tool_get_food_calories` and `test_single_tool_get_user_email`
(0/10 with the rule, 10/10 without it). The rule appears to prime the
model to bifurcate its output into reasoning vs content channels and
land the answer in the wrong one — the opposite of its intent. The
direct-API success it showed in isolation (8/8) does not transfer to
the deepagents harness's longer prompt. Cluster E should be closed at
the integration layer (a middleware that copies `reasoning_content` →
`content` when content is empty and there are no tool calls), not at
the prompt layer.

The suffix is appended to whatever `base_system_prompt` is ultimately
assembled for the agent, so it layers cleanly on top of user- or
SDK-provided base prompts without fighting them.

This module exists as the audit anchor for the model: its presence
documents that GLM-5p1 has been profiled against eval data. If a
future GLM revision changes its training or output discipline, add
the new key here (mirroring the Codex `_CODEX_MODEL_SPECS` pattern)
rather than reusing this exact key.
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

## Parallel Tool Use

- When tool calls do not depend on each other's outputs, batch them \
into a single response (for example, reading multiple reference \
files at once).
- Do not parallelize tool calls when one depends on the result of \
another. Never use placeholders or guess missing parameters.

## Stop Conditions

- A task is complete when every requested action has succeeded, every \
TODO created via `write_todos` is resolved (done, blocked with a \
one-sentence reason, or cancelled), and you have communicated the \
outcome to the user. Stop the turn when these hold.
- Do not repeat a successful mutation to "double-check" — repeated \
calls accumulate side effects on the underlying system.
- Before finishing, write a brief confirmation of what changed (or \
what did not change) so the user does not have to re-derive it from \
the tool trace."""
"""Text appended to the assembled base system prompt."""


def register() -> None:
    """Register the built-in Fireworks GLM-5p1 harness profile."""
    _register_harness_profile_impl(
        _GLM_5P1_MODEL_SPEC,
        HarnessProfile(system_prompt_suffix=_SYSTEM_PROMPT_SUFFIX),
    )
