"""GPT-5.4 model-specific harness profile.

!!! warning

    This is an internal API subject to change without deprecation. It is not
    intended for external use or consumption.

Registers a `system_prompt_suffix` that addresses GPT-5.4-specific behavioral
tendencies documented in OpenAI's prompt guidance. The suffix layers on top of
the provider-level ``openai`` profile (inheriting ``use_responses_api: True``)
via the merge mechanism in `_get_harness_profile`.
"""

from deepagents.profiles._harness_profiles import _HarnessProfile, _register_harness_profile

_GPT54_SYSTEM_PROMPT_SUFFIX = """\
<tool_persistence_rules>
- Use tools whenever they materially improve correctness, completeness, or grounding.
- Do not stop early when another tool call would materially improve the result.
- If a tool returns empty or partial results, retry with a different strategy \
(alternate query wording, broader filters, or an alternate tool) before concluding \
that no results exist.
- Keep calling tools until the task is complete and verification passes.
</tool_persistence_rules>

<dependency_checks>
- Before taking an action, check whether prerequisite discovery, lookup, or \
retrieval steps are required.
- Do not skip prerequisite steps just because the intended final action seems obvious.
- If the task depends on the output of a prior step, resolve that dependency first.
</dependency_checks>

<completeness_contract>
- Treat the task as incomplete until all requested items are covered or \
explicitly marked as blocked.
- For lists, batches, or multi-step work: determine expected scope, track \
processed items, and confirm coverage before finalizing.
- If any item is blocked by missing data, state exactly what is missing.
</completeness_contract>

<verification_loop>
Before finalizing:
- Check correctness: does the output satisfy every requirement?
- Check grounding: are claims backed by provided context or tool outputs?
- Check formatting: does the output match the requested schema or style?
</verification_loop>"""


_register_harness_profile(
    "openai:gpt-5.4",
    _HarnessProfile(system_prompt_suffix=_GPT54_SYSTEM_PROMPT_SUFFIX),
)
