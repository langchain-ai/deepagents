"""Anthropic provider harness profile.

!!! warning

    This is an internal API subject to change without deprecation. It is not
    intended for external use or consumption.

Registers a provider-level profile for Anthropic models with a system prompt
suffix that improves agentic performance. Based on Anthropic's prompting best
practices for Claude's latest models.
"""
# ruff: noqa: E501

from deepagents.profiles._harness_profiles import _HarnessProfile, _register_harness_profile

_ANTHROPIC_SYSTEM_PROMPT_SUFFIX = """\
<parallel_tool_calls>
If you intend to call multiple tools and there are no dependencies between the calls, make all of the independent calls in the same message. Read multiple files simultaneously, run independent searches in parallel, or check multiple resources at once. Only sequence calls when one depends on the output of another. Never use placeholders for values you don't have yet — if a call depends on a prior result, wait for it.
</parallel_tool_calls>

<grounded_responses>
Never speculate about code, files, or system state you have not directly observed. If a task involves a specific file or function, read it before making claims about its contents or behavior. When uncertain about current state, investigate first — do not guess.
</grounded_responses>

<tool_result_reflection>
After receiving tool results, reflect on their quality and relevance before taking the next action. Use this reflection to determine whether your current approach is still the best path, rather than mechanically continuing a plan that new information may have invalidated.
</tool_result_reflection>

<decisive_execution>
When approaching a problem, choose a strategy and commit to it. Avoid revisiting decisions unless you encounter evidence that directly contradicts your reasoning. If weighing multiple approaches, pick the most promising one and execute — course-correct only if it fails.
</decisive_execution>"""
"""System prompt suffix appended for all Anthropic models.

Covers four areas that complement the base ``BASE_AGENT_PROMPT`` without
duplicating it:

* **Parallel tool calls** — explicit guidance to batch independent calls,
  the single largest latency improvement for Claude in agentic loops.
* **Grounded responses** — hard constraint against speculating about
  unobserved code/files (upgrades the softer "read relevant files" in the
  base prompt).
* **Tool-result reflection** — leverages Claude's interleaved thinking to
  re-evaluate the plan after each tool round.
* **Decisive execution** — prevents waffling between strategies, reducing
  wasted tokens without conflicting with the base prompt's "analyze *why*
  on failure" guidance.
"""

_register_harness_profile(
    "anthropic",
    _HarnessProfile(system_prompt_suffix=_ANTHROPIC_SYSTEM_PROMPT_SUFFIX),
)
