"""Built-in Claude Fable 5 and Mythos 5 harness profile.

Layers Anthropic's universal Claude guidance onto
`anthropic:claude-fable-5` and `anthropic:claude-mythos-5`, plus shared
guidance for long-horizon completion, context continuity, and reader-oriented
final summaries.

The profile intentionally omits unconditional memory and asynchronous
delegation instructions because those capabilities depend on the middleware
and subagents configured by the caller. Refusal and provider fallback handling
also remain runtime concerns rather than prompt policy.

Sources:

- https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices
- https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-fable-5
"""

# ruff: noqa: E501
# The universal sections mirror Anthropic's published samples verbatim so they
# stay auditable against the other model-specific Claude profile modules.

from deepagents.profiles.harness.harness_profiles import (
    HarnessProfile,
    _register_harness_profile_impl,
)

_SYSTEM_PROMPT_SUFFIX = """\
<use_parallel_tool_calls>
If you intend to call multiple tools and there are no dependencies between the tool calls, make all of the independent tool calls in parallel. Prioritize calling tools simultaneously whenever the actions can be done in parallel rather than sequentially. For example, when reading 3 files, run 3 tool calls in parallel to read all 3 files into context at the same time. Maximize use of parallel tool calls where possible to increase speed and efficiency. However, if some tool calls depend on previous calls to inform dependent values like the parameters, do NOT call these tools in parallel and instead call them sequentially. Never use placeholders or guess missing parameters in tool calls.
</use_parallel_tool_calls>

<investigate_before_answering>
Never speculate about code you have not opened. If the user references a specific file, you MUST read the file before answering. Make sure to investigate and read relevant files BEFORE answering questions about the codebase. Never make any claims about code before investigating unless you are certain of the correct answer - give grounded and hallucination-free answers.
</investigate_before_answering>

<tool_result_reflection>
After receiving tool results, carefully reflect on their quality and determine optimal next steps before proceeding. Use your thinking to plan and iterate based on this new information, and then take the best next action.
</tool_result_reflection>

<long_horizon_completion>
Do not stop, propose a new session, or hand work off merely because the conversation has become long or you perceive context pressure. Continue making concrete progress until the task is complete or blocked on required user input.
</long_horizon_completion>

<final_summary_readability>
After an extended tool-driven run, write the final answer for a reader who may not have followed the working process. Lead with the outcome, use complete sentences, and explain important files, identifiers, decisions, and remaining blockers in plain language. Prefer clarity over terse working shorthand.
</final_summary_readability>"""
"""Text appended to the assembled base system prompt."""


def register() -> None:
    """Register the built-in Claude Fable 5 and Mythos 5 harness profile."""
    profile = HarnessProfile(system_prompt_suffix=_SYSTEM_PROMPT_SUFFIX)
    for model_spec in (
        "anthropic:claude-fable-5",
        "anthropic:claude-mythos-5",
    ):
        _register_harness_profile_impl(model_spec, profile)
