"""Built-in GPT-6.1 Sol harness profile.

Replaces the `task` tool and general-purpose subagent descriptions for
`openai:gpt-6.1-sol` with selective delegation guidance: keep routine work
with the parent and reserve subagents for substantial, independent work.
"""

# ruff: noqa: E501
# Model-facing prompt lines are kept unwrapped.

from deepagents.profiles.harness.harness_profiles import (
    GeneralPurposeSubagentProfile,
    HarnessProfile,
    _register_harness_profile_impl,
)

_TASK_DESCRIPTION = """Launch an ephemeral subagent to handle a well-scoped task.

Available agent types and the tools they have access to:
{available_agents}

Specify subagent_type to select the agent. Usage notes:
- Select an agent according to its description and the user's explicit requirements.
- Launch multiple agents concurrently when their tasks are independent and their write scopes do not conflict, using a single message with multiple tool calls.
- Each invocation is stateless by default: the agent sees only the prompt you give it and returns a single final report. Put sufficient context in the prompt — unless an agent type below says it inherits your conversation instead. State a bounded deliverable, relevant constraints, and the expected result.
- The agent's report is not shown to the user; relay a summary yourself.
- Tell the agent whether to create content, analyze, or only research, since it can't necessarily see the user's intent unless it inherits your conversation, as noted per agent type below.
- If an agent's description says to use it proactively, do so without waiting to be asked."""
"""`task` tool description template; `{available_agents}` lists the subagents."""

_GENERAL_PURPOSE_DESCRIPTION = (
    "General-purpose agent for substantial, well-scoped work that benefits from a separate investigation, "
    "an independent review of a concrete risk, or parallel work with a distinct deliverable. "
    "Prefer direct tools for routine searches, small edits, and tightly coupled next steps. "
    "Complexity or uncertainty about finding a file is not sufficient reason to delegate. "
    "Keep discovery, implementation, and relevant checks for one small change with one implementation owner "
    "instead of launching a fresh agent for each phase. A targeted independent review can still be useful. "
    "This agent has the same capabilities as the main agent."
)
"""General-purpose subagent description shown in the `task` tool listing."""


def register() -> None:
    """Register the built-in GPT-6.1 Sol harness profile."""
    _register_harness_profile_impl(
        "openai:gpt-6.1-sol",
        HarnessProfile(
            tool_description_overrides={"task": _TASK_DESCRIPTION},
            general_purpose_subagent=GeneralPurposeSubagentProfile(description=_GENERAL_PURPOSE_DESCRIPTION),
        ),
    )
