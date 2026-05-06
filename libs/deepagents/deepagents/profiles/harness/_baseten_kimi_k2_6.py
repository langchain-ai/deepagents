"""Built-in Moonshot Kimi K2.6 (via Baseten) harness profile.

Layers a system-prompt suffix and a `compact_conversation` description
override onto `baseten:moonshotai/Kimi-K2.6` to counter the failure
modes observed in the
[GHA evals run](https://github.com/langchain-ai/deepagents/actions/runs/25403883357):

- Multi-turn execution discipline (planning, no redundant reads,
  user-facing communication on every turn) — addresses the
  `tau2_airline` and `bfcl multi_turn_*` failures, the dominant
  cluster in the run.
- Followup discipline — addresses the `vague_send_report` and
  `detailed_calendar_brief` failures where the model re-asked for
  details the user had already specified.
- Doc grounding — addresses the `nexus_*` failures where the model
  invented syntax (pipe operators, missing wrappers) instead of
  following the provided `api_reference.md` / `syntax.md`.
- `compact_conversation` description override — Kimi never reached for
  the tool in either `test_compact_tool_*` case; the rewritten
  description spells out the trigger conditions imperatively rather
  than as a soft "use proactively" hint.

The suffix is appended to whatever `base_system_prompt` is ultimately
assembled for the agent, so it layers cleanly on top of user- or
SDK-provided base prompts without fighting them. This profile is
prototype-quality — derived from a single eval run — and is expected to
be tuned further as additional Kimi runs accumulate.
"""

# ruff: noqa: E501
# Prompt sections are single lines by design to keep tag boundaries
# obvious in agent transcripts; hard-wrapping them would also mean
# wrapping mid-sentence in ways that obscure the rule the model is
# meant to follow.

from types import MappingProxyType

from deepagents.profiles.harness.harness_profiles import (
    HarnessProfile,
    _register_harness_profile_impl,
)

_SYSTEM_PROMPT_SUFFIX = """\
<plan_before_mutate>
Before calling any tool that modifies state — bookings, reservations, certificates, files, messages, or any database write — first complete the reads you need, then write a numbered plan listing the exact mutations and their arguments. Begin mutating only when reads are complete and the plan is final. Do not interleave reads and writes once the plan is set; if a write reveals new state that invalidates the plan, stop, re-read, and produce a fresh plan before continuing.
</plan_before_mutate>

<no_redundant_reads>
If a tool has already returned a value in this conversation, do not call the same tool with the same arguments again. Refer back to the prior observation. Re-fetching information you already have wastes turns, can cause stale-data inconsistency, and exhausts the model-call budget before you reach the mutation step.
</no_redundant_reads>

<communicate_each_turn>
After every tool call or batch of parallel tool calls, send a one- or two-sentence message to the user describing what you found and what you will do next. Silence between tool calls causes the user to disengage and end the conversation before the task is complete. Even brief acknowledgements ("got the user details — searching flights now") count and are required.
</communicate_each_turn>

<respect_user_specifications>
Read the user's most recent message carefully and treat any specific value they provide as fully specified. If the user said "every week", do not ask which day; if they said "daily", do not ask what time; if they specified a count, range, or named entity, do not re-ask for it. Ask only about facts the user did not provide. When the task asks for a single focused followup, ask exactly one question — do not append additional questions, even if they would be useful.
</respect_user_specifications>

<ground_in_provided_docs>
When a task supplies reference materials (e.g. `api_reference.md`, `syntax.md`, schemas under `/cases/`), follow the documented syntax exactly. Do not invent operators, function-call shapes, or composition idioms — pipe operators (`|>`), threading macros, method chaining, or wrappers — that are not present in those docs. If unsure how to compose a call, quote the relevant pattern from the docs verbatim before writing your answer.
</ground_in_provided_docs>"""
"""Text appended to the assembled base system prompt.

Each section maps to a distinct failure cluster from the GHA run:

- `<plan_before_mutate>`: db_state mismatch on tau2_airline tasks
    14/23/27/33/44 and bfcl multi_turn_composite_97/199 / miss_func_55.
- `<no_redundant_reads>`: redundant `get_reservation_details` x9 /
    `get_flight_status` x10 patterns observed in tau2 task_27 and
    task_44 trajectories.
- `<communicate_each_turn>`: tau2 task_7 hit COMM=0.00 / `user_stop`
    despite a perfect 5/5 actions match.
- `<respect_user_specifications>`: `vague_send_report` and
    `detailed_calendar_brief` failed because the agent re-asked for
    schedule details the user had already specified.
- `<ground_in_provided_docs>`: all three `nexus_*` failures
    (`nvd_nested_13`, `placesapi_15`, `multiversemath_18`) involved the
    model inventing syntax not present in the supplied docs.
"""


_COMPACT_CONVERSATION_DESCRIPTION = (
    "Compact the conversation by summarizing older messages into a concise "
    "summary. Required whenever you are asked to summarize a large file "
    "(roughly >300 lines), switch to an unrelated new task, or when the "
    "conversation has accumulated many tool results. Calling this tool "
    "frees up the context window so you have room to complete the current "
    "task; not calling it when the conversation is long will cause you to "
    "exhaust the model-call budget before finishing. This tool takes no "
    "arguments."
)
"""Override for `compact_conversation`'s description.

The stock description softly suggests "use this proactively when the
conversation is getting long." Kimi K2.6 never picked up on the hint
in `test_compact_tool_new_task` or `test_compact_tool_large_reads` —
both failures had the agent stuck in a `read_file` loop until the run
limit fired. The rewritten description names the three concrete
triggers (large file summary, task switch, accumulated tool results)
imperatively and spells out the consequence of skipping the call.
"""


_TOOL_DESCRIPTION_OVERRIDES = MappingProxyType(
    {
        "compact_conversation": _COMPACT_CONVERSATION_DESCRIPTION,
    },
)
"""Per-tool description replacements applied to this profile."""


def register() -> None:
    """Register the built-in Kimi K2.6 harness profile."""
    _register_harness_profile_impl(
        "baseten:moonshotai/Kimi-K2.6",
        HarnessProfile(
            system_prompt_suffix=_SYSTEM_PROMPT_SUFFIX,
            tool_description_overrides=_TOOL_DESCRIPTION_OVERRIDES,
        ),
    )
