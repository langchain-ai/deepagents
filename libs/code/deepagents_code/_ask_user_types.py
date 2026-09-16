"""Lightweight types and shared rendering for the ask-user interrupt protocol.

Extracted from `ask_user` so `textual_adapter` can import `AskUserRequest` at
module level — and `app` can reference the types at type-check time — without
pulling in the langchain middleware stack.

This is the shared wire format for the tool, the TUI, and `auto_mode`, colocated
so no consumer has to import another. Each symbol's own docstring records which
side produces it and which reads it.
"""

from __future__ import annotations

import json
from typing import Annotated, Literal, NotRequired, assert_never, get_args

from pydantic import AfterValidator, Field
from typing_extensions import TypedDict

QuestionType = Literal["text", "multiple_choice", "multi_select"]
"""Supported `ask_user` question types."""

QUESTION_TYPES: frozenset[str] = frozenset(get_args(QuestionType))
"""Runtime membership view of `QuestionType`.

Derived via `get_args` so the membership checks in `ask_user` and `auto_mode`
cannot drift from the alias when a new question type is added. That drift would
be silent: an unrecognized type makes `_ask_user_question_count` return `None`,
which drops the user's answers as same-turn authorization without any error.
"""


def _requires_choices(question_type: QuestionType) -> bool:
    """Return whether `question_type` needs a non-empty `choices` list.

    Args:
        question_type: A member of `QuestionType`.

    Returns:
        True if questions of this type must define `choices`.
    """
    if question_type == "text":
        return False
    if question_type in {"multiple_choice", "multi_select"}:
        return True
    # Exhaustiveness guard: adding a `QuestionType` member without deciding here
    # whether it needs choices fails type checking rather than silently landing
    # on the wrong side of `CHOICE_QUESTION_TYPES`.
    assert_never(question_type)


CHOICE_QUESTION_TYPES: frozenset[str] = frozenset(
    question_type
    for question_type in get_args(QuestionType)
    if _requires_choices(question_type)
)
"""Question types that require a non-empty `choices` list.

Derived from `_requires_choices` rather than written out, so it cannot omit a
new choice-based `QuestionType` member. Note that requiring `choices` is not the
same as being *rendered* as a choice list: the TUI has its own exhaustive
dispatch in `_QuestionWidget.compose`.
"""


def encode_multi_select_answer(values: list[str]) -> str:
    """Encode the selected values of a `multi_select` answer for the wire.

    The answer stays one string so `answers: list[str]` keeps one slot per
    question, positionally matched. JSON is self-delimiting, so values carrying
    commas, quotes, or newlines round-trip exactly.

    The encoding is opaque to a consumer that only moves the answer around. A
    consumer that tests whether the user answered at all must go through
    `ask_user_answer_is_empty`: an unselected `multi_select` encodes as the
    *truthy* string `[]`, so a bare `.strip()` reads it as answered.

    Args:
        values: Selected values — toggled predefined choices in choice-list
            order, then custom Other values in slot order.

    Returns:
        The values as a JSON array, e.g. `["a", "b"]`. An empty selection
        encodes as `[]`. The result never contains a line feed or carriage
        return, so it cannot split a blank-line-separated transcript block.
        `ensure_ascii=False` does pass U+2028 and U+2029 through literally, so
        do not read this as "the result is a single line to every consumer" —
        the guarantee is about the block separator only.
    """
    return json.dumps(values, ensure_ascii=False)


def decode_multi_select_answer(raw: str) -> list[str] | None:
    """Decode a `multi_select` answer produced by `encode_multi_select_answer`.

    Args:
        raw: The raw answer string for a `multi_select` question. Must already
            be a `str`: a non-string raises `TypeError` out of `json.loads`
            rather than returning `None`. Every call site establishes that
            first — `_parse_answers` coerces non-string answers,
            `_validated_ask_user_answers` checks `isinstance(answer, str)`, and
            the display path passes a slice of the transcript.

    Returns:
        The selected values, or `None` when `raw` is not a JSON array of
            strings. Callers choose the policy for `None`; the one thing none of
            them may do is fall back to splitting on `", "`, which mis-splits a
            value that contains a comma.
    """
    try:
        decoded = json.loads(raw)
    except (json.JSONDecodeError, RecursionError):
        return None
    if not isinstance(decoded, list) or not all(
        isinstance(value, str) for value in decoded
    ):
        return None
    return decoded


def ask_user_answer_is_empty(answer: str, question_type: object) -> bool:
    """Return whether an `ask_user` answer counts as "no answer".

    The single definition of emptiness for an answer, shared by the TUI (which
    blocks submitting a required question) and by the Auto authorization path
    (which keeps unanswered questions out of the consent evidence). Both sides
    must agree: an answer the TUI accepts as empty must not read as consent
    evidence downstream.

    A bare `.strip()` is wrong for `multi_select`, whose empty encoding is the
    truthy string `[]`. Every other type is empty when it is blank.

    Args:
        answer: The raw answer string for the question.
        question_type: The paired question's `type`. Anything other than
            `'multi_select'` — including a malformed or missing type — takes the
            blank test.

    Returns:
        `True` when the question has no substantive answer.
    """
    if question_type == "multi_select":
        # A malformed encoding counts as empty, which is the fail-closed side of
        # both call sites: the TUI re-prompts a *required* question (an optional
        # one submits unchanged), and Auto withholds the answer from the consent
        # evidence rather than passing an undecodable string to the classifier.
        #
        # This makes emptiness asymmetric across types for the `(cancelled)` and
        # `(error: ...)` placeholders, which are not JSON: on a `multi_select`
        # they read as empty and are withheld from the evidence, while the same
        # placeholder on a `text` question becomes a row. Withholding is the
        # safe direction, so the asymmetry is acceptable — but it is real, and a
        # future caller counting answered questions across types will see it.
        return not decode_multi_select_answer(answer)
    return not answer.strip()


def _validate_question_text(text: str) -> str:
    """Reject a `question` that is empty or all whitespace.

    This is the only rule that *rejects* blank question text.
    `messages._ask_user_question_count` also tests it, but degrades to a count
    of zero instead of rejecting.

    Do not reorder this annotation and the `Field` beside it. As the inner
    annotation it runs before `min_length=1`, which therefore never rejects
    anything — that constraint is kept only because it is what puts
    `minLength: 1` in the JSON schema the model reads. Swapping the two moves
    the empty-string rejection onto `min_length` and changes the error the
    model sees from this function's message to `string_too_short`. A string
    like `"   "` would render as a visually blank prompt.

    Args:
        text: The parsed `question` text to check.

    Returns:
        The same `text`, unchanged.

    Raises:
        ValueError: If `text` has no non-whitespace character.
    """
    if not text.strip():
        msg = "question text must not be blank"
        raise ValueError(msg)
    return text


def _validate_choice(choice: Choice) -> Choice:
    """Reject a choice whose `value` is blank.

    A blank value would render as an unlabelled option the user can select but
    whose answer reads as "no answer".

    Attached to the item annotation inside `Question.choices`, not to `Choice`
    itself, so `TypeAdapter(Choice)` does not apply it.

    Callers must pass a parsed `Choice`. A missing `value` never reaches here —
    it is a required key, so pydantic rejects it as `choices.N.value: Field
    required` before the choice-level validators run. On a raw dict this would
    raise `KeyError`, which is not a `ValidationError` and would halt the run.

    Args:
        choice: The parsed `Choice` to check.

    Returns:
        The same `choice`, unchanged.

    Raises:
        ValueError: If `value` is blank or whitespace-only.
    """
    if not choice["value"].strip():
        msg = f"choice has a blank 'value': {choice!r}"
        raise ValueError(msg)
    return choice


class Choice(TypedDict):
    """A single choice option for a multiple choice or multi-select question."""

    value: Annotated[
        str,
        Field(
            description=(
                "The display label for this choice. Also the text returned as "
                "the answer when this choice is selected. A 'multi_select' answer "
                "is a JSON array, so a value may contain commas, quotes, and "
                "newlines; JSON escaping keeps it exact. A 'multiple_choice' "
                "value is returned on its own with no escaping, so keep that "
                "one to a single line."
            )
        ),
    ]


def _validate_question(question: Question) -> Question:
    """Apply the cross-field rules of a single question.

    These are the rules no single field type can express: choice questions need
    a non-empty `choices` list, and non-choice questions must not define one.

    `Literal` already rejects an unknown `type` and `strict=True` a non-boolean
    `required` before this runs, so they are not re-checked here.

    Args:
        question: The parsed `Question` to check.

    Returns:
        The same `question`, unchanged.

    Raises:
        ValueError: If the question violates one of the rules above.
    """
    question_type = question["type"]
    question_text = question["question"]
    choices = question.get("choices")
    if question_type in CHOICE_QUESTION_TYPES:
        if not choices:
            msg = (
                f"{question_type} question {question_text!r} requires a "
                f"non-empty 'choices' list"
            )
            raise ValueError(msg)
    elif choices:
        msg = f"{question_type} question {question_text!r} must not define 'choices'"
        raise ValueError(msg)
    return question


class Question(TypedDict):
    """A question to ask the user."""

    question: Annotated[
        str,
        AfterValidator(_validate_question_text),
        Field(description="The question text to display.", min_length=1),
    ]

    type: Annotated[
        QuestionType,
        Field(
            description=(
                "Question type. 'text' for free-form input, 'multiple_choice' for "
                "picking exactly one predefined option, 'multi_select' for picking "
                "one or more predefined options. Both choice types always append an "
                "'Other' free-form option automatically; multi-select can accept "
                "multiple custom Other values (filling one reveals another). A "
                "'multi_select' answer comes back as a JSON array of the selected "
                'values (including any custom Other text), e.g. ["a", "b"]; if '
                "nothing is selected on an optional question the answer is []."
            )
        ),
    ]

    choices: NotRequired[
        Annotated[
            list[Annotated[Choice, AfterValidator(_validate_choice)]],
            Field(
                description=(
                    "Options for 'multiple_choice' and 'multi_select' questions. "
                    "Every choice needs a non-empty 'value'. An 'Other' free-form "
                    "option is always appended automatically for both types; "
                    "multi-select may collect multiple custom Other values. A "
                    "'multi_select' answer is a JSON array of the selected "
                    "values, so values (including custom Other text) may "
                    "contain commas, quotes, and newlines."
                )
            ),
        ]
    ]

    required: NotRequired[
        Annotated[
            bool,
            # `strict=True` so a stringly-typed `"false"` is rejected at the tool
            # boundary rather than coerced. Without it pydantic accepts the string,
            # the prompt renders, and then `_ask_user_question_count` — which reads
            # the *raw* tool args and requires a real bool — returns `None`, which
            # drops every answer in the call as same-turn authorization with no
            # error. Strict here keeps the two boundaries agreeing, and makes the
            # coercion loud instead of silently expensive.
            Field(
                description=(
                    "Whether the user must answer. Defaults to true if omitted."
                ),
                strict=True,
            ),
        ]
    ]


ValidatedQuestion = Annotated[Question, AfterValidator(_validate_question)]
"""A `Question` with its cross-field rules applied during pydantic parsing.

Only the cross-field `choices` rules are scoped to this alias. The field-level
rules on `Question.question` and on `Question.choices` are unconditional, so
they also apply wherever `Question` itself is pydantic-parsed — notably
`TypeAdapter(AskUserRequest)` in `tui.textual_adapter`, which re-validates the
interrupt payload client-side and re-raises on failure. Keep that in mind
before adding another field-level rule here.

Where a `ValueError` from one of these validators ends up depends on the path.
On the tool path it becomes a tool-call `ValidationError`, which `ToolNode`
turns into an error `ToolMessage` the model can correct and retry from. On the
client re-validation path there is no tool call and `textual_adapter` logs and
re-raises instead."""


def _validate_questions(questions: list[ValidatedQuestion]) -> list[ValidatedQuestion]:
    """Reject an empty `questions` list.

    Per-question rules live on `ValidatedQuestion`; this covers the one rule
    about the list itself. Attached both to the tool's `questions` parameter
    and to `AskUserRequest.questions`, so the client re-validation boundary
    rejects an empty list too — `AskUserMenu([])` would otherwise build a
    titled prompt with no question widgets and nothing focusable.

    Args:
        questions: The parsed `questions` argument to check.

    Returns:
        The same `questions`, unchanged.

    Raises:
        ValueError: If the list is empty.
    """
    if not questions:
        msg = "ask_user requires at least one question"
        raise ValueError(msg)
    return questions


class AskUserRequest(TypedDict):
    """Request payload sent via interrupt when asking the user questions."""

    type: Literal["ask_user"]
    """Discriminator tag, always `'ask_user'`."""

    questions: Annotated[list[ValidatedQuestion], AfterValidator(_validate_questions)]
    """Questions to present to the user.

    `ValidatedQuestion` rather than `Question`, and carrying
    `_validate_questions`, so every rule the tool applies also applies where
    `tui.textual_adapter` re-validates this payload. A choice question with no
    `choices` would otherwise reach the client and degrade to a text box, and
    an empty list would render a prompt with no questions in it.
    """

    tool_call_id: str
    """ID of the originating tool call, used to route the response back."""


ASK_USER_AUTHORIZATION_METADATA_KEY = "deepagents_code_ask_user_authorization"
MAX_ASK_USER_AUTHORIZATION_ANSWER_CHARS = 4000
# These limits bound the receipt-anchored question/answer evidence copied into
# Auto's classifier prompt. Questions are model-generated and otherwise have no
# schema length constraint, so they must not be able to exhaust that context.
# The answer budget is measured on the answer as it travels the wire, which for
# `multi_select` is the JSON encoding: escaping and quoting count toward the
# limit, so a selection can cross it while its decoded values would not.
# Crossing it withholds the receipt, which is the fail-closed direction.
MAX_ASK_USER_AUTHORIZATION_QUESTION_CHARS = 4000
MAX_ASK_USER_AUTHORIZATION_QUESTION_TOTAL_CHARS = 8000


class AskUserAuthorizationReceipt(TypedDict):
    """Trusted same-turn authorization recorded after an ask_user response."""

    version: Literal[1]
    thread_id: str
    turn_id: str
    tool_call_id: str
    answers: list[str]


class AskUserAnswered(TypedDict):
    """Widget result when the user submits answers."""

    type: Literal["answered"]
    """Discriminator tag, always `'answered'`."""

    answers: list[str]
    """User-provided answers, one per question."""


class AskUserCancelled(TypedDict):
    """Widget result when the user cancels the prompt."""

    type: Literal["cancelled"]
    """Discriminator tag, always `'cancelled'`."""


AskUserWidgetResult = AskUserAnswered | AskUserCancelled
"""Discriminated union for the ask_user widget Future result."""


ASK_USER_NOTHING_SELECTED = "(nothing selected)"
"""Display-only placeholder for a `multi_select` the user left unselected.

Distinct from `ASK_USER_NO_ANSWER`, which marks a *missing* answer: this one
marks an answer that arrived and selected nothing. Only
`render_ask_user_transcript_for_display` produces it, and only for a human — the
transcript the model reads keeps the `[]` encoding.

In-band like the other placeholders, so a `multi_select` value spelled
`(nothing selected)` renders indistinguishably from an empty selection. The
result feeds no trust decision, so this costs legibility in a corner case
rather than correctness.
"""

ASK_USER_NO_ANSWER = "(no answer)"
"""Placeholder for a question with no positionally matching answer.

Unreachable from `_parse_answers` today: every branch there emits exactly
`len(questions)` answers, and the answered branch rejects a mismatch outright
rather than formatting it. Kept as a guard for any future caller that formats a
transcript without that check. A question the user deliberately left blank
renders as an empty `A:` line instead, never as this placeholder.
"""

ASK_USER_CANCELLED_ANSWER = "(cancelled)"
"""Placeholder recorded for every question when the user cancels the prompt."""

ASK_USER_ERROR_ANSWER_PREFIX = "(error: "
"""Prefix of the placeholder recorded for every question when the prompt fails.

The full placeholder is `(error: <detail>)`. Producer-side only: no production
code matches on it — `test_ask_user_types` is the only consumer, and it asserts
against this constant. The TUI derives its row summary from the recorded tool
status instead, because the placeholder is in-band — the prefix alone cannot
distinguish a failed prompt from a user who typed `(error: ...)` as their answer.
"""

AskUserRowSummary = Literal["User answered", "Question failed"]
"""The summaries an `ask_user` row may collapse to.

Narrows `ToolCallMessage.defer_success` so the transcript cannot be passed where
a summary belongs — the constraint that keeps user-typed answers out of
`tool.result` hook payloads.

Of the four `*_SUMMARY` constants below, only `ASK_USER_ANSWERED_SUMMARY` and
`ASK_USER_FAILED_SUMMARY` are members: the other two are hook bodies that no row
ever renders. `Literal[...]` cannot reference a plain string constant, so those
two values are restated here; both are annotated with this alias, so `ty` rejects
a reword that drifts. `test_row_summary_alias_matches_its_constants` pins it too.
"""

ASK_USER_ANSWERED_SUMMARY: AskUserRowSummary = "User answered"
"""One-line summary shown for an answered `ask_user` row before it is expanded.

Doubles as the `tool.result` hook payload's `tool_output` for an answered prompt
whose `ToolMessage` arrived, deliberately in place of the transcript, so
user-typed answers are not forwarded to hook scripts. When no `ToolMessage`
arrives the hook reports `ASK_USER_ANSWERED_NO_RESULT_SUMMARY` instead, while
the row still settles to this string. Rewording changes that hook contract as
well as the row; see `textual_adapter` and its `tool.result` tests.
"""

ASK_USER_ANSWERED_NO_RESULT_SUMMARY = "User answered (no tool result)"
"""The `tool.result` hook payload for an answered prompt that never completed.

Reported by `_dispatch_terminal_tool_result_hooks` when a teardown closes out a
row carrying a deferred success — the agent crashed, the stream ended, or the
user cancelled the turn. The guard there matches an already-settled row too, not
only one still awaiting its `ToolMessage`; see
`ToolCallMessage.deferred_success_output`.

The status stays `"success"` because the answers did reach the graph and only the
tool's own completion was lost, and `ask_user` results double as authorization
records; this distinct body is what lets an audit consumer tell "answers
delivered and the tool completed" from "answers delivered, then the turn died".
When the answers were never delivered at all the hook reports
`ASK_USER_ANSWERED_NOT_DELIVERED_SUMMARY` with `tool_status="error"` instead.
Hook contract: rewording it changes what those consumers see.
"""

ASK_USER_ANSWERED_NOT_DELIVERED_SUMMARY = "User answered (answers not delivered)"
"""The `tool.result` hook payload for answers the turn discarded before resuming.

Reported with `tool_status="error"` when a sibling question in the same batch was
cancelled: `textual_adapter` aborts the turn *before* `Command(resume=...)`, so
the resume payload — including this prompt's answers — is dropped, and the inline
widget is already unmounted, making the answers unrecoverable. The user answered,
but nothing downstream ever saw it.

Distinct from `ASK_USER_ANSWERED_NO_RESULT_SUMMARY`, and an error rather than a
success, precisely because `ask_user` results double as authorization records: a
`"success"` here would record an authorization that never took effect. Hook
contract: rewording it changes what audit consumers see.
"""

ASK_USER_CANCELLED_SUMMARY = "Question cancelled"
"""The `tool.result` hook payload's `tool_output` for the cancelled path.

Rewording it changes that hook contract. Not rendered on any row: a live cancel
calls `set_rejected` (which records no output), and a transcript of `(cancelled)`
placeholders from a non-TUI client is summarized from the recorded status like
any other, so it reads as `ASK_USER_ANSWERED_SUMMARY`. The dismissal banner in
`textual_adapter` deliberately does not use this constant, and no longer even
shares its wording — the banner says "dismissed" where this says "cancelled".
That divergence is intentional: the banner is user-facing prose free to be
reworded, this is the hook contract. Do not "de-duplicate" them.
"""

ASK_USER_FAILED_SUMMARY: AskUserRowSummary = "Question failed"
"""One-line summary shown for an `ask_user` prompt the middleware reported failed.

Also the `tool.result` hook payload's `tool_output` on that path — an answered
prompt whose `ToolMessage` came back with `status="error"` — deliberately in
place of the transcript, whose `(error: ...)` placeholders carry an arbitrary
detail string. Live widget failures are not this: they report their own error
text (see `textual_adapter`'s invalid-payload and cancel branches). Rewording
changes that hook contract as well as the row.
"""


def format_ask_user_error_answer(detail: str) -> str:
    """Render the placeholder answer recorded for every question on failure.

    Args:
        detail: Human-readable reason the prompt failed.

    Returns:
        The `(error: <detail>)` placeholder.
    """
    return f"{ASK_USER_ERROR_ANSWER_PREFIX}{detail})"


def format_ask_user_transcript(questions: list[Question], answers: list[str]) -> str:
    r"""Render questions and answers as the `Q:`/`A:` transcript.

    This is the text the `ask_user` tool returns to the model and persists in the
    thread. The TUI renders that authoritative text literally, except for the
    display-only re-render in `render_ask_user_transcript_for_display`, which
    anchors on the known question text and gives up rather than guess.

    Answers of every type are interpolated unescaped, so the encoding is not
    unambiguously decodable: an answer containing a blank line followed by a
    literal `Q: <text>\nA:` header is indistinguishable from a real block
    boundary. Only the model reads it that way today. Any future decoder must
    anchor on the known question text rather than on a generic `Q: ` pattern,
    or a crafted answer can fabricate an extra question/answer pair.

    For an answer `encode_multi_select_answer` produced, the JSON encoding does
    close the hazard above: no encoded value can carry a raw line feed, so it
    cannot fabricate a block boundary. This function cannot *rely* on that,
    because it does not validate answers and not every answer is encoded — the
    cancel and error paths in `_parse_answers` substitute `(cancelled)` and
    `(error: ...)` placeholders for every question whatever its type, and a
    non-TUI client resuming the interrupt can put arbitrary text in a
    `multi_select` slot.

    Args:
        questions: Questions that were asked. Callers must pass questions whose
            `question` text is a non-empty string; `_validate_question_text`,
            attached to `Question.question`, enforces that before interrupting.
            The empty default below only keeps a caller that skips validation
            from raising `KeyError`.
        answers: Answers, positionally matched to `questions`. A missing entry
            falls back to `ASK_USER_NO_ANSWER`; extra entries are dropped.

    Returns:
        Blank-line separated `Q: ...\nA: ...` blocks, one per question.
    """
    blocks = [
        f"Q: {question.get('question', '')}\n"
        f"A: {answers[i] if i < len(answers) else ASK_USER_NO_ANSWER}"
        for i, question in enumerate(questions)
    ]
    return "\n\n".join(blocks)


def render_ask_user_transcript_for_display(
    questions: list[Question], transcript: str
) -> str | None:
    r"""Re-render a transcript with `multi_select` answers unpacked for a human.

    The transcript is built for the model, where a `multi_select` answer is a
    JSON array. A person reading the same text sees the quoting and, worse, sees
    a multi-line custom answer flattened to a literal `\n`. This unpacks those
    answers back to one value per line, leaving every other answer untouched.

    Recovering the answers means splitting the transcript, which
    `format_ask_user_transcript` warns is not unambiguously decodable. So this
    anchors on the known question text, the mitigation that docstring
    prescribes, and gives up rather than guessing: a transcript that does not
    match the questions exactly returns `None` and must be rendered literally.
    An answer may itself contain a later `Q: ...` block verbatim (pasting
    transcript-like text into a text answer is legitimate), so a separator must
    be unique to be trusted — with duplicates, the first occurrence would
    swallow the real blocks into the wrong answer and misattribute the rest.
    The result is display-only and feeds no trust decision.

    Two things this deliberately does not do. It does not reject trailing
    content: the last answer runs to the end of the transcript, so junk appended
    after the final block is indistinguishable from answer text and is
    re-emitted verbatim. And it does not handle three or more identically worded
    questions — their shared anchor makes the separator non-unique, so a
    perfectly well-formed transcript falls back to literal rendering. Both are
    display-only outcomes.

    Args:
        questions: The questions that produced `transcript`, in order.
        transcript: Output of `format_ask_user_transcript` for those questions.

    Returns:
        The re-rendered transcript, or `None` when `transcript` does not parse
            as exactly these questions — including when no answer needed
            unpacking, so the caller keeps its literal rendering. The two
            cases share one sentinel because the caller wants the same literal
            rendering either way. A caller that wants to report an unexpected
            `None` should first check that a `multi_select` is present at all,
            which is what the TUI row does.
    """
    if not questions:
        return None
    anchors = [f"Q: {question.get('question', '')}\nA: " for question in questions]
    answers: list[str] = []
    position = 0
    for index, anchor in enumerate(anchors):
        if not transcript.startswith(anchor, position):
            return None
        position += len(anchor)
        if index + 1 == len(anchors):
            # The final answer runs to the end, so there is nothing left to
            # check: trailing junk cannot be told apart from answer text.
            answers.append(transcript[position:])
            break
        separator = f"\n\n{anchors[index + 1]}"
        end = transcript.find(separator, position)
        # `end` indexes the first of the two newlines, so `end + 2` is the `Q`
        # of the next anchor. Re-searching from there cannot re-find this match,
        # and cannot miss one that overlaps it: an occurrence starting at
        # `end + 1` would need `transcript[end + 2]` to be a newline, and it is
        # the `Q`. So the check establishes exactly one separator ahead.
        if end == -1 or transcript.find(separator, end + 2) != -1:
            return None
        answers.append(transcript[position:end])
        position = end + 2

    changed = False
    blocks: list[str] = []
    for question, answer in zip(questions, answers, strict=True):
        rendered = answer
        if question.get("type") == "multi_select":
            values = decode_multi_select_answer(answer)
            if values is not None:
                # One per line, so a value that itself spans lines stays
                # legible. This trades away distinguishability: `["a", "b"]` and
                # `["a\nb"]` render identically. Legibility is the point here and
                # nothing downstream parses the result, so the trade is fine.
                rendered = "\n".join(values) if values else ASK_USER_NOTHING_SELECTED
                changed = changed or rendered != answer
        blocks.append(f"Q: {question.get('question', '')}\nA: {rendered}")
    if not changed:
        return None
    return "\n\n".join(blocks)
