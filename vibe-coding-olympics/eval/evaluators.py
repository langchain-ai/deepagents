"""LLM-as-judge evaluators for vibe coding competition website scoring.

Each evaluator expects:
    inputs: The prompt given to the contestant.
    outputs: The generated HTML source.
    screenshot_b64: Base64-encoded PNG screenshot of the rendered page.

All evaluators use continuous scoring (0.0 to 1.0). The set is intentionally
decomposed — keep one signal per axis so regressions are diagnosable in
LangSmith rather than hidden inside a blended score.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from openevals.llm import create_llm_as_judge

if TYPE_CHECKING:
    from collections.abc import Callable

# Must support vision — evaluators send page screenshots as image_url content blocks.
_JUDGE_MODEL = "openai:gpt-5.4"

_ROLE_PREAMBLE = (
    "You are an expert web designer judging a 5-minute vibe coding competition. "
    "Contestants used an AI coding tool to generate a website from a short prompt. "
    "Calibrate your expectations accordingly — this was built in 5 minutes, not 5 days."
)

_SCORE_INSTRUCTION = (
    "Score from 0.0 (terrible) to 1.0 (exceptional for a 5-minute build)."
)


def _make_judge_prompt(
    criteria: str,
) -> Callable[..., list[dict[str, Any]]]:
    """Return a prompt callable for the given scoring criteria.

    Args:
        criteria: Axis-specific rubric inserted between context and scoring
            instruction.

    Returns:
        A callable compatible with the openevals prompt-callable contract.
    """

    def _prompt(
        *,
        inputs: str | dict | None = None,
        outputs: str | dict | None = None,
        screenshot_b64: str = "",
        **kwargs: object,  # noqa: ARG001  # required by openevals contract
    ) -> list[dict[str, Any]]:
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"{_ROLE_PREAMBLE}\n\n"
                            f"**Prompt given to the contestant:** {inputs}\n\n"
                            f"**HTML source:**\n```html\n{outputs}\n```\n\n"
                            f"{criteria}\n\n"
                            f"{_SCORE_INSTRUCTION}"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{screenshot_b64}",
                        },
                    },
                ],
            }
        ]

    return _prompt


_EVALUATOR_SPECS: list[tuple[str, str]] = [
    (
        "color",
        (
            "Evaluate the **color palette** only. Ignore layout, typography, "
            "and content for this axis.\n"
            "1. Cohesion — colors feel like a chosen palette, not random\n"
            "2. Contrast — foreground/background combinations are legible\n"
            "3. Appropriateness — palette matches the subject's tone\n"
            "4. Restraint — not too many competing hues"
        ),
    ),
    (
        "typography",
        (
            "Evaluate **typography** only. Ignore color, layout, and content.\n"
            "1. Readability — body copy is comfortable to read\n"
            "2. Hierarchy — heading/body sizes are clearly differentiated\n"
            "3. Consistency — font choices are stable across the page\n"
            "4. Character — type choices suit the subject"
        ),
    ),
    (
        "layout",
        (
            "Evaluate **layout, spacing, and composition**.\n"
            "1. Visual hierarchy — the most important thing reads first\n"
            "2. Alignment — content sits on a coherent grid\n"
            "3. Spacing rhythm — margins/paddings feel considered, not cramped or empty\n"
            "4. Balance — the page is not lopsided or unintentionally asymmetric\n"
            "5. Flow — the eye is guided through the page naturally"
        ),
    ),
    (
        "content_completeness",
        (
            "Evaluate **content completeness**.\n"
            "1. Meaningful, relevant text — not lorem ipsum or placeholders\n"
            "2. Expected sections are present for the chosen theme\n"
            "3. Logical information hierarchy\n"
            "4. Substance without filler\n"
            "5. Functional elements (links, buttons, CTAs) make sense"
        ),
    ),
    (
        "creativity",
        (
            "Evaluate **creativity and memorability**.\n"
            "1. Novel visual or interaction ideas\n"
            "2. Animations, transitions, or dynamic effects that earn their place\n"
            "3. Personality — does the page have a voice?\n"
            "4. Memorability — would a judge remember it tomorrow?\n"
            "5. Clever technique — creative use of CSS, SVG, or layout tricks"
        ),
    ),
    (
        "interpretation_quality",
        (
            "Evaluate how the contestant **interpreted** the prompt.\n"
            "Prompts in this competition are often deliberately open-ended. "
            "Reward defensible creative interpretation — do NOT penalize the "
            "contestant for filling in gaps the prompt left open.\n\n"
            "1. Did they commit to a coherent reading of the prompt?\n"
            "2. Are their additions beyond the literal prompt thematic and reasonable?\n"
            "3. Does the execution read as intentional rather than accidental?\n"
            "4. If the prompt was vague, did they take a confident stance rather than hedge?\n"
            "5. Would a reasonable judge agree this is a valid take on the prompt?"
        ),
    ),
]


EVALUATORS_BY_AXIS: dict[str, Callable[..., dict[str, Any]]] = {
    key: create_llm_as_judge(
        prompt=_make_judge_prompt(criteria),
        feedback_key=key,
        model=_JUDGE_MODEL,
        continuous=True,
    )
    for key, criteria in _EVALUATOR_SPECS
}

LLM_AXES: tuple[str, ...] = tuple(EVALUATORS_BY_AXIS.keys())
