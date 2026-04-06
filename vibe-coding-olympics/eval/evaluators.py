"""LLM-as-judge evaluators for vibe coding competition website scoring.

Each evaluator expects:
    inputs: The prompt given to the contestant.
    outputs: The generated HTML source.
    screenshot_b64: Base64-encoded PNG screenshot of the rendered page.

All evaluators use continuous scoring (0.0 to 1.0).
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


# ---------------------------------------------------------------------------
# Shared prompt builder
# ---------------------------------------------------------------------------


def _make_judge_prompt(
    criteria: str,
) -> Callable[..., list[dict[str, Any]]]:
    """Return a prompt callable for the given scoring criteria."""

    def _prompt(
        *,
        inputs: str | dict | None = None,
        outputs: str | dict | None = None,
        screenshot_b64: str = "",
        **kwargs: object,  # noqa: ARG001  # required by openevals prompt callable contract
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


# ---------------------------------------------------------------------------
# Evaluator specs — single source of truth for keys and criteria
# ---------------------------------------------------------------------------

_EVALUATOR_SPECS: list[tuple[str, str]] = [
    (
        "visual_design",
        (
            "Evaluate the **visual design** of this website based on:\n"
            "1. Layout structure and visual hierarchy\n"
            "2. Color palette — cohesive and appropriate\n"
            "3. Typography — readable, consistent sizing/weight\n"
            "4. Whitespace and spacing — not cramped or empty\n"
            "5. Overall polish — does it feel intentional?"
        ),
    ),
    (
        "content_completeness",
        (
            "Evaluate the **content completeness** of this website based on:\n"
            "1. Meaningful, relevant text — not lorem ipsum or placeholders\n"
            "2. All expected sections present for the theme\n"
            "3. Logical information hierarchy and structure\n"
            "4. Content density — enough substance without filler\n"
            "5. Functional elements — links, buttons, and CTAs make sense"
        ),
    ),
    (
        "creativity",
        (
            "Evaluate the **creativity** of this website based on:\n"
            "1. Novel visual or interaction approaches\n"
            "2. Animations, transitions, or dynamic effects\n"
            "3. Personality and voice — does it stand out?\n"
            "4. Memorability — would you remember this site?\n"
            "5. Clever use of CSS, SVG, or other techniques"
        ),
    ),
    (
        "prompt_adherence",
        (
            "Evaluate how well this website **adheres to the prompt** based on:\n"
            "1. Theme match — does it clearly address the prompt?\n"
            "2. Appropriate tone and style for the subject\n"
            "3. All explicit requirements from the prompt are met\n"
            "4. Implicit expectations — what a reasonable person would expect\n"
            "5. Cohesion — every element supports the prompt's intent"
        ),
    ),
]


# ---------------------------------------------------------------------------
# Build evaluators from specs
# ---------------------------------------------------------------------------

ALL_EVALUATORS = [
    create_llm_as_judge(
        prompt=_make_judge_prompt(criteria),
        feedback_key=key,
        model=_JUDGE_MODEL,
        continuous=True,
    )
    for key, criteria in _EVALUATOR_SPECS
]

FEEDBACK_KEYS = [key for key, _ in _EVALUATOR_SPECS]

# Named references for direct imports.
visual_design_evaluator = ALL_EVALUATORS[0]
content_completeness_evaluator = ALL_EVALUATORS[1]
creativity_evaluator = ALL_EVALUATORS[2]
prompt_adherence_evaluator = ALL_EVALUATORS[3]
