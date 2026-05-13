"""LLM-as-judge evaluator for vibe coding competition website scoring.

The judge returns every LLM-scored axis from a single multimodal call. Keeping
the axes separate preserves diagnosability while avoiding one network round
trip per rubric dimension during the live event.
"""

from __future__ import annotations

from typing import Any

from langchain_openai import ChatOpenAI

# Must support vision.
_JUDGE_MODEL = "gpt-5.5"
_REASONING_EFFORT = "low"
_MODEL_TIMEOUT_SECS = 12.0

LLM_AXES: tuple[str, ...] = (
    "color",
    "typography",
    "layout",
    "content_completeness",
    "creativity",
    "interpretation_quality",
)

_ROLE_PREAMBLE = (
    "You are an expert web designer judging a 5-minute vibe coding competition. "
    "Contestants used an AI coding tool to generate a website from a short prompt. "
    "Calibrate your expectations accordingly -- this was built in 5 minutes, not 5 days."
)

_SCORE_INSTRUCTION = (
    "Return a JSON object with one numeric score for each axis. "
    "Each score must be from 0.0 (absent or broken) to 1.0 "
    "(exceptional for a 5-minute build). Be generous for live-show scoring: "
    "a coherent but basic working page should usually land around 0.55 to "
    "0.75, a strong 5-minute page around 0.75 to 0.95, and reserve scores "
    "below 0.25 for absent, broken, or irrelevant work. Use decimal values; "
    "do not round small positive scores down to 0.0."
)

_RUBRIC = """Evaluate these axes independently:

color:
1. Cohesion -- colors feel like a chosen palette, not random
2. Contrast -- foreground/background combinations are legible
3. Appropriateness -- palette matches the subject's tone
4. Restraint -- not too many competing hues

typography:
1. Readability -- body copy is comfortable to read
2. Hierarchy -- heading/body sizes are clearly differentiated
3. Consistency -- font choices are stable across the page
4. Character -- type choices suit the subject

layout:
1. Visual hierarchy -- the most important thing reads first
2. Alignment -- content sits on a coherent grid
3. Spacing rhythm -- margins/paddings feel considered, not cramped or empty
4. Balance -- the page is not lopsided or unintentionally asymmetric
5. Flow -- the eye is guided through the page naturally

content_completeness:
1. Meaningful, relevant text -- not lorem ipsum or placeholders
2. Expected sections are present for the chosen theme
3. Logical information hierarchy
4. Substance without filler
5. Functional elements (links, buttons, CTAs) make sense

creativity:
1. Novel visual or interaction ideas
2. Animations, transitions, or dynamic effects that earn their place
3. Personality -- does the page have a voice?
4. Memorability -- would a judge remember it tomorrow?
5. Clever technique -- creative use of CSS, SVG, or layout tricks

interpretation_quality:
Prompts are often deliberately open-ended. Reward defensible creative
interpretation; do not penalize contestants for filling in gaps the prompt
left open.
1. Did they commit to a coherent reading of the prompt?
2. Are additions beyond the literal prompt thematic and reasonable?
3. Does the execution read as intentional rather than accidental?
4. If the prompt was vague, did they take a confident stance rather than hedge?
5. Would a reasonable judge agree this is a valid take on the prompt?
"""

_OUTPUT_PROPERTIES: dict[str, Any] = {
    "reasoning": {
        "type": "string",
        "description": (
            "Brief calibration note explaining the scores. End with "
            "`Thus, the scores should be assigned as above.`"
        ),
    },
    **{
        axis: {
            "type": "number",
            "description": f"Continuous score for `{axis}` from 0.0 to 1.0.",
        }
        for axis in LLM_AXES
    },
}

_OUTPUT_SCHEMA: dict[str, Any] = {
    "title": "vibe_scores",
    "description": "Per-axis website scores for a 5-minute coding competition.",
    "type": "object",
    "additionalProperties": False,
    "properties": _OUTPUT_PROPERTIES,
    "required": ["reasoning", *LLM_AXES],
}


def _make_judge() -> Any:
    """Create the structured multimodal judge model.

    Returns:
        A LangChain chat model configured to return the axis-score schema.
    """
    judge = ChatOpenAI(
        model=_JUDGE_MODEL,
        reasoning={"effort": _REASONING_EFFORT},
        timeout=_MODEL_TIMEOUT_SECS,
        max_retries=0,
    )
    return judge.with_structured_output(_OUTPUT_SCHEMA)


def _make_message(
    *,
    prompt: str,
    html: str,
    screenshot_b64: str,
) -> list[dict[str, Any]]:
    """Build the multimodal judge message.

    Args:
        prompt: The original prompt given to the contestant.
        html: Rendered HTML source of the page.
        screenshot_b64: Base64-encoded PNG screenshot.

    Returns:
        OpenAI-compatible chat messages for the judge.
    """
    text = (
        f"{_ROLE_PREAMBLE}\n\n"
        f"Prompt given to the contestant: {prompt}\n\n"
        f"HTML source:\n```html\n{html}\n```\n\n"
        f"{_RUBRIC}\n\n"
        f"{_SCORE_INSTRUCTION}"
    )
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{screenshot_b64}",
                    },
                },
            ],
        }
    ]


def _coerce_score(value: object) -> float | None:
    """Convert one model-produced score to the expected unit interval."""
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if score > 1.0:
        score /= 10.0
    return max(0.0, min(1.0, score))


def score_llm_axes(
    *,
    prompt: str,
    html: str,
    screenshot_b64: str,
) -> dict[str, float | None]:
    """Score all LLM axes with a single judge call.

    Args:
        prompt: The original prompt given to the contestant.
        html: Rendered HTML source of the page.
        screenshot_b64: Base64-encoded PNG screenshot.

    Returns:
        Mapping of axis name to score in [0.0, 1.0], or `None` when the
        axis could not be parsed.
    """
    raw = _make_judge().invoke(
        _make_message(
            prompt=prompt,
            html=html,
            screenshot_b64=screenshot_b64,
        )
    )
    if not isinstance(raw, dict):
        return dict.fromkeys(LLM_AXES, None)
    return {axis: _coerce_score(raw.get(axis)) for axis in LLM_AXES}
