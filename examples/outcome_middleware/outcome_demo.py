"""Offline demo of `OutcomeMiddleware`.

Walks through the four terminal outcomes the middleware exposes:

1. **satisfied first try** -- agent nails it on the first attempt.
2. **needs_revision then satisfied** -- grader rejects iteration 0,
   middleware injects feedback, agent revises on iteration 1.
3. **max_iterations_reached** -- grader rejects every attempt; the loop
   terminates at the configured cap.
4. **grader exception -> failed** -- grader raises; middleware records a
   `failed` evaluation and terminates cleanly without crashing the run.

Each scenario uses a scripted main chat model and a stubbed grader, so the
demo runs entirely offline. The script asserts the final `outcome_status`
matches the expectation for each scenario, so this doubles as a smoke
test you can run by hand: `python outcome_demo.py`.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from langchain.agents import create_agent
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from deepagents.middleware.outcomes import GraderResponse, OutcomeMiddleware


class _ScriptedChatModel(BaseChatModel):
    """Returns a pre-scripted sequence of `AIMessage`s, one per call.

    Inlined here so the example has no dependency on the test suite.
    """

    messages: Iterator[AIMessage] = Field(exclude=True)

    def bind_tools(self, tools, **_kwargs):  # type: ignore[no-untyped-def]
        return self

    def _generate(
        self,
        messages: list[BaseMessage],  # noqa: ARG002
        stop: list[str] | None = None,  # noqa: ARG002
        run_manager: CallbackManagerForLLMRun | None = None,  # noqa: ARG002
        **_kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=next(self.messages))])

    @property
    def _llm_type(self) -> str:
        return "scripted-chat-model"


@dataclass
class Scenario:
    """One end-to-end run with scripted inputs and an expected outcome."""

    name: str
    description: str
    agent_replies: list[str]
    grader_script: list[GraderResponse] | BaseException
    max_iterations: int
    expected_status: str


USER_PROMPT = "Write me a haiku."
RUBRIC = "- exactly 3 lines\n- mentions the sea\n- syllable pattern 5/7/5"


SCENARIOS: list[Scenario] = [
    Scenario(
        name="satisfied first try",
        description=(
            "Agent's first draft satisfies every criterion; grader signs off "
            "immediately. One model call, one grader call."
        ),
        agent_replies=[
            "Salt on the wind blows\nSea birds wheel above the foam\nTides answer the moon",
        ],
        grader_script=[
            GraderResponse(
                result="satisfied",
                explanation="all criteria met on the first attempt",
                criteria=[
                    {"name": "exactly 3 lines", "passed": True},
                    {"name": "mentions the sea", "passed": True},
                ],
            ),
        ],
        max_iterations=3,
        expected_status="satisfied",
    ),
    Scenario(
        name="needs_revision then satisfied",
        description=(
            "First draft misses the rubric; grader returns needs_revision with "
            "per-criterion gaps; middleware injects feedback as a tagged "
            "HumanMessage and loops; second draft satisfies."
        ),
        agent_replies=[
            "An ancient pond stills\nA frog jumps in with a splash\nSilence returns now",
            "Salt on the wind blows\nSea birds wheel above the foam\nTides answer the moon",
        ],
        grader_script=[
            GraderResponse(
                result="needs_revision",
                explanation="the haiku does not mention the sea",
                criteria=[
                    {"name": "exactly 3 lines", "passed": True},
                    {
                        "name": "mentions the sea",
                        "passed": False,
                        "gap": "no mention of the sea in any line",
                    },
                ],
            ),
            GraderResponse(
                result="satisfied",
                explanation="all criteria met after revision",
                criteria=[
                    {"name": "exactly 3 lines", "passed": True},
                    {"name": "mentions the sea", "passed": True},
                ],
            ),
        ],
        max_iterations=3,
        expected_status="satisfied",
    ),
    Scenario(
        name="max_iterations_reached",
        description=(
            "Every grader pass returns needs_revision. With max_iterations=2, "
            "the second rejection terminates with max_iterations_reached "
            "instead of looping again."
        ),
        agent_replies=[
            "first draft (off-rubric)",
            "second draft (still off-rubric)",
        ],
        grader_script=[
            GraderResponse(
                result="needs_revision",
                explanation="still missing the sea reference",
                criteria=[
                    {"name": "mentions the sea", "passed": False, "gap": "no sea"},
                ],
            ),
            GraderResponse(
                result="needs_revision",
                explanation="still missing the sea reference",
                criteria=[
                    {"name": "mentions the sea", "passed": False, "gap": "still no sea"},
                ],
            ),
        ],
        max_iterations=2,
        expected_status="max_iterations_reached",
    ),
    Scenario(
        name="grader exception -> failed",
        description=(
            "Grader raises before producing a verdict; middleware records a "
            "failed evaluation with the exception text and terminates cleanly."
        ),
        agent_replies=["only draft"],
        grader_script=RuntimeError("grader sub-agent unavailable"),
        max_iterations=3,
        expected_status="failed",
    ),
]


def _stub_grader(
    middleware: OutcomeMiddleware,
    grader_script: list[GraderResponse] | BaseException,
) -> None:
    """Wire `_grade` / `_agrade` to either return canned responses or raise."""
    if isinstance(grader_script, BaseException):
        exc = grader_script

        def _grade(_state, _iteration):  # type: ignore[no-untyped-def]
            raise exc

        async def _agrade(_state, _iteration):  # type: ignore[no-untyped-def]
            raise exc
    else:
        iterator = iter(grader_script)

        def _grade(_state, _iteration):  # type: ignore[no-untyped-def]
            return next(iterator), None

        async def _agrade(_state, _iteration):  # type: ignore[no-untyped-def]
            return next(iterator), None

    middleware._grade = _grade  # type: ignore[assignment]
    middleware._agrade = _agrade  # type: ignore[assignment]


def _run_scenario(scenario: Scenario) -> None:
    print()
    print(f"--- {scenario.name} ---")
    print(scenario.description)

    agent_model = _ScriptedChatModel(
        messages=iter([AIMessage(content=reply) for reply in scenario.agent_replies])
    )
    middleware = OutcomeMiddleware(max_iterations=scenario.max_iterations)
    _stub_grader(middleware, scenario.grader_script)
    agent = create_agent(model=agent_model, tools=[], middleware=[middleware])

    result = agent.invoke(
        {
            "messages": [HumanMessage(content=USER_PROMPT)],
            "rubric": RUBRIC,
        }
    )

    for evaluation in result.get("outcome_evaluations", []):
        print(
            f"  iter {evaluation['iteration']}: "
            f"{evaluation['result']} -- {evaluation['explanation']}"
        )

    final_status = result["outcome_status"]
    print(f"  final outcome_status: {final_status}")

    assert final_status == scenario.expected_status, (
        f"Expected outcome_status={scenario.expected_status!r}, "
        f"got {final_status!r}"
    )


def main() -> None:
    # The grader-exception scenario intentionally triggers `logger.exception`
    # inside the middleware. Silence it so the demo's stdout/stderr stay clean.
    # Real applications should leave this logger at its default level.
    logging.getLogger("deepagents.middleware.outcomes").setLevel(logging.CRITICAL)
    for scenario in SCENARIOS:
        _run_scenario(scenario)
    print()
    print(f"All {len(SCENARIOS)} scenarios completed successfully.")


if __name__ == "__main__":
    main()
