from __future__ import annotations

from contextvars import ContextVar
from threading import Lock
from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage

from tests.evals import llm_judge as llm_judge_module
from tests.evals.llm_judge import LLMJudge
from tests.evals.utils import AgentStep, AgentTrajectory

if TYPE_CHECKING:
    from collections.abc import Callable


def _make_trajectory(answer: str) -> AgentTrajectory:
    return AgentTrajectory(
        steps=[AgentStep(index=1, action=AIMessage(content=answer), observations=[])],
        files={},
    )


def test_threaded_judge_preserves_caller_contextvars(monkeypatch) -> None:
    active_run: ContextVar[str | None] = ContextVar("active_run", default=None)
    seen: list[tuple[str, str | None]] = []
    lock = Lock()

    def fake_create_llm_as_judge(**_kwargs: object) -> Callable[..., dict[str, object]]:
        def evaluator(*, outputs: str, criterion: str) -> dict[str, object]:
            with lock:
                seen.append((criterion, active_run.get()))
            return {"score": True, "comment": outputs}

        return evaluator

    monkeypatch.setattr(llm_judge_module, "create_llm_as_judge", fake_create_llm_as_judge)
    monkeypatch.setattr(llm_judge_module.t, "log_feedback", lambda **_kwargs: None)

    token = active_run.set("langsmith-test-context")
    try:
        results = LLMJudge(criteria=("correctness", "safety"))._grade(
            _make_trajectory("final answer")
        )
    finally:
        active_run.reset(token)

    assert [result["score"] for result in results] == [True, True]
    assert sorted(seen) == [
        ("correctness", "langsmith-test-context"),
        ("safety", "langsmith-test-context"),
    ]
