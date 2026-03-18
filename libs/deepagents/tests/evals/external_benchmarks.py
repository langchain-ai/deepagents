from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from deepagents import create_deep_agent
from tests.evals.utils import (
    AgentTrajectory,
    SuccessAssertion,
    TrajectoryScorer,
    run_agent,
)

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel


@dataclass(frozen=True)
class _NormalizedSubstringsPresent(SuccessAssertion):
    """Fail unless all expected snippets appear after whitespace normalization."""

    snippets: tuple[str, ...]

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", "", text).lower().replace("'", "").replace('"', "").replace("`", "")

    def check(self, trajectory: AgentTrajectory) -> bool:
        answer = self._normalize(trajectory.answer)
        return all(self._normalize(snippet) in answer for snippet in self.snippets)

    def describe_failure(self, trajectory: AgentTrajectory) -> str:
        return (
            "Expected final text to contain all normalized snippets "
            f"{list(self.snippets)}, got {trajectory.answer!r}"
        )


_DATA_DIR = Path(__file__).parent / "data" / "benchmark_samples"

_FRAMES_IDS = {
    "frames_10",
    "frames_11",
    "frames_12",
    "frames_16",
    "frames_18",
}
_NEXUS_IDS = {
    "nexus_nvd_nested_13",
    "nexus_nvd_nested_14",
    "nexus_placesapi_15",
    "nexus_multiversemath_17",
    "nexus_multiversemath_18",
}
_BFCL_V3_IDS = {
    "multi_turn_composite_97",
    "multi_turn_composite_116",
    "multi_turn_composite_199",
    "multi_turn_miss_func_55",
    "multi_turn_miss_param_55",
}

_FILE_BACKED_SYSTEM_PROMPT = (
    "Use the files already present in the workspace to solve the task. "
    "Return only the final answer requested by the prompt. "
    "Do not use the task tool or any subagent delegation."
)


def _load_final(name: str) -> list[dict[str, Any]]:
    path = _DATA_DIR / f"{name}_final.json"
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        return data.get("tasks", data.get("cases", []))
    return data


FRAMES_CASES: list[dict[str, Any]] = [
    case for case in _load_final("frames") if case["id"] in _FRAMES_IDS
]
NEXUS_CASES: list[dict[str, Any]] = [
    case for case in _load_final("nexus") if case["id"] in _NEXUS_IDS
]
BFCL_V3_CASES: list[dict[str, Any]] = [
    case for case in _load_final("bfcl_v3") if case["id"] in _BFCL_V3_IDS
]


def _external_eval_metadata(
    *,
    case_id: str,
    vertical: str,
    origin_benchmark: str,
    difficulty: str,
    tags: list[str] | None = None,
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "vertical": vertical,
        "origin_benchmark": origin_benchmark,
        "difficulty": difficulty,
        "subset": "curated_external_hard",
        "tags": tags or [],
    }


def _create_file_backed_agent(model: BaseChatModel):
    return create_deep_agent(model=model, system_prompt=_FILE_BACKED_SYSTEM_PROMPT)


def _create_text_scorer(case: dict[str, Any]) -> TrajectoryScorer:
    return TrajectoryScorer().success(
        _NormalizedSubstringsPresent(snippets=tuple(case["answer_snippets"]))
    )


def run_frames_case(case: dict[str, Any], model: BaseChatModel) -> None:
    agent = _create_file_backed_agent(model)
    run_agent(
        agent,
        model=model,
        query=case["prompt"],
        initial_files=case["files"],
        scorer=_create_text_scorer(case),
        eval_metadata=_external_eval_metadata(
            case_id=case["id"],
            vertical="retrieval",
            origin_benchmark="frames",
            difficulty=str(case.get("difficulty", "hard")),
            tags=["curated", "hard", "retrieval", *case.get("axes", [])],
        ),
    )


def run_nexus_case(case: dict[str, Any], model: BaseChatModel) -> None:
    agent = _create_file_backed_agent(model)
    run_agent(
        agent,
        model=model,
        query=case["prompt"],
        initial_files=case["files"],
        scorer=_create_text_scorer(case),
        eval_metadata=_external_eval_metadata(
            case_id=case["id"],
            vertical="reasoning",
            origin_benchmark="nexus",
            difficulty=str(case.get("difficulty", "hard")),
            tags=["curated", "hard", "reasoning", *case.get("axes", [])],
        ),
    )


def run_bfcl_case(case: dict[str, Any], model: BaseChatModel) -> None:
    agent = _create_file_backed_agent(model)
    run_agent(
        agent,
        model=model,
        query=case["prompt"],
        initial_files=case["files"],
        scorer=_create_text_scorer(case),
        eval_metadata=_external_eval_metadata(
            case_id=case["id"],
            vertical="tool_calling",
            origin_benchmark="bfcl",
            difficulty=str(case.get("difficulty", "hard")),
            tags=["curated", "hard", "tool_calling", *case.get("axes", [])],
        ),
    )
