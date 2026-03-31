"""Eval tests drawn from curated external benchmarks.

Runs a focused hard-set of 15 cases across three public benchmarks:
- FRAMES: multi-hop retrieval with arithmetic/temporal reasoning
- Nexus: deeply nested function composition (depth 4-6)
- BFCL v3: multi-turn stateful tool calling across API domains

Each benchmark's runner and scoring logic lives in external_benchmarks.py.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from tests.evals.external_benchmarks import (
    BFCL_V3_CASES,
    FRAMES_CASES,
    NEXUS_CASES,
    run_bfcl_case,
    run_frames_case,
    run_nexus_case,
)

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

# ---------------------------------------------------------------------------
# Per-case tier classification (based on cross-model frontier eval results)
# ---------------------------------------------------------------------------

_FRAMES_HILLCLIMB = {"frames_12"}
_NEXUS_HILLCLIMB = {"nexus_placesapi_15", "nexus_multiversemath_18"}
_BFCL_V3_HILLCLIMB = {
    "multi_turn_composite_97",
    "multi_turn_composite_199",
    "multi_turn_miss_func_55",
}


def _tiered_params(
    cases: list[dict[str, Any]], hillclimb_ids: set[str]
) -> list[Any]:
    """Wrap each case in `pytest.param` with the appropriate eval_tier mark."""
    return [
        pytest.param(
            c,
            marks=pytest.mark.eval_tier(
                "hillclimb" if c["id"] in hillclimb_ids else "baseline"
            ),
            id=c["id"],
        )
        for c in cases
    ]


# ---------------------------------------------------------------------------
# Focused hard-set: 15 examples across 3 benchmarks
# ---------------------------------------------------------------------------


@pytest.mark.eval_category("retrieval")
@pytest.mark.langsmith
@pytest.mark.parametrize("case", _tiered_params(FRAMES_CASES, _FRAMES_HILLCLIMB))
def test_frames(model: BaseChatModel, case: dict[str, Any]) -> None:
    """FRAMES: multi-hop retrieval with arithmetic/temporal reasoning."""
    run_frames_case(case, model)


@pytest.mark.eval_category("tool_use")
@pytest.mark.langsmith
@pytest.mark.parametrize("case", _tiered_params(NEXUS_CASES, _NEXUS_HILLCLIMB))
def test_nexus(model: BaseChatModel, case: dict[str, Any]) -> None:
    """Nexus: deeply nested function composition (depth 4-6)."""
    run_nexus_case(case, model)


@pytest.mark.eval_category("tool_use")
@pytest.mark.langsmith
@pytest.mark.parametrize("case", _tiered_params(BFCL_V3_CASES, _BFCL_V3_HILLCLIMB))
def test_bfcl_v3(model: BaseChatModel, case: dict[str, Any]) -> None:
    """BFCL v3: multi-turn tool use across API domains."""
    run_bfcl_case(case, model)
