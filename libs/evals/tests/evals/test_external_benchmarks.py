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
# Focused hard-set: 15 examples across 3 benchmarks
# ---------------------------------------------------------------------------


@pytest.mark.eval_category("retrieval")
@pytest.mark.langsmith
@pytest.mark.parametrize("case", FRAMES_CASES, ids=[case["id"] for case in FRAMES_CASES])
def test_frames(model: BaseChatModel, case: dict[str, Any]) -> None:
    """FRAMES: multi-hop retrieval with arithmetic/temporal reasoning."""
    run_frames_case(case, model)


@pytest.mark.eval_category("tool_use")
@pytest.mark.langsmith
@pytest.mark.parametrize("case", NEXUS_CASES, ids=[case["id"] for case in NEXUS_CASES])
def test_nexus(model: BaseChatModel, case: dict[str, Any]) -> None:
    """Nexus: deeply nested function composition (depth 4-6)."""
    run_nexus_case(case, model)


@pytest.mark.eval_category("tool_use")
@pytest.mark.langsmith
@pytest.mark.parametrize("case", BFCL_V3_CASES, ids=[case["id"] for case in BFCL_V3_CASES])
def test_bfcl_v3(model: BaseChatModel, case: dict[str, Any]) -> None:
    """BFCL v3: multi-turn tool use across API domains."""
    run_bfcl_case(case, model)
