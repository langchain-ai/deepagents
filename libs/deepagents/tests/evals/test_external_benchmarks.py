from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests.evals.external_benchmarks import HOTPOT_CASES, TOOL_CASES, run_hotpot_case, run_tool_case

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel


@pytest.mark.langsmith
@pytest.mark.parametrize("case", TOOL_CASES, ids=[case["id"] for case in TOOL_CASES])
def test_curated_external_tool_cases(model: BaseChatModel, case: dict) -> None:
    run_tool_case(case, model)


@pytest.mark.langsmith
@pytest.mark.parametrize("case", HOTPOT_CASES, ids=[case["id"] for case in HOTPOT_CASES])
def test_curated_external_hotpot_cases(model: BaseChatModel, case: dict) -> None:
    run_hotpot_case(case, model)
