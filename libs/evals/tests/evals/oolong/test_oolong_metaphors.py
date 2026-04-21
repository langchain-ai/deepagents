"""Oolong / metaphors — metaphor-label aggregation over a seeded context.

Mirrors ``evals/oolong/datasets/metaphors.eval.test.ts``. Per-test
logging/scoring and the ``(runner, task)`` grid construction live in
:mod:`_test_body`.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

from tests.evals.oolong._test_body import (
    load_dataset_tasks,
    parametrize_cases,
    run_oolong_case,
)

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from tests.evals.oolong.data_utils import OolongTask

# Off-catalog — see ``test_oolong_trec_coarse`` for context.

_LANGSMITH_CONFIGURED = bool(os.environ.get("LANGSMITH_API_KEY"))
_langsmith_mark = pytest.mark.langsmith if _LANGSMITH_CONFIGURED else lambda f: f

_CASES, _IDS = parametrize_cases(load_dataset_tasks("metaphors"))


@_langsmith_mark
@pytest.mark.parametrize(("runner_name", "task"), _CASES, ids=_IDS)
async def test_metaphors(
    runner_name: str,
    task: OolongTask,
    model: BaseChatModel,
) -> None:
    await run_oolong_case(runner_name, task, model)
