"""OOLONG-synth: plain `create_deep_agent` vs. code-interpreter.

Same examples, asymmetric model setup (matching Zhang et al. 2025):

- **Root model** (``--model``, e.g. ``openai:gpt-5``): drives the
  orchestrator in both arms.
- **Sub-model** (``--sub-model``, default ``openai:gpt-5-mini``):
  unused by both arms (kept for conftest fixture compatibility).

Both arms run as parameterized pytest cases so a single eval
invocation captures both into one LangSmith project. Each case is
tagged with ``arm`` and ``task_group`` metadata so you can slice the
LangSmith dashboard by either axis.

Default smoke subset (``context_len=8192``, 5 examples from the
``agnews`` input subset) is intentionally tiny — pass
``n_examples=None`` to `load_oolong_examples` for the full 50-task
bucket that matches the paper's task count.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

from tests.evals.oolong_benchmarks import (
    TASK_GROUPS,
    load_oolong_examples,
    run_oolong_case,
)

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from tests.evals.oolong_benchmarks import Arm, OolongExample


pytestmark = [
    pytest.mark.eval_category("long_context_aggregation"),
    pytest.mark.eval_tier("hillclimb"),
    pytest.mark.langsmith,
]
"""All cases in this module are long-context aggregation, hillclimb-tier."""


def _example_params() -> list[pytest.param]:
    """Build the (arm, example) parameter matrix.

    Set ``OOLONG_N_EXAMPLES=0`` (or empty) for the full 50-task bucket.
    Set ``OOLONG_CONTEXT_LEN`` to override the context bucket (default 8192).
    Set ``OOLONG_DATASET`` to change the input subset (default ``agnews``).
      Use ``trec_coarse`` for the paper's exact dataset (validation split).
    """
    raw = os.environ.get("OOLONG_N_EXAMPLES", "5")
    n_examples: int | None = int(raw) if raw and raw != "0" else None
    context_len = int(os.environ.get("OOLONG_CONTEXT_LEN", "8192"))
    input_subset = os.environ.get("OOLONG_DATASET", "agnews")
    examples = load_oolong_examples(
        input_subset=input_subset, context_len=context_len, n_examples=n_examples
    )
    params: list[pytest.param] = []
    for arm in ("plain", "code_interpreter"):
        params.extend(pytest.param(arm, ex, id=f"{arm}-{ex.task_group}-{ex.id}") for ex in examples)
    return params


@pytest.mark.parametrize(("arm", "example"), _example_params())
def test_oolong_synth(
    arm: Arm,
    example: OolongExample,
    model: BaseChatModel,
    sub_model_name: str,
) -> None:
    """Run a single OOLONG-synth example against the given arm."""
    assert example.task_group in TASK_GROUPS
    run_oolong_case(
        example,
        arm,
        root_model=model,
        sub_model_id=sub_model_name,
    )
