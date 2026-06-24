"""OOLONG-synth: plain subagents vs. code-interpreter (RLM).

Same examples, asymmetric model setup (matching Zhang et al. 2025):

- **Root model** (``--model``, e.g. ``claude-sonnet-4-6``): drives the
  orchestrator in both arms.
- **Sub-model** (``--sub-model``, default ``openai:gpt-5-mini``): the
  general-purpose subagent that reads/classifies context chunks.

The arm is the experiment dimension, not an input: select it with
``OOLONG_ARM=plain`` or ``OOLONG_ARM=code_interpreter`` and give each run its
own ``LANGSMITH_EXPERIMENT`` name. Run each arm in its own session — see
``README.md`` in this folder for the full run matrix and workspace setup.

Default subset (``OOLONG_DATASET=trec_coarse``, ``OOLONG_CONTEXT_LEN=65536``,
``OOLONG_N_EXAMPLES=5``) matches the north-star LangSmith dataset bucket — set
``OOLONG_N_EXAMPLES=0`` for the full 50-task bucket, or drop
``OOLONG_CONTEXT_LEN`` (e.g. ``8192``) for a faster smoke run.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

from tests.evals.oolong.benchmarks import (
    TASK_GROUPS,
    load_oolong_examples,
    resolve_arm,
    run_oolong_case,
)

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from tests.evals.oolong.benchmarks import OolongExample


pytestmark = [
    pytest.mark.eval_category("long_context_aggregation"),
    pytest.mark.eval_tier("hillclimb"),
    pytest.mark.langsmith,
]
"""All cases in this module are long-context aggregation, hillclimb-tier."""


def _example_params() -> list[pytest.param]:
    """Build the per-example parameter matrix (one row per example, not per arm).

    Set ``OOLONG_N_EXAMPLES=0`` (or empty) for the full 50-task bucket.
    Set ``OOLONG_CONTEXT_LEN`` to override the context bucket (default 65536,
      matching the north-star dataset).
    Set ``OOLONG_DATASET`` to change the subset (default ``trec_coarse`` — the
      paper's exact dataset, validation split).

    The arm is *not* a parameter — it is read from ``OOLONG_ARM`` at runtime so
    both arms produce the same example identity and line up for comparison.
    """
    raw = os.environ.get("OOLONG_N_EXAMPLES", "5")
    n_examples: int | None = int(raw) if raw and raw != "0" else None
    context_len = int(os.environ.get("OOLONG_CONTEXT_LEN", "65536"))
    dataset = os.environ.get("OOLONG_DATASET", "trec_coarse")

    examples = load_oolong_examples(dataset=dataset, context_len=context_len, n_examples=n_examples)
    return [pytest.param(ex, id=f"{ex.task_group}-{ex.task_id}") for ex in examples]


@pytest.mark.parametrize("example", _example_params())
def test_oolong_synth(
    example: OolongExample,
    model: BaseChatModel,
    sub_model_name: str,
) -> None:
    """Run a single OOLONG-synth example against the arm given by ``OOLONG_ARM``."""
    assert example.task_group in TASK_GROUPS
    run_oolong_case(
        example,
        resolve_arm(),
        root_model=model,
        sub_model_id=sub_model_name,
    )
