"""Pytest entry point for the large-scale compaction benchmark.

Parametrizes ``(instance, technique)`` and drives the shared runner,
then emits per-checkpoint and per-category LangSmith feedback. This
module is the *only* compaction-bench test that actually invokes a
model, which is why it is tagged ``eval_tier("hillclimb")`` - it is
not a regression gate and must not run on every PR.

### Feedback keys emitted

For each run, the following keys are logged to LangSmith:

- ``compaction_<Gxx>``: per-checkpoint score (e.g. ``compaction_G1``).
- ``compaction_<category>``: per-category weighted score (e.g.
  ``compaction_goal_drift``). Categories correspond to the
  ``FailureMode`` enum.
- ``compaction_weighted_total``: overall weighted score across all
  executed checkpoints.

Per-checkpoint keys make it easy to drill into a single regression
(e.g. "G13 dropped from 1.0 to 0.3 after the prompt change"). The
per-category rollups are what drives the technique-comparison UI.

### Failure policy

This test does **not** call ``pytest.fail`` on low scores. The eval is
hillclimb, not pass/fail: techniques are compared relative to each
other, and an absolute score of 0.6 may be perfectly acceptable for a
hard instance. Only runtime errors (missing API key, agent crash) fail
the test.

### Running

```bash
uv run --group test pytest tests/evals/compaction_bench/ \
    --eval-tier hillclimb -v
```

Set ``OPENAI_API_KEY`` if running the ``openai_compact`` technique; it
will be skipped otherwise.
"""

from __future__ import annotations

import contextlib
import logging
import os
from typing import TYPE_CHECKING

import pytest
from langsmith import testing as t

from tests.evals.compaction_bench.instance_001_partnerco import INSTANCE
from tests.evals.compaction_bench.runner import run_and_grade
from tests.evals.compaction_bench.task_spec import FailureMode, Instance
from tests.evals.compaction_bench.techniques import TECHNIQUES, SummarizationTechnique

if TYPE_CHECKING:
    from pathlib import Path

    from langchain_core.language_models import BaseChatModel

    from tests.evals.compaction_bench.scorecard import Scorecard


pytestmark = [
    pytest.mark.eval_category("summarization"),
    pytest.mark.eval_tier("hillclimb"),
]
"""Summarization category (compaction *is* summarization under pressure);
hillclimb tier since each run is expensive and not a regression gate."""

logger = logging.getLogger(__name__)

_LANGSMITH_CONFIGURED = bool(os.environ.get("LANGSMITH_API_KEY"))
_langsmith_mark = pytest.mark.langsmith if _LANGSMITH_CONFIGURED else lambda f: f


# ---------------------------------------------------------------------------
# Registry-based parametrization
# ---------------------------------------------------------------------------

# v1 ships one hand-authored instance. A second instance drops in here
# as a second element without any other changes required.
_INSTANCES: list[Instance] = [INSTANCE]
_TECHNIQUES: list[SummarizationTechnique] = list(TECHNIQUES.values())


# ---------------------------------------------------------------------------
# LangSmith helpers
# ---------------------------------------------------------------------------


def _log_feedback(*, key: str, value: float | str | None) -> None:
    """Log feedback to LangSmith, silently no-op when not configured.

    The broad exception suppression mirrors memory_agent_bench's
    pattern - LangSmith errors must not turn a slow eval into a hard
    failure after the agent run completed successfully.
    """
    with contextlib.suppress(Exception):
        t.log_feedback(key=key, value=value)


def _log_clean_inputs(
    *,
    model: BaseChatModel,
    instance: Instance,
    technique: SummarizationTechnique,
) -> None:
    """Override LangSmith's auto-captured inputs with a minimal payload.

    Without this, the ``model`` fixture object is serialized wholesale
    into the dataset example inputs, which is noisy and leaks fixture
    internals.
    """
    with contextlib.suppress(Exception):
        t.log_inputs(
            {
                "instance_id": instance.id,
                "technique": technique.name,
                "model": str(getattr(model, "model", None) or getattr(model, "model_name", "")),
            }
        )


def _log_scorecard_feedback(scorecard: Scorecard) -> None:
    """Emit per-checkpoint, per-category, and weighted-total feedback.

    Three granularities because we have three question types:

    - "Did G13 regress?" (per-checkpoint)
    - "Is this technique better at preserving decisions?"
      (per-category)
    - "Overall, is this technique improving?" (weighted total)
    """
    for result in scorecard.all_results:
        _log_feedback(key=f"compaction_{result.checkpoint_id}", value=result.score)

    for category in FailureMode:
        cat_score = scorecard.categories.get(category)
        if cat_score is not None:
            _log_feedback(
                key=f"compaction_{category.value}",
                value=cat_score.weighted_score,
            )

    _log_feedback(key="compaction_weighted_total", value=scorecard.weighted_total)


# ---------------------------------------------------------------------------
# Skip gates
# ---------------------------------------------------------------------------


def _require_openai_key_for(technique: SummarizationTechnique) -> None:
    """Skip when a technique needs OpenAI but no key is set.

    ``openai_compact`` instantiates an OpenAI model as its summarizer;
    running without a key would surface as an opaque init error deep
    in the first compaction event. Failing fast here is clearer.
    """
    if technique.name == "openai_compact" and not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("openai_compact technique requires OPENAI_API_KEY")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _instance_id(instance: Instance) -> str:
    return instance.id


def _technique_id(technique: SummarizationTechnique) -> str:
    return technique.name


@_langsmith_mark
@pytest.mark.parametrize("instance", _INSTANCES, ids=_instance_id)
@pytest.mark.parametrize("technique", _TECHNIQUES, ids=_technique_id)
def test_compaction_bench(
    instance: Instance,
    technique: SummarizationTechnique,
    model: BaseChatModel,
    tmp_path: Path,
) -> None:
    """Drive one ``(instance, technique)`` run and emit per-checkpoint feedback.

    The runner handles all the mechanics (seeding, turn loop,
    snapshotting, grading); this function exists to (a) wire pytest's
    ``model`` and ``tmp_path`` fixtures in, (b) skip cleanly when a
    technique's credentials are missing, and (c) translate the
    ``Scorecard`` into LangSmith feedback keys.

    Args:
        instance: The instance to run.
        technique: The summarization technique under test.
        model: The chat model fixture (parametrized at session level).
        tmp_path: Per-test temp directory for the fixture mini-repo.
    """
    _log_clean_inputs(model=model, instance=instance, technique=technique)
    _require_openai_key_for(technique)

    artifacts, scorecard = run_and_grade(
        instance=instance,
        technique=technique,
        model=model,
        root_dir=tmp_path,
        include_judge=False,
        include_subprocess=False,
    )

    logger.info(
        "compaction_bench %s/%s weighted_total=%.3f",
        instance.id,
        technique.name,
        scorecard.weighted_total,
    )
    logger.info("per-category summary:\n%s", scorecard.summary())

    # LangSmith side effects are last so a feedback-logging bug cannot
    # mask a grader bug: the trace and categorical scores are visible
    # in logs even if the feedback emitter silently fails.
    _log_scorecard_feedback(scorecard)

    # Keep the artifacts reference alive for debugging (otherwise pytest
    # discards it immediately). Intentional no-op at runtime.
    _ = artifacts
