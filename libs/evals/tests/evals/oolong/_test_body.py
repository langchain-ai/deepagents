"""Shared body for the three Oolong dataset test modules.

Factored out so ``test_oolong_trec_coarse.py`` / ``test_oolong_multinli.py``
/ ``test_oolong_metaphors.py`` stay small and the scoring + feedback
logic lives in one place. Each per-dataset module parametrizes over
``(runner_name, task)`` and calls :func:`run_oolong_case` inside its
own pytest function.

Correctness flows through :class:`OolongCorrect`, a
:class:`SuccessAssertion` that runs the partial-credit scorer, emits
the full LangSmith feedback bundle, and then returns the ``correct``
bool back to the :class:`TrajectoryScorer` runner. The scorer raises
``pytest.fail`` on a ``False`` return, which the LangSmith pytest
plugin surfaces as a failed test in the experiment view.

LangSmith input/output wiring here matches the house pattern in
``tests/evals/memory_agent_bench/test_memory_agent_bench.py``:

- ``t.log_inputs`` + ``run_tree.inputs`` assignment fire BEFORE the
  agent runs, so the dataset row has clean inputs even if the
  scorer fails the test mid-run.
- ``t.log_outputs`` + per-key ``t.log_feedback`` fire from inside
  ``OolongCorrect.check`` so the partial-credit bundle lands before
  the scorer's ``pytest.fail``.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

from langsmith import testing as t
from langsmith.run_helpers import get_current_run_tree

from tests.evals.oolong.data_utils import OolongTask, load_oolong_tasks
from tests.evals.oolong.runners import RUNNERS
from tests.evals.oolong.scoring import parse_gold, score_output
from tests.evals.utils import (
    AgentTrajectory,
    SuccessAssertion,
    TrajectoryScorer,
    run_agent_async,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel


def load_dataset_tasks(dataset_name: str) -> list[OolongTask]:
    """Load the subset of Oolong tasks for one source dataset.

    Used at ``pytest.mark.parametrize`` collection time. Streams HF
    and stops as soon as the per-dataset cap is hit (see
    :mod:`data_utils`).

    HF load failures re-raise: a silent empty list at collection time
    would just produce a misleading ``[... NOTSET] SKIPPED`` with no
    indication of what actually went wrong (e.g. disk-space, network,
    auth). Let pytest report the real error at import time instead.
    """
    return load_oolong_tasks(dataset_name)


def case_id(runner_name: str, task: OolongTask) -> str:
    """Build the pytest ``id`` string for one (runner, task) pair.

    Runner first, then ``[len] task_type::id`` (matching the JS side)
    so the LangSmith dashboards group by runner when the id list is
    sorted.
    """
    return f"{runner_name}/[{task.context_len}] {task.task}::{task.id}"


def parametrize_cases(
    tasks: list[OolongTask],
) -> tuple[list[tuple[str, OolongTask]], list[str]]:
    """Build the ``(runner_name, task)`` parameter grid + pytest ids.

    Cross-product of ``RUNNERS`` × ``tasks``, flattened so pytest can
    consume it via a single ``parametrize``. Returning both the
    ``argvalues`` list and a matched ``ids`` list keeps the two in
    sync at one call site — easier to audit than threading them
    through each per-dataset test module.
    """
    argvalues: list[tuple[str, OolongTask]] = []
    ids: list[str] = []
    for runner_name in RUNNERS:
        for task in tasks:
            argvalues.append((runner_name, task))
            ids.append(case_id(runner_name, task))
    return argvalues, ids


@dataclass(frozen=True)
class OolongCorrect(SuccessAssertion):
    """Assert that the agent's answer is correct under Oolong scoring.

    Runs :func:`score_output` in ``check()`` and side-effects the full
    LangSmith feedback bundle (``prediction``, ``gold_answer``,
    ``score``, ``final_text``, ``runner`` outputs + per-strategy
    feedback keys) before returning the pass/fail bool. Logging from
    inside ``check`` is a deliberate choice: the scorer raises
    ``pytest.fail`` on a ``False`` return, so anything logged
    *after* the scorer runs wouldn't land on failing runs. Firing the
    logs from within ``check`` keeps the full feedback on both
    passing and failing cases.

    ``describe_failure`` re-runs :func:`score_output` to build the
    terse one-liner pytest shows. That's a second scoring pass per
    failing case; :func:`score_output` is pure and O(len(output)), so
    the cost is negligible compared with the agent run itself.
    """

    runner_name: str
    task: OolongTask
    gold: str

    def check(self, trajectory: AgentTrajectory) -> bool:
        final_text = trajectory.answer
        score = score_output(final_text, self.gold, self.task.answer_type)

        # Opt-in debug print — enable with ``OOLONG_DEBUG_SCORING=1``
        # when diagnosing why a run did/didn't pass. Prints to stderr
        # so it shows up with pytest ``-s`` but doesn't pollute
        # LangSmith feedback.
        if os.environ.get("OOLONG_DEBUG_SCORING"):
            print(
                "\n[oolong-debug]\n"
                f"  runner={self.runner_name} task_id={self.task.id} "
                f"dataset={self.task.dataset} "
                f"answer_type={self.task.answer_type}\n"
                f"  raw_output={final_text!r}\n"
                f"  pred={score.pred!r}\n"
                f"  gold={self.gold!r}\n"
                f"  score.correct={score.correct} "
                f"score.score={score.score:.3f}\n"
                f"  strategies: exact={score.exact_match} "
                f"norm={score.normalized_match} "
                f"contains={score.contains_match} "
                f"numeric={score.numeric_match}",
                file=sys.stderr,
                flush=True,
            )

        # Structured outputs match the JS logging shape: prediction,
        # gold, score, final_text. Runner goes in outputs (not inputs)
        # so LangSmith can group/colour by it without forking dataset
        # rows.
        t.log_outputs(
            {
                "prediction": score.pred,
                "gold_answer": score.gold,
                "score": score.score,
                "final_text": final_text,
                "runner": self.runner_name,
            }
        )

        # Feedback keys match the JS side. Pass ``score=`` (not
        # ``value=``) so LangSmith renders numeric feedback columns
        # in the UI. ``correct`` drives the "pass" visualization.
        t.log_feedback(key="correct", score=1 if score.correct else 0)
        t.log_feedback(key="score", score=score.score)
        t.log_feedback(key="exact_match", score=1 if score.exact_match else 0)
        t.log_feedback(
            key="normalized_match", score=1 if score.normalized_match else 0
        )
        t.log_feedback(
            key="contains_match", score=1 if score.contains_match else 0
        )
        t.log_feedback(key="numeric_match", score=1 if score.numeric_match else 0)

        return score.correct

    def describe_failure(self, trajectory: AgentTrajectory) -> str:
        score = score_output(trajectory.answer, self.gold, self.task.answer_type)
        return (
            f"Oolong task {self.task.id} (runner={self.runner_name}) "
            f"scored {score.score:.3f}; pred={score.pred!r} "
            f"gold={self.gold!r}"
        )


async def run_oolong_case(
    runner_name: str,
    task: OolongTask,
    model: BaseChatModel,
) -> None:
    """Execute one (runner, task) pair and emit LangSmith feedback.

    Builds the runner context, pins the clean dataset inputs, runs
    the question through ``run_agent_async`` with an
    :class:`OolongCorrect` scorer, and tears the runner down. The
    scorer raises ``pytest.fail`` on an incorrect answer, which shows
    up as a failed test in the LangSmith experiment view.

    Runs via ``agent.ainvoke`` (through ``run_agent_async``) rather
    than ``agent.invoke``. The sync path routes tool calls through a
    thread pool inside ``ToolNode``, which breaks the REPL runners:
    ``quickjs_rs.Context`` is ``!Send`` and panics if invoked from a
    thread other than the one that created it. The async path awaits
    tool coroutines on the caller's event loop, no thread-handoff.
    Baseline (no REPL) works fine either way; async is the safe
    default.
    """
    build_runner = RUNNERS[runner_name]
    ctx = await build_runner(model=model, task=task)

    # Override auto-captured dataset inputs BEFORE running the agent
    # so the dataset row has clean inputs even if the scorer fails
    # the test mid-run. Mirrors the JS eval's
    # ``ls.test({inputs: {...}, referenceOutputs: {...}}, ...)`` shape
    # so Python and JS datasets line up row-for-row.
    #
    # ``t.log_inputs`` alone is NOT enough — the LangSmith pytest
    # plugin auto-captures fixture values at submit time. We also
    # overwrite ``run_tree.inputs`` directly, mirroring the
    # ``tau2_airline`` pattern (``test_tau2_airline.py:95-107``).
    #
    # The runner name is deliberately NOT in inputs: inputs are the
    # example's identity key for LangSmith example-upsert. Putting
    # ``runner`` there would fork one dataset row per runner and kill
    # the "one dataset, one experiment per runner" comparison. Runner
    # goes on the run as a per-run outputs field instead (see
    # :meth:`OolongCorrect.check`).
    clean_inputs = {
        "task_id": task.id,
        "dataset": task.dataset,
        "context_len": task.context_len,
        "task_type": task.task,
        "task_group": task.task_group,
        "answer_type": task.answer_type,
        "input_subset": task.input_subset,
        "question": task.question,
    }
    t.log_inputs(clean_inputs)
    t.log_reference_outputs({"answer": task.answer})
    run_tree = get_current_run_tree()
    if run_tree is not None:
        run_tree.inputs = clean_inputs
    else:
        logger.warning(
            "get_current_run_tree() returned None; "
            "LangSmith dataset inputs will fall back to auto-capture",
        )

    gold_answer = parse_gold(task.answer)
    scorer = TrajectoryScorer().success(
        OolongCorrect(runner_name=runner_name, task=task, gold=gold_answer)
    )

    try:
        query = task.question + ctx.query_addendum
        await run_agent_async(
            ctx.agent,
            model=model,
            query=query,
            initial_files=ctx.initial_files,
            scorer=scorer,
        )
    finally:
        ctx.teardown()
