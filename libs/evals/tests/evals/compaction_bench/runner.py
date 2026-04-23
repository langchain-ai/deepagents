"""End-to-end runner for a single compaction-bench ``(instance, technique)`` pair.

The runner is the only place in the compaction bench that actually
invokes a model. Everything else - the fixture, the script, the
graders, the scorecard - is pure-data and pure-function.

### What it does, top to bottom

1. Seeds a fresh temp directory with the instance's mini-repo fixture.
2. Builds a ``FilesystemBackend`` rooted at that directory so the
   agent's file tools (``ls``, ``read_file``, ``write_file``, ``grep``,
   ``glob``) operate on the fixture.
3. Asks the ``SummarizationTechnique`` to construct a middleware stack
   (summarization middleware + any supporting middleware).
4. Creates a ``create_deep_agent`` graph with an in-memory checkpointer
   so a single ``thread_id`` threads state across all scripted user
   turns.
5. Replays the instance's scripted ``UserMessage`` sequence turn-by-turn,
   sharing the ``thread_id`` so the compaction middleware sees the full
   conversation and fires on schedule.
6. After each turn, slices out the trajectory *delta* (the new steps
   added by that turn) so trajectory graders can attribute tool calls
   to the correct phase.
7. After the final turn, snapshots the filesystem into a
   ``dict[str, str]`` that matches the graders' expected shape
   (leading-slash keys).
8. Runs ``grade_all`` against everything and aggregates into a
   ``Scorecard``.

### Why this shape

Keeping the runner free of LangSmith feedback logging and free of
pytest wiring means the same function can be driven from the pytest
entry point (which handles feedback) or from an ad-hoc script
(single-instance smoke test, local debugging, notebook).

### What is intentionally *not* here

- **LangSmith feedback emission.** Done by the pytest entry point.
  The runner returns a ``Scorecard``; the caller decides what to do
  with it.
- **Retry / resume logic.** Not needed at instance-001 scale. A 20-turn
  run takes a handful of minutes and costs pennies; a failed run is
  just rerun.
- **Token-budget assertion.** Lives in a dedicated pre-flight unit
  test; the runner itself does not care whether compaction actually
  fired.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

from deepagents import create_deep_agent
from deepagents.backends.filesystem import FilesystemBackend
from langgraph.checkpoint.memory import MemorySaver

from tests.evals.compaction_bench.graders import (
    GraderContext,
    PerTurnTrajectory,
    grade_all,
    load_fixture,
)
from tests.evals.compaction_bench.scorecard import Scorecard
from tests.evals.utils import AgentStep, AgentTrajectory, run_agent

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from langchain_core.language_models import BaseChatModel
    from langgraph.graph.state import CompiledStateGraph

    from tests.evals.compaction_bench.task_spec import Instance
    from tests.evals.compaction_bench.techniques import SummarizationTechnique

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunArtifacts:
    """Everything the graders need from one end-to-end run.

    Kept separate from ``Scorecard`` so callers can inspect the raw
    artifacts for debugging (e.g. "why did G13 fire? let me look at
    turn 12's trajectory") without re-running the agent.

    Attributes:
        instance_id: The instance that was run.
        technique_name: The technique that was used.
        thread_id: The ``thread_id`` threaded through every turn.
        fixture_files: The fixture snapshot the run started from.
        final_files: The filesystem snapshot after the final turn.
        per_turn_trajectories: Per-turn trajectory slices, one entry per
            scripted ``UserMessage`` actually executed.
    """

    instance_id: str
    technique_name: str
    thread_id: str
    fixture_files: Mapping[str, str]
    final_files: Mapping[str, str]
    per_turn_trajectories: tuple[PerTurnTrajectory, ...]


# ---------------------------------------------------------------------------
# Filesystem seeding and snapshotting
# ---------------------------------------------------------------------------
#
# FilesystemBackend reads and writes real files on disk. That means: we
# need to physically copy the fixture to a temp directory before a run
# (so the agent can modify it without mutating the source tree) and we
# need to walk that temp directory after the run to reconstruct the
# ``dict[str, str]`` the graders expect.


def seed_fixture_to_disk(fixture_files: Mapping[str, str], root_dir: Path) -> None:
    """Materialize ``fixture_files`` under ``root_dir`` as real files.

    The mapping keys are the leading-slash paths produced by
    ``load_fixture``; they are joined onto ``root_dir`` without the
    leading slash. Intermediate directories are created as needed.
    Existing files with the same path are overwritten.

    Args:
        fixture_files: Fixture content, keyed by leading-slash path.
        root_dir: Directory to seed into. Must exist and be writable.

    Raises:
        ValueError: If any key is missing its leading slash (indicates
            the caller used ``pathlib.Path.relative_to`` incorrectly).
    """
    for key, content in fixture_files.items():
        if not key.startswith("/"):
            msg = f"fixture key must start with '/', got {key!r}"
            raise ValueError(msg)
        target = root_dir / key.lstrip("/")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


def snapshot_filesystem(root_dir: Path) -> dict[str, str]:
    """Walk ``root_dir`` and return its text contents keyed by leading-slash path.

    The format matches ``load_fixture`` exactly (leading slash,
    ``__pycache__`` skipped, binary files skipped) so graders can
    diff fixture vs final without any path massaging.

    Args:
        root_dir: Directory to snapshot.

    Returns:
        Mapping from leading-slash path (e.g. ``"/webhooks/partnerco.py"``)
        to file content.
    """
    return load_fixture(root_dir)


# ---------------------------------------------------------------------------
# Agent construction
# ---------------------------------------------------------------------------


def build_agent(
    *,
    model: BaseChatModel,
    technique: SummarizationTechnique,
    root_dir: Path,
) -> CompiledStateGraph:
    """Compose a ``create_deep_agent`` graph for one run.

    The backend is a ``FilesystemBackend`` with ``virtual_mode=True`` so
    the agent sees stable leading-slash paths (``/webhooks/partnerco.py``)
    rather than absolute host paths. That keeps grader trajectory
    analysis meaningful across different temp-dir locations.

    Implementation detail: ``create_deep_agent`` always installs a
    default ``SummarizationMiddleware`` in its base stack, and
    ``langchain.create_agent`` rejects two middleware instances sharing
    a name. We *replace* the default factory with one that returns the
    technique's configured middleware for the duration of the build,
    rather than layering our middleware on top. The patch is scoped to
    a try/finally so a failure in ``create_deep_agent`` cannot leave
    the module-level factory mutated.

    Args:
        model: The chat model the agent should use.
        technique: Supplies the summarization middleware for this run.
        root_dir: Directory the ``FilesystemBackend`` is rooted at.

    Returns:
        The compiled state graph, ready to invoke.
    """
    # Imports are local (not module-level) so ``deepagents.graph``'s
    # heavy import chain is only paid when an agent is actually being
    # built, and so the patch target is resolved at call time.
    from deepagents import graph as deepagents_graph

    backend = FilesystemBackend(root_dir=root_dir, virtual_mode=True)
    checkpointer = MemorySaver()

    def _technique_factory(
        inner_model: BaseChatModel,
        inner_backend: object,
    ) -> object:
        """Drop-in replacement for ``create_summarization_middleware``.

        Returns the technique's configured middleware regardless of the
        arguments deepagents would otherwise pass. We honor the model
        argument (in case deepagents routes a subagent model through
        here in a future change) by forwarding it to the technique.
        """
        _ = inner_backend  # Backend is supplied via closure; argument unused.
        return technique.build_summarization_middleware(
            consumer_model=inner_model,
            backend=backend,
        )

    original_factory = deepagents_graph.create_summarization_middleware
    deepagents_graph.create_summarization_middleware = (
        _technique_factory  # ty: ignore[invalid-assignment]
    )
    try:
        return create_deep_agent(
            model=model,
            backend=backend,
            checkpointer=checkpointer,
        )
    finally:
        deepagents_graph.create_summarization_middleware = original_factory


# ---------------------------------------------------------------------------
# Turn-loop execution
# ---------------------------------------------------------------------------


def _slice_new_steps(full_trajectory: AgentTrajectory, previous_step_count: int) -> AgentTrajectory:
    """Return a trajectory containing only the steps beyond ``previous_step_count``.

    ``run_agent`` returns a cumulative trajectory built from the full
    conversation state, so on turn N it already reflects the work of
    turns 1..N-1. Graders that reason per-phase (G12, G13) need the
    delta only; this helper carves it out without recomputing anything.

    The ``files`` view is intentionally carried over unchanged - no
    grader consults ``PerTurnTrajectory.trajectory.files`` today, and
    the per-turn files dict would be a confusing half-state (tools
    executed during the turn, not a stable post-turn snapshot). The
    authoritative filesystem snapshot lives in ``RunArtifacts.final_files``.

    Args:
        full_trajectory: The cumulative trajectory returned by ``run_agent``.
        previous_step_count: Number of steps present before this turn.

    Returns:
        A new ``AgentTrajectory`` scoped to the delta steps. Step
        indices are renumbered from 1 so downstream consumers can
        treat each slice as a standalone trajectory.
    """
    new_steps: list[AgentStep] = []
    for offset, old_step in enumerate(full_trajectory.steps[previous_step_count:], start=1):
        new_steps.append(
            AgentStep(
                index=offset,
                action=old_step.action,
                observations=list(old_step.observations),
            )
        )
    return AgentTrajectory(steps=new_steps, files=full_trajectory.files)


def execute_run(
    agent: CompiledStateGraph,
    instance: Instance,
    *,
    model: BaseChatModel,
    thread_id: str | None = None,
    eval_metadata: Mapping[str, object] | None = None,
) -> tuple[tuple[PerTurnTrajectory, ...], str]:
    """Replay the instance script turn-by-turn against ``agent``.

    Invokes ``run_agent`` once per scripted ``UserMessage``, sharing a
    single ``thread_id`` so the compaction middleware sees the full
    accumulated conversation and fires when its trigger is met.

    Args:
        agent: The compiled agent graph.
        instance: The instance whose ``messages`` drive the loop.
        model: The chat model (used by ``run_agent`` for LangSmith logging).
        thread_id: Optional thread ID. Generated if omitted.
        eval_metadata: Optional metadata attached to each ``run_agent`` call
            (will appear on LangSmith runs).

    Returns:
        A ``(per_turn_trajectories, thread_id)`` tuple. The list has one
        entry per executed turn, in script order.
    """
    if thread_id is None:
        thread_id = str(uuid.uuid4())

    per_turn: list[PerTurnTrajectory] = []
    previous_step_count = 0

    for message in instance.messages:
        logger.info(
            "compaction_bench turn=%d phase=%s",
            message.turn,
            message.phase.value,
        )
        cumulative = run_agent(
            agent,
            model=model,
            thread_id=thread_id,
            query=message.content,
            eval_metadata=dict(eval_metadata) if eval_metadata else None,
        )
        new_slice = _slice_new_steps(cumulative, previous_step_count)
        per_turn.append(
            PerTurnTrajectory(
                turn=message.turn,
                phase=message.phase,
                trajectory=new_slice,
            )
        )
        previous_step_count = len(cumulative.steps)

    return tuple(per_turn), thread_id


# ---------------------------------------------------------------------------
# Top-level convenience
# ---------------------------------------------------------------------------


def run_instance(
    *,
    instance: Instance,
    technique: SummarizationTechnique,
    model: BaseChatModel,
    root_dir: Path,
    thread_id: str | None = None,
    eval_metadata: Mapping[str, object] | None = None,
) -> RunArtifacts:
    """Seed, run, and snapshot a single ``(instance, technique)`` pair.

    ``root_dir`` is seeded with the instance's fixture, an agent is
    constructed, the instance's ``messages`` are replayed, and the
    filesystem is snapshotted afterwards.

    The caller owns ``root_dir``'s lifecycle - this function does not
    create or delete it. Typical pytest usage passes ``tmp_path``.

    Args:
        instance: The instance to run.
        technique: The summarization technique under test.
        model: The consumer model the agent runs on.
        root_dir: Pre-existing directory the fixture will be seeded into.
        thread_id: Optional shared thread ID. Generated if omitted.
        eval_metadata: Optional metadata for LangSmith tagging.

    Returns:
        ``RunArtifacts`` capturing fixture, final filesystem, and
        per-turn trajectories - ready to pass to ``grade_run``.
    """
    fixture_files = load_fixture(instance.fixture_dir)
    seed_fixture_to_disk(fixture_files, root_dir)

    agent = build_agent(model=model, technique=technique, root_dir=root_dir)
    per_turn, thread_id = execute_run(
        agent,
        instance,
        model=model,
        thread_id=thread_id,
        eval_metadata=eval_metadata,
    )

    final_files = snapshot_filesystem(root_dir)

    return RunArtifacts(
        instance_id=instance.id,
        technique_name=technique.name,
        thread_id=thread_id,
        fixture_files=fixture_files,
        final_files=final_files,
        per_turn_trajectories=per_turn,
    )


# ---------------------------------------------------------------------------
# Grading a run
# ---------------------------------------------------------------------------


def grade_run(
    artifacts: RunArtifacts,
    instance: Instance,
    *,
    include_judge: bool = False,
    include_subprocess: bool = False,
    judge_model: str | None = None,
) -> Scorecard:
    """Run all graders against a ``RunArtifacts`` and aggregate.

    Separated from ``run_instance`` so callers can (1) drive the
    expensive run once and (2) re-grade cheaply when graders evolve,
    without re-invoking the model. Artifacts are serializable enough
    (modulo ``AIMessage`` internals) that they can be pickled for
    later re-grading during grader development.

    Args:
        artifacts: The artifacts returned by ``run_instance``.
        instance: The instance metadata (canonical paths, rejection
            turns, etc).
        include_judge: Whether to run the LLM-judge-backed graders
            (G15). Off by default because it costs model calls.
        include_subprocess: Whether to run the pytest-subprocess grader
            (G16). Off by default because it shells out.
        judge_model: Optional override for the judge model identifier.

    Returns:
        ``Scorecard`` aggregating the checkpoint results.
    """
    ctx = GraderContext(
        instance=instance,
        fixture_files=artifacts.fixture_files,
        final_files=artifacts.final_files,
        per_turn_trajectories=artifacts.per_turn_trajectories,
    )
    results = grade_all(
        ctx,
        include_judge=include_judge,
        include_subprocess=include_subprocess,
        judge_model=judge_model,
    )
    return Scorecard.from_results(
        instance_id=artifacts.instance_id,
        technique=artifacts.technique_name,
        results=results,
    )


def run_and_grade(
    *,
    instance: Instance,
    technique: SummarizationTechnique,
    model: BaseChatModel,
    root_dir: Path,
    include_judge: bool = False,
    include_subprocess: bool = False,
    judge_model: str | None = None,
    thread_id: str | None = None,
    eval_metadata: Mapping[str, object] | None = None,
) -> tuple[RunArtifacts, Scorecard]:
    """Convenience wrapper that chains ``run_instance`` and ``grade_run``.

    Callers that only want a scorecard (most scripts, the pytest entry
    point) use this; callers that want to do something bespoke with
    the artifacts (inspecting per-turn trajectories, caching for
    re-grading) call the two halves separately.

    Args:
        instance: The instance to run.
        technique: Summarization technique under test.
        model: Consumer model.
        root_dir: Pre-existing directory for the fixture.
        include_judge: Whether to enable the LLM-judge grader.
        include_subprocess: Whether to enable the pytest-subprocess grader.
        judge_model: Optional judge model override.
        thread_id: Optional shared thread ID.
        eval_metadata: Optional LangSmith metadata.

    Returns:
        The ``(artifacts, scorecard)`` pair.
    """
    artifacts = run_instance(
        instance=instance,
        technique=technique,
        model=model,
        root_dir=root_dir,
        thread_id=thread_id,
        eval_metadata=eval_metadata,
    )
    scorecard = grade_run(
        artifacts,
        instance,
        include_judge=include_judge,
        include_subprocess=include_subprocess,
        judge_model=judge_model,
    )
    return artifacts, scorecard


__all__ = [
    "RunArtifacts",
    "build_agent",
    "execute_run",
    "grade_run",
    "run_and_grade",
    "run_instance",
    "seed_fixture_to_disk",
    "snapshot_filesystem",
]
