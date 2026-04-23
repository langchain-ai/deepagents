"""Unit tests for the compaction-bench runner.

The runner is the one module that would normally invoke a model. These
tests cover its mechanics without any model call by patching
``tests.evals.compaction_bench.runner.run_agent`` with a deterministic
fake that grows a cumulative trajectory turn-by-turn.

The coverage targets:

- ``seed_fixture_to_disk`` materializes files correctly (including
  nested paths and overwrites), rejects malformed keys.
- ``snapshot_filesystem`` round-trips cleanly with ``load_fixture``.
- ``_slice_new_steps`` returns *only* the delta and renumbers indices.
- ``execute_run`` produces one ``PerTurnTrajectory`` per scripted turn,
  with the correct phase and delta steps.
- ``grade_run`` returns a ``Scorecard`` aggregating real grader output.
- ``run_and_grade`` end-to-end wiring (with patched run_agent +
  patched build_agent).
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from tests.evals.compaction_bench import runner
from tests.evals.compaction_bench.graders import (
    PerTurnTrajectory,
    load_fixture,
)
from tests.evals.compaction_bench.instance_001_partnerco import INSTANCE
from tests.evals.compaction_bench.runner import (
    RunArtifacts,
    _slice_new_steps,
    execute_run,
    grade_run,
    run_and_grade,
    seed_fixture_to_disk,
    snapshot_filesystem,
)
from tests.evals.compaction_bench.scorecard import Scorecard
from tests.evals.compaction_bench.task_spec import FIXTURES_ROOT
from tests.evals.utils import AgentStep, AgentTrajectory

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ai_step(
    index: int, *, text: str = "", tool_calls: list[dict[str, Any]] | None = None
) -> AgentStep:
    """Build an ``AgentStep`` with an ``AIMessage`` and no observations.

    The ``id`` is stable-per-index so step-equality checks don't
    incidentally catch ``uuid`` randomness.
    """
    action = AIMessage(content=text, tool_calls=tool_calls or [])
    return AgentStep(index=index, action=action, observations=[])


def _ai_step_with_observations(
    index: int,
    *,
    tool_name: str,
    tool_args: dict[str, Any],
    tool_output: str = "ok",
) -> AgentStep:
    """Build an ``AgentStep`` that makes one tool call + one observation."""
    tool_id = f"call_{index}"
    action = AIMessage(
        content="",
        tool_calls=[{"name": tool_name, "args": tool_args, "id": tool_id}],
    )
    obs = ToolMessage(content=tool_output, tool_call_id=tool_id)
    return AgentStep(index=index, action=action, observations=[obs])


class _TrajectoryGrower:
    """Deterministic stand-in for ``run_agent`` that grows a trajectory.

    Each call appends a configurable number of steps (default 1) to the
    trajectory and returns a fresh ``AgentTrajectory`` reflecting the
    full cumulative state - exactly matching how the real ``run_agent``
    behaves when invoked against an agent with a shared ``thread_id``.

    This is how we simulate an agent making multiple steps per user
    turn without standing up a real model.
    """

    def __init__(self, steps_per_turn: list[int] | None = None) -> None:
        self._steps: list[AgentStep] = []
        self._calls = 0
        self._steps_per_turn = steps_per_turn or []
        self.queries: list[str] = []

    def __call__(self, agent: object, **kwargs: Any) -> AgentTrajectory:
        _ = agent
        query = kwargs.get("query")
        if isinstance(query, str):
            self.queries.append(query)
        count = self._steps_per_turn[self._calls] if self._calls < len(self._steps_per_turn) else 1
        for _i in range(count):
            idx = len(self._steps) + 1
            self._steps.append(_ai_step(idx, text=f"turn_call_{self._calls}_step_{idx}"))
        self._calls += 1
        return AgentTrajectory(steps=list(self._steps), files={})


# ---------------------------------------------------------------------------
# seed_fixture_to_disk / snapshot_filesystem
# ---------------------------------------------------------------------------


class TestSeedFixtureToDisk:
    def test_materializes_files_with_nested_paths(self, tmp_path: Path) -> None:
        fixture = {
            "/README.md": "# hi\n",
            "/webhooks/partnerco.py": "print('hello')\n",
            "/common/logger.py": "def log(): pass\n",
        }
        seed_fixture_to_disk(fixture, tmp_path)

        assert (tmp_path / "README.md").read_text() == "# hi\n"
        assert (tmp_path / "webhooks" / "partnerco.py").read_text() == "print('hello')\n"
        assert (tmp_path / "common" / "logger.py").read_text() == "def log(): pass\n"

    def test_overwrites_existing_files(self, tmp_path: Path) -> None:
        (tmp_path / "README.md").write_text("old content")
        seed_fixture_to_disk({"/README.md": "new content"}, tmp_path)
        assert (tmp_path / "README.md").read_text() == "new content"

    def test_rejects_keys_without_leading_slash(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="leading slash|'/'"):
            seed_fixture_to_disk({"README.md": "content"}, tmp_path)

    def test_empty_fixture_creates_no_files(self, tmp_path: Path) -> None:
        seed_fixture_to_disk({}, tmp_path)
        assert list(tmp_path.iterdir()) == []


class TestSnapshotFilesystem:
    def test_round_trips_with_seed(self, tmp_path: Path) -> None:
        fixture = {
            "/a.py": "x = 1\n",
            "/dir/b.py": "y = 2\n",
            "/dir/nested/c.py": "z = 3\n",
        }
        seed_fixture_to_disk(fixture, tmp_path)
        snap = snapshot_filesystem(tmp_path)
        assert snap == fixture

    def test_captures_new_files_created_under_root(self, tmp_path: Path) -> None:
        seed_fixture_to_disk({"/original.py": "a"}, tmp_path)
        (tmp_path / "NOTES.md").write_text("# review\n")
        snap = snapshot_filesystem(tmp_path)
        assert snap["/NOTES.md"] == "# review\n"
        assert snap["/original.py"] == "a"

    def test_real_instance_001_fixture_round_trips(self, tmp_path: Path) -> None:
        """Seeding + snapshot preserves the real fixture byte-for-byte."""
        fixture = load_fixture(FIXTURES_ROOT / "instance_001")
        seed_fixture_to_disk(fixture, tmp_path)
        assert snapshot_filesystem(tmp_path) == fixture


# ---------------------------------------------------------------------------
# _slice_new_steps
# ---------------------------------------------------------------------------


class TestSliceNewSteps:
    def test_empty_previous_returns_all_renumbered(self) -> None:
        original = AgentTrajectory(
            steps=[_ai_step(1), _ai_step(2), _ai_step(3)],
            files={},
        )
        sliced = _slice_new_steps(original, previous_step_count=0)
        assert [s.index for s in sliced.steps] == [1, 2, 3]
        assert sliced.steps[0].action.text == original.steps[0].action.text

    def test_slices_off_first_n_steps(self) -> None:
        original = AgentTrajectory(
            steps=[_ai_step(i, text=f"step{i}") for i in range(1, 6)],
            files={},
        )
        sliced = _slice_new_steps(original, previous_step_count=2)
        assert [s.index for s in sliced.steps] == [1, 2, 3]
        # Indices are renumbered, content preserved.
        assert sliced.steps[0].action.text == "step3"
        assert sliced.steps[2].action.text == "step5"

    def test_previous_equals_len_returns_empty_steps(self) -> None:
        original = AgentTrajectory(steps=[_ai_step(1), _ai_step(2)], files={})
        sliced = _slice_new_steps(original, previous_step_count=2)
        assert sliced.steps == []

    def test_previous_greater_than_len_returns_empty_steps(self) -> None:
        """Defensive: misordered state shouldn't explode."""
        original = AgentTrajectory(steps=[_ai_step(1)], files={})
        sliced = _slice_new_steps(original, previous_step_count=99)
        assert sliced.steps == []

    def test_preserves_observations(self) -> None:
        step = _ai_step_with_observations(
            1, tool_name="read_file", tool_args={"file_path": "/x.py"}
        )
        original = AgentTrajectory(steps=[step], files={})
        sliced = _slice_new_steps(original, previous_step_count=0)
        assert len(sliced.steps[0].observations) == 1
        assert sliced.steps[0].observations[0].content == "ok"

    def test_files_dict_is_carried_over(self) -> None:
        original = AgentTrajectory(steps=[_ai_step(1)], files={"/x.py": "content"})
        sliced = _slice_new_steps(original, previous_step_count=0)
        assert sliced.files == {"/x.py": "content"}


# ---------------------------------------------------------------------------
# execute_run
# ---------------------------------------------------------------------------


class TestExecuteRun:
    def test_one_per_turn_trajectory_per_scripted_message(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        grower = _TrajectoryGrower()
        monkeypatch.setattr(runner, "run_agent", grower)

        per_turn, thread_id = execute_run(
            MagicMock(),
            INSTANCE,
            model=MagicMock(),
            thread_id="fixed-thread",
        )

        assert len(per_turn) == len(INSTANCE.messages)
        assert thread_id == "fixed-thread"
        # Each queried content matches the scripted message content.
        assert grower.queries == [m.content for m in INSTANCE.messages]

    def test_generates_thread_id_when_omitted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        grower = _TrajectoryGrower()
        monkeypatch.setattr(runner, "run_agent", grower)

        _, thread_id = execute_run(MagicMock(), INSTANCE, model=MagicMock())

        # Not empty, and parses as a UUID (we do not care which one).
        assert thread_id
        uuid.UUID(thread_id)

    def test_per_turn_phase_matches_script(self, monkeypatch: pytest.MonkeyPatch) -> None:
        grower = _TrajectoryGrower()
        monkeypatch.setattr(runner, "run_agent", grower)

        per_turn, _ = execute_run(MagicMock(), INSTANCE, model=MagicMock())

        for pt, msg in zip(per_turn, INSTANCE.messages, strict=True):
            assert pt.turn == msg.turn
            assert pt.phase is msg.phase

    def test_delta_step_count_matches_fake_emission(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Turn 1 emits 3 steps; turn 2 emits 2 steps; remaining turns
        # emit the default (1).
        grower = _TrajectoryGrower(steps_per_turn=[3, 2])
        monkeypatch.setattr(runner, "run_agent", grower)

        per_turn, _ = execute_run(MagicMock(), INSTANCE, model=MagicMock())

        assert len(per_turn[0].trajectory.steps) == 3
        assert len(per_turn[1].trajectory.steps) == 2
        # Remaining turns default to 1 step each.
        assert all(len(p.trajectory.steps) == 1 for p in per_turn[2:])

    def test_per_turn_indices_renumber_from_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        grower = _TrajectoryGrower(steps_per_turn=[3])
        monkeypatch.setattr(runner, "run_agent", grower)

        per_turn, _ = execute_run(MagicMock(), INSTANCE, model=MagicMock())

        # Each delta starts at 1, not the cumulative index.
        assert [s.index for s in per_turn[0].trajectory.steps] == [1, 2, 3]
        # Turn 2 also starts at 1 in its *own* slice.
        assert per_turn[1].trajectory.steps[0].index == 1


# ---------------------------------------------------------------------------
# grade_run
# ---------------------------------------------------------------------------


class TestGradeRun:
    def test_returns_scorecard_with_correct_ids(self) -> None:
        fixture = load_fixture(FIXTURES_ROOT / "instance_001")
        artifacts = RunArtifacts(
            instance_id=INSTANCE.id,
            technique_name="deepagents",
            thread_id="t",
            fixture_files=fixture,
            final_files=fixture,  # no edits
            per_turn_trajectories=(),
        )
        sc = grade_run(artifacts, INSTANCE)
        assert isinstance(sc, Scorecard)
        assert sc.instance_id == INSTANCE.id
        assert sc.technique == "deepagents"
        # Something was scored.
        assert sc.all_results

    def test_final_files_identical_to_fixture_produces_nonzero_goal_drift(self) -> None:
        """A no-op run keeps billing untouched and deps unchanged -> G1/G2 pass."""
        fixture = load_fixture(FIXTURES_ROOT / "instance_001")
        artifacts = RunArtifacts(
            instance_id=INSTANCE.id,
            technique_name="deepagents",
            thread_id="t",
            fixture_files=fixture,
            final_files=fixture,
            per_turn_trajectories=(),
        )
        sc = grade_run(artifacts, INSTANCE)
        g1 = next(r for r in sc.all_results if r.checkpoint_id == "G1")
        g2 = next(r for r in sc.all_results if r.checkpoint_id == "G2")
        assert g1.score == 1.0
        assert g2.score == 1.0


# ---------------------------------------------------------------------------
# run_and_grade (end-to-end wiring)
# ---------------------------------------------------------------------------


class TestRunAndGrade:
    def test_wires_seeding_execution_and_grading(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Smoke test: patch run_agent + build_agent, exercise full pipeline."""
        grower = _TrajectoryGrower()
        monkeypatch.setattr(runner, "run_agent", grower)
        # ``build_agent`` normally calls ``create_deep_agent``, which pulls in
        # a real compiled graph; a mock is sufficient here because
        # ``run_agent`` is already patched to ignore the agent argument.
        monkeypatch.setattr(runner, "build_agent", lambda **_kwargs: MagicMock())

        fake_technique = MagicMock()
        fake_technique.name = "fake_tech"

        artifacts, scorecard = run_and_grade(
            instance=INSTANCE,
            technique=fake_technique,
            model=MagicMock(),
            root_dir=tmp_path,
        )

        # Fixture got seeded to disk.
        assert (tmp_path / "webhooks" / "generic_handler.py").exists()
        # Final-files snapshot matches the seeded fixture (no writes happened).
        assert artifacts.final_files == artifacts.fixture_files
        # Per-turn trajectories: one per scripted turn.
        assert len(artifacts.per_turn_trajectories) == len(INSTANCE.messages)
        assert all(isinstance(pt, PerTurnTrajectory) for pt in artifacts.per_turn_trajectories)
        # Scorecard names propagate from artifacts.
        assert scorecard.technique == "fake_tech"
        assert scorecard.instance_id == INSTANCE.id

    def test_scorecard_flags_unchanged_canonical_handler(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If the agent writes nothing, G3 (canonical handler created) fails."""
        grower = _TrajectoryGrower()
        monkeypatch.setattr(runner, "run_agent", grower)
        monkeypatch.setattr(runner, "build_agent", lambda **_kwargs: MagicMock())

        fake_technique = MagicMock()
        fake_technique.name = "no_op"

        _, scorecard = run_and_grade(
            instance=INSTANCE,
            technique=fake_technique,
            model=MagicMock(),
            root_dir=tmp_path,
        )
        g3 = next(r for r in scorecard.all_results if r.checkpoint_id == "G3")
        assert g3.score == 0.0
        assert "canonical_handler_new" in g3.evidence or "partnerco.py" in g3.evidence
