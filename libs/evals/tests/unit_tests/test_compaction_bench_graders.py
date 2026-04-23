"""Unit tests for the compaction-bench graders.

These tests drive each grader with hand-crafted inputs - no agent run,
no model call, no filesystem I/O beyond a temp directory for the
pytest-subprocess grader. They cover the happy path and the most
important failure mode for every grader, plus the aggregation logic
that bundles results into a ``Scorecard``.

The tests are intentionally shaped so that ``make test`` in ``libs/evals/``
runs them with zero network access.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

from tests.evals.compaction_bench import graders
from tests.evals.compaction_bench.graders import (
    GraderContext,
    PerTurnTrajectory,
    grade_g1,
    grade_g2,
    grade_g3,
    grade_g4,
    grade_g5,
    grade_g6,
    grade_g7,
    grade_g8,
    grade_g9,
    grade_g10,
    grade_g11,
    grade_g12,
    grade_g13,
    grade_g14,
    grade_g16,
    load_fixture,
)
from tests.evals.compaction_bench.instance_001_partnerco import INSTANCE
from tests.evals.compaction_bench.scorecard import (
    CheckpointResult,
    Scorecard,
    diff,
)
from tests.evals.compaction_bench.task_spec import (
    CHECKPOINTS,
    CHECKPOINTS_BY_ID,
    FIXTURES_ROOT,
    Constraint,
    FailureMode,
    Instance,
    Phase,
    Rejection,
    UserMessage,
)
from tests.evals.utils import AgentStep, AgentTrajectory

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fixture_files() -> dict[str, str]:
    """Real instance_001 fixture contents. Cached per module for speed."""
    return load_fixture(FIXTURES_ROOT / "instance_001")


@pytest.fixture
def instance() -> Instance:
    """The canonical instance_001 object."""
    return INSTANCE


def _ctx(
    instance: Instance,
    fixture_files: dict[str, str],
    *,
    overlay: dict[str, str] | None = None,
    removed: set[str] | None = None,
    per_turn: Sequence[PerTurnTrajectory] = (),
) -> GraderContext:
    """Build a ``GraderContext`` where ``final_files`` = fixture + overlay minus removed.

    Args:
        instance: The instance under test.
        fixture_files: Base fixture contents.
        overlay: Files to add or overwrite on top of the fixture.
        removed: Paths to drop from the fixture entirely.
        per_turn: Per-turn trajectories for trajectory-dependent graders.

    Returns:
        A fresh ``GraderContext``.
    """
    final: dict[str, str] = dict(fixture_files)
    if overlay:
        final.update(overlay)
    if removed:
        for path in removed:
            final.pop(path, None)
    return GraderContext(
        instance=instance,
        fixture_files=fixture_files,
        final_files=final,
        per_turn_trajectories=tuple(per_turn),
    )


def _ai_message_with_tool_calls(calls: Sequence[tuple[str, dict]]) -> AIMessage:
    """Build an ``AIMessage`` carrying the given tool calls.

    Args:
        calls: Sequence of ``(tool_name, args_dict)`` pairs.

    Returns:
        An ``AIMessage`` with populated ``tool_calls``.
    """
    return AIMessage(
        content="",
        tool_calls=[
            {"name": name, "args": args, "id": f"call_{i}"} for i, (name, args) in enumerate(calls)
        ],
    )


def _per_turn(turn: int, phase: Phase, read_paths: Sequence[str]) -> PerTurnTrajectory:
    """Build a ``PerTurnTrajectory`` that reads the given file paths in order."""
    step = AgentStep(
        index=1,
        action=_ai_message_with_tool_calls([("read_file", {"file_path": p}) for p in read_paths]),
        observations=[],
    )
    return PerTurnTrajectory(
        turn=turn,
        phase=phase,
        trajectory=AgentTrajectory(steps=[step], files={}),
    )


# Minimal, correct "happy path" handler content that passes every
# canonical-handler grader (G3, G4, G5). Tests that want to break a
# single grader overlay a slightly-mangled variant of this.
_HAPPY_HANDLER = """\
from common.audit import log_event
from common.idempotency import already_processed, mark_processed
from common.logger import get_logger

logger = get_logger(__name__)


def handle(event, *, secret, signature, raw):
    event_id = event["id"]
    if already_processed(event_id):
        return
    log_event("webhook.partnerco", {"id": event_id})
    mark_processed(event_id)
"""

# Minimal retry helper that satisfies G6 on both bounds.
_HAPPY_RETRY = """\
import time


MAX_ATTEMPTS = 5
MAX_DELAY = 30  # seconds, exponential cap


def with_retry(fn, *args, **kwargs):
    for attempt in range(MAX_ATTEMPTS):
        try:
            return fn(*args, **kwargs)
        except Exception:
            delay = min(MAX_DELAY, 2 ** attempt)
            time.sleep(delay)
    return None
"""

_HAPPY_RATE_CONFIG = """\
tiers:
  standard: 10
  priority: 50
  enterprise: 200

overrides:
  acme:
    enterprise: 500
"""

_HAPPY_NOTES = """\
# PartnerCo integration NOTES

## Constraints followed
- No changes to the billing module (Feature A).
- No new package dependencies (Feature A).
- Webhook events are deduplicated for idempotency by event id.
- Uses common.logger.get_logger across the new handler.
- No blocking IO in the hot path; keeps p99 latency under 50ms.
- Audit logging via common.audit on every event accepted.
- Retries use exponential backoff capped at 30 seconds, max 5 attempts (Feature B).
- Rate limiting has three tiers with per-partner overrides (Feature C).

## Alternatives rejected
- Extending the generic handler would have pulled in a new cryptography library,
  violating the no-new-deps rule.
- Using tenacity for retries was rejected for the same no-new-dependencies reason.
"""


def _ideal_overlay(instance: Instance) -> dict[str, str]:
    """Return an overlay that makes every deterministic grader pass on the fixture."""
    return {
        instance.canonical_handler_new: _HAPPY_HANDLER,
        "/webhooks/retry.py": _HAPPY_RETRY,
        "/ratelimit/tiered_config.yaml": _HAPPY_RATE_CONFIG,
        instance.notes_path: _HAPPY_NOTES,
    }


# ---------------------------------------------------------------------------
# task_spec catalog
# ---------------------------------------------------------------------------


class TestCheckpointCatalog:
    """Self-consistency of the checkpoint catalog."""

    def test_ids_are_unique(self) -> None:
        ids = [c.id for c in CHECKPOINTS]
        assert len(ids) == len(set(ids))

    def test_every_failure_mode_has_at_least_one_checkpoint(self) -> None:
        seen = {c.failure_mode for c in CHECKPOINTS}
        for mode in FailureMode:
            assert mode in seen, f"no checkpoint for {mode!r}"

    def test_catalog_index_is_complete(self) -> None:
        assert set(CHECKPOINTS_BY_ID) == {c.id for c in CHECKPOINTS}


# ---------------------------------------------------------------------------
# Instance smoke tests
# ---------------------------------------------------------------------------


class TestInstance001:
    """Self-consistency of the hand-authored instance."""

    def test_turn_numbers_are_contiguous(self) -> None:
        turns = [m.turn for m in INSTANCE.messages]
        assert turns == list(range(1, len(turns) + 1))

    def test_every_constraint_is_introduced_exactly_once(self) -> None:
        introduced: list[Constraint] = []
        for msg in INSTANCE.messages:
            introduced.extend(msg.introduces)
        assert len(introduced) == len(set(introduced))
        assert set(introduced) == set(Constraint)

    def test_rejections_map_to_real_turns(self) -> None:
        for rejection in Rejection:
            turn = INSTANCE.rejection_turn(rejection)
            assert INSTANCE.phase_at_turn(turn) in {
                Phase.FEATURE_A,
                Phase.FEATURE_B,
                Phase.FEATURE_C,
            }

    def test_fixture_dir_exists(self) -> None:
        assert INSTANCE.fixture_dir.is_dir()

    def test_canonical_handler_not_in_fixture(self, fixture_files: dict[str, str]) -> None:
        # The agent must *create* the canonical handler; it should not
        # already be in the fixture.
        assert INSTANCE.canonical_handler_new not in fixture_files

    def test_rejected_generic_handler_is_in_fixture(self, fixture_files: dict[str, str]) -> None:
        assert INSTANCE.rejected_generic_handler in fixture_files


# ---------------------------------------------------------------------------
# G1 — billing untouched
# ---------------------------------------------------------------------------


class TestG1BillingUntouched:
    def test_passes_when_billing_unchanged(
        self, instance: Instance, fixture_files: dict[str, str]
    ) -> None:
        ctx = _ctx(instance, fixture_files, overlay=_ideal_overlay(instance))
        result = grade_g1(ctx)
        assert result.score == 1.0

    def test_fails_when_billing_file_modified(
        self, instance: Instance, fixture_files: dict[str, str]
    ) -> None:
        ctx = _ctx(
            instance,
            fixture_files,
            overlay={"/billing/ratelimiter.py": "# tampered\n"},
        )
        result = grade_g1(ctx)
        assert result.score == 0.0
        assert "billing" in result.evidence

    def test_fails_when_billing_file_deleted(
        self, instance: Instance, fixture_files: dict[str, str]
    ) -> None:
        ctx = _ctx(instance, fixture_files, removed={"/billing/ratelimiter.py"})
        result = grade_g1(ctx)
        assert result.score == 0.0
        assert "deleted" in result.evidence

    def test_fails_when_new_file_added_under_billing(
        self, instance: Instance, fixture_files: dict[str, str]
    ) -> None:
        ctx = _ctx(instance, fixture_files, overlay={"/billing/new.py": "x = 1\n"})
        result = grade_g1(ctx)
        assert result.score == 0.0


# ---------------------------------------------------------------------------
# G2 — no new dependencies
# ---------------------------------------------------------------------------


class TestG2NoNewDeps:
    def test_passes_with_unchanged_pyproject(
        self, instance: Instance, fixture_files: dict[str, str]
    ) -> None:
        result = grade_g2(_ctx(instance, fixture_files))
        assert result.score == 1.0

    def test_fails_when_dep_added(self, instance: Instance, fixture_files: dict[str, str]) -> None:
        fixture_py = fixture_files["/pyproject.toml"]
        tampered = fixture_py.replace(
            '"pyyaml>=6.0.2",',
            '"pyyaml>=6.0.2",\n    "tenacity>=9.0.0",',
        )
        ctx = _ctx(instance, fixture_files, overlay={"/pyproject.toml": tampered})
        result = grade_g2(ctx)
        assert result.score == 0.0
        assert "tenacity" in result.evidence

    def test_passes_when_dep_removed(
        self, instance: Instance, fixture_files: dict[str, str]
    ) -> None:
        # Removing a dep should still pass (we only disallow additions).
        fixture_py = fixture_files["/pyproject.toml"]
        trimmed = fixture_py.replace('\n    "pyyaml>=6.0.2",', "")
        ctx = _ctx(instance, fixture_files, overlay={"/pyproject.toml": trimmed})
        assert grade_g2(ctx).score == 1.0


# ---------------------------------------------------------------------------
# G3 — idempotency in canonical handler
# ---------------------------------------------------------------------------


class TestG3Idempotency:
    def test_passes_for_correct_handler(
        self, instance: Instance, fixture_files: dict[str, str]
    ) -> None:
        ctx = _ctx(
            instance,
            fixture_files,
            overlay={instance.canonical_handler_new: _HAPPY_HANDLER},
        )
        assert grade_g3(ctx).score == 1.0

    def test_fails_when_handler_missing(
        self, instance: Instance, fixture_files: dict[str, str]
    ) -> None:
        result = grade_g3(_ctx(instance, fixture_files))
        assert result.score == 0.0
        assert "missing" in result.evidence

    def test_fails_when_mark_processed_not_called(
        self, instance: Instance, fixture_files: dict[str, str]
    ) -> None:
        handler = _HAPPY_HANDLER.replace("mark_processed(event_id)", "pass")
        ctx = _ctx(
            instance,
            fixture_files,
            overlay={instance.canonical_handler_new: handler},
        )
        result = grade_g3(ctx)
        assert result.score == 0.0
        assert "mark_processed" in result.evidence


# ---------------------------------------------------------------------------
# G4 — existing logger used
# ---------------------------------------------------------------------------


class TestG4ExistingLogger:
    def test_passes_when_only_common_logger_used(
        self, instance: Instance, fixture_files: dict[str, str]
    ) -> None:
        ctx = _ctx(
            instance,
            fixture_files,
            overlay={instance.canonical_handler_new: _HAPPY_HANDLER},
        )
        assert grade_g4(ctx).score == 1.0

    def test_partial_credit_when_both_used(
        self, instance: Instance, fixture_files: dict[str, str]
    ) -> None:
        handler = _HAPPY_HANDLER + "\nimport logging\nlogging.getLogger('x')\n"
        ctx = _ctx(
            instance,
            fixture_files,
            overlay={instance.canonical_handler_new: handler},
        )
        assert grade_g4(ctx).score == 0.5

    def test_fails_when_common_logger_absent(
        self, instance: Instance, fixture_files: dict[str, str]
    ) -> None:
        handler = "import logging\nlog = logging.getLogger(__name__)\n"
        ctx = _ctx(
            instance,
            fixture_files,
            overlay={instance.canonical_handler_new: handler},
        )
        assert grade_g4(ctx).score == 0.0


# ---------------------------------------------------------------------------
# G5 — audit logging
# ---------------------------------------------------------------------------


class TestG5AuditLogging:
    def test_passes_when_audit_wired(
        self, instance: Instance, fixture_files: dict[str, str]
    ) -> None:
        ctx = _ctx(
            instance,
            fixture_files,
            overlay={instance.canonical_handler_new: _HAPPY_HANDLER},
        )
        assert grade_g5(ctx).score == 1.0

    def test_fails_when_no_log_event_call(
        self, instance: Instance, fixture_files: dict[str, str]
    ) -> None:
        handler = _HAPPY_HANDLER.replace("log_event(", "# log_event(")
        ctx = _ctx(
            instance,
            fixture_files,
            overlay={instance.canonical_handler_new: handler},
        )
        assert grade_g5(ctx).score == 0.0


# ---------------------------------------------------------------------------
# G6 — retry bounds
# ---------------------------------------------------------------------------


class TestG6RetryBounds:
    def test_passes_when_both_bounds_present(
        self, instance: Instance, fixture_files: dict[str, str]
    ) -> None:
        ctx = _ctx(
            instance,
            fixture_files,
            overlay={"/webhooks/retry.py": _HAPPY_RETRY},
        )
        assert grade_g6(ctx).score == 1.0

    def test_partial_when_only_cap_present(
        self, instance: Instance, fixture_files: dict[str, str]
    ) -> None:
        retry = _HAPPY_RETRY.replace("MAX_ATTEMPTS = 5", "MAX_ATTEMPTS = 9")
        retry = retry.replace("range(MAX_ATTEMPTS)", "range(MAX_ATTEMPTS)")
        ctx = _ctx(instance, fixture_files, overlay={"/webhooks/retry.py": retry})
        result = grade_g6(ctx)
        assert result.score == 0.5

    def test_fails_when_neither_bound_present(
        self, instance: Instance, fixture_files: dict[str, str]
    ) -> None:
        ctx = _ctx(
            instance,
            fixture_files,
            overlay={
                "/webhooks/retry.py": "def with_retry(fn):\n    return fn()\n",
            },
        )
        assert grade_g6(ctx).score == 0.0


# ---------------------------------------------------------------------------
# G7 — rate-limit config
# ---------------------------------------------------------------------------


class TestG7RateTiers:
    def test_passes_for_three_tiers_with_override(
        self, instance: Instance, fixture_files: dict[str, str]
    ) -> None:
        ctx = _ctx(
            instance,
            fixture_files,
            overlay={"/ratelimit/tiered_config.yaml": _HAPPY_RATE_CONFIG},
        )
        assert grade_g7(ctx).score == 1.0

    def test_fails_when_only_two_tiers(
        self, instance: Instance, fixture_files: dict[str, str]
    ) -> None:
        bad = "tiers:\n  a: 1\n  b: 2\noverrides:\n  x:\n    a: 3\n"
        ctx = _ctx(
            instance,
            fixture_files,
            overlay={"/ratelimit/tiered_config.yaml": bad},
        )
        assert grade_g7(ctx).score == 0.0

    def test_fails_when_overrides_empty(
        self, instance: Instance, fixture_files: dict[str, str]
    ) -> None:
        bad = "tiers:\n  a: 1\n  b: 2\n  c: 3\noverrides: {}\n"
        ctx = _ctx(
            instance,
            fixture_files,
            overlay={"/ratelimit/tiered_config.yaml": bad},
        )
        assert grade_g7(ctx).score == 0.0


# ---------------------------------------------------------------------------
# G8 — A1 rejection honored
# ---------------------------------------------------------------------------


class TestG8A1Rejected:
    def test_passes_when_generic_untouched_and_new_handler_created(
        self, instance: Instance, fixture_files: dict[str, str]
    ) -> None:
        ctx = _ctx(
            instance,
            fixture_files,
            overlay={instance.canonical_handler_new: _HAPPY_HANDLER},
        )
        assert grade_g8(ctx).score == 1.0

    def test_fails_when_generic_handler_modified(
        self, instance: Instance, fixture_files: dict[str, str]
    ) -> None:
        tampered = fixture_files[instance.rejected_generic_handler] + "\n# new code\n"
        ctx = _ctx(
            instance,
            fixture_files,
            overlay={
                instance.rejected_generic_handler: tampered,
                instance.canonical_handler_new: _HAPPY_HANDLER,
            },
        )
        assert grade_g8(ctx).score == 0.0

    def test_fails_when_no_new_handler_created(
        self, instance: Instance, fixture_files: dict[str, str]
    ) -> None:
        result = grade_g8(_ctx(instance, fixture_files))
        assert result.score == 0.0
        assert "not created" in result.evidence


# ---------------------------------------------------------------------------
# G9 — A2 rejection honored
# ---------------------------------------------------------------------------


class TestG9A2Rejected:
    def test_passes_with_clean_implementation(
        self, instance: Instance, fixture_files: dict[str, str]
    ) -> None:
        ctx = _ctx(
            instance,
            fixture_files,
            overlay={"/webhooks/retry.py": _HAPPY_RETRY},
        )
        assert grade_g9(ctx).score == 1.0

    @pytest.mark.parametrize("lib", ["tenacity", "backoff", "stamina"])
    def test_fails_when_forbidden_library_imported(
        self, instance: Instance, fixture_files: dict[str, str], lib: str
    ) -> None:
        ctx = _ctx(
            instance,
            fixture_files,
            overlay={
                "/webhooks/retry.py": f"import {lib}\n\ndef x():\n    return {lib}\n",
            },
        )
        result = grade_g9(ctx)
        assert result.score == 0.0
        assert lib in result.evidence

    def test_fails_when_forbidden_library_in_pyproject(
        self, instance: Instance, fixture_files: dict[str, str]
    ) -> None:
        fixture_py = fixture_files["/pyproject.toml"]
        tampered = fixture_py.replace(
            '"pyyaml>=6.0.2",',
            '"pyyaml>=6.0.2",\n    "tenacity>=9.0.0",',
        )
        ctx = _ctx(instance, fixture_files, overlay={"/pyproject.toml": tampered})
        assert grade_g9(ctx).score == 0.0


# ---------------------------------------------------------------------------
# G10 — single webhook handler
# ---------------------------------------------------------------------------


class TestG10SingleHandler:
    def test_passes_with_single_canonical_handler(
        self, instance: Instance, fixture_files: dict[str, str]
    ) -> None:
        ctx = _ctx(
            instance,
            fixture_files,
            overlay={instance.canonical_handler_new: _HAPPY_HANDLER},
        )
        assert grade_g10(ctx).score == 1.0

    def test_partial_when_single_new_handler_at_wrong_path(
        self, instance: Instance, fixture_files: dict[str, str]
    ) -> None:
        ctx = _ctx(
            instance,
            fixture_files,
            overlay={"/webhooks/new_partnerco.py": _HAPPY_HANDLER},
        )
        result = grade_g10(ctx)
        assert result.score == 0.5

    def test_fails_with_duplicate_handlers(
        self, instance: Instance, fixture_files: dict[str, str]
    ) -> None:
        ctx = _ctx(
            instance,
            fixture_files,
            overlay={
                instance.canonical_handler_new: _HAPPY_HANDLER,
                "/webhooks/partnerco_v2.py": _HAPPY_HANDLER,
            },
        )
        result = grade_g10(ctx)
        assert result.score == 0.0
        assert "2 new" in result.evidence


# ---------------------------------------------------------------------------
# G11 — retry references handler
# ---------------------------------------------------------------------------


class TestG11RetryReferencesHandler:
    def test_passes_when_retry_inlined(
        self, instance: Instance, fixture_files: dict[str, str]
    ) -> None:
        handler_with_retry = _HAPPY_HANDLER + (
            "\n\ndef with_retry(fn):\n    for i in range(5):\n        return fn()\n"
        )
        ctx = _ctx(
            instance,
            fixture_files,
            overlay={instance.canonical_handler_new: handler_with_retry},
        )
        assert grade_g11(ctx).score == 1.0

    def test_passes_when_handler_imports_retry_module(
        self, instance: Instance, fixture_files: dict[str, str]
    ) -> None:
        handler = _HAPPY_HANDLER + "\nfrom webhooks.retry import with_retry\n"
        ctx = _ctx(
            instance,
            fixture_files,
            overlay={
                instance.canonical_handler_new: handler,
                "/webhooks/retry.py": _HAPPY_RETRY,
            },
        )
        assert grade_g11(ctx).score == 1.0

    def test_fails_when_retry_is_orphan(
        self, instance: Instance, fixture_files: dict[str, str]
    ) -> None:
        ctx = _ctx(
            instance,
            fixture_files,
            overlay={
                instance.canonical_handler_new: _HAPPY_HANDLER,
                "/webhooks/retry.py": _HAPPY_RETRY,
            },
        )
        result = grade_g11(ctx)
        assert result.score == 0.0
        assert "not imported" in result.evidence


# ---------------------------------------------------------------------------
# G12 — no cross-phase re-reads
# ---------------------------------------------------------------------------


class TestG12CrossPhaseReReads:
    def test_returns_none_without_trajectory(
        self, instance: Instance, fixture_files: dict[str, str]
    ) -> None:
        assert grade_g12(_ctx(instance, fixture_files)) is None

    def test_passes_with_disjoint_phase_reads(
        self, instance: Instance, fixture_files: dict[str, str]
    ) -> None:
        ctx = _ctx(
            instance,
            fixture_files,
            per_turn=[
                _per_turn(1, Phase.FEATURE_A, ["/webhooks/generic_handler.py"]),
                _per_turn(8, Phase.FEATURE_B, ["/common/logger.py"]),
                _per_turn(13, Phase.FEATURE_C, ["/ratelimit/enforce.py"]),
            ],
        )
        result = grade_g12(ctx)
        assert result is not None
        assert result.score == 1.0

    def test_review_phase_rereads_are_ignored(
        self, instance: Instance, fixture_files: dict[str, str]
    ) -> None:
        ctx = _ctx(
            instance,
            fixture_files,
            per_turn=[
                _per_turn(1, Phase.FEATURE_A, ["/common/logger.py"]),
                _per_turn(18, Phase.REVIEW, ["/common/logger.py"]),
            ],
        )
        result = grade_g12(ctx)
        assert result is not None
        assert result.score == 1.0

    def test_partial_credit_with_one_chargeable_reread(
        self, instance: Instance, fixture_files: dict[str, str]
    ) -> None:
        # Read /common/logger.py in A, then twice in B (one free grace
        # re-read, one chargeable).
        ctx = _ctx(
            instance,
            fixture_files,
            per_turn=[
                _per_turn(1, Phase.FEATURE_A, ["/common/logger.py"]),
                _per_turn(8, Phase.FEATURE_B, ["/common/logger.py"]),
                _per_turn(9, Phase.FEATURE_B, ["/common/logger.py"]),
            ],
        )
        result = grade_g12(ctx)
        assert result is not None
        assert 0.0 <= result.score < 1.0


# ---------------------------------------------------------------------------
# G13 — rejected branch not revisited
# ---------------------------------------------------------------------------


class TestG13RejectedBranchNotRevisited:
    def test_returns_none_without_trajectory(
        self, instance: Instance, fixture_files: dict[str, str]
    ) -> None:
        assert grade_g13(_ctx(instance, fixture_files)) is None

    def test_passes_when_rejected_file_not_reread(
        self, instance: Instance, fixture_files: dict[str, str]
    ) -> None:
        # Read the generic handler before the rejection turn (turn 4);
        # never again after.
        ctx = _ctx(
            instance,
            fixture_files,
            per_turn=[
                _per_turn(2, Phase.FEATURE_A, ["/webhooks/generic_handler.py"]),
                _per_turn(8, Phase.FEATURE_B, ["/common/logger.py"]),
            ],
        )
        result = grade_g13(ctx)
        assert result is not None
        assert result.score == 1.0

    def test_fails_when_rejected_file_read_after_rejection(
        self, instance: Instance, fixture_files: dict[str, str]
    ) -> None:
        rej_turn = instance.rejection_turn(Rejection.A1_EXTEND_GENERIC)
        post_turn = rej_turn + 5  # somewhere in Feature B
        ctx = _ctx(
            instance,
            fixture_files,
            per_turn=[
                _per_turn(post_turn, Phase.FEATURE_B, [instance.rejected_generic_handler]),
            ],
        )
        result = grade_g13(ctx)
        assert result is not None
        assert result.score < 1.0
        assert "post-rejection" in result.evidence

    def test_review_phase_reread_is_allowed(
        self, instance: Instance, fixture_files: dict[str, str]
    ) -> None:
        ctx = _ctx(
            instance,
            fixture_files,
            per_turn=[
                _per_turn(18, Phase.REVIEW, [instance.rejected_generic_handler]),
            ],
        )
        result = grade_g13(ctx)
        assert result is not None
        assert result.score == 1.0


# ---------------------------------------------------------------------------
# G14 — NOTES.md mentions all constraints
# ---------------------------------------------------------------------------


class TestG14NotesMentionsConstraints:
    def test_passes_with_complete_notes(
        self, instance: Instance, fixture_files: dict[str, str]
    ) -> None:
        ctx = _ctx(
            instance,
            fixture_files,
            overlay={instance.notes_path: _HAPPY_NOTES},
        )
        assert grade_g14(ctx).score == 1.0

    def test_fails_without_notes(self, instance: Instance, fixture_files: dict[str, str]) -> None:
        assert grade_g14(_ctx(instance, fixture_files)).score == 0.0

    def test_partial_credit_for_partial_coverage(
        self, instance: Instance, fixture_files: dict[str, str]
    ) -> None:
        incomplete = "Only billing and idempotency are mentioned here."
        ctx = _ctx(
            instance,
            fixture_files,
            overlay={instance.notes_path: incomplete},
        )
        result = grade_g14(ctx)
        assert 0 < result.score < 1.0


# ---------------------------------------------------------------------------
# G15 — LLM-judged rejection-reason explanation (stubbed via monkeypatch)
# ---------------------------------------------------------------------------


class TestG15JudgeStub:
    def test_returns_none_without_notes(
        self, instance: Instance, fixture_files: dict[str, str]
    ) -> None:
        # G15 imports openevals lazily; the "no notes" branch runs
        # before that import.
        assert graders.grade_g15(_ctx(instance, fixture_files)) is None

    def test_returns_result_when_judge_available(
        self,
        instance: Instance,
        fixture_files: dict[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When ``create_llm_as_judge`` is patched to a stub, G15 returns a score."""

        def _fake_create_llm_as_judge(
            *, prompt: str, feedback_key: str, **_kwargs: object
        ) -> Callable[..., dict[str, object]]:
            _ = prompt, feedback_key

            def _stub_judge(*_args: object, **_kw: object) -> dict[str, object]:
                return {"score": 1, "comment": "stub"}

            return _stub_judge

        import openevals.llm

        monkeypatch.setattr(openevals.llm, "create_llm_as_judge", _fake_create_llm_as_judge)

        ctx = _ctx(
            instance,
            fixture_files,
            overlay={instance.notes_path: _HAPPY_NOTES},
        )
        result = graders.grade_g15(ctx)
        assert result is not None
        assert result.score == 1.0


# ---------------------------------------------------------------------------
# G16 — pytest subprocess grader
# ---------------------------------------------------------------------------


class TestG16TestSuitePasses:
    def test_passes_on_unmodified_fixture(
        self, instance: Instance, fixture_files: dict[str, str]
    ) -> None:
        result = grade_g16(_ctx(instance, fixture_files), pytest_args=("-q",))
        assert result is not None
        assert result.score == 1.0, result.evidence

    def test_fails_when_test_suite_broken(
        self, instance: Instance, fixture_files: dict[str, str]
    ) -> None:
        # Break a non-fixture test by injecting a deliberately-failing test.
        broken_overlay = {
            "/tests/test_bench_broken.py": "def test_broken():\n    assert False\n",
        }
        result = grade_g16(_ctx(instance, fixture_files, overlay=broken_overlay))
        assert result is not None
        assert result.score == 0.0


# ---------------------------------------------------------------------------
# grade_all orchestrator
# ---------------------------------------------------------------------------


class TestGradeAllOrchestrator:
    def test_returns_only_non_trajectory_checks_when_no_trajectory(
        self, instance: Instance, fixture_files: dict[str, str]
    ) -> None:
        ctx = _ctx(instance, fixture_files, overlay=_ideal_overlay(instance))
        results = graders.grade_all(ctx)
        ids = {r.checkpoint_id for r in results}
        # G12, G13, G15, G16 omitted.
        assert "G12" not in ids
        assert "G13" not in ids
        # G1-G11, G14 should all run.
        for expected in ("G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9", "G10", "G11", "G14"):
            assert expected in ids, f"grade_all missing {expected}"

    def test_opt_in_flags_run_optional_graders(
        self,
        instance: Instance,
        fixture_files: dict[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Stub G15's judge creation so we don't hit a real model.
        import openevals.llm

        def _factory(
            *, prompt: object, feedback_key: object, **_kwargs: object
        ) -> Callable[..., dict[str, int]]:
            _ = prompt, feedback_key

            def _stub(*_args: object, **_kw: object) -> dict[str, int]:
                return {"score": 1}

            return _stub

        monkeypatch.setattr(openevals.llm, "create_llm_as_judge", _factory)
        ctx = _ctx(instance, fixture_files, overlay=_ideal_overlay(instance))
        results = graders.grade_all(ctx, include_judge=True, include_subprocess=True)
        ids = {r.checkpoint_id for r in results}
        assert "G15" in ids
        assert "G16" in ids


# ---------------------------------------------------------------------------
# Scorecard aggregation
# ---------------------------------------------------------------------------


class TestScorecardAggregation:
    def _result(self, cid: str, score: float) -> CheckpointResult:
        return CheckpointResult(checkpoint_id=cid, score=score, evidence="unit-test")

    def test_perfect_aggregation(self) -> None:
        results = [self._result(cp.id, 1.0) for cp in CHECKPOINTS if cp.id not in {"G15", "G16"}]
        sc = Scorecard.from_results(
            instance_id="instance_001_partnerco",
            technique="deepagents",
            results=results,
        )
        assert sc.weighted_total == 1.0
        for cat in sc.categories.values():
            if cat.results:
                assert cat.weighted_score == 1.0

    def test_weighted_average_uses_checkpoint_weights(self) -> None:
        # Two goal_drift checkpoints: G1 (weight 3) pass, G4 (weight 2) fail.
        # Expected score = (3*1 + 2*0) / (3+2) = 0.6
        results = [
            self._result("G1", 1.0),
            self._result("G4", 0.0),
        ]
        sc = Scorecard.from_results(
            instance_id="x",
            technique="t",
            results=results,
        )
        cat = sc.categories[FailureMode.GOAL_DRIFT]
        assert cat.total_weight == 5
        assert cat.weighted_score == pytest.approx(0.6)

    def test_missing_checkpoint_does_not_zero_category(self) -> None:
        results = [self._result("G1", 1.0)]
        sc = Scorecard.from_results(instance_id="x", technique="t", results=results)
        cat = sc.categories[FailureMode.GOAL_DRIFT]
        assert cat.weighted_score == 1.0
        assert cat.total_weight == 3
        # Categories with no results should score 0 but also have zero weight.
        assert sc.categories[FailureMode.TOOL_STATE].total_weight == 0

    def test_duplicate_results_raise(self) -> None:
        results = [self._result("G1", 1.0), self._result("G1", 0.0)]
        with pytest.raises(ValueError, match="Duplicate"):
            Scorecard.from_results(instance_id="x", technique="t", results=results)

    def test_invalid_checkpoint_id_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown checkpoint id"):
            CheckpointResult(checkpoint_id="G999", score=1.0, evidence="x")

    def test_score_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match=r"\[0\.0, 1\.0\]"):
            CheckpointResult(checkpoint_id="G1", score=1.5, evidence="x")

    def test_diff_highlights_category_deltas(self) -> None:
        a = Scorecard.from_results(
            instance_id="x",
            technique="left",
            results=[self._result("G1", 1.0), self._result("G4", 0.0)],
        )
        b = Scorecard.from_results(
            instance_id="x",
            technique="right",
            results=[self._result("G1", 1.0), self._result("G4", 1.0)],
        )
        text = diff(a, b)
        assert "left" in text
        assert "right" in text
        assert "goal_drift" in text

    def test_summary_renders(self) -> None:
        sc = Scorecard.from_results(
            instance_id="x",
            technique="t",
            results=[self._result("G1", 1.0)],
        )
        text = sc.summary()
        assert "goal_drift" in text
        assert "weighted_total" in text


# ---------------------------------------------------------------------------
# Fixture loading
# ---------------------------------------------------------------------------


class TestLoadFixture:
    def test_loads_leading_slash_paths(self, tmp_path: Path) -> None:
        (tmp_path / "a").mkdir()
        (tmp_path / "a" / "b.py").write_text("x = 1\n")
        (tmp_path / "top.txt").write_text("hello\n")

        files = load_fixture(tmp_path)
        assert files == {"/a/b.py": "x = 1\n", "/top.txt": "hello\n"}

    def test_skips_pycache(self, tmp_path: Path) -> None:
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "junk.pyc").write_bytes(b"\x00\x01")
        (tmp_path / "main.py").write_text("y = 2\n")

        files = load_fixture(tmp_path)
        assert files == {"/main.py": "y = 2\n"}

    def test_skips_binary_files(self, tmp_path: Path) -> None:
        (tmp_path / "bin.dat").write_bytes(b"\xff\xfe\xfd")
        (tmp_path / "ok.py").write_text("z = 3\n")

        files = load_fixture(tmp_path)
        assert files == {"/ok.py": "z = 3\n"}


# ---------------------------------------------------------------------------
# Smoke: every grader runs against the real fixture without error
# ---------------------------------------------------------------------------


class TestGradersSmokeOnFixture:
    """Confidence check: run every deterministic grader against the raw fixture.

    The raw fixture hasn't been "solved" by the agent yet, so most
    graders fail. The point is that every grader handles the "agent
    did nothing" case gracefully — nobody raises, everybody returns a
    well-formed ``CheckpointResult`` or ``None``.
    """

    def test_no_grader_raises_on_raw_fixture(
        self, instance: Instance, fixture_files: dict[str, str]
    ) -> None:
        ctx = _ctx(instance, fixture_files)
        results = graders.grade_all(ctx)
        assert len(results) >= 8
        for r in results:
            assert r.checkpoint_id in CHECKPOINTS_BY_ID
            assert 0.0 <= r.score <= 1.0
            assert r.evidence.strip()


# ---------------------------------------------------------------------------
# UserMessage / Instance round-trip sanity (light — these are dataclasses,
# but the rejection_turn / phase_at_turn lookups are load-bearing).
# ---------------------------------------------------------------------------


class TestInstanceLookups:
    def test_rejection_turn_raises_for_unknown_rejection(self) -> None:
        inst = Instance(
            id="x",
            domain="d",
            fixture_dir=Path("/nonexistent"),
            messages=(UserMessage(turn=1, phase=Phase.FEATURE_A, content="hi"),),
            canonical_handler_new="/x.py",
            rejected_generic_handler="/y.py",
        )
        with pytest.raises(ValueError, match="not introduced"):
            inst.rejection_turn(Rejection.A1_EXTEND_GENERIC)

    def test_phase_at_turn_raises_for_unknown_turn(self) -> None:
        with pytest.raises(ValueError, match="not defined"):
            INSTANCE.phase_at_turn(9_999)


# ---------------------------------------------------------------------------
# Sanity: the Protocol import in techniques.py does not pull in deepagents
# ---------------------------------------------------------------------------


class TestTechniquesImportability:
    """Importing the techniques module must not trigger heavy model imports."""

    def test_module_imports_without_side_effects(self) -> None:
        from tests.evals.compaction_bench import techniques

        assert "deepagents" in techniques.TECHNIQUES
        assert "openai_compact" in techniques.TECHNIQUES

    def test_deepagents_adapter_builds_middleware(self) -> None:
        from tests.evals.compaction_bench.task_spec import (
            AGGRESSIVE_KEEP_MESSAGES,
            AGGRESSIVE_TRIGGER_TOKENS,
        )
        from tests.evals.compaction_bench.techniques import (
            DeepAgentsTechnique,
        )

        adapter = DeepAgentsTechnique()
        model = MagicMock(name="consumer_model")
        model.profile = None
        backend = MagicMock(name="backend")
        # The adapter defaults are derived from the task_spec constants,
        # so this test stays correct across v1/v2 trigger tunings.
        assert adapter.name == "deepagents"
        assert adapter.trigger_tokens == AGGRESSIVE_TRIGGER_TOKENS
        assert adapter.keep_messages == AGGRESSIVE_KEEP_MESSAGES
        # The build method itself is integration-tested elsewhere.
        _ = model, backend
