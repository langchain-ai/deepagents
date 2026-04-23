"""Task specification for the compaction benchmark.

Defines the enums, dataclasses, and constants that describe an eval
instance: phases, constraints, rejected-branch catalog, and graded
checkpoints. The types here are the shared vocabulary used by graders,
scorecard, and individual instance modules.

Keeping this file free of any filesystem or model dependencies makes it
trivially importable from unit tests — the grader tests in
``tests/unit_tests/test_compaction_bench_graders.py`` depend only on
these types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class Phase(str, Enum):
    """Logical phases within a single compaction-bench session.

    Each user message in an instance is tagged with the phase it
    belongs to. Phase boundaries are used by trajectory graders to
    distinguish e.g. "files read in Feature A" from "files re-read in
    Feature B after Feature A was already compacted away".
    """

    FEATURE_A = "feature_a"
    """Webhook-handler feature; introduces C1-C6 and rejection A1."""

    FEATURE_B = "feature_b"
    """Retry-logic feature; introduces C7-C8 and rejection A2."""

    FEATURE_C = "feature_c"
    """Rate-limiting feature; introduces C9-C10."""

    REVIEW = "review"
    """Part E comprehensive-review turn producing ``NOTES.md``."""


class Constraint(str, Enum):
    """Load-bearing user constraints introduced once, reused across phases.

    Each constraint is introduced in user prose at a specific turn and
    is expected to survive any number of compactions. The ``load_bearing_phases``
    attribute on ``Constraint`` would be nice but enums don't carry
    metadata cleanly; the mapping lives on ``Instance`` instead.
    """

    C1_BILLING_OFFLIMITS = "c1_billing_offlimits"
    """Do not modify the ``billing/`` module; introduced in Feature A."""

    C2_NO_NEW_DEPS = "c2_no_new_deps"
    """No new package dependencies; introduced in Feature A."""

    C3_IDEMPOTENCY = "c3_idempotency"
    """Events must be idempotent by event ID."""

    C4_EXISTING_LOGGER = "c4_existing_logger"
    """Must use the existing ``common/logger.py`` helper."""

    C5_LATENCY = "c5_latency"
    """Handler must avoid blocking calls (p99 < 50ms)."""

    C6_AUDIT_LOGGING = "c6_audit_logging"
    """Audit logging via ``common/audit.py`` required for external events."""

    C7_C8_RETRY_BOUNDS = "c7_c8_retry_bounds"
    """Exponential backoff with max 30s cap and max 5 attempts.

    Two related constraints merged into a single checkpoint: both are
    introduced in the same turn in Feature B and both are verified by
    the same regex+config grader.
    """

    C9_C10_RATE_TIERS = "c9_c10_rate_tiers"
    """Three rate-limit tiers with per-partner overrides.

    Same merge rationale as ``C7_C8_RETRY_BOUNDS``: both are
    introduced in the same turn in Feature C and both are verified by
    parsing the rate-limit config file.
    """


class Rejection(str, Enum):
    """Rejected branches that must not be revisited after compaction.

    Each rejection is introduced in exactly one turn by the user (or
    implicitly by the agent realizing the constraint violation). After
    the rejection turn, files uniquely associated with the rejected
    branch must not be re-read in later phases.
    """

    A1_EXTEND_GENERIC = "a1_extend_generic"
    """Extending ``webhooks/generic_handler.py`` would require a new crypto dep."""

    A2_EXTERNAL_RETRY_LIB = "a2_external_retry_lib"
    """Using an external retry library violates C2 (no new deps)."""


class FailureMode(str, Enum):
    """Per-category scoring surface; each checkpoint maps to exactly one."""

    GOAL_DRIFT = "goal_drift"
    """User constraints not honored after being introduced."""

    DECISION_HISTORY = "decision_history"
    """Rejected alternatives re-proposed or re-explored after compaction."""

    ARTIFACT_CONTINUITY = "artifact_continuity"
    """Canonical files duplicated / re-created instead of being edited."""

    TOOL_STATE = "tool_state"
    """Tool-observable repo knowledge lost across compactions."""

    DIRECT_RECALL = "direct_recall"
    """LLM-judge-graded recall in the Part E review turn."""

    OVERALL_CORRECTNESS = "overall_correctness"
    """Did the task actually succeed (e.g. test suite passes)."""


@dataclass(frozen=True)
class Checkpoint:
    """A single graded checkpoint.

    Attributes:
        id: Short, stable identifier used as the scorecard key
            (e.g. ``"G1"``). Stable across runs so LangSmith feedback
            keys don't drift.
        name: Human-readable one-line description.
        failure_mode: Category this checkpoint rolls up into.
        weight: Integer weight used when computing the per-category
            weighted score. Relative within a category; absolute values
            don't matter across categories.
        description: Longer explanation, including what a failure looks
            like. Shown in diff reports.
    """

    id: str
    name: str
    failure_mode: FailureMode
    weight: int
    description: str


# ---------------------------------------------------------------------------
# Checkpoint catalog
# ---------------------------------------------------------------------------
#
# Fifteen checkpoints organized by failure mode. The weights are relative
# within a category; the absolute numbers don't matter cross-category
# because aggregation is per-category.
#
# G1-G7 -> goal_drift (constraint preservation)
# G8-G9 -> decision_history (rejection preservation)
# G10-G11 -> artifact_continuity
# G12-G13 -> tool_state
# G14-G15 -> direct_recall (LLM-judged)
# G16 -> overall_correctness
#
# This is the v1 simplified checkpoint set, not the full 28-checkpoint
# matrix from the original design document.

CHECKPOINTS: tuple[Checkpoint, ...] = (
    # ---------- goal_drift ----------
    Checkpoint(
        id="G1",
        name="billing_untouched",
        failure_mode=FailureMode.GOAL_DRIFT,
        weight=3,
        description=(
            "No files under ``billing/`` differ from the fixture. Enforces "
            "C1 across all three features and the review phase."
        ),
    ),
    Checkpoint(
        id="G2",
        name="no_new_dependencies",
        failure_mode=FailureMode.GOAL_DRIFT,
        weight=3,
        description=(
            "``pyproject.toml`` dependency list is a subset of the fixture's. Enforces C2."
        ),
    ),
    Checkpoint(
        id="G3",
        name="idempotency_in_handler",
        failure_mode=FailureMode.GOAL_DRIFT,
        weight=3,
        description=(
            "Webhook handler imports/uses ``common.idempotency`` and keys by "
            "the event ID. Enforces C3 across Features A and B."
        ),
    ),
    Checkpoint(
        id="G4",
        name="existing_logger_used",
        failure_mode=FailureMode.GOAL_DRIFT,
        weight=2,
        description=(
            "No `logging.getLogger` or ad-hoc ``print`` calls in new code; "
            "``common.logger`` is imported instead. Enforces C4."
        ),
    ),
    Checkpoint(
        id="G5",
        name="audit_logging_wired",
        failure_mode=FailureMode.GOAL_DRIFT,
        weight=2,
        description=(
            "Webhook handler calls ``common.audit.log_event`` at least once. Enforces C6."
        ),
    ),
    Checkpoint(
        id="G6",
        name="retry_bounds_respected",
        failure_mode=FailureMode.GOAL_DRIFT,
        weight=2,
        description=(
            "Retry code contains both a 30s cap and a max-5-attempts bound. Enforces C7+C8."
        ),
    ),
    Checkpoint(
        id="G7",
        name="rate_limit_tiers_configured",
        failure_mode=FailureMode.GOAL_DRIFT,
        weight=2,
        description=(
            "Rate-limit config file defines three tiers and at least one "
            "per-partner override. Enforces C9+C10."
        ),
    ),
    # ---------- decision_history ----------
    Checkpoint(
        id="G8",
        name="a1_rejection_honored",
        failure_mode=FailureMode.DECISION_HISTORY,
        weight=3,
        description=(
            "``webhooks/generic_handler.py`` is not modified (byte-identical "
            "to fixture); a new handler file was created instead of "
            "extending the generic one."
        ),
    ),
    Checkpoint(
        id="G9",
        name="a2_rejection_honored",
        failure_mode=FailureMode.DECISION_HISTORY,
        weight=3,
        description=(
            "No ``tenacity``, ``backoff``, ``retrying``, ``stamina``, or "
            "``resilient`` import appears anywhere in the final filesystem."
        ),
    ),
    # ---------- artifact_continuity ----------
    Checkpoint(
        id="G10",
        name="single_webhook_handler",
        failure_mode=FailureMode.ARTIFACT_CONTINUITY,
        weight=2,
        description=(
            "Exactly one non-fixture handler file exists under ``webhooks/`` "
            "(no ``partnerco_v2.py`` or ``new_partnerco.py`` duplicates)."
        ),
    ),
    Checkpoint(
        id="G11",
        name="retry_references_handler",
        failure_mode=FailureMode.ARTIFACT_CONTINUITY,
        weight=2,
        description=(
            "Retry code is either integrated into the canonical handler or "
            "imported by it — not a standalone orphan."
        ),
    ),
    # ---------- tool_state ----------
    Checkpoint(
        id="G12",
        name="no_cross_phase_file_rereads",
        failure_mode=FailureMode.TOOL_STATE,
        weight=2,
        description=(
            "A file first read in Feature A is not re-read in Feature B or C "
            "(allowing at most one re-read grace per file). Indicates the "
            "agent retained repo-layout knowledge across compactions."
        ),
    ),
    Checkpoint(
        id="G13",
        name="rejected_branch_not_revisited",
        failure_mode=FailureMode.TOOL_STATE,
        weight=2,
        description=(
            "After rejection A1, ``webhooks/generic_handler.py`` is not "
            "read in later phases; after A2, files referencing external "
            "retry libraries are not read. Grace: the Part E review turn is "
            "allowed to read anything."
        ),
    ),
    # ---------- direct_recall ----------
    Checkpoint(
        id="G14",
        name="notes_mentions_all_constraints",
        failure_mode=FailureMode.DIRECT_RECALL,
        weight=4,
        description=(
            "``NOTES.md`` contains a specific keyword / phrase for each of "
            "the seven constraints. Combines keyword regex with an LLM "
            "judge that checks the constraint is actually explained."
        ),
    ),
    Checkpoint(
        id="G15",
        name="notes_explains_rejections",
        failure_mode=FailureMode.DIRECT_RECALL,
        weight=4,
        description=(
            "``NOTES.md`` correctly identifies why each rejected branch was "
            "rejected — not just that it was rejected. LLM-judged."
        ),
    ),
    # ---------- overall_correctness ----------
    Checkpoint(
        id="G16",
        name="test_suite_passes",
        failure_mode=FailureMode.OVERALL_CORRECTNESS,
        weight=3,
        description=(
            "``pytest`` against the final filesystem state exits zero. Subprocess grader."
        ),
    ),
)
"""The ordered, frozen catalog of checkpoints for the v1 bench.

Graders in ``graders.py`` are keyed by ``Checkpoint.id`` (``"G1"`` …
``"G16"``) so the catalog is the single source of truth for the ids.
"""


CHECKPOINTS_BY_ID: dict[str, Checkpoint] = {c.id: c for c in CHECKPOINTS}
"""Lookup by checkpoint id. Populated once at import time."""


# ---------------------------------------------------------------------------
# Instance scaffolding
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UserMessage:
    """A single scripted user message within an instance.

    The agent takes as many internal steps as it needs between user
    messages; each ``UserMessage`` corresponds to one ``run_agent``
    invocation with a shared ``thread_id``.

    Attributes:
        turn: 1-indexed user-message number within the session.
        phase: The phase this message belongs to.
        content: The literal text of the user message.
        introduces: Constraints first stated in this turn (the grader
            uses this to know e.g. "C3 was introduced at turn 1").
        rejects: Optional rejection this turn triggers. If set, later
            graders treat anything after this turn as "post-rejection".
    """

    turn: int
    phase: Phase
    content: str
    introduces: tuple[Constraint, ...] = ()
    rejects: Rejection | None = None


@dataclass(frozen=True)
class Instance:
    """A single compaction-bench instance.

    Attributes:
        id: Stable identifier (e.g. ``"instance_001_partnerco"``).
        domain: Human-readable domain label for reports (e.g.
            ``"partnerco_webhooks"``).
        fixture_dir: Filesystem path to the mini-repo fixture. Contents
            are copied into the agent's working directory at the start
            of the run.
        messages: Ordered user-message script.
        canonical_handler_new: Path of the new (non-fixture) handler
            the agent is expected to create in Feature A. Used by
            artifact-continuity graders to spot duplicates.
        rejected_generic_handler: Path of the generic handler whose
            extension is rejected (branch A1). Used by the rejected-
            branch-revisit grader.
        notes_path: Where the Part E review writes ``NOTES.md``.
    """

    id: str
    domain: str
    fixture_dir: Path
    messages: tuple[UserMessage, ...]
    canonical_handler_new: str
    rejected_generic_handler: str
    notes_path: str = "/NOTES.md"

    # ---- derived views (cheap; computed on access, not cached) ----

    def rejection_turn(self, rejection: Rejection) -> int:
        """Return the 1-indexed user-message turn where ``rejection`` was made.

        Args:
            rejection: The rejection to look up.

        Returns:
            Turn index.

        Raises:
            ValueError: If ``rejection`` is not present in ``messages``.
        """
        for msg in self.messages:
            if msg.rejects is rejection:
                return msg.turn
        msg = f"Rejection {rejection!r} is not introduced in instance {self.id!r}."
        raise ValueError(msg)

    def phase_at_turn(self, turn: int) -> Phase:
        """Return the phase for a given 1-indexed turn.

        Args:
            turn: Turn index (1-indexed).

        Returns:
            The phase containing that turn.

        Raises:
            ValueError: If ``turn`` is outside the message range.
        """
        for msg in self.messages:
            if msg.turn == turn:
                return msg.phase
        msg = f"Turn {turn} is not defined in instance {self.id!r}."
        raise ValueError(msg)


# ---------------------------------------------------------------------------
# Session-wide constants
# ---------------------------------------------------------------------------


EXTERNAL_RETRY_LIBS: frozenset[str] = frozenset(
    {"tenacity", "backoff", "retrying", "stamina", "resilient"}
)
"""Python package names that would violate the A2 rejection.

Graders scan ``pyproject.toml`` and final Python sources for imports of
these names. Picked to match the design doc's ``a2_rejection_cause``
templating axis, minus the instance-specific one picked per run.
"""


AGGRESSIVE_TRIGGER_TOKENS: int = 15_000
"""Token count at which summarization fires under the aggressive config.

### v1 tuning note

The original design doc specified 80k, chosen to be well below the
production 150k default so compaction would fire 2-3 times during a
~120k-token session. That math assumes a fixture + script large enough
to generate ~120k of transcript. The v1 ``instance_001`` fixture is
intentionally small (~4-5k raw tokens across ~16 files) - a quick
spin-up, not a realistic ~150-file production repo.

At 80k, compaction reliably does not fire against the v1 fixture.
At 15k, compaction fires once or twice across the scripted 18 turns,
giving a real summarization event to measure against. That is enough
signal to compare techniques on goal-drift, decision-history, and
artifact-continuity failure modes - which is all v1 claims to do.

v2 plans to expand the fixture into a genuinely production-sized
mini-repo, at which point this can be restored to 80k. See
``README.md`` ("what v2 adds") for the roadmap.

The preflight test ``test_compaction_bench_token_budget.py`` enforces
that the instance content still has enough substance to cross this
threshold under realistic agent overhead.
"""


AGGRESSIVE_KEEP_MESSAGES: int = 15
"""Recent messages retained verbatim after each summarization event."""


FIXTURES_ROOT: Path = Path(__file__).parent / "fixtures"
"""Directory containing per-instance mini-repo fixtures.

Instances reference subdirectories of this path via ``Instance.fixture_dir``.
"""


__all__ = [
    "AGGRESSIVE_KEEP_MESSAGES",
    "AGGRESSIVE_TRIGGER_TOKENS",
    "CHECKPOINTS",
    "CHECKPOINTS_BY_ID",
    "EXTERNAL_RETRY_LIBS",
    "FIXTURES_ROOT",
    "Checkpoint",
    "Constraint",
    "FailureMode",
    "Instance",
    "Phase",
    "Rejection",
    "UserMessage",
]


# ---------------------------------------------------------------------------
# Self-consistency check
# ---------------------------------------------------------------------------


def _validate_catalog() -> None:
    """Fail loudly at import time if the checkpoint catalog is malformed.

    This catches two subtle mistakes: duplicate ids (a copy-paste regression)
    and non-positive weights (which would silently zero-out a checkpoint's
    contribution to its category's score).

    Raises:
        ValueError: If the catalog contains duplicate ids or non-positive
            weights.
    """
    seen: set[str] = set()
    for cp in CHECKPOINTS:
        if cp.id in seen:
            msg = f"Duplicate checkpoint id in CHECKPOINTS: {cp.id!r}"
            raise ValueError(msg)
        if cp.weight <= 0:
            msg = f"Checkpoint {cp.id!r} has non-positive weight {cp.weight!r}"
            raise ValueError(msg)
        seen.add(cp.id)


_validate_catalog()


# Unused import guard: keep `field` available for future extension without
# triggering an unused-import warning if callers start using it downstream.
_ = field
