"""Scorecard data types and per-category aggregation.

A ``Scorecard`` is the canonical output of one (instance, technique)
run. It is designed so that:

- Per-category scores (``goal_drift``, ``decision_history``, …) are
  the primary comparison surface. The ``weighted_total`` is included
  for convenience but never drives conclusions on its own.
- Every checkpoint result carries evidence — a short human-readable
  string explaining *why* it passed or failed. This makes the
  scorecard self-describing; a reviewer can read one scorecard and
  understand what happened without re-running the bench.
- The aggregation is a pure function (``Scorecard.from_results``).
  It does not touch the network, filesystem, or the agent runtime,
  which lets grader unit tests assemble hand-crafted results and
  verify aggregation without any model calls.

Partial credit is allowed: a checkpoint's ``score`` is a ``float``
in ``[0.0, 1.0]``. Deterministic graders will usually produce
``0.0`` or ``1.0``; LLM-judge graders that evaluate N sub-criteria
(e.g. "NOTES.md mentions all 7 constraints") produce fractional
scores like ``5/7 = 0.714``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from tests.evals.compaction_bench.task_spec import (
    CHECKPOINTS,
    CHECKPOINTS_BY_ID,
    FailureMode,
)

if TYPE_CHECKING:
    from collections.abc import Iterable


@dataclass(frozen=True)
class CheckpointResult:
    """Outcome of a single grader.

    Attributes:
        checkpoint_id: The ``Checkpoint.id`` this result corresponds to
            (e.g. ``"G1"``). Must exist in ``CHECKPOINTS_BY_ID``.
        score: Score in ``[0.0, 1.0]``. Deterministic graders produce
            ``0.0`` (fail) or ``1.0`` (pass); fractional values
            represent partial credit from LLM-judge graders.
        evidence: Short human-readable explanation of the outcome.
            Intended to be a one-line answer to "why this score?".
    """

    checkpoint_id: str
    score: float
    evidence: str

    def __post_init__(self) -> None:
        """Validate ids and clamp the score to a sane range.

        Raises:
            ValueError: If ``checkpoint_id`` isn't in the catalog or
                ``score`` is outside ``[0.0, 1.0]``.
        """
        if self.checkpoint_id not in CHECKPOINTS_BY_ID:
            msg = f"Unknown checkpoint id: {self.checkpoint_id!r}"
            raise ValueError(msg)
        if not 0.0 <= self.score <= 1.0:
            msg = (
                f"CheckpointResult.score must be in [0.0, 1.0], "
                f"got {self.score!r} for {self.checkpoint_id!r}"
            )
            raise ValueError(msg)


@dataclass(frozen=True)
class CategoryScore:
    """Aggregated score for a single failure mode.

    Attributes:
        failure_mode: Category this score covers.
        weighted_score: Weighted average in ``[0.0, 1.0]``.

            Zero when no contributing checkpoints exist.
        total_weight: Sum of weights across contributing checkpoints.
            Useful for diagnosing thin categories (e.g. "this score is
            based on only one weight-2 checkpoint").
        results: The underlying per-checkpoint results in catalog order.
    """

    failure_mode: FailureMode
    weighted_score: float
    total_weight: int
    results: tuple[CheckpointResult, ...]

    @property
    def is_perfect(self) -> bool:
        """Return whether every contributing checkpoint scored 1.0."""
        return all(r.score >= 1.0 for r in self.results)


@dataclass(frozen=True)
class Scorecard:
    """Top-level scorecard for one (instance, technique) run.

    Attributes:
        instance_id: The ``Instance.id`` this scorecard is for.
        technique: The ``SummarizationTechnique.name`` used.
        categories: Per-category aggregates, one per ``FailureMode``.
        weighted_total: Cross-category weighted average. Convenience
            only; reviewers should compare categories directly.
        all_results: Flat list of every ``CheckpointResult`` in the
            canonical catalog order, for ad-hoc inspection and diffing.
    """

    instance_id: str
    technique: str
    categories: dict[FailureMode, CategoryScore]
    weighted_total: float
    all_results: tuple[CheckpointResult, ...] = field(default=())

    @classmethod
    def from_results(
        cls,
        *,
        instance_id: str,
        technique: str,
        results: Iterable[CheckpointResult],
    ) -> Scorecard:
        """Build a ``Scorecard`` by aggregating a set of checkpoint results.

        The input results need not be in catalog order and need not
        cover every checkpoint; missing checkpoints are simply absent
        from the output (they do not count against any category). This
        matters for graders that are expensive or optional (LLM-judge
        graders in particular) — omitting them yields a scorecard with
        fewer checkpoints rather than spuriously-zeroed categories.

        Args:
            instance_id: Instance identifier.
            technique: Technique identifier.
            results: Checkpoint results in any order.

        Returns:
            A fully-populated ``Scorecard``.

        Raises:
            ValueError: If any result references an unknown checkpoint id,
                or if two results share the same ``checkpoint_id``.
        """
        results_tuple = tuple(results)

        # Detect duplicate ids: silently overwriting would be a hazard
        # for graders that accidentally fire twice.
        seen: set[str] = set()
        for r in results_tuple:
            if r.checkpoint_id in seen:
                msg = f"Duplicate result for checkpoint {r.checkpoint_id!r}"
                raise ValueError(msg)
            seen.add(r.checkpoint_id)

        by_id = {r.checkpoint_id: r for r in results_tuple}

        categories: dict[FailureMode, CategoryScore] = {}
        total_weighted = 0.0
        total_weight = 0

        for failure_mode in FailureMode:
            cat_results: list[CheckpointResult] = []
            cat_weighted = 0.0
            cat_weight = 0

            for cp in CHECKPOINTS:
                if cp.failure_mode is not failure_mode:
                    continue
                result = by_id.get(cp.id)
                if result is None:
                    continue
                cat_results.append(result)
                cat_weighted += result.score * cp.weight
                cat_weight += cp.weight

            categories[failure_mode] = CategoryScore(
                failure_mode=failure_mode,
                weighted_score=(cat_weighted / cat_weight) if cat_weight > 0 else 0.0,
                total_weight=cat_weight,
                results=tuple(cat_results),
            )
            total_weighted += cat_weighted
            total_weight += cat_weight

        ordered_results = tuple(by_id[cp.id] for cp in CHECKPOINTS if cp.id in by_id)

        return cls(
            instance_id=instance_id,
            technique=technique,
            categories=categories,
            weighted_total=(total_weighted / total_weight) if total_weight > 0 else 0.0,
            all_results=ordered_results,
        )

    def summary(self) -> str:
        """Return a compact multi-line summary useful for console output.

        Each category is printed on its own line as
        ``<name>: <score> (<n_checkpoints>/<total_weight>)``, followed
        by the ``weighted_total`` on a final line.

        Returns:
            A newline-separated summary string.
        """
        lines: list[str] = []
        for failure_mode in FailureMode:
            cat = self.categories.get(failure_mode)
            if cat is None or not cat.results:
                lines.append(f"  {failure_mode.value:>22s}: (no results)")
                continue
            lines.append(
                f"  {failure_mode.value:>22s}: {cat.weighted_score:.2f} "
                f"({len(cat.results)} checkpoints, weight={cat.total_weight})"
            )
        lines.append(f"  {'weighted_total':>22s}: {self.weighted_total:.2f}")
        return f"Scorecard({self.instance_id}, {self.technique}):\n" + "\n".join(lines)


def diff(a: Scorecard, b: Scorecard) -> str:
    """Return a short side-by-side per-category diff of two scorecards.

    Useful for the technique A/B view: run two techniques on the same
    instance, pass both scorecards in, print the result.

    Args:
        a: Scorecard on the left.
        b: Scorecard on the right.

    Returns:
        A multi-line string showing per-category deltas.
    """
    lines = [f"diff: {a.technique} (L) vs {b.technique} (R) on {a.instance_id}"]
    for failure_mode in FailureMode:
        ca = a.categories.get(failure_mode)
        cb = b.categories.get(failure_mode)
        la = ca.weighted_score if ca and ca.results else None
        lb = cb.weighted_score if cb and cb.results else None
        if la is None and lb is None:
            continue
        la_str = "  n/a" if la is None else f"{la:.2f}"
        lb_str = "  n/a" if lb is None else f"{lb:.2f}"
        delta_str = "      " if la is None or lb is None else f"({lb - la:+.2f})"
        lines.append(f"  {failure_mode.value:>22s}: {la_str} -> {lb_str}  {delta_str}")
    lines.append(
        f"  {'weighted_total':>22s}: {a.weighted_total:.2f} -> {b.weighted_total:.2f}  "
        f"({b.weighted_total - a.weighted_total:+.2f})"
    )
    return "\n".join(lines)


__all__ = [
    "CategoryScore",
    "CheckpointResult",
    "Scorecard",
    "diff",
]
