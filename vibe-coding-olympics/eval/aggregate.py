"""Pure helpers for composing per-axis scores into a single number.

Aggregation lives outside the judge so callers (tournament runner, CI,
notebooks) can reweight axes per round without touching the judge code.
"""

from __future__ import annotations

from collections.abc import Mapping

from capture import AccessibilityReport

_AXE_PENALTY_BUDGET = 15.0

DEFAULT_WEIGHTS: dict[str, float] = {
    "color": 0.10,
    "typography": 0.10,
    "layout": 0.20,
    "content_completeness": 0.20,
    "creativity": 0.15,
    "interpretation_quality": 0.15,
    "accessibility": 0.10,
}


def aggregate(
    axes: Mapping[str, float | None],
    weights: Mapping[str, float] | None = None,
) -> float:
    """Compute a weighted mean over per-axis scores, ignoring `None` axes.

    Weights are renormalized across present axes, so skipping one axis (e.g.
    `accessibility=None` when axe failed to inject) does not punish the
    overall score — the remaining axes simply absorb the weight.

    Args:
        axes: Mapping of axis name to score in [0.0, 1.0], or `None` if the
            axis could not be evaluated.
        weights: Optional weight mapping. Unknown axes get weight 0.
            Defaults to `DEFAULT_WEIGHTS`.

    Returns:
        Weighted mean in [0.0, 1.0]. Returns `0.0` if no axes have signal.
    """
    weights = weights or DEFAULT_WEIGHTS
    present = {k: v for k, v in axes.items() if v is not None}
    total_weight = sum(weights.get(k, 0.0) for k in present)
    if total_weight == 0:
        return 0.0
    weighted = sum(present[k] * weights.get(k, 0.0) for k in present)
    return weighted / total_weight


def score_accessibility(report: AccessibilityReport | None) -> float | None:
    """Convert an axe-core report into a [0, 1] score.

    Serious and critical violations count double. The penalty budget is
    tuned for 5-minute builds — roughly a dozen minor WCAG issues drop the
    axis to zero.

    Args:
        report: The axe report, or `None` if the audit could not run.

    Returns:
        Score in [0.0, 1.0], or `None` to signal the axis should be
        excluded from aggregation.
    """
    if report is None:
        return None
    penalty = report.violation_count + report.serious_violation_count
    return max(0.0, 1.0 - penalty / _AXE_PENALTY_BUDGET)
