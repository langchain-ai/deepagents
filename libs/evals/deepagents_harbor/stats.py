"""Statistical utilities for eval score reporting.

Provides Wilson score confidence intervals and minimum detectable effect
estimation, as recommended by Anthropic's infrastructure noise research.
"""

from __future__ import annotations

import math


def wilson_ci(
    successes: int,
    total: int,
    *,
    z: float = 1.96,
) -> tuple[float, float]:
    """Compute Wilson score confidence interval for a binomial proportion.

    More accurate than the normal approximation for small samples and
    proportions near 0 or 1. Recommended by Anthropic's infrastructure noise
    research for eval score reporting.

    Args:
        successes: Number of successes (e.g., passed tasks).
        total: Total number of trials.
        z: Z-score for desired confidence level (1.96 = 95% CI).

    Returns:
        Tuple of `(lower_bound, upper_bound)` as proportions in `[0, 1]`.
    """
    if total == 0:
        return (0.0, 0.0)

    p = successes / total
    z2 = z * z
    denom = 1 + z2 / total
    center = (p + z2 / (2 * total)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / total + z2 / (4 * total * total))

    return (max(0.0, center - margin), min(1.0, center + margin))


def format_ci(
    successes: int,
    total: int,
    *,
    z: float = 1.96,
) -> str:
    """Format a success rate with Wilson confidence interval.

    Args:
        successes: Number of successes.
        total: Total number of trials.
        z: Z-score for desired confidence level.

    Returns:
        Formatted string like `'72.3% [68.1%, 76.2%] (95% CI, n=90)'`.
    """
    if total == 0:
        return "N/A (no trials)"

    rate = (successes / total) * 100
    lo, hi = wilson_ci(successes, total, z=z)
    confidence = math.erf(z / math.sqrt(2)) * 100
    return f"{rate:.1f}% [{lo * 100:.1f}%, {hi * 100:.1f}%] ({confidence:.0f}% CI, n={total})"


def min_detectable_effect(total: int, *, z: float = 1.96, p: float = 0.5) -> float:
    """Estimate minimum detectable effect size for a given sample count.

    The MDE is the smallest difference in success rates between two runs that
    can be considered statistically significant. If two runs score 72% and 78%
    but the MDE is 14pp, that 6pp gap is indistinguishable from noise at the
    chosen confidence level.

    Derived from the standard error of the difference between two independent
    proportions: `MDE = z * sqrt(2 * p * (1-p) / n)`. Assumes equal sample sizes
    in both runs. Defaults to `p=0.5` because that maximizes `p*(1-p)`, giving
    the most conservative (widest) estimate.

    Args:
        total: Number of tasks per run (assumes both runs have the same count).
        z: Z-score for desired confidence level (1.96 = 95% CI).
        p: Assumed base proportion. 0.5 is the conservative default
            since it maximizes variance.

    Returns:
        Minimum detectable difference as a proportion (e.g., `0.042 = 4.2pp`).
    """
    if total == 0:
        return 1.0
    # Two-sample proportion test: MDE ≈ z * sqrt(2 * p * (1-p) / n)
    return z * math.sqrt(2 * p * (1 - p) / total)
