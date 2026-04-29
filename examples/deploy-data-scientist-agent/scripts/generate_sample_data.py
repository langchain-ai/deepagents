"""Generate synthetic SaaS metrics data for the data scientist agent example.

The generated CSV is intentionally fake but shaped like a realistic business
dataset so the agent can practice trend analysis, segmentation, visualization,
and executive reporting.
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from datetime import date
from pathlib import Path


SEED = 42
EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATHS = [
    EXAMPLE_ROOT / "data" / "sample_saas_metrics.csv",
    EXAMPLE_ROOT / "skills" / "data" / "sample_saas_metrics.csv",
]


@dataclass(frozen=True)
class SegmentProfile:
    """Business assumptions used to create coherent synthetic segment behavior."""

    base_customers: int
    monthly_growth: float
    churn_rate: float
    base_nps: int
    revenue_multiplier: float


SEGMENT_PROFILES = {
    "startup": SegmentProfile(
        base_customers=140,
        monthly_growth=0.050,
        churn_rate=0.045,
        base_nps=38,
        revenue_multiplier=0.8,
    ),
    "mid_market": SegmentProfile(
        base_customers=95,
        monthly_growth=0.038,
        churn_rate=0.030,
        base_nps=46,
        revenue_multiplier=1.4,
    ),
    "enterprise": SegmentProfile(
        base_customers=45,
        monthly_growth=0.026,
        churn_rate=0.018,
        base_nps=54,
        revenue_multiplier=3.2,
    ),
}

REGION_MULTIPLIERS = {
    "north_america": 1.18,
    "emea": 1.00,
    "apac": 0.86,
    "latin_america": 0.72,
}

PLAN_MULTIPLIERS = {
    "starter": 1.0,
    "growth": 2.4,
    "enterprise": 5.8,
}

CHANNEL_MULTIPLIERS = {
    "organic": 0.95,
    "paid_search": 1.10,
    "partner": 1.18,
    "sales_outbound": 1.28,
}


def iter_month_starts(year: int) -> list[date]:
    """Return the first day of each month for the requested year."""

    return [date(year, month, 1) for month in range(1, 13)]


def clamp(value: float, lower: int, upper: int) -> int:
    """Clamp a numeric value to an integer range."""

    return max(lower, min(upper, round(value)))


def build_rows() -> list[dict[str, str | int | float]]:
    """Build deterministic synthetic monthly SaaS metrics rows."""

    random.seed(SEED)
    rows: list[dict[str, str | int | float]] = []

    for month_index, month_start in enumerate(iter_month_starts(2025)):
        seasonal_factor = 1 + 0.06 * random.choice([-1, 0, 1])

        for region, region_multiplier in REGION_MULTIPLIERS.items():
            for segment, profile in SEGMENT_PROFILES.items():
                for plan, plan_multiplier in PLAN_MULTIPLIERS.items():
                    for channel, channel_multiplier in CHANNEL_MULTIPLIERS.items():
                        growth_factor = (1 + profile.monthly_growth) ** month_index
                        base_active = (
                            profile.base_customers
                            * growth_factor
                            * region_multiplier
                            * plan_multiplier
                            * channel_multiplier
                            * seasonal_factor
                        )
                        active_customers = clamp(
                            base_active + random.gauss(0, base_active * 0.04),
                            lower=5,
                            upper=10_000,
                        )

                        new_customers = clamp(
                            active_customers
                            * (profile.monthly_growth + random.uniform(0.005, 0.028)),
                            lower=0,
                            upper=2_000,
                        )
                        churned_customers = clamp(
                            active_customers
                            * (profile.churn_rate + random.uniform(-0.006, 0.009)),
                            lower=0,
                            upper=2_000,
                        )

                        average_revenue_per_customer = (
                            79
                            * profile.revenue_multiplier
                            * plan_multiplier
                            * region_multiplier
                        )
                        discount_rate = round(
                            max(
                                0.0,
                                min(
                                    0.35,
                                    random.gauss(0.08 if channel == "sales_outbound" else 0.04, 0.025),
                                ),
                            ),
                            3,
                        )
                        monthly_revenue = round(
                            active_customers
                            * average_revenue_per_customer
                            * (1 - discount_rate),
                            2,
                        )

                        support_tickets = clamp(
                            active_customers
                            * random.uniform(0.06, 0.16)
                            * (1.2 if plan == "starter" else 0.9),
                            lower=0,
                            upper=5_000,
                        )
                        avg_response_time_hours = round(
                            max(
                                0.4,
                                random.gauss(
                                    7.5 if plan == "starter" else 4.5,
                                    1.2 if segment == "enterprise" else 1.8,
                                ),
                            ),
                            2,
                        )
                        nps_score = clamp(
                            profile.base_nps
                            + (6 if plan == "enterprise" else 0)
                            - (avg_response_time_hours * 1.15)
                            - (churned_customers / max(active_customers, 1) * 100)
                            + random.gauss(0, 4),
                            lower=-20,
                            upper=85,
                        )

                        rows.append(
                            {
                                "date": month_start.isoformat(),
                                "region": region,
                                "segment": segment,
                                "plan": plan,
                                "channel": channel,
                                "active_customers": active_customers,
                                "new_customers": new_customers,
                                "churned_customers": churned_customers,
                                "monthly_revenue": monthly_revenue,
                                "discount_rate": discount_rate,
                                "support_tickets": support_tickets,
                                "avg_response_time_hours": avg_response_time_hours,
                                "nps_score": nps_score,
                            }
                        )

    return rows


def write_csv(rows: list[dict[str, str | int | float]]) -> None:
    """Write generated rows to the local and deploy-bundled data paths."""

    if not rows:
        raise ValueError("No rows generated")

    for output_path in OUTPUT_PATHS:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    """Generate the sample CSV."""

    rows = build_rows()
    write_csv(rows)
    for output_path in OUTPUT_PATHS:
        print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
