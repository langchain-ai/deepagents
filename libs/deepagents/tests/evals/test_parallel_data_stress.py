"""Eval tests for data-heavy parallel subagent stress scenarios.

Tests the supervisor agent's ability to coordinate many concurrent subagents,
each returning noisy/complex data, and synthesize correct answers from the
combined results.

Six scenarios with increasing difficulty:

1. Extreme noisy extraction (port of JS Eval E)
2. Full-dataset aggregation
3. Cross-referencing between subagent result sets
4. Contradictions between subagents
5. High parallelism (25 concurrent subagents)
6. Temporal ordering sensitivity
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from langchain_core.tools import tool

from deepagents import create_deep_agent
from tests.evals.utils import (
    TrajectoryScorer,
    final_text_contains,
    run_agent,
    tool_call,
)

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel


# ===========================================================================
# Eval 1: Extreme noisy extraction (port of JS Eval E)
#
# 11 regions x 8 suppliers = 88 total. Dense wall-of-text prose with
# contradictory sources, historical vs. current figures, near-identical
# values across suppliers.
#
# Ground truth:
#   1. Highest audited units_produced: Pacifica Alloys (487,200)
#      Runner-up: Nordic Forge (486,800) — difference of 400 units
#      Trap: Pacifica's trade journal figure is 489,000
#
#   2. Lowest defect rate (defects / audited units_produced):
#      Helvetia Precision: 112 / 385,000 = 0.000291
#      Runner-up: Rhine Components: 98 / 334,500 = 0.000293
#      Trap: Helvetia's prior-year defects were 89 (lower, wrong year)
#
#   3. lead_time > 45 AND cost > $150:
#      Ural Heavy Industries (lead 52, cost $178, quality 7.1)
#      Gobi Materials (lead 48, cost $162, quality 8.3) <-- highest quality
#      Patagonia Steel (lead 51, cost $155, quality 6.9)
#      Saharan Metals (lead 47, cost $171, quality 7.8)
#      Answer: Gobi Materials (quality 8.3)
# ===========================================================================

_SUPPLIER_KEYS = (
    "name",
    "units_produced",
    "units_reported_by_trade_journal",
    "defects",
    "defects_prior_year",
    "cost_per_unit",
    "cost_per_unit_competitor_estimate",
    "lead_time_days",
    "lead_time_days_last_quarter",
    "quality_score",
    "quality_score_industry_avg",
    "workforce",
    "temp_workers",
    "capacity_utilization",
    "year_established",
)

_REGION_RAW: dict[str, tuple[str, list[tuple[Any, ...]]]] = {
    "East Asia": (
        "East Asia",
        [
            ("Yangtze Manufacturing", 412_300, 415_000, 287, 312, 89, 92, 21, 23, 8.7, 7.9, 4200, 1100, 87.3, 1994),
            ("Shenzhen Dynamics", 378_500, 380_200, 198, 221, 76, 79, 18, 19, 9.1, 7.9, 3800, 900, 91.2, 2001),
            ("Osaka Precision Works", 295_800, 298_000, 134, 142, 142, 138, 28, 30, 9.4, 7.9, 2100, 350, 78.6, 1967),
            ("Seoul Components Ltd", 341_200, 343_500, 245, 267, 98, 101, 24, 26, 8.2, 7.9, 2900, 650, 82.1, 1988),
            ("Taipei MicroFab", 267_400, 269_100, 156, 173, 118, 115, 22, 24, 8.9, 7.9, 1800, 420, 84.7, 2005),
            ("Hanoi Industrial Group", 198_700, 201_000, 312, 345, 62, 65, 32, 35, 6.8, 7.9, 5100, 2200, 69.4, 2011),
            ("Manila Metalworks", 156_300, 158_000, 278, 301, 58, 61, 35, 38, 6.2, 7.9, 4800, 2000, 65.8, 2014),
            ("Jakarta Foundry Co", 183_900, 186_000, 267, 289, 71, 74, 29, 31, 7.1, 7.9, 3600, 1400, 72.3, 2008),
        ],
    ),
    "Western Europe": (
        "Western Europe",
        [
            ("Rhine Components", 334_500, 336_800, 98, 105, 187, 182, 31, 33, 9.6, 8.4, 2800, 400, 81.5, 1952),
            ("Loire Fabrication", 278_100, 280_500, 167, 178, 168, 164, 34, 36, 8.8, 8.4, 2200, 380, 76.2, 1971),
            ("Lombardy Castings", 312_800, 315_000, 189, 201, 145, 141, 27, 29, 8.5, 8.4, 2600, 520, 83.7, 1983),
            ("Ruhr Steelworks", 398_200, 401_000, 234, 251, 132, 128, 25, 27, 8.1, 8.4, 3400, 700, 88.9, 1948),
            ("Basque Alloys", 245_600, 247_800, 178, 192, 156, 152, 30, 32, 8.3, 8.4, 1900, 340, 74.8, 1996),
            ("Flanders Precision", 221_400, 223_000, 112, 119, 198, 193, 36, 38, 9.2, 8.4, 1600, 250, 71.3, 1965),
            ("Iberian Metals", 189_700, 191_500, 201, 218, 112, 108, 33, 35, 7.4, 8.4, 2100, 580, 68.9, 2002),
            ("Danube Industries", 267_300, 269_500, 156, 167, 141, 137, 28, 30, 8.6, 8.4, 2400, 460, 79.1, 1978),
        ],
    ),
    "North America": (
        "North America",
        [
            ("Great Lakes Foundry", 445_100, 448_000, 312, 334, 124, 120, 19, 21, 8.4, 8.0, 3900, 850, 89.5, 1956),
            ("Cascadia Materials", 312_400, 314_800, 187, 201, 138, 134, 23, 25, 8.9, 8.0, 2700, 500, 82.3, 1979),
            ("Appalachian Steel", 389_700, 392_000, 267, 289, 108, 105, 21, 23, 7.8, 8.0, 3500, 780, 86.1, 1962),
            ("Sonoran Fabrication", 234_800, 236_500, 178, 192, 95, 92, 26, 28, 7.5, 8.0, 2800, 920, 75.4, 1998),
            ("Prairie Components", 278_600, 280_200, 156, 168, 119, 116, 22, 24, 8.6, 8.0, 2200, 410, 80.7, 1985),
            ("Alberta Precision", 198_300, 200_100, 112, 121, 167, 163, 29, 31, 9.3, 8.0, 1500, 280, 73.2, 1974),
            ("Gulf Coast Metals", 356_900, 359_200, 289, 312, 102, 99, 20, 22, 7.6, 8.0, 3200, 890, 87.8, 1969),
            ("Chesapeake Alloys", 267_100, 269_000, 145, 157, 148, 144, 25, 27, 8.8, 8.0, 2000, 350, 78.9, 1982),
        ],
    ),
    "South America": (
        "South America",
        [
            ("Patagonia Steel", 234_500, 236_800, 312, 338, 155, 151, 51, 54, 6.9, 7.2, 3100, 1200, 71.2, 1987),
            ("Amazonia Metals", 189_200, 191_000, 267, 289, 88, 85, 42, 45, 6.5, 7.2, 4200, 1800, 66.3, 2003),
            ("Andean Fabrication", 156_800, 158_500, 198, 215, 102, 99, 38, 41, 7.1, 7.2, 2800, 950, 68.7, 1995),
            ("Pampas Industrial", 278_900, 281_200, 234, 253, 115, 112, 35, 37, 7.6, 7.2, 2400, 680, 74.5, 1991),
            ("Orinoco Castings", 134_500, 136_200, 189, 205, 78, 75, 44, 47, 6.3, 7.2, 3500, 1500, 62.1, 2009),
            ("Cerrado Components", 212_700, 214_500, 178, 193, 97, 94, 37, 39, 7.3, 7.2, 2600, 780, 70.8, 1999),
            ("Altiplano Alloys", 145_600, 147_300, 212, 231, 84, 81, 41, 44, 6.7, 7.2, 3200, 1300, 64.5, 2006),
            ("Llanos Metalworks", 167_800, 169_500, 223, 241, 91, 88, 39, 42, 6.8, 7.2, 2900, 1100, 67.2, 2001),
        ],
    ),
    "Scandinavia": (
        "Scandinavia",
        [
            ("Nordic Forge", 486_800, 489_500, 156, 167, 178, 174, 26, 28, 9.2, 8.8, 3100, 420, 92.1, 1938),
            ("Fjord Industries", 334_200, 336_500, 123, 131, 192, 187, 30, 32, 9.5, 8.8, 2400, 310, 85.4, 1955),
            ("Baltic Fabrication", 267_500, 269_800, 145, 156, 165, 161, 28, 30, 8.9, 8.8, 2000, 350, 80.6, 1972),
            ("Lappland Materials", 198_900, 200_500, 89, 96, 215, 210, 34, 36, 9.7, 8.8, 1200, 180, 72.8, 1961),
            ("Aland Precision", 156_700, 158_200, 78, 84, 234, 228, 37, 39, 9.8, 8.8, 900, 120, 68.1, 1949),
            ("Gothenburg Works", 378_400, 380_800, 198, 213, 152, 148, 24, 26, 8.7, 8.8, 2800, 510, 88.3, 1945),
            ("Stavanger Alloys", 289_100, 291_400, 134, 143, 175, 171, 27, 29, 9.1, 8.8, 2100, 340, 82.7, 1968),
            ("Helsinki Castings", 223_600, 225_200, 112, 120, 189, 184, 31, 33, 9.3, 8.8, 1700, 260, 77.4, 1958),
        ],
    ),
    "Alps": (
        "Alpine Region",
        [
            ("Helvetia Precision", 385_000, 387_500, 112, 89, 205, 200, 32, 34, 9.5, 9.0, 2500, 320, 84.2, 1943),
            ("Tyrolean Metalworks", 267_300, 269_500, 134, 143, 178, 174, 29, 31, 9.1, 9.0, 1800, 280, 79.6, 1956),
            ("Bavarian Components", 345_200, 347_600, 178, 190, 162, 158, 26, 28, 8.8, 9.0, 2900, 480, 86.8, 1964),
            ("Dolomite Alloys", 198_400, 200_100, 98, 105, 221, 216, 35, 37, 9.6, 9.0, 1300, 190, 73.5, 1951),
            ("Jura Fabrication", 234_700, 236_300, 145, 156, 185, 180, 31, 33, 9.2, 9.0, 1600, 240, 77.1, 1959),
            ("Engadin Works", 312_600, 314_800, 167, 179, 155, 151, 27, 29, 8.7, 9.0, 2200, 390, 83.4, 1971),
            ("Vorarlberg Castings", 178_900, 180_400, 89, 95, 238, 232, 38, 40, 9.7, 9.0, 1100, 160, 69.8, 1947),
            ("Bernese Metals", 289_400, 291_700, 156, 167, 168, 164, 28, 30, 9.0, 9.0, 2000, 330, 81.2, 1966),
        ],
    ),
    "Central Asia": (
        "Central Asia",
        [
            ("Ural Heavy Industries", 423_100, 426_000, 378, 412, 178, 173, 52, 56, 7.1, 6.8, 5800, 2400, 78.9, 1941),
            ("Kazakh Metals Corp", 312_400, 314_800, 289, 312, 92, 89, 43, 46, 6.4, 6.8, 4500, 1900, 72.1, 1953),
            ("Gobi Materials", 178_500, 180_200, 145, 156, 162, 158, 48, 51, 8.3, 6.8, 1800, 420, 68.4, 1998),
            ("Silk Road Fabrication", 234_700, 236_500, 234, 253, 108, 105, 39, 42, 7.0, 6.8, 3200, 1200, 74.6, 1985),
            ("Tashkent Alloys", 267_800, 269_500, 256, 278, 98, 95, 41, 44, 6.6, 6.8, 3800, 1600, 76.3, 1967),
            ("Caspian Components", 198_200, 200_000, 198, 214, 115, 112, 37, 40, 7.2, 6.8, 2900, 980, 70.5, 1979),
            ("Altai Precision", 145_600, 147_200, 123, 134, 145, 141, 35, 37, 7.8, 6.8, 1600, 450, 65.8, 1992),
            ("Aral Metalworks", 189_300, 191_000, 212, 229, 105, 102, 40, 43, 6.7, 6.8, 3400, 1400, 71.2, 1974),
        ],
    ),
    "Africa": (
        "Sub-Saharan Africa",
        [
            ("Saharan Metals", 198_400, 200_200, 267, 289, 171, 167, 47, 50, 7.8, 6.5, 3800, 1600, 69.1, 1989),
            ("Zambezi Foundry", 145_200, 146_800, 198, 215, 82, 79, 44, 47, 6.1, 6.5, 4200, 1900, 62.4, 2007),
            ("Highveld Components", 289_700, 292_000, 234, 253, 118, 115, 36, 38, 7.5, 6.5, 2800, 920, 75.3, 1982),
            ("Rift Valley Alloys", 112_300, 113_800, 178, 193, 74, 71, 49, 52, 5.8, 6.5, 5100, 2300, 58.7, 2012),
            ("Congo Basin Works", 134_800, 136_500, 212, 231, 68, 65, 53, 57, 5.4, 6.5, 5800, 2700, 55.2, 2015),
            ("Limpopo Steel", 223_500, 225_300, 189, 205, 105, 102, 38, 41, 7.2, 6.5, 3200, 1100, 72.8, 1993),
            ("Nile Delta Fabrication", 178_600, 180_300, 201, 218, 92, 89, 42, 45, 6.6, 6.5, 3600, 1400, 67.5, 2000),
            ("Kalahari Metals", 156_900, 158_400, 178, 193, 88, 85, 40, 43, 6.8, 6.5, 3400, 1300, 65.1, 2004),
        ],
    ),
    "Oceania": (
        "Oceania & Pacific",
        [
            ("Pacifica Alloys", 487_200, 489_000, 198, 213, 145, 141, 28, 30, 8.9, 7.8, 3500, 680, 91.7, 1958),
            ("Outback Foundry", 356_800, 359_100, 234, 253, 112, 109, 24, 26, 8.1, 7.8, 2900, 620, 86.3, 1971),
            ("Tasman Components", 278_400, 280_600, 167, 179, 138, 134, 27, 29, 8.6, 7.8, 2200, 410, 82.5, 1984),
            ("Coral Sea Metals", 189_700, 191_400, 145, 156, 128, 125, 31, 33, 8.3, 7.8, 1800, 350, 76.8, 1992),
            ("Polynesia Steel", 112_500, 113_800, 178, 193, 95, 92, 38, 41, 6.9, 7.8, 2600, 980, 64.2, 2008),
            ("Kimberley Fabrication", 312_100, 314_400, 201, 217, 125, 121, 25, 27, 8.2, 7.8, 2500, 520, 84.1, 1976),
            ("Canterbury Alloys", 234_600, 236_200, 145, 156, 152, 148, 29, 31, 8.7, 7.8, 1700, 300, 79.3, 1965),
            ("Torres Strait Works", 167_300, 169_000, 189, 205, 108, 105, 33, 35, 7.5, 7.8, 2100, 780, 71.6, 1999),
        ],
    ),
    "South Asia": (
        "South Asia",
        [
            ("Deccan Manufacturing", 423_800, 426_500, 345, 378, 72, 69, 31, 34, 7.2, 6.9, 6200, 2800, 82.4, 1976),
            ("Bengal Industrial", 312_500, 314_800, 289, 312, 65, 62, 34, 37, 6.8, 6.9, 5400, 2400, 78.1, 1983),
            ("Indus Precision", 234_100, 236_000, 178, 193, 88, 85, 29, 31, 7.6, 6.9, 3200, 980, 75.6, 1991),
            ("Tamil Components", 378_200, 380_500, 312, 338, 68, 65, 32, 35, 7.0, 6.9, 5800, 2600, 80.9, 1979),
            ("Ganges Alloys", 289_600, 291_300, 256, 278, 75, 72, 30, 33, 7.1, 6.9, 4600, 1900, 77.3, 1986),
            ("Lankan Fabrication", 156_400, 158_100, 198, 215, 58, 55, 37, 40, 6.3, 6.9, 4100, 1800, 66.5, 2005),
            ("Himalayan Metals", 178_700, 180_400, 212, 230, 82, 79, 35, 38, 6.9, 6.9, 3800, 1500, 70.8, 1997),
            ("Rajasthan Castings", 267_900, 269_600, 234, 253, 78, 75, 33, 36, 7.3, 6.9, 4200, 1700, 76.4, 1988),
        ],
    ),
    "Middle East": (
        "Middle East & North Africa",
        [
            ("Arabian Gulf Industries", 378_500, 381_000, 234, 253, 135, 131, 26, 28, 8.2, 7.4, 3400, 1200, 84.6, 1978),
            ("Nile Fabrication", 234_200, 236_000, 189, 205, 98, 95, 33, 36, 7.3, 7.4, 2800, 950, 76.2, 1991),
            ("Mesopotamia Metals", 312_700, 315_000, 267, 289, 112, 109, 29, 31, 7.6, 7.4, 3100, 1100, 80.5, 1985),
            ("Levant Components", 189_800, 191_500, 156, 168, 145, 141, 31, 33, 8.1, 7.4, 2000, 520, 73.8, 1996),
            ("Maghreb Alloys", 267_400, 269_200, 212, 229, 108, 105, 30, 32, 7.5, 7.4, 2600, 880, 78.1, 1988),
            ("Persian Works", 345_100, 347_400, 245, 265, 122, 118, 27, 29, 7.9, 7.4, 3000, 1050, 82.3, 1981),
            ("Anatolian Steel", 398_600, 401_000, 278, 301, 118, 115, 25, 27, 7.8, 7.4, 3600, 1300, 86.1, 1973),
            ("Sinai Fabrication", 156_300, 158_000, 178, 193, 85, 82, 36, 39, 6.8, 7.4, 2200, 780, 68.4, 2003),
        ],
    ),
}

_REGION_DATABASE: dict[str, dict[str, Any]] = {}
for _rk, (_rn, _rs) in _REGION_RAW.items():
    _REGION_DATABASE[_rk] = {
        "region": _rn,
        "suppliers": [dict(zip(_SUPPLIER_KEYS, s, strict=True)) for s in _rs],
    }


def _generate_extreme_report(region: dict[str, Any]) -> str:
    """Produce a wall-of-text analyst report with maximum noise."""
    parts: list[str] = []
    suppliers: list[dict[str, Any]] = region["suppliers"]

    parts.append(
        f"The {region['region']} manufacturing landscape presents a complex picture for the current "
        f"fiscal period. Regional analysts from multiple independent research firms have compiled "
        f"production data, though as is often the case with cross-border manufacturing statistics, "
        f"there are notable discrepancies between official company filings, trade journal estimates, "
        f"and third-party audit results. The following synthesis attempts to reconcile these sources "
        f"while noting where significant disagreements persist. All cost figures are normalised to "
        f"USD at prevailing exchange rates. Defect counts reflect the standardised ISO 9001 "
        f"methodology unless otherwise noted, and quality scores are on the universal 1-10 composite "
        f"scale incorporating process maturity, output consistency, and customer satisfaction metrics."
    )

    for s in suppliers:
        total_wf = s["workforce"] + s["temp_workers"]
        prior_units = round(s["units_produced"] * 0.95)
        alt_defect_pct = f"{s['defects'] / s['units_produced'] * 100:.4f}"
        prior_defect_pct = f"{s['defects_prior_year'] / prior_units * 100:.4f}"

        parts.append(
            f"Turning to {s['name']}, which was established in {s['year_established']} and operates "
            f"with a total workforce of approximately {total_wf:,} individuals (of whom "
            f"{s['workforce']:,} are classified as permanent full-time employees and the remaining "
            f"{s['temp_workers']:,} are temporary or contract workers engaged on varying terms), "
            f"the company reported audited production output of {s['units_produced']:,} units for "
            f"the period under review. It is worth noting that the {region['region']} Trade Journal "
            f"published a somewhat higher figure of {s['units_reported_by_trade_journal']:,} units, "
            f"which appears to include units in final quality assurance stages but had not yet passed "
            f"final inspection at the reporting date; the company itself has confirmed the lower "
            f"figure of {s['units_produced']:,} as the definitive audited count. In the prior fiscal "
            f"year, production was estimated at roughly {prior_units:,} units, suggesting year-on-year "
            f"growth of approximately {(1 / 0.95 - 1) * 100:.1f}% on an absolute basis though some "
            f"of this may reflect capacity expansion rather than efficiency gains given that capacity "
            f"utilisation currently stands at {s['capacity_utilization']}% compared to the regional "
            f"average of roughly {s['capacity_utilization'] * 0.95:.1f}%. The defect count under "
            f"ISO 9001 criteria was recorded at {s['defects']} units, yielding a per-unit defect "
            f"rate of {alt_defect_pct}%, which represents a "
            f"{'favourable improvement' if s['defects'] < s['defects_prior_year'] else 'concerning increase'} "
            f"compared to the prior year figure of {s['defects_prior_year']} defects (approximately "
            f"{prior_defect_pct}% on the then-lower production base). An alternative methodology used "
            f"by some regional auditors, which counts partial defects at half weight, would yield an "
            f"adjusted defect count closer to {round(s['defects'] * 0.7)} but this is not the "
            f"standard measure. Unit production cost was reported at ${s['cost_per_unit']} per unit "
            f"according to company filings, although a competitor intelligence report circulated in "
            f"the industry suggests the true all-in cost may be closer to "
            f"${s['cost_per_unit_competitor_estimate']} when accounting for unreported logistics "
            f"surcharges and quality remediation expenses; for the purposes of this analysis we rely "
            f"on the company-reported figure of ${s['cost_per_unit']}. Lead time from order placement "
            f"to delivery averaged {s['lead_time_days']} calendar days during the current period, "
            f"compared to {s['lead_time_days_last_quarter']} days in the previous quarter, and the "
            f"composite quality score assigned by the independent rating consortium stands at "
            f"{s['quality_score']} against a regional industry average of "
            f"{s['quality_score_industry_avg']}. It should be noted that quality scoring "
            f"methodologies were revised in the most recent assessment cycle, and direct comparison "
            f"with scores published more than two years ago requires a correction factor of "
            f"approximately 0.3 points upward on the legacy scale."
        )

    parts.append(
        f"This concludes the regional assessment for {region['region']}. Readers are cautioned "
        f"that all figures represent point-in-time estimates and may be subject to revision in "
        f"subsequent reporting periods as audits are finalised and exchange rate adjustments are "
        f"applied retroactively."
    )
    return " ".join(parts)


@tool
def analyze_region(region: str) -> str:
    """Retrieve a detailed supply chain analyst report for a manufacturing region.

    Covers 8 suppliers with production units, defects, costs, lead times,
    quality scores, and workforce data.

    Args:
        region: Region name (e.g. 'East Asia', 'Western Europe').
    """
    key = next(
        (k for k in _REGION_DATABASE if k.lower() == region.lower() or _REGION_DATABASE[k]["region"].lower() == region.lower()),
        None,
    )
    if key is None:
        return f"No data found for region: {region}"
    return _generate_extreme_report(_REGION_DATABASE[key])


_EVAL1_REGIONS = [
    "East Asia",
    "Western Europe",
    "North America",
    "South America",
    "Scandinavia",
    "Alps",
    "Central Asia",
    "Africa",
    "Oceania",
    "South Asia",
    "Middle East",
]

_EVAL1_QUERY = (
    f"Analyze all 11 regions: {', '.join(_EVAL1_REGIONS)}. "
    "Use the AUDITED production figures (not trade journal estimates) and "
    "company-reported costs (not competitor estimates). Use the ISO 9001 "
    "defect count for the CURRENT period (not prior year, not adjusted). "
    "Then answer these questions precisely:\n"
    "1. Which supplier across all regions has the highest audited "
    "units_produced? Give the exact name and number.\n"
    "2. Which supplier has the lowest defect rate (defects divided by "
    "audited units_produced)? Give the exact name and the ratio.\n"
    "3. List every supplier where lead_time_days > 45 AND cost_per_unit > "
    "$150. Among those, which has the highest quality_score? Give the "
    "name and score."
)


@pytest.mark.langsmith
def test_extreme_noisy_extraction(model: BaseChatModel) -> None:
    """Eval 1: 96 suppliers across 11 regions with wall-of-text noise."""
    agent = create_deep_agent(
        model=model,
        system_prompt=(
            "You are a global supply chain supervisor. Delegate each region "
            "analysis to your regional_analyst subagent in parallel (one task "
            "call per region). After receiving all results, answer the user's "
            "questions using exact numbers. For defect rate, compute "
            "defects / audited units_produced for each supplier. Be precise — "
            "many suppliers have very similar numbers so small differences matter."
        ),
        subagents=[
            {
                "name": "regional_analyst",
                "description": (
                    "Analyzes a manufacturing region using the analyze_region tool. "
                    "Returns data on 8 suppliers including production units, defects, "
                    "costs, lead times, and quality scores."
                ),
                "system_prompt": (
                    "You are a supply chain analyst. When given a region, use the "
                    "analyze_region tool to retrieve the report. Extract data for "
                    "each supplier carefully. Use audited figures, not trade journal "
                    "estimates. Use current-period defects, not prior year. Use "
                    "company-reported costs, not competitor estimates."
                ),
                "tools": [analyze_region],
            }
        ],
    )
    run_agent(
        agent,
        query=_EVAL1_QUERY,
        model=model,
        scorer=(
            TrajectoryScorer()
            .expect(
                agent_steps=2,
                tool_call_requests=11,
                tool_calls=[tool_call(name="task", args_contains={"subagent_type": "regional_analyst"})],
            )
            .success(
                final_text_contains("Pacifica", case_insensitive=True),
                final_text_contains("Helvetia", case_insensitive=True),
                final_text_contains("Gobi", case_insensitive=True),
            )
        ),
    )


# ===========================================================================
# Eval 2: Full-dataset aggregation
#
# 8 sectors x 5 companies = 40 total. Query requires statistics that
# cannot be answered without ALL subagent results: median, percentage,
# and ranking.
#
# Ground truth (pre-computed):
#   Median revenue-per-employee: ~$251,411
#     (20th value: TitanCore 248,276 | 21st value: NeuralPath 254,545)
#
#   Companies with risk > 7.0 — 11 out of 40 = 27.5%
#     BioVantage 7.2 | FluxLayer 7.8 | AtomicHorizon 7.1
#     IronForge 8.4 | CobaltRidge 7.6 | RareEarth 7.3
#     DeepCurrentAI 8.7 | SynapticWave 7.4 | CortexAI 7.9
#     AquaPure 8.1 | CoinVault 7.5
#
#   Top 3 sectors by avg annual growth — AI 19.56%, Fintech 12.46%, Cloud 11.58%
# ===========================================================================

_COMPANY_KEYS = (
    "name",
    "revenue",
    "prev_year_revenue",
    "employees",
    "contractors",
    "growth_rate_annual",
    "growth_rate_quarterly",
    "market_cap",
    "enterprise_value",
    "risk_score",
)

_SECTOR_RAW: dict[str, tuple[str, list[tuple[Any, ...]]]] = {
    "Technology": (
        "Technology",
        [
            ("QuantumLeap Systems", 3_800_000_000, 3_400_000_000, 14200, 3100, 11.8, 3.2, 42_000_000_000, 45_500_000_000, 5.2),
            ("ByteForge Inc", 2_100_000_000, 1_950_000_000, 8900, 2200, 7.7, 1.8, 18_000_000_000, 19_200_000_000, 4.1),
            ("NeuralPath Corp", 5_600_000_000, 5_100_000_000, 22000, 5500, 9.8, 2.6, 67_000_000_000, 71_000_000_000, 3.8),
            ("CipherDyn", 890_000_000, 820_000_000, 3400, 800, 8.5, 2.1, 7_200_000_000, 7_800_000_000, 6.3),
            ("VoltGrid Tech", 1_450_000_000, 1_380_000_000, 6100, 1400, 5.1, 1.3, 12_500_000_000, 13_100_000_000, 4.7),
        ],
    ),
    "Healthcare": (
        "Healthcare",
        [
            ("NovaPharma", 4_200_000_000, 3_750_000_000, 8200, 2100, 12.0, 3.4, 52_000_000_000, 55_000_000_000, 5.9),
            ("MediSync Global", 2_800_000_000, 2_600_000_000, 11500, 3200, 7.7, 1.9, 28_000_000_000, 30_500_000_000, 4.5),
            ("BioVantage Labs", 1_600_000_000, 1_480_000_000, 6700, 1800, 8.1, 2.0, 19_000_000_000, 20_200_000_000, 7.2),
            ("HelixCure", 950_000_000, 870_000_000, 4100, 950, 9.2, 2.5, 11_000_000_000, 11_800_000_000, 6.8),
            ("PulsePoint Diagnostics", 3_100_000_000, 2_900_000_000, 13200, 2800, 6.9, 1.7, 35_000_000_000, 37_000_000_000, 3.4),
        ],
    ),
    "Cloud": (
        "Cloud Infrastructure",
        [
            ("StratoCloud", 5_100_000_000, 4_500_000_000, 10100, 4200, 13.3, 3.6, 78_000_000_000, 82_000_000_000, 4.9),
            ("NimbusScale", 2_300_000_000, 2_050_000_000, 7800, 2100, 12.2, 3.1, 31_000_000_000, 33_000_000_000, 5.5),
            ("EdgeVault", 1_700_000_000, 1_550_000_000, 5400, 1300, 9.7, 2.4, 20_000_000_000, 21_500_000_000, 6.1),
            ("TerraNode Systems", 3_400_000_000, 3_100_000_000, 12800, 3500, 9.7, 2.5, 41_000_000_000, 44_000_000_000, 4.2),
            ("FluxLayer", 780_000_000, 690_000_000, 2900, 700, 13.0, 3.5, 9_500_000_000, 10_100_000_000, 7.8),
        ],
    ),
    "Energy": (
        "Energy",
        [
            ("SolarPeak", 2_900_000_000, 2_600_000_000, 9500, 4100, 11.5, 3.0, 33_000_000_000, 36_000_000_000, 5.7),
            ("FusionDrive Energy", 4_800_000_000, 4_400_000_000, 18000, 6200, 9.1, 2.3, 55_000_000_000, 59_000_000_000, 4.3),
            ("WindCrest Power", 1_200_000_000, 1_100_000_000, 5200, 1800, 9.1, 2.2, 14_000_000_000, 15_200_000_000, 6.5),
            ("AtomicHorizon", 6_200_000_000, 5_800_000_000, 25000, 7500, 6.9, 1.7, 71_000_000_000, 76_000_000_000, 7.1),
            ("GreenArc Solutions", 850_000_000, 780_000_000, 3600, 900, 9.0, 2.3, 8_200_000_000, 8_900_000_000, 5.0),
        ],
    ),
    "Mining": (
        "Mining & Materials",
        [
            ("IronForge Global", 7_100_000_000, 6_500_000_000, 32000, 11000, 9.2, 2.4, 48_000_000_000, 53_000_000_000, 8.4),
            ("CobaltRidge", 2_400_000_000, 2_200_000_000, 9800, 3400, 9.1, 2.2, 22_000_000_000, 24_000_000_000, 7.6),
            ("TitanCore Minerals", 3_600_000_000, 3_300_000_000, 14500, 5200, 9.1, 2.3, 37_000_000_000, 40_000_000_000, 6.9),
            ("RareEarth Dynamics", 1_100_000_000, 980_000_000, 4800, 1500, 12.2, 3.2, 13_000_000_000, 14_000_000_000, 7.3),
            ("SilverVein Corp", 1_800_000_000, 1_700_000_000, 7600, 2800, 5.9, 1.4, 16_000_000_000, 17_500_000_000, 5.8),
        ],
    ),
    "AI": (
        "Artificial Intelligence",
        [
            ("DeepCurrentAI", 3_200_000_000, 2_700_000_000, 6800, 2400, 18.5, 5.1, 58_000_000_000, 61_000_000_000, 8.7),
            ("SynapticWave", 1_900_000_000, 1_600_000_000, 5200, 1800, 18.8, 5.0, 34_000_000_000, 36_000_000_000, 7.4),
            ("CortexAI Labs", 750_000_000, 580_000_000, 2100, 800, 29.3, 7.8, 21_000_000_000, 22_000_000_000, 7.9),
            ("LogicMesh", 2_600_000_000, 2_300_000_000, 9400, 3100, 13.0, 3.4, 39_000_000_000, 41_500_000_000, 5.6),
            ("PerceptronX", 1_300_000_000, 1_100_000_000, 4500, 1200, 18.2, 4.8, 25_000_000_000, 26_500_000_000, 6.8),
        ],
    ),
    "Water": (
        "Water & Utilities",
        [
            ("AquaPure Systems", 1_500_000_000, 1_400_000_000, 7200, 2500, 7.1, 1.8, 3_100_000_000, 3_800_000_000, 8.1),
            ("HydroVolt", 2_200_000_000, 2_050_000_000, 9100, 3000, 7.3, 1.9, 11_500_000_000, 12_800_000_000, 4.8),
            ("ClearStream Global", 3_800_000_000, 3_500_000_000, 15800, 4200, 8.6, 2.1, 27_000_000_000, 29_500_000_000, 3.9),
            ("TidalForce", 900_000_000, 830_000_000, 3800, 1100, 8.4, 2.1, 7_800_000_000, 8_400_000_000, 5.3),
            ("ReservoirTech", 1_700_000_000, 1_580_000_000, 6900, 2000, 7.6, 1.9, 14_000_000_000, 15_200_000_000, 4.1),
        ],
    ),
    "Fintech": (
        "Financial Technology",
        [
            ("LedgerPrime", 2_700_000_000, 2_400_000_000, 5800, 2000, 12.5, 3.3, 44_000_000_000, 46_500_000_000, 6.2),
            ("PayCircuit", 4_500_000_000, 4_100_000_000, 16000, 4500, 9.8, 2.5, 62_000_000_000, 65_000_000_000, 4.0),
            ("CoinVault Exchange", 1_800_000_000, 1_500_000_000, 4200, 1300, 20.0, 5.4, 29_000_000_000, 30_500_000_000, 7.5),
            ("InsurTech Nexus", 1_100_000_000, 1_000_000_000, 4900, 1100, 10.0, 2.6, 15_000_000_000, 16_000_000_000, 5.1),
            ("WealthGrid", 3_300_000_000, 3_000_000_000, 11200, 3200, 10.0, 2.6, 40_000_000_000, 42_500_000_000, 4.4),
        ],
    ),
}

_SECTOR_DATABASE: dict[str, dict[str, Any]] = {}
for _sk, (_sn, _sc) in _SECTOR_RAW.items():
    _SECTOR_DATABASE[_sk] = {
        "sector": _sn,
        "companies": [dict(zip(_COMPANY_KEYS, c, strict=True)) for c in _sc],
    }


def _generate_sector_report(sector: dict[str, Any]) -> str:
    """Produce a noisy analyst report for a market sector."""
    lines: list[str] = [
        f"SECTOR ANALYSIS REPORT: {sector['sector'].upper()}",
        "",
        f"This comprehensive quarterly review covers the {sector['sector']} sector. "
        f"The following assessments are based on audited financial statements, independent "
        f"analyst estimates, and proprietary risk modelling. Note that figures may differ "
        f"slightly from other published sources due to methodological differences in revenue "
        f"recognition and employee classification. All monetary values are in USD.",
        "",
    ]
    for c in sector["companies"]:
        rev_b = c["revenue"] / 1_000_000_000
        prev_b = c["prev_year_revenue"] / 1_000_000_000
        cap_b = c["market_cap"] / 1_000_000_000
        ev_b = c["enterprise_value"] / 1_000_000_000
        total_wf = c["employees"] + c["contractors"]
        lines.extend(
            [
                f"--- {c['name']} ---",
                "",
                f"{c['name']} reported annual revenue of approximately ${rev_b:.1f} billion, "
                f"up from ${prev_b:.1f} billion in the prior year. Some analysts cite a higher "
                f"estimate of ${rev_b * 1.02:.1f} billion including deferred revenue adjustments, "
                f"though the audited figure remains ${c['revenue'] / 1_000_000_000:.2f} billion. "
                f"Annual growth rate stands at {c['growth_rate_annual']}%, while the most recent "
                f"quarterly growth was {c['growth_rate_quarterly']}%. Historical five-year CAGR "
                f"is estimated at roughly {c['growth_rate_annual'] * 0.85:.1f}%.",
                "",
                f"The company employs {c['employees']:,} full-time staff with an additional "
                f"{c['contractors']:,} contractors (total workforce ~{total_wf:,}). The previous "
                f"year's headcount was roughly {round(c['employees'] * 0.95):,} FTEs. When "
                f"evaluating productivity metrics, use the full-time employee count of "
                f"{c['employees']:,} rather than the total workforce figure.",
                "",
                f"Market cap: ${cap_b:.1f}B. Enterprise value: ${ev_b:.1f}B. EV/Revenue "
                f"multiple: ~{c['enterprise_value'] / c['revenue']:.1f}x. Risk score: "
                f"{c['risk_score']} on a 1-10 scale (sector avg: "
                f"{sum(x['risk_score'] for x in sector['companies']) / len(sector['companies']):.1f}).",
                "",
            ]
        )
    lines.append(f"END OF {sector['sector'].upper()} SECTOR REPORT.")
    return "\n".join(lines)


@tool
def analyze_sector(sector: str) -> str:
    """Retrieve a detailed analyst report for a market sector.

    Covers 5 companies with revenue, employees, growth, market cap, and risk.

    Args:
        sector: Sector name (e.g. 'Technology', 'Healthcare', 'AI').
    """
    key = next(
        (k for k in _SECTOR_DATABASE if k.lower() == sector.lower() or _SECTOR_DATABASE[k]["sector"].lower() == sector.lower()),
        None,
    )
    if key is None:
        return f"No data found for sector: {sector}"
    return _generate_sector_report(_SECTOR_DATABASE[key])


_EVAL2_SECTORS = [
    "Technology",
    "Healthcare",
    "Cloud",
    "Energy",
    "Mining",
    "AI",
    "Water",
    "Fintech",
]

_EVAL2_QUERY = (
    f"Analyze all 8 sectors: {', '.join(_EVAL2_SECTORS)}. "
    "For each company, use the full-time employee count (NOT total workforce "
    "including contractors). Then answer these questions:\n"
    "1. Across all 40 companies, what is the median revenue-per-employee "
    "ratio? Show the value.\n"
    "2. What percentage of the 40 companies have a risk score strictly "
    "above 7.0?\n"
    "3. Rank the 8 sectors by average annual growth rate and name the top 3."
)


@pytest.mark.langsmith
def test_full_dataset_aggregation(model: BaseChatModel) -> None:
    """Eval 2: 40 companies across 8 sectors requiring median/percentage/ranking."""
    agent = create_deep_agent(
        model=model,
        system_prompt=(
            "You are a market research supervisor. Delegate each sector analysis "
            "to your sector_analyst subagent in parallel (one task call per sector). "
            "After receiving all results, compute the requested metrics precisely "
            "using exact numbers. Revenue-per-employee means dividing annual revenue "
            "by full-time employee count. Show your work."
        ),
        subagents=[
            {
                "name": "sector_analyst",
                "description": (
                    "Analyzes a market sector using the analyze_sector tool. Returns "
                    "data on 5 companies including revenue, employees, growth, market "
                    "cap, and risk scores."
                ),
                "system_prompt": (
                    "You are a sector analysis agent. When given a sector name, use "
                    "the analyze_sector tool to retrieve the report and extract the "
                    "data for each company. Use the full-time employee count, not "
                    "total workforce. Use the audited revenue figure."
                ),
                "tools": [analyze_sector],
            }
        ],
    )
    run_agent(
        agent,
        query=_EVAL2_QUERY,
        model=model,
        scorer=(
            TrajectoryScorer()
            .expect(
                agent_steps=2,
                tool_call_requests=8,
                tool_calls=[tool_call(name="task", args_contains={"subagent_type": "sector_analyst"})],
            )
            .success(
                final_text_contains("19.5", case_insensitive=True),
                final_text_contains("27.5", case_insensitive=True),
                final_text_contains("Cloud", case_insensitive=True),
            )
        ),
    )


# ===========================================================================
# Eval 3: Cross-referencing between subagent result sets
#
# Two subagent types return data about the same 20 projects keyed by name.
# Financial analysts return budget/revenue; operations analysts return
# engineers/months. The supervisor must JOIN by project name to compute
# derived metrics.
#
# Projects split into 4 batches of 5: Alpha, Beta, Gamma, Delta
#
# Ground truth:
#   Highest budget-per-engineer: Lighthouse = 6,800,000 / 8 = $850,000
#     Runner-up: Titan = 7,200,000 / 9 = $800,000 (highest raw budget)
#   Highest revenue-per-month: Quantum = 25,500,000 / 6 = $4,250,000
#     Runner-up: Falcon = 15,500,000 / 4 = $3,875,000 (short project)
# ===========================================================================

_PROJECT_FINANCIAL: dict[str, dict[str, list[dict[str, Any]]]] = {
    "Alpha": {
        "projects": [
            {"name": "Atlas", "budget": 2_400_000, "revenue": 18_000_000, "overhead_pct": 12.3},
            {"name": "Beacon", "budget": 1_800_000, "revenue": 12_000_000, "overhead_pct": 9.8},
            {"name": "Cascade", "budget": 3_200_000, "revenue": 8_500_000, "overhead_pct": 15.1},
            {"name": "Dynamo", "budget": 950_000, "revenue": 6_200_000, "overhead_pct": 8.4},
            {"name": "Eclipse", "budget": 5_100_000, "revenue": 22_000_000, "overhead_pct": 14.7},
        ]
    },
    "Beta": {
        "projects": [
            {"name": "Falcon", "budget": 1_200_000, "revenue": 15_500_000, "overhead_pct": 7.2},
            {"name": "Granite", "budget": 2_800_000, "revenue": 9_800_000, "overhead_pct": 11.5},
            {"name": "Horizon", "budget": 4_500_000, "revenue": 31_000_000, "overhead_pct": 13.8},
            {"name": "Impulse", "budget": 750_000, "revenue": 4_800_000, "overhead_pct": 6.9},
            {"name": "Javelin", "budget": 3_600_000, "revenue": 14_200_000, "overhead_pct": 12.1},
        ]
    },
    "Gamma": {
        "projects": [
            {"name": "Keystone", "budget": 2_100_000, "revenue": 11_500_000, "overhead_pct": 10.3},
            {"name": "Lighthouse", "budget": 6_800_000, "revenue": 28_000_000, "overhead_pct": 16.5},
            {"name": "Meridian", "budget": 1_500_000, "revenue": 7_200_000, "overhead_pct": 8.7},
            {"name": "Nexus", "budget": 890_000, "revenue": 5_600_000, "overhead_pct": 7.1},
            {"name": "Olympus", "budget": 4_200_000, "revenue": 19_500_000, "overhead_pct": 13.2},
        ]
    },
    "Delta": {
        "projects": [
            {"name": "Pinnacle", "budget": 3_100_000, "revenue": 13_800_000, "overhead_pct": 11.9},
            {"name": "Quantum", "budget": 1_900_000, "revenue": 25_500_000, "overhead_pct": 9.4},
            {"name": "Redstone", "budget": 5_500_000, "revenue": 16_000_000, "overhead_pct": 14.3},
            {"name": "Summit", "budget": 2_600_000, "revenue": 9_100_000, "overhead_pct": 10.8},
            {"name": "Titan", "budget": 7_200_000, "revenue": 21_000_000, "overhead_pct": 15.6},
        ]
    },
}

_PROJECT_OPERATIONS: dict[str, dict[str, list[dict[str, Any]]]] = {
    "Alpha": {
        "projects": [
            {"name": "Atlas", "engineers": 12, "months": 24, "milestones": 8, "on_time_pct": 87.5},
            {"name": "Beacon", "engineers": 8, "months": 18, "milestones": 6, "on_time_pct": 83.3},
            {"name": "Cascade", "engineers": 15, "months": 12, "milestones": 5, "on_time_pct": 80.0},
            {"name": "Dynamo", "engineers": 4, "months": 15, "milestones": 4, "on_time_pct": 100.0},
            {"name": "Eclipse", "engineers": 22, "months": 30, "milestones": 12, "on_time_pct": 75.0},
        ]
    },
    "Beta": {
        "projects": [
            {"name": "Falcon", "engineers": 6, "months": 4, "milestones": 3, "on_time_pct": 100.0},
            {"name": "Granite", "engineers": 14, "months": 14, "milestones": 5, "on_time_pct": 80.0},
            {"name": "Horizon", "engineers": 20, "months": 36, "milestones": 15, "on_time_pct": 73.3},
            {"name": "Impulse", "engineers": 5, "months": 10, "milestones": 3, "on_time_pct": 100.0},
            {"name": "Javelin", "engineers": 16, "months": 22, "milestones": 9, "on_time_pct": 77.8},
        ]
    },
    "Gamma": {
        "projects": [
            {"name": "Keystone", "engineers": 10, "months": 16, "milestones": 6, "on_time_pct": 83.3},
            {"name": "Lighthouse", "engineers": 8, "months": 32, "milestones": 14, "on_time_pct": 71.4},
            {"name": "Meridian", "engineers": 7, "months": 12, "milestones": 4, "on_time_pct": 100.0},
            {"name": "Nexus", "engineers": 4, "months": 8, "milestones": 3, "on_time_pct": 100.0},
            {"name": "Olympus", "engineers": 18, "months": 24, "milestones": 10, "on_time_pct": 80.0},
        ]
    },
    "Delta": {
        "projects": [
            {"name": "Pinnacle", "engineers": 12, "months": 20, "milestones": 8, "on_time_pct": 87.5},
            {"name": "Quantum", "engineers": 9, "months": 6, "milestones": 3, "on_time_pct": 100.0},
            {"name": "Redstone", "engineers": 25, "months": 18, "milestones": 7, "on_time_pct": 71.4},
            {"name": "Summit", "engineers": 11, "months": 14, "milestones": 5, "on_time_pct": 80.0},
            {"name": "Titan", "engineers": 9, "months": 28, "milestones": 11, "on_time_pct": 72.7},
        ]
    },
}


def _generate_financial_report(batch: str) -> str:
    data = _PROJECT_FINANCIAL.get(batch)
    if data is None:
        return f"No financial data found for batch: {batch}"
    lines = [
        f"FINANCIAL DATA REPORT — Batch {batch}",
        "",
        "The following financial summary covers projects in this batch. All figures are "
        "from the most recent fiscal audit. Note that some projects have ongoing revenue "
        "recognition adjustments that may alter final figures by up to 3%. Overhead "
        "percentages include administrative costs, facilities, and shared services "
        "allocations. Budget figures represent total approved spend including contingency "
        "reserves; actual spend may differ.",
        "",
    ]
    for p in data["projects"]:
        adj_budget = round(p["budget"] * 1.03)
        adj_revenue = round(p["revenue"] * 0.97)
        lines.extend(
            [
                f"Project {p['name']}: Total approved budget of ${p['budget']:,} (some internal "
                f"forecasts suggest adjusted spend closer to ${adj_budget:,} after contingency "
                f"drawdowns). Revenue generated to date: ${p['revenue']:,}, though preliminary "
                f"reconciliation suggests a conservative estimate of ${adj_revenue:,} before "
                f"deferred recognition adjustments. Overhead allocation: {p['overhead_pct']}% "
                f"of budget. The project's ROI excluding overhead is approximately "
                f"{p['revenue'] / p['budget']:.1f}x.",
                "",
            ]
        )
    return "\n".join(lines)


def _generate_operations_report(batch: str) -> str:
    data = _PROJECT_OPERATIONS.get(batch)
    if data is None:
        return f"No operations data found for batch: {batch}"
    lines = [
        f"OPERATIONS DATA REPORT — Batch {batch}",
        "",
        "The following operational metrics cover projects in this batch. Engineer counts "
        "represent dedicated full-time equivalents; some projects also utilise shared "
        "platform engineers not counted here. Duration in months is measured from kickoff "
        "to current reporting date. Milestone completion is tracked against the approved "
        "project plan.",
        "",
    ]
    for p in data["projects"]:
        total_staff = p["engineers"] + round(p["engineers"] * 0.3)
        lines.extend(
            [
                f"Project {p['name']}: Staffed with {p['engineers']} dedicated engineers "
                f"(total team including shared resources is approximately {total_staff}). "
                f"Project duration: {p['months']} months from kickoff. {p['milestones']} "
                f"milestones defined, with {p['on_time_pct']}% completed on time. The "
                f"prior quarter's staffing was {p['engineers'] - 1} engineers before a "
                f"recent hire.",
                "",
            ]
        )
    return "\n".join(lines)


@tool
def get_financial_data(batch: str) -> str:
    """Retrieve financial data (budget, revenue) for a batch of projects.

    Args:
        batch: Batch name — one of 'Alpha', 'Beta', 'Gamma', 'Delta'.
    """
    return _generate_financial_report(batch)


@tool
def get_operations_data(batch: str) -> str:
    """Retrieve operations data (engineers, timeline) for a batch of projects.

    Args:
        batch: Batch name — one of 'Alpha', 'Beta', 'Gamma', 'Delta'.
    """
    return _generate_operations_report(batch)


_EVAL3_QUERY = (
    "Get both financial and operations data for all 20 projects across all "
    "4 batches (Alpha, Beta, Gamma, Delta). Then compute these metrics:\n"
    "1. For each project, compute budget-per-engineer (budget divided by "
    "number of dedicated engineers). Which project has the HIGHEST "
    "budget-per-engineer? Give the project name and exact value.\n"
    "2. For each project, compute revenue-per-month (revenue divided by "
    "project duration in months). Which project has the HIGHEST "
    "revenue-per-month? Give the project name and exact value."
)


@pytest.mark.langsmith
def test_cross_referencing(model: BaseChatModel) -> None:
    """Eval 3: 20 projects requiring joins between financial and operations data."""
    agent = create_deep_agent(
        model=model,
        system_prompt=(
            "You are a portfolio supervisor. You have two types of subagents: "
            "financial_analyst (for budget/revenue data) and operations_analyst "
            "(for engineer/timeline data). Data is organized into 4 batches: "
            "Alpha, Beta, Gamma, Delta (5 projects each). Delegate all 8 lookups "
            "in parallel (4 financial + 4 operations). After receiving all results, "
            "match projects by name to compute the requested cross-referenced metrics."
        ),
        subagents=[
            {
                "name": "financial_analyst",
                "description": ("Retrieves financial data (budget, revenue) for a batch of projects using the get_financial_data tool."),
                "system_prompt": (
                    "You are a financial analyst. Use the get_financial_data tool "
                    "with the batch name to retrieve financial metrics. Report the "
                    "budget and revenue for each project. Use the approved budget "
                    "figure, not adjusted estimates."
                ),
                "tools": [get_financial_data],
            },
            {
                "name": "operations_analyst",
                "description": ("Retrieves operations data (engineers, timeline) for a batch of projects using the get_operations_data tool."),
                "system_prompt": (
                    "You are an operations analyst. Use the get_operations_data "
                    "tool with the batch name to retrieve operational metrics. "
                    "Report the dedicated engineer count and duration in months "
                    "for each project. Use dedicated engineers, not total staff."
                ),
                "tools": [get_operations_data],
            },
        ],
    )
    run_agent(
        agent,
        query=_EVAL3_QUERY,
        model=model,
        scorer=(
            TrajectoryScorer()
            .expect(
                agent_steps=2,
                tool_call_requests=8,
                tool_calls=[
                    tool_call(name="task", args_contains={"subagent_type": "financial_analyst"}),
                    tool_call(name="task", args_contains={"subagent_type": "operations_analyst"}),
                ],
            )
            .success(
                final_text_contains("Lighthouse", case_insensitive=True),
                final_text_contains("Quantum", case_insensitive=True),
            )
        ),
    )


# ===========================================================================
# Eval 4: Contradictions between subagents
#
# 3 auditors report on the same 10 companies with deliberately different
# numbers. The supervisor must apply median-resolution.
#
# Ground truth (medians):
#   Highest revenue: Eos Industries (median 8,900M)
#     Runner-up: Gemini Solutions (median 8,850M) — only 50M gap
#     Trap: Auditor B reports Gemini at 9,200M (higher raw value)
#   Lowest operating cost: Helix Systems (median 850M)
#     Runner-up: Jupiter Ventures (median 870M) — only 20M gap
#     Trap: Auditor B reports Jupiter at 870M which equals its own median
# ===========================================================================

_AUDITOR_DATA: dict[str, list[dict[str, Any]]] = {
    "Auditor A": [
        {"name": "Apex Corp", "revenue": 5200, "operating_cost": 1420, "headcount": 12400},
        {"name": "Boreal Inc", "revenue": 3800, "operating_cost": 980, "headcount": 7200},
        {"name": "Crescent Ltd", "revenue": 7100, "operating_cost": 2100, "headcount": 18500},
        {"name": "Delphi Group", "revenue": 4500, "operating_cost": 1250, "headcount": 9800},
        {"name": "Eos Industries", "revenue": 8900, "operating_cost": 3100, "headcount": 22000},
        {"name": "Flux Dynamics", "revenue": 6300, "operating_cost": 1800, "headcount": 14200},
        {"name": "Gemini Solutions", "revenue": 8850, "operating_cost": 2400, "headcount": 21000},
        {"name": "Helix Systems", "revenue": 2900, "operating_cost": 850, "headcount": 5800},
        {"name": "Ionic Labs", "revenue": 5800, "operating_cost": 1600, "headcount": 11500},
        {"name": "Jupiter Ventures", "revenue": 4100, "operating_cost": 860, "headcount": 8400},
    ],
    "Auditor B": [
        {"name": "Apex Corp", "revenue": 5100, "operating_cost": 1380, "headcount": 12200},
        {"name": "Boreal Inc", "revenue": 3900, "operating_cost": 1020, "headcount": 7400},
        {"name": "Crescent Ltd", "revenue": 7400, "operating_cost": 2050, "headcount": 18800},
        {"name": "Delphi Group", "revenue": 4300, "operating_cost": 1300, "headcount": 9600},
        {"name": "Eos Industries", "revenue": 8600, "operating_cost": 3200, "headcount": 21500},
        {"name": "Flux Dynamics", "revenue": 6500, "operating_cost": 1750, "headcount": 14500},
        {"name": "Gemini Solutions", "revenue": 9200, "operating_cost": 2350, "headcount": 21500},
        {"name": "Helix Systems", "revenue": 3100, "operating_cost": 820, "headcount": 6000},
        {"name": "Ionic Labs", "revenue": 5600, "operating_cost": 1550, "headcount": 11200},
        {"name": "Jupiter Ventures", "revenue": 4200, "operating_cost": 870, "headcount": 8600},
    ],
    "Auditor C": [
        {"name": "Apex Corp", "revenue": 5300, "operating_cost": 1460, "headcount": 12600},
        {"name": "Boreal Inc", "revenue": 3700, "operating_cost": 960, "headcount": 7100},
        {"name": "Crescent Ltd", "revenue": 7200, "operating_cost": 2150, "headcount": 18300},
        {"name": "Delphi Group", "revenue": 4600, "operating_cost": 1200, "headcount": 10000},
        {"name": "Eos Industries", "revenue": 9100, "operating_cost": 2950, "headcount": 22400},
        {"name": "Flux Dynamics", "revenue": 6100, "operating_cost": 1850, "headcount": 14000},
        {"name": "Gemini Solutions", "revenue": 8500, "operating_cost": 2500, "headcount": 20800},
        {"name": "Helix Systems", "revenue": 2800, "operating_cost": 890, "headcount": 5600},
        {"name": "Ionic Labs", "revenue": 6000, "operating_cost": 1650, "headcount": 11800},
        {"name": "Jupiter Ventures", "revenue": 3900, "operating_cost": 940, "headcount": 8200},
    ],
}


def _generate_audit_report(auditor: str) -> str:
    companies = _AUDITOR_DATA.get(auditor)
    if companies is None:
        return f"No audit report found for: {auditor}"
    lines = [
        f"INDEPENDENT AUDIT REPORT — {auditor}",
        "",
        f"This report by {auditor} presents financial findings for 10 companies based on "
        f"our proprietary assessment methodology. Figures are in millions of USD. Revenue "
        f"and operating cost estimates reflect our independent analysis and may differ from "
        f"other auditors' findings due to differences in revenue recognition timing, cost "
        f"allocation methods, and headcount classification standards. Our methodology "
        f"weights Q3-Q4 performance more heavily for revenue estimation.",
        "",
    ]
    for c in companies:
        adj_rev = round(c["revenue"] * 1.05)
        lines.extend(
            [
                f"{c['name']}: Revenue assessed at ${c['revenue']:,}M (our growth-adjusted "
                f"forecast suggests potential upward revision to ${adj_rev:,}M pending Q4 "
                f"finalization). Operating cost: ${c['operating_cost']:,}M. Total headcount "
                f"surveyed: {c['headcount']:,} employees. Operating margin: "
                f"{(c['revenue'] - c['operating_cost']) / c['revenue'] * 100:.1f}%.",
                "",
            ]
        )
    return "\n".join(lines)


@tool
def get_audit_report(auditor: str) -> str:
    """Retrieve an independent audit report from one of three auditing firms.

    Each auditor assesses the same 10 companies but may report different
    figures due to methodological differences.

    Args:
        auditor: Auditor name — 'Auditor A', 'Auditor B', or 'Auditor C'.
    """
    return _generate_audit_report(auditor)


_EVAL4_QUERY = (
    "Get audit reports from all 3 auditors: Auditor A, Auditor B, and "
    "Auditor C. Each auditor reports on the same 10 companies but their "
    "figures differ. Apply this conflict resolution rule: where figures "
    "disagree between auditors, use the MEDIAN of the three values.\n"
    "Then answer:\n"
    "1. Which company has the highest MEDIAN revenue? Give the name and "
    "the median revenue figure.\n"
    "2. Which company has the lowest MEDIAN operating cost? Give the name "
    "and the median operating cost figure.\n"
    "Cite the specific auditor values that led to each median."
)


@pytest.mark.langsmith
def test_contradictions(model: BaseChatModel) -> None:
    """Eval 4: 3 auditors with conflicting data requiring median resolution."""
    agent = create_deep_agent(
        model=model,
        system_prompt=(
            "You are an audit reconciliation supervisor. Delegate to your "
            "auditor subagent to retrieve reports from all 3 auditing firms "
            "in parallel. After receiving all results, compare figures across "
            "auditors for each company and apply the user's conflict resolution "
            "rules precisely."
        ),
        subagents=[
            {
                "name": "auditor",
                "description": (
                    "Retrieves an independent audit report from a named auditing "
                    "firm using the get_audit_report tool. Returns revenue, "
                    "operating cost, and headcount for 10 companies."
                ),
                "system_prompt": (
                    "You are an audit retrieval agent. Use the get_audit_report "
                    "tool with the auditor name to retrieve their report. Report "
                    "the revenue and operating cost figures for each company."
                ),
                "tools": [get_audit_report],
            }
        ],
    )
    run_agent(
        agent,
        query=_EVAL4_QUERY,
        model=model,
        scorer=(
            TrajectoryScorer()
            .expect(
                agent_steps=2,
                tool_call_requests=3,
                tool_calls=[tool_call(name="task", args_contains={"subagent_type": "auditor"})],
            )
            .success(
                final_text_contains("Eos", case_insensitive=True),
                final_text_contains("Helix", case_insensitive=True),
            )
        ),
    )


# ===========================================================================
# Eval 5: High parallelism (25 concurrent subagents)
#
# 25 districts x 4 stores = 100 stores. Simple data but massive concurrency.
#
# Ground truth:
#   Highest sales: Summit Flagship ($4,820,000) in Hillcrest
#   Best satisfaction with return_rate < 5%: Heritage Boutique (9.8, 2.1%)
#     Trap: Luxe Corner has satisfaction 9.9 but return_rate 5.2%
#   Stores with foot_traffic > 10,000: 37 (see data below)
# ===========================================================================

# Fields: store_name, sales, foot_traffic, avg_transaction, return_rate, satisfaction
_DISTRICT_DATA: dict[str, list[tuple[str, int, int, float, float, float]]] = {
    "Northgate": [
        ("Pine Street Market", 2_450_000, 12_300, 48.5, 3.2, 8.7),
        ("Oak Avenue", 1_890_000, 9_800, 42.1, 4.8, 7.9),
        ("Elm Boulevard", 3_120_000, 15_200, 55.2, 2.9, 9.1),
        ("Cedar Junction", 1_560_000, 7_400, 38.9, 6.1, 7.2),
    ],
    "Southpark": [
        ("Main Street Shop", 2_780_000, 11_500, 51.3, 3.8, 8.4),
        ("Park Lane", 1_650_000, 8_900, 39.8, 5.5, 7.5),
        ("Garden Row", 2_340_000, 10_200, 46.7, 4.1, 8.1),
        ("Fountain Plaza", 1_920_000, 7_800, 41.2, 3.5, 8.3),
    ],
    "Eastridge": [
        ("Ridge Road Market", 3_450_000, 14_100, 58.4, 2.7, 9.0),
        ("Valley View", 1_780_000, 8_200, 37.6, 5.9, 7.1),
        ("Sunrise Corner", 2_100_000, 9_500, 44.3, 4.4, 8.0),
        ("Hilltop Bazaar", 2_670_000, 11_800, 52.1, 3.3, 8.6),
    ],
    "Westfield": [
        ("Field House", 1_950_000, 10_400, 43.5, 4.7, 7.8),
        ("Prairie Point", 2_890_000, 13_200, 54.6, 3.1, 8.9),
        ("Meadow Walk", 1_340_000, 6_800, 35.2, 6.8, 6.9),
        ("Horizon Market", 2_560_000, 9_100, 49.8, 4.3, 8.2),
    ],
    "Lakeside": [
        ("Lakefront Store", 3_210_000, 15_600, 56.9, 2.5, 9.2),
        ("Shore Road Market", 1_890_000, 8_400, 40.1, 5.2, 7.6),
        ("Marina Point", 2_670_000, 11_200, 50.4, 3.6, 8.5),
        ("Pier Walk Shop", 1_450_000, 7_100, 36.8, 6.3, 7.0),
    ],
    "Hillcrest": [
        ("Summit Flagship", 4_820_000, 18_900, 72.3, 2.1, 9.5),
        ("Crest View Market", 2_340_000, 10_800, 47.5, 4.0, 8.3),
        ("Ridgeway Store", 1_780_000, 7_600, 38.4, 5.4, 7.4),
        ("Peak Plaza", 3_100_000, 14_500, 57.8, 2.8, 9.0),
    ],
    "Riverside": [
        ("River Road Shop", 2_560_000, 11_400, 49.2, 3.7, 8.4),
        ("Bridge Street", 1_890_000, 8_600, 41.5, 4.9, 7.8),
        ("Waterfront Market", 2_910_000, 13_800, 55.6, 3.0, 8.8),
        ("Dock Side Bazaar", 1_670_000, 7_200, 37.3, 5.7, 7.3),
    ],
    "Oakmont": [
        ("Grand Oak Store", 2_780_000, 12_100, 52.8, 3.4, 8.6),
        ("Acorn Lane", 1_540_000, 7_900, 36.4, 5.8, 7.1),
        ("Timber Walk", 2_230_000, 10_600, 45.9, 4.2, 8.0),
        ("Canopy Market", 1_890_000, 9_200, 41.7, 4.6, 7.7),
    ],
    "Pinewood": [
        ("Forest Edge Store", 2_120_000, 10_300, 44.8, 4.5, 7.9),
        ("Pine Trail Market", 3_340_000, 14_800, 59.1, 2.6, 9.1),
        ("Woodland Plaza", 1_670_000, 8_100, 38.2, 5.3, 7.4),
        ("Grove Corner", 2_450_000, 11_700, 50.6, 3.9, 8.3),
    ],
    "Cedarville": [
        ("Cedar Falls Store", 1_980_000, 9_600, 42.3, 4.4, 8.0),
        ("Redwood Row", 2_670_000, 12_400, 53.5, 3.2, 8.7),
        ("Spruce Lane", 1_450_000, 7_300, 35.8, 6.0, 7.2),
        ("Juniper Market", 2_340_000, 10_900, 47.2, 3.8, 8.4),
    ],
    "Maplewood": [
        ("Maple Center", 2_890_000, 13_500, 54.7, 3.0, 8.8),
        ("Autumn Lane", 1_760_000, 8_800, 39.5, 5.1, 7.6),
        ("Sycamore Store", 2_230_000, 10_100, 46.3, 4.3, 8.1),
        ("Birch Walk", 1_560_000, 7_500, 37.1, 5.6, 7.3),
    ],
    "Birchdale": [
        ("Birchdale Central", 2_450_000, 11_600, 49.8, 3.5, 8.5),
        ("Silver Birch Store", 1_890_000, 9_300, 41.2, 4.7, 7.8),
        ("White Bark Market", 3_010_000, 14_200, 56.4, 2.8, 8.9),
        ("Aspen Corner", 1_670_000, 7_800, 38.6, 5.4, 7.4),
    ],
    "Ironbridge": [
        ("Iron Works Store", 2_340_000, 10_700, 47.9, 4.1, 8.2),
        ("Forge Lane", 1_780_000, 8_500, 39.8, 5.0, 7.7),
        ("Anvil Market", 2_670_000, 12_800, 53.2, 3.3, 8.6),
        ("Foundry Plaza", 1_450_000, 7_100, 36.5, 5.9, 7.1),
    ],
    "Stonewall": [
        ("Granite Store", 2_560_000, 11_900, 50.5, 3.6, 8.4),
        ("Quarry Lane", 1_980_000, 9_400, 43.1, 4.5, 7.9),
        ("Marble Market", 2_890_000, 13_600, 55.8, 2.9, 8.8),
        ("Slate Corner", 1_340_000, 6_900, 35.4, 6.2, 7.0),
    ],
    "Greenfield": [
        ("Green Meadow Store", 2_120_000, 10_500, 45.6, 4.2, 8.1),
        ("Pasture Lane", 1_670_000, 8_300, 38.7, 5.3, 7.5),
        ("Clover Market", 2_780_000, 13_100, 52.9, 3.1, 8.7),
        ("Thistle Plaza", 1_560_000, 7_600, 37.2, 5.7, 7.3),
    ],
    "Fairview": [
        ("Vista Store", 3_120_000, 14_900, 58.7, 2.5, 9.2),
        ("Outlook Lane", 1_890_000, 9_100, 41.8, 4.8, 7.8),
        ("Panorama Market", 2_450_000, 11_300, 48.4, 3.7, 8.4),
        ("Terrace Corner", 1_780_000, 8_000, 39.3, 5.2, 7.6),
    ],
    "Clearwater": [
        ("Crystal Store", 2_670_000, 12_200, 51.7, 3.4, 8.5),
        ("Springs Lane", 1_540_000, 7_700, 36.9, 5.6, 7.2),
        ("Brook Market", 2_340_000, 10_800, 47.1, 4.0, 8.2),
        ("Current Plaza", 1_920_000, 9_500, 42.6, 4.4, 8.0),
    ],
    "Sunridge": [
        ("Sunbeam Store", 2_890_000, 13_400, 54.3, 3.0, 8.8),
        ("Dawn Lane", 1_780_000, 8_600, 39.7, 5.1, 7.6),
        ("Solstice Market", 2_230_000, 10_400, 46.5, 4.3, 8.1),
        ("Radiance Corner", 1_670_000, 7_900, 38.1, 5.5, 7.4),
    ],
    "Windmill": [
        ("Windmill Central", 2_560_000, 11_800, 50.1, 3.6, 8.4),
        ("Gust Lane", 1_450_000, 7_200, 36.2, 6.0, 7.1),
        ("Breeze Market", 2_780_000, 13_000, 53.4, 3.2, 8.7),
        ("Turbine Plaza", 1_980_000, 9_600, 43.8, 4.5, 7.9),
    ],
    "Crossroads": [
        ("Junction Store", 2_340_000, 10_900, 48.2, 3.9, 8.3),
        ("Intersection Lane", 1_890_000, 9_200, 41.4, 4.6, 7.8),
        ("Roundabout Market", 2_670_000, 12_500, 52.6, 3.3, 8.6),
        ("Bypass Corner", 1_560_000, 7_400, 37.5, 5.8, 7.2),
    ],
    "Harborview": [
        ("Harbor Store", 2_910_000, 13_700, 55.9, 2.9, 8.9),
        ("Jetty Lane", 1_780_000, 8_400, 39.1, 5.0, 7.7),
        ("Wharf Market", 2_120_000, 10_200, 44.7, 4.4, 8.0),
        ("Anchor Plaza", 1_670_000, 8_100, 38.3, 5.3, 7.5),
    ],
    "Bayshore": [
        ("Luxe Corner", 3_560_000, 16_200, 65.4, 5.2, 9.9),
        ("Tide Lane", 1_890_000, 9_000, 40.8, 4.7, 7.8),
        ("Sandbar Market", 2_450_000, 11_100, 49.5, 3.8, 8.4),
        ("Coral Plaza", 1_540_000, 7_600, 36.7, 5.6, 7.3),
    ],
    "Cliffside": [
        ("Cliff Edge Store", 2_670_000, 12_600, 52.3, 3.3, 8.6),
        ("Bluff Lane", 1_980_000, 9_700, 43.4, 4.5, 7.9),
        ("Ledge Market", 2_340_000, 10_500, 47.8, 4.1, 8.2),
        ("Overlook Corner", 1_450_000, 7_300, 36.1, 5.9, 7.1),
    ],
    "Meadowbrook": [
        ("Heritage Boutique", 2_230_000, 9_800, 46.2, 2.1, 9.8),
        ("Brook Lane", 1_780_000, 8_700, 39.4, 5.1, 7.6),
        ("Wildflower Market", 2_560_000, 11_500, 50.8, 3.6, 8.4),
        ("Fern Plaza", 1_670_000, 7_900, 38.0, 5.4, 7.4),
    ],
    "Thornhill": [
        ("Thorn Gate Store", 2_890_000, 13_300, 54.1, 3.1, 8.7),
        ("Bramble Lane", 1_560_000, 7_500, 37.4, 5.7, 7.3),
        ("Thistle Market", 2_120_000, 10_100, 45.1, 4.2, 8.1),
        ("Hawthorn Plaza", 1_890_000, 9_400, 41.9, 4.6, 7.8),
    ],
}

_STORE_FIELDS = ("name", "sales", "foot_traffic", "avg_transaction", "return_rate", "satisfaction")
_STORE_DATABASE: dict[str, list[dict[str, Any]]] = {
    district: [dict(zip(_STORE_FIELDS, s, strict=True)) for s in stores] for district, stores in _DISTRICT_DATA.items()
}

# Pre-compute ground truth for foot_traffic > 10,000
_HIGH_TRAFFIC_COUNT = sum(1 for stores in _STORE_DATABASE.values() for s in stores if s["foot_traffic"] > 10_000)


def _generate_district_report(district: str) -> str:
    stores = _STORE_DATABASE.get(district)
    if stores is None:
        return f"No data found for district: {district}"
    lines = [
        f"DISTRICT PERFORMANCE REPORT — {district}",
        "",
        f"Quarterly performance review for {len(stores)} retail locations in the "
        f"{district} district. Figures reflect the most recent reporting period. "
        f"Sales figures are annualized from trailing 12-month actuals. Foot traffic "
        f"is measured via in-store sensors (monthly average). Return rates and "
        f"satisfaction scores are computed from customer surveys and POS data.",
        "",
    ]
    for s in stores:
        monthly_sales = round(s["sales"] / 12)
        prior_sales = round(s["sales"] * 0.93)
        lines.extend(
            [
                f"{s['name']}: Annual sales of ${s['sales']:,} (approximately ${monthly_sales:,} "
                f"per month; prior year was ${prior_sales:,}). Monthly foot traffic: "
                f"{s['foot_traffic']:,} visitors. Average transaction value: "
                f"${s['avg_transaction']:.2f}. Return rate: {s['return_rate']}%. "
                f"Customer satisfaction score: {s['satisfaction']} out of 10.",
                "",
            ]
        )
    return "\n".join(lines)


@tool
def analyze_district(district: str) -> str:
    """Retrieve a performance report for a retail district covering 4 stores.

    Returns sales, foot traffic, transaction values, return rates, and
    satisfaction scores.

    Args:
        district: District name (e.g. 'Northgate', 'Hillcrest').
    """
    return _generate_district_report(district)


_EVAL5_DISTRICTS = list(_DISTRICT_DATA.keys())

_EVAL5_QUERY = (
    f"Analyze all 25 districts: {', '.join(_EVAL5_DISTRICTS)}. "
    "Then answer these questions:\n"
    "1. Across all 100 stores, which single store has the highest annual "
    "sales? Give the store name and exact sales figure.\n"
    "2. Among stores with a return_rate strictly below 5%, which has the "
    "highest customer satisfaction score? Give the store name, satisfaction "
    "score, and return rate.\n"
    f"3. How many stores have monthly foot traffic above 10,000?"
)


@pytest.mark.langsmith
def test_high_parallelism(model: BaseChatModel) -> None:
    """Eval 5: 100 stores across 25 districts testing massive concurrency."""
    agent = create_deep_agent(
        model=model,
        system_prompt=(
            "You are a retail analytics supervisor. Delegate each district "
            "analysis to your district_analyst subagent in parallel (one task "
            "call per district — 25 total). After receiving all results, "
            "aggregate the data across all 100 stores to answer the user's "
            "questions precisely."
        ),
        subagents=[
            {
                "name": "district_analyst",
                "description": (
                    "Analyzes a retail district using the analyze_district tool. "
                    "Returns performance data for 4 stores including sales, foot "
                    "traffic, return rates, and satisfaction scores."
                ),
                "system_prompt": (
                    "You are a retail analyst. Use the analyze_district tool with "
                    "the district name to retrieve the performance report. Report "
                    "all metrics for each store."
                ),
                "tools": [analyze_district],
            }
        ],
    )
    run_agent(
        agent,
        query=_EVAL5_QUERY,
        model=model,
        scorer=(
            TrajectoryScorer()
            .expect(
                agent_steps=2,
                tool_call_requests=25,
                tool_calls=[tool_call(name="task", args_contains={"subagent_type": "district_analyst"})],
            )
            .success(
                final_text_contains("Summit Flagship", case_insensitive=True),
                final_text_contains("Heritage Boutique", case_insensitive=True),
                final_text_contains(str(_HIGH_TRAFFIC_COUNT)),
            )
        ),
    )


# ===========================================================================
# Eval 6: Temporal ordering sensitivity
#
# 8 regions x 3 products x 4 quarters. Reports interleave historical
# context, prior-year comparisons, and "preliminary" vs "adjusted" labels.
#
# Ground truth:
#   Largest Q3->Q4 improvement: GammaChip in Pacific (+5,400 units)
#     Runner-up: AlphaWidget in Southwest (+5,200 units) — only 200-unit gap
#
#   Most consistent growth (lowest variance in QoQ changes):
#     Midwest: changes are ~480-530 per quarter, variance ~247
#     Runner-up: Northwest ~480-550 per quarter, variance ~432
# ===========================================================================

# (Q1, Q2, Q3, Q4) units sold
_QUARTERLY_DATA: dict[str, dict[str, tuple[int, int, int, int]]] = {
    "Northeast": {
        "AlphaWidget": (12_000, 13_500, 14_200, 15_000),
        "BetaSensor": (8_500, 9_200, 10_100, 10_800),
        "GammaChip": (5_200, 5_800, 6_500, 7_100),
    },
    "Southeast": {
        "AlphaWidget": (15_000, 14_200, 13_800, 16_500),
        "BetaSensor": (11_000, 12_500, 11_800, 13_200),
        "GammaChip": (7_800, 8_200, 7_900, 9_100),
    },
    "Midwest": {
        "AlphaWidget": (10_000, 10_520, 11_010, 11_540),
        "BetaSensor": (7_000, 7_490, 7_990, 8_510),
        "GammaChip": (4_500, 4_980, 5_490, 5_990),
    },
    "Southwest": {
        "AlphaWidget": (9_000, 11_500, 7_200, 12_400),
        "BetaSensor": (6_500, 8_200, 7_100, 9_500),
        "GammaChip": (3_800, 5_100, 4_200, 6_800),
    },
    "Northwest": {
        "AlphaWidget": (11_000, 11_530, 12_080, 12_600),
        "BetaSensor": (7_500, 8_010, 8_530, 9_040),
        "GammaChip": (4_000, 4_480, 4_970, 5_460),
    },
    "Pacific": {
        "AlphaWidget": (14_000, 14_500, 14_800, 15_200),
        "BetaSensor": (9_500, 10_200, 10_800, 11_500),
        "GammaChip": (6_000, 6_800, 9_800, 15_200),
    },
    "Mountain": {
        "AlphaWidget": (8_500, 9_200, 9_800, 10_500),
        "BetaSensor": (6_000, 6_800, 7_300, 7_900),
        "GammaChip": (3_500, 4_100, 4_800, 5_200),
    },
    "Atlantic": {
        "AlphaWidget": (13_000, 13_800, 14_500, 15_100),
        "BetaSensor": (9_000, 9_800, 10_500, 11_200),
        "GammaChip": (5_500, 6_200, 6_900, 7_400),
    },
}

_QUARTER_LABELS = ("Q1", "Q2", "Q3", "Q4")


def _generate_quarterly_report(region: str) -> str:
    products = _QUARTERLY_DATA.get(region)
    if products is None:
        return f"No quarterly data found for region: {region}"
    parts: list[str] = []
    parts.append(
        f"The {region} region's quarterly performance report presents a nuanced picture "
        f"across three product lines. Analysts should note that preliminary figures "
        f"released mid-quarter were subsequently revised in the adjusted totals below; "
        f"only the adjusted figures should be used for year-end comparisons. Prior-year "
        f"figures are included for context but reflect the old reporting methodology "
        f"and are not directly comparable without a correction factor of approximately "
        f"1.05x. Seasonal patterns vary significantly by product line — AlphaWidget "
        f"typically sees stronger Q4 performance due to holiday demand, while BetaSensor "
        f"demand is more industrial and less seasonal. GammaChip sales can be volatile "
        f"due to supply chain dynamics."
    )
    for product, quarters in products.items():
        prior_year = tuple(round(q * 0.88) for q in quarters)
        preliminary = tuple(round(q * 0.96) for q in quarters)
        parts.append(
            f"For {product} in {region}, the preliminary figures initially released "
            f"were {preliminary[0]:,}, {preliminary[1]:,}, {preliminary[2]:,}, and "
            f"{preliminary[3]:,} units for Q1 through Q4 respectively, but these were "
            f"revised upward in the final adjusted totals to {quarters[0]:,} (Q1), "
            f"{quarters[1]:,} (Q2), {quarters[2]:,} (Q3), and {quarters[3]:,} (Q4). "
            f"For reference, the prior year's figures under the old methodology were "
            f"approximately {prior_year[0]:,}, {prior_year[1]:,}, {prior_year[2]:,}, "
            f"and {prior_year[3]:,} for Q1-Q4. The Q2-to-Q3 change was "
            f"{quarters[2] - quarters[1]:+,} units while the Q3-to-Q4 change was "
            f"{quarters[3] - quarters[2]:+,} units. Year-over-year growth for Q4 "
            f"specifically was approximately {(quarters[3] - prior_year[3]) / prior_year[3] * 100:.1f}% "
            f"but this comparison should be interpreted cautiously given the "
            f"methodology change."
        )
    parts.append(f"This concludes the {region} quarterly report. All adjusted figures are final for reporting purposes.")
    return " ".join(parts)


@tool
def get_quarterly_report(region: str) -> str:
    """Retrieve a quarterly performance report for a region covering 3 product lines.

    Returns Q1-Q4 units sold for AlphaWidget, BetaSensor, and GammaChip.
    Use the ADJUSTED figures (not preliminary, not prior-year).

    Args:
        region: Region name (e.g. 'Northeast', 'Pacific', 'Midwest').
    """
    return _generate_quarterly_report(region)


_EVAL6_REGIONS = list(_QUARTERLY_DATA.keys())

_EVAL6_QUERY = (
    f"Analyze all 8 regions: {', '.join(_EVAL6_REGIONS)}. "
    "Use the ADJUSTED quarterly figures (not preliminary, not prior-year). "
    "Then answer these questions:\n"
    "1. Across all regions and products, which specific product-in-region "
    "combination had the largest Q3-to-Q4 improvement in units sold? "
    "Give the product name, region, and the exact increase.\n"
    "2. Which region had the most consistent quarter-over-quarter growth? "
    "Define consistency as having the lowest variance in QoQ unit changes "
    "across all 3 products and all quarter transitions (Q1->Q2, Q2->Q3, "
    "Q3->Q4). Give the region name."
)


@pytest.mark.langsmith
def test_temporal_ordering(model: BaseChatModel) -> None:
    """Eval 6: 8 regions x 3 products x 4 quarters with noisy time-series prose."""
    agent = create_deep_agent(
        model=model,
        system_prompt=(
            "You are a product analytics supervisor. Delegate each region's "
            "quarterly report to your quarterly_analyst subagent in parallel "
            "(one task call per region — 8 total). After receiving all results, "
            "use only the ADJUSTED figures to answer the user's questions. "
            "Compute exact QoQ changes (Q2-Q1, Q3-Q2, Q4-Q3) for each "
            "product in each region."
        ),
        subagents=[
            {
                "name": "quarterly_analyst",
                "description": (
                    "Retrieves a quarterly performance report for a region using "
                    "the get_quarterly_report tool. Returns Q1-Q4 units sold for "
                    "3 product lines."
                ),
                "system_prompt": (
                    "You are a quarterly analyst. Use the get_quarterly_report "
                    "tool with the region name. Extract the ADJUSTED Q1-Q4 "
                    "figures for each product. Do NOT use preliminary or "
                    "prior-year figures."
                ),
                "tools": [get_quarterly_report],
            }
        ],
    )
    run_agent(
        agent,
        query=_EVAL6_QUERY,
        model=model,
        scorer=(
            TrajectoryScorer()
            .expect(
                agent_steps=2,
                tool_call_requests=8,
                tool_calls=[tool_call(name="task", args_contains={"subagent_type": "quarterly_analyst"})],
            )
            .success(
                final_text_contains("GammaChip", case_insensitive=True),
                final_text_contains("Pacific", case_insensitive=True),
                final_text_contains("Midwest", case_insensitive=True),
            )
        ),
    )
