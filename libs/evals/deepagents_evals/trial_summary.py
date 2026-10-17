"""Helpers for rendering trial summary tables in the GHA step summary."""

from __future__ import annotations


def _esc(value: object) -> str:
    """Escape characters that would break a markdown table row.

    Pipes terminate cells, backslashes need to be doubled before pipe
    escapes survive a second pass, and newlines split a row in two —
    none of which the renderer notices until the table is already broken.
    """
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("\n", " ")
        .replace("\r", " ")
        .strip()
    )


def _fmt(value: float | None, places: int) -> str:
    return "-" if value is None else f"{value:.{places}f}"


def render_per_trial_category_matrix(
    trials: list[dict],
    cat_keys: list[str],
    labels: dict[str, str] | None = None,
    *,
    places: int = 3,
) -> list[str]:
    """Build the per-trial-by-per-category correctness table as markdown lines.

    Each row is one trial; each column is one category. Cells are
    correctness scores formatted to `places` decimals; categories that
    did not run in a given trial render as `-` so a missing column is
    visually distinct from a 0.0 score.

    Args:
        trials: Per-trial summary dicts (each must carry `trial_index`
            and optionally `category_scores`).
        cat_keys: Category keys to render as columns, in display order.
        labels: Optional human-friendly labels keyed by category. Falls
            back to the raw key when a label is missing.
        places: Decimal places for score cells.

    Returns:
        Markdown lines (blank line, heading, blank line, header row,
        separator, data rows). An empty list when `trials` or `cat_keys`
        is empty so callers can unconditionally extend a buffer.
    """
    if not trials or not cat_keys:
        return []

    labels = labels or {}
    header = "| # | " + " | ".join(_esc(labels.get(c, c)) for c in cat_keys) + " |"
    sep = "|---:|" + "|".join("---:" for _ in cat_keys) + "|"
    lines = ["", "### Per-trial correctness by category", "", header, sep]
    for trial in trials:
        scores = trial.get("category_scores") or {}
        cells = [_fmt(scores.get(c), places) for c in cat_keys]
        lines.append(f"| {trial.get('trial_index')} | " + " | ".join(cells) + " |")
    return lines
