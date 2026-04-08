"""Radar chart generation for eval results.

Produces per-model radar (spider) charts where each axis represents an
eval category (e.g. file_operations, memory, tool_use) and the radial position
encodes the score (0-1 correctness).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, cast

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from matplotlib.projections.polar import PolarAxes

_CATEGORIES_JSON = Path(__file__).parent / "categories.json"
try:
    _categories_raw = json.loads(_CATEGORIES_JSON.read_text(encoding="utf-8"))
except FileNotFoundError:
    msg = (
        f"categories.json not found at {_CATEGORIES_JSON}. "
        "Ensure the deepagents_evals package is installed correctly "
        "(the file should be included via [tool.setuptools.package-data])."
    )
    raise FileNotFoundError(msg) from None
except (json.JSONDecodeError, KeyError) as exc:
    msg = f"Failed to parse {_CATEGORIES_JSON}: {exc}"
    raise ValueError(msg) from exc

ALL_CATEGORIES: list[str] = _categories_raw["categories"]
"""All eval category names, including unit tests that don't appear on radar charts."""

EVAL_CATEGORIES: list[str] = _categories_raw.get("radar_categories", _categories_raw["categories"])
"""Radar-eligible eval category names.

Order determines axis placement on the radar chart (clockwise from top).
Categories like ``unit_test`` that verify SDK plumbing rather than model
capability are excluded from this list.
"""

CATEGORY_LABELS: dict[str, str] = _categories_raw["labels"]
"""Human-friendly display labels for radar chart axes, keyed by category name."""

del _categories_raw


@dataclass(frozen=True)
class _Theme:
    """Color scheme for a radar chart."""

    bg: str
    grid: str
    label: str
    tick: str
    watermark_alpha: float
    fill_alpha: float
    pill_alpha: float
    colors: tuple[str, ...]


_LIGHT = _Theme(
    bg="#f8f9fa",
    grid="#d5d8dc",
    label="#2c3e50",
    tick="#7f8c8d",
    watermark_alpha=0.6,
    fill_alpha=0.08,
    pill_alpha=0.85,
    colors=(
        "#1b4f72",  # navy
        "#b03a2e",  # burgundy
        "#1e8449",  # forest
        "#6c3483",  # plum
        "#ca6f1e",  # amber
        "#148f77",  # teal
        "#a04000",  # rust
        "#2e4053",  # slate
    ),
)

_DARK = _Theme(
    bg="#0d1117",
    grid="#30363d",
    label="#c9d1d9",
    tick="#8b949e",
    watermark_alpha=0.45,
    fill_alpha=0.12,
    pill_alpha=0.92,
    colors=(
        "#58a6ff",  # blue
        "#f97583",  # coral
        "#56d364",  # green
        "#d2a8ff",  # lavender
        "#f0883e",  # orange
        "#39d2c0",  # teal
        "#ff7eb6",  # pink
        "#e3b341",  # gold
    ),
)

_THEMES: dict[str, _Theme] = {"light": _LIGHT, "dark": _DARK}

THEMES: list[str] = list(_THEMES.keys())
"""Supported chart themes, derived from the internal theme registry."""


@dataclass(frozen=True)
class ModelResult:
    """Eval scores for a single model across categories.

    Attributes:
        model: Model identifier (e.g. `anthropic:claude-sonnet-4-6`).
        scores: Mapping of category name to correctness score in `[0, 1]`.
    """

    model: str
    scores: dict[str, float] = field(default_factory=dict)


def generate_radar(
    results: list[ModelResult],
    *,
    categories: list[str] | None = None,
    title: str = "Eval Results",
    output: str | Path | None = None,
    figsize: tuple[float, float] = (10, 10),
    theme: str = "light",
    _color_offset: int = 0,
) -> Figure:
    """Generate a radar chart comparing models across eval categories.

    Args:
        results: One `ModelResult` per model to plot.
        categories: Category axes to include. Defaults to `EVAL_CATEGORIES`.
        title: Chart title.
        output: If provided, save the figure to this path (PNG/SVG/PDF).
        figsize: Figure size in inches.
        theme: Color scheme — `"light"` or `"dark"`.

            Unrecognized values fall back to `"light"`.

    Returns:
        The matplotlib `Figure` object.
    """
    if plt is None:
        msg = "matplotlib is required for chart generation. Install: pip install deepagents-evals[charts]"
        raise ImportError(msg)

    try:
        t = _THEMES[theme]
    except KeyError:
        msg = f"Unknown theme {theme!r}; expected one of {sorted(_THEMES)}"
        raise ValueError(msg) from None

    cats = categories or EVAL_CATEGORIES
    n = len(cats)
    labels = [CATEGORY_LABELS.get(c, c) for c in cats]

    # Compute angle for each axis (evenly spaced, starting from top).
    angles = [i * 2 * math.pi / n for i in range(n)]
    angles.append(angles[0])  # close the polygon

    fig, ax_raw = plt.subplots(figsize=figsize, subplot_kw={"polar": True})
    ax = cast("PolarAxes", ax_raw)
    fig.patch.set_facecolor(t.bg)
    ax.set_facecolor(t.bg)

    # Start from top (90 degrees) and go clockwise.
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)

    # --- Grid & spine styling ---
    ax.grid(color=t.grid, linewidth=0.6, linestyle="-", alpha=0.7)
    ax.spines["polar"].set_color(t.grid)
    ax.spines["polar"].set_linewidth(0.8)

    # Axis labels — placed manually so they always render above the data.
    # Default tick labels sit inside the axes z-order stack and get covered
    # by polar fills regardless of zorder settings.
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([])  # hide default labels

    label_radius = 1.22
    # The top label (angle ≈ 0) sits directly below the title — pull it in
    # slightly so the two don't collide.
    top_radius = 1.12
    for angle, text in zip(angles[:-1], labels, strict=True):
        ha = "center"
        if 0 < angle < math.pi:
            ha = "left"
        elif math.pi < angle < 2 * math.pi:
            ha = "right"

        is_top = abs(angle) < 0.01 or abs(angle - 2 * math.pi) < 0.01  # noqa: PLR2004
        r = top_radius if is_top else label_radius

        ax.text(
            angle,
            r,
            text,
            ha=ha,
            va="center",
            fontsize=11,
            fontweight="semibold",
            color=t.label,
            zorder=15,
            bbox={
                "facecolor": t.bg,
                "edgecolor": "none",
                "pad": 3,
                "alpha": 0.95,
            },
        )

    # Radial ticks at 0.2 intervals.
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(
        ["20%", "40%", "60%", "80%", "100%"],
        fontsize=7,
        color=t.tick,
    )
    ax.set_ylim(0, 1.05)

    # Plot each model as a filled polygon.
    for idx, result in enumerate(results):
        color = t.colors[(idx + _color_offset) % len(t.colors)]
        values = [result.scores.get(c, 0.0) for c in cats]
        values.append(values[0])  # close polygon

        # Filled area.
        ax.fill(angles, values, alpha=t.fill_alpha, color=color, zorder=2)

        # Primary line.
        ax.plot(
            angles,
            values,
            "-",
            linewidth=2.2,
            color=color,
            label=_short_model_name(result.model),
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=3,
        )

        # Data points.
        ax.plot(
            angles,
            values,
            "o",
            color=color,
            markersize=5,
            markeredgecolor=t.bg,
            markeredgewidth=1.2,
            zorder=4,
        )

        # Score annotations with background pill.  When multiple models are
        # plotted, stagger radially outward along each spoke so pills don't
        # pile up at the same point.
        multi = len(results) > 1
        radial_bump = 0.05 + idx * 0.06 if multi else 0.04

        for angle, val in zip(angles[:-1], values[:-1], strict=True):
            ax.text(
                angle,
                val + radial_bump,
                f"{val:.0%}",
                ha="center",
                va="center",
                fontsize=6.5 if multi else 7,
                fontweight="medium",
                color=color,
                zorder=6,
                bbox={
                    "boxstyle": "round,pad=0.2",
                    "facecolor": t.bg,
                    "edgecolor": color,
                    "linewidth": 0.5,
                    "alpha": t.pill_alpha,
                },
            )

    # Legend — anchored to the figure's upper-right to avoid wasting space.
    legend = fig.legend(
        loc="upper right",
        fontsize=9,
        frameon=True,
        fancybox=True,
        shadow=False,
        framealpha=0.95,
        edgecolor=t.grid,
        labelcolor=t.label,
    )
    legend.get_frame().set_linewidth(0.8)
    legend.get_frame().set_facecolor(t.bg)

    # Title.
    ax.set_title(
        title,
        fontsize=15,
        fontweight="bold",
        color=t.label,
        pad=60,
    )
    # Watermark.
    fig.text(
        0.98,
        0.02,
        "langchain-ai/deep-agents",
        ha="right",
        va="bottom",
        fontsize=7,
        color=t.tick,
        alpha=t.watermark_alpha,
        style="italic",
    )
    fig.tight_layout(pad=2.0)

    if output is not None:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=t.bg)
        plt.close(fig)

    return fig


def generate_individual_radars(
    results: list[ModelResult],
    *,
    categories: list[str] | None = None,
    output_dir: str | Path = "charts/individual",
    title_prefix: str = "Eval Results",
    figsize: tuple[float, float] = (10, 10),
    theme: str = "light",
) -> list[Path]:
    """Generate one radar chart per model.

    Each chart is saved as `<sanitized_model_name>.png` inside `output_dir`.

    Args:
        results: One `ModelResult` per model.
        categories: Category axes to include. Defaults to `EVAL_CATEGORIES`.
        output_dir: Directory to write per-model PNGs.
        title_prefix: Prefix for each chart title (model name is appended).
        figsize: Figure size in inches.
        theme: Color scheme — `"light"` or `"dark"`.

            Unrecognized values fall back to `"light"`.

    Returns:
        List of paths to the saved PNG files.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    for idx, result in enumerate(results):
        name = _short_model_name(result.model)
        safe = _safe_filename(result.model)
        dest = out / f"{safe}.png"
        generate_radar(
            [result],
            categories=categories,
            title=f"{title_prefix} — {name}",
            output=dest,
            figsize=figsize,
            theme=theme,
            _color_offset=idx,
        )
        paths.append(dest)
    return paths


def _safe_filename(model: str) -> str:
    """Convert a model identifier into a filesystem-safe filename stem.

    Replaces colons, slashes, and spaces with hyphens, then strips leading/
    trailing hyphens.

    Args:
        model: Full model identifier.

    Returns:
        Sanitized string safe for use as a filename (without extension).
    """
    safe = model.replace(":", "-").replace("/", "-").replace(" ", "-")
    return safe.strip("-") or "unknown"


def _short_model_name(model: str) -> str:
    """Shorten `provider:model-name-version` to a readable label.

    Strips the `provider:` prefix if present and truncates to 30 characters.

    Args:
        model: Full model identifier.

    Returns:
        Shortened display name.
    """
    max_len = 30
    # Strip provider prefix if present.
    if ":" in model:
        model = model.split(":", 1)[1]
    # Truncate long names.
    if len(model) > max_len:
        model = model[: max_len - 3] + "..."
    return model


def load_results_from_summary(path: str | Path) -> list[ModelResult]:
    """Load model results from an `evals_summary.json` file.

    The summary file is a JSON array of objects. Each object must have a
    `category_scores` dict mapping category names to `[0, 1]` correctness
    floats. The `model` key defaults to `"unknown"` if absent.

    Args:
        path: Path to `evals_summary.json`.

    Returns:
        List of `ModelResult` objects.

    Raises:
        FileNotFoundError: If `path` does not exist.
        json.JSONDecodeError: If the file contains invalid JSON.
        ValueError: If a score value in `category_scores` is not numeric.
        KeyError: If an entry is missing `category_scores`.
    """
    import json

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    results: list[ModelResult] = []
    for entry in data:
        model = str(entry.get("model", "unknown"))
        scores = {k: float(v) for k, v in entry["category_scores"].items()}
        results.append(ModelResult(model=model, scores=scores))
    return results


def toy_data() -> list[ModelResult]:
    """Generate toy eval data for experimentation.

    Returns:
        List of `ModelResult` with plausible scores across all categories.
    """
    return [
        ModelResult(
            model="anthropic:claude-sonnet-4-6",
            scores={
                "file_operations": 0.92,
                "retrieval": 0.76,
                "tool_use": 0.85,
                "memory": 0.83,
                "conversation": 0.80,
                "summarization": 0.90,
            },
        ),
        ModelResult(
            model="openai:gpt-5.4",
            scores={
                "file_operations": 0.88,
                "retrieval": 0.72,
                "tool_use": 0.86,
                "memory": 0.79,
                "conversation": 0.75,
                "summarization": 0.85,
            },
        ),
        ModelResult(
            model="google_genai:gemini-2.5-pro",
            scores={
                "file_operations": 0.85,
                "retrieval": 0.68,
                "tool_use": 0.80,
                "memory": 0.80,
                "conversation": 0.70,
                "summarization": 0.88,
            },
        ),
        ModelResult(
            model="anthropic:claude-opus-4-6",
            scores={
                "file_operations": 0.95,
                "retrieval": 0.81,
                "tool_use": 0.90,
                "memory": 0.90,
                "conversation": 0.85,
                "summarization": 0.94,
            },
        ),
    ]
