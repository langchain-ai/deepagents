"""Radar chart generation for eval results.

Produces per-model radar (spider) charts where each axis represents an
eval category (e.g. file_operations, memory, hitl) and the radial position
encodes the score (0-1 correctness).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, cast

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

EVAL_CATEGORIES: list[str] = _categories_raw["categories"]
"""Canonical eval category names.

Order determines axis placement on the radar chart (clockwise from top).
"""

CATEGORY_LABELS: dict[str, str] = _categories_raw["labels"]
"""Human-friendly display labels for radar chart axes, keyed by category name."""

del _categories_raw

_COLORS: list[str] = [
    "#2563eb",  # blue
    "#dc2626",  # red
    "#16a34a",  # green
    "#9333ea",  # purple
    "#ea580c",  # orange
    "#0891b2",  # cyan
    "#be185d",  # pink
    "#854d0e",  # brown
]
"""Eight visually distinct hex colors, cycled across models on the radar chart."""


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
    _color_offset: int = 0,
) -> Figure:
    """Generate a radar chart comparing models across eval categories.

    Args:
        results: One `ModelResult` per model to plot.
        categories: Category axes to include. Defaults to `EVAL_CATEGORIES`.
        title: Chart title.
        output: If provided, save the figure to this path (PNG/SVG/PDF).
        figsize: Figure size in inches.

    Returns:
        The matplotlib `Figure` object.
    """
    import matplotlib.pyplot as plt

    cats = categories or EVAL_CATEGORIES
    n = len(cats)
    labels = [CATEGORY_LABELS.get(c, c) for c in cats]

    # Compute angle for each axis (evenly spaced, starting from top).
    angles = [i * 2 * math.pi / n for i in range(n)]
    angles.append(angles[0])  # close the polygon

    fig, ax_raw = plt.subplots(figsize=figsize, subplot_kw={"polar": True})
    ax = cast("PolarAxes", ax_raw)

    # Start from top (90 degrees) and go clockwise.
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)

    # Draw grid and axis labels.
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11, fontweight="bold")

    # Radial ticks at 0.2 intervals.
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8, color="grey")
    ax.set_ylim(0, 1.05)

    # Plot each model as a filled polygon.
    for idx, result in enumerate(results):
        color = _COLORS[(idx + _color_offset) % len(_COLORS)]
        values = [result.scores.get(c, 0.0) for c in cats]
        values.append(values[0])  # close polygon

        ax.plot(
            angles,
            values,
            "o-",
            linewidth=2,
            color=color,
            label=_short_model_name(result.model),
            markersize=5,
        )
        ax.fill(angles, values, alpha=0.1, color=color)

        # Annotate each point with its score.
        for angle, val in zip(angles[:-1], values[:-1], strict=True):
            ax.annotate(
                f"{val:.0%}",
                xy=(angle, val),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=7,
                color=color,
            )

    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    fig.tight_layout()

    if output is not None:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


def generate_individual_radars(
    results: list[ModelResult],
    *,
    categories: list[str] | None = None,
    output_dir: str | Path = "charts/individual",
    title_prefix: str = "Eval Results",
    figsize: tuple[float, float] = (10, 10),
) -> list[Path]:
    """Generate one radar chart per model.

    Each chart is saved as `<sanitized_model_name>.png` inside `output_dir`.

    Args:
        results: One `ModelResult` per model.
        categories: Category axes to include. Defaults to `EVAL_CATEGORIES`.
        output_dir: Directory to write per-model PNGs.
        title_prefix: Prefix for each chart title (model name is appended).
        figsize: Figure size in inches.

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
                "skills": 0.88,
                "hitl": 0.95,
                "memory": 0.83,
                "summarization": 0.90,
                "subagents": 0.78,
                "system_prompt": 0.97,
                "tool_usage": 0.85,
                "followup_quality": 0.91,
                "external_benchmarks": 0.76,
                "tau2_airline": 0.70,
                "memory_agent_bench": 0.82,
            },
        ),
        ModelResult(
            model="openai:gpt-4.1",
            scores={
                "file_operations": 0.88,
                "skills": 0.82,
                "hitl": 0.80,
                "memory": 0.79,
                "summarization": 0.85,
                "subagents": 0.75,
                "system_prompt": 0.90,
                "tool_usage": 0.88,
                "followup_quality": 0.85,
                "external_benchmarks": 0.72,
                "tau2_airline": 0.65,
                "memory_agent_bench": 0.78,
            },
        ),
        ModelResult(
            model="google_genai:gemini-2.5-pro",
            scores={
                "file_operations": 0.85,
                "skills": 0.78,
                "hitl": 0.72,
                "memory": 0.80,
                "summarization": 0.88,
                "subagents": 0.70,
                "system_prompt": 0.85,
                "tool_usage": 0.82,
                "followup_quality": 0.80,
                "external_benchmarks": 0.68,
                "tau2_airline": 0.60,
                "memory_agent_bench": 0.75,
            },
        ),
        ModelResult(
            model="anthropic:claude-opus-4-6",
            scores={
                "file_operations": 0.95,
                "skills": 0.92,
                "hitl": 0.93,
                "memory": 0.90,
                "summarization": 0.94,
                "subagents": 0.85,
                "system_prompt": 0.98,
                "tool_usage": 0.91,
                "followup_quality": 0.94,
                "external_benchmarks": 0.81,
                "tau2_airline": 0.75,
                "memory_agent_bench": 0.88,
            },
        ),
    ]
