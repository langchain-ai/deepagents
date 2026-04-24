"""Single-site judge CLI for the Vibe Coding Olympics.

Captures a website, runs the LLM-as-judge evaluators, scores accessibility
from an axe-core audit, prints a per-axis table, writes JSON, and
optionally logs to LangSmith. Callers that want to evaluate multiple sites
should call `evaluate_site()` concurrently with `asyncio.gather`.

Aggregation into a single overall score is intentionally deferred to the
caller — see `aggregate.py` for the helper.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from aggregate import score_accessibility
from capture import AccessibilityReport, Capture, capture_site
from evaluators import EVALUATORS_BY_AXIS
from langsmith_log import log_to_langsmith


@dataclass
class SiteScore:
    """Per-site evaluation result."""

    site_name: str
    url: str
    prompt: str
    axes: dict[str, float | None]
    metadata: dict[str, Any] = field(default_factory=dict)
    round_num: int | None = None


_AXIS_DISPLAY: dict[str, str] = {
    "color": "Color",
    "typography": "Typography",
    "layout": "Layout",
    "content_completeness": "Content",
    "creativity": "Creativity",
    "interpretation_quality": "Interpretation",
    "accessibility": "Accessibility",
}


def _run_llm_evaluators(
    prompt: str, html: str, screenshot_b64: str
) -> dict[str, float | None]:
    """Run every LLM evaluator against one captured site.

    Each evaluator is isolated — a single failure never poisons the others.

    Args:
        prompt: The original prompt given to the contestant.
        html: Rendered HTML source of the page.
        screenshot_b64: Base64-encoded PNG screenshot.

    Returns:
        Mapping of axis name to score in [0.0, 1.0], or `None` on failure.
    """
    scores: dict[str, float | None] = {}
    for axis, evaluator in EVALUATORS_BY_AXIS.items():
        try:
            result = evaluator(
                inputs=prompt,
                outputs=html,
                screenshot_b64=screenshot_b64,
            )
            scores[axis] = float(result["score"])
        except Exception as exc:
            msg = f"Evaluator '{axis}' failed: {exc}"
            print(f"  WARNING: {msg}", file=sys.stderr)
            scores[axis] = None
    return scores


def _build_metadata(report: AccessibilityReport | None) -> dict[str, Any]:
    """Extract structured metadata from the capture for logging/debugging.

    Args:
        report: The axe-core report, if one ran.

    Returns:
        Metadata dict suitable for embedding in the JSON output.
    """
    if report is None:
        return {"accessibility_audit": "skipped_or_failed"}
    return {
        "accessibility_audit": "ok",
        "accessibility_violations": report.violation_count,
        "accessibility_serious_violations": report.serious_violation_count,
        "accessibility_passes": report.passes_count,
    }


async def evaluate_site(
    *,
    url: str,
    site_name: str,
    prompt: str,
    round_num: int | None = None,
    run_axe: bool = True,
) -> tuple[SiteScore, Capture]:
    """Capture and score one site end-to-end.

    Designed to be safe to call concurrently — all state (screenshot files,
    JSON output) is keyed on `site_name` so parallel callers do not collide
    as long as they pass distinct names.

    Args:
        url: The site URL to fetch.
        site_name: Display name used for file paths and LangSmith metadata.
        prompt: The original prompt given to the contestant.
        round_num: Optional round number; influences output filenames and
            LangSmith experiment prefix when set.
        run_axe: If `True`, run the axe-core accessibility audit and
            include an `accessibility` axis in the score.

    Returns:
        Tuple of (`SiteScore`, `Capture`). The capture is returned so
        callers can persist screenshots, forward HTML to LangSmith, or
        feed downstream tooling without re-fetching.
    """
    capture = await capture_site(url, run_axe=run_axe)
    screenshot_b64 = base64.b64encode(capture.screenshot_png).decode()

    axes: dict[str, float | None] = _run_llm_evaluators(
        prompt, capture.html, screenshot_b64
    )
    if run_axe:
        axes["accessibility"] = score_accessibility(capture.accessibility)

    score = SiteScore(
        site_name=site_name,
        url=url,
        prompt=prompt,
        axes=axes,
        metadata=_build_metadata(capture.accessibility),
        round_num=round_num,
    )
    return score, capture


def print_score(score: SiteScore) -> None:
    """Print a single-site per-axis scoreboard to stdout.

    Args:
        score: The `SiteScore` to render.
    """
    bar = "═" * 56
    print()
    print(bar)
    header = f"  {score.site_name}"
    if score.round_num is not None:
        header += f"  —  Round {score.round_num}"
    print(header)
    print(f'  Prompt: "{score.prompt}"')
    print(f"  URL:    {score.url}")
    print(bar)
    for axis, value in score.axes.items():
        label = _AXIS_DISPLAY.get(axis, axis)
        if value is None:
            rendered = "  n/a"
        else:
            rendered = f"{round(value * 10):>4d}/10"
        print(f"  {label:22s}{rendered}")
    print(bar)
    print()


def write_json(score: SiteScore, out_dir: Path) -> Path:
    """Write a `SiteScore` to JSON under `out_dir`.

    Filename is `round-{n}-{site}.json` when `round_num` is set, else
    `{site}.json`.

    Args:
        score: Result to serialize.
        out_dir: Directory to write into; created if missing.

    Returns:
        Path to the written JSON file.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = score.site_name.replace(" ", "_")
    if score.round_num is not None:
        filename = f"round-{score.round_num}-{safe_name}.json"
    else:
        filename = f"{safe_name}.json"
    path = out_dir / filename
    path.write_text(json.dumps(asdict(score), indent=2) + "\n")
    return path


def _save_screenshot(
    png: bytes, site_name: str, round_num: int | None
) -> None:
    """Persist the raw screenshot to `screenshots/` for debugging."""
    try:
        ss_dir = Path("screenshots")
        ss_dir.mkdir(exist_ok=True)
        safe_name = site_name.replace(" ", "_")
        if round_num is not None:
            filename = f"round-{round_num}-{safe_name}.png"
        else:
            filename = f"{safe_name}.png"
        (ss_dir / filename).write_bytes(png)
    except OSError as exc:
        msg = f"Warning: could not save screenshot: {exc}"
        print(msg, file=sys.stderr)


async def _async_cli(args: argparse.Namespace) -> None:
    """CLI pipeline: capture -> score -> print -> write -> log.

    Args:
        args: Parsed argparse namespace.
    """
    print(f"Capturing {args.url}...")
    score, capture = await evaluate_site(
        url=args.url,
        site_name=args.name,
        prompt=args.prompt,
        round_num=args.round,
        run_axe=not args.no_axe,
    )

    _save_screenshot(capture.screenshot_png, args.name, args.round)
    print_score(score)

    try:
        path = write_json(score, Path(args.out))
        print(f"Results written to {path}")
    except OSError as exc:
        msg = f"Warning: could not write results JSON: {exc}"
        print(msg, file=sys.stderr)

    if not args.no_langsmith:
        screenshot_b64 = base64.b64encode(capture.screenshot_png).decode()
        log_to_langsmith(
            site_name=score.site_name,
            prompt=score.prompt,
            html=capture.html,
            screenshot_b64=screenshot_b64,
            axes=score.axes,
            round_num=score.round_num,
            metadata=score.metadata,
        )


def main() -> None:
    """Parse CLI arguments and run the single-site judge."""
    parser = argparse.ArgumentParser(
        description=(
            "Vibe Coding Olympics judge — score a single website against "
            "a prompt. Run multiple instances concurrently to score "
            "multiple sites."
        ),
    )
    parser.add_argument("--url", required=True, help="Website URL.")
    parser.add_argument("--name", required=True, help="Site display name.")
    parser.add_argument(
        "--prompt", required=True, help="The prompt given to the contestant."
    )
    parser.add_argument(
        "--round",
        type=int,
        default=None,
        help="Optional round number for filenames and LangSmith metadata.",
    )
    parser.add_argument(
        "--out",
        default="results",
        help="Output directory for JSON results. (default: %(default)s)",
    )
    parser.add_argument(
        "--no-axe",
        action="store_true",
        help="Skip the axe-core accessibility audit.",
    )
    parser.add_argument(
        "--no-langsmith",
        action="store_true",
        help="Skip LangSmith logging (also controlled by env/config).",
    )

    args = parser.parse_args()
    try:
        asyncio.run(_async_cli(args))
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
