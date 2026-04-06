"""CLI judge for the Vibe Coding Olympics.

Captures screenshots of two player websites, runs LLM evaluators, prints a
results table, writes results to JSON, and optionally logs to LangSmith.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
from itertools import starmap
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

from evaluators import ALL_EVALUATORS, FEEDBACK_KEYS

# ---------------------------------------------------------------------------
# Screenshot capture
# ---------------------------------------------------------------------------


async def capture(url: str) -> tuple[str, bytes]:
    """Capture a full-page screenshot and HTML source from a URL.

    Uses a headless Chromium browser with a 1280x900 viewport. Waits for
    network idle before capturing (15s timeout).

    Args:
        url: The website URL to capture.

    Returns:
        Tuple of (html_source, png_bytes).

    Raises:
        RuntimeError: If the page fails to load or screenshot.
    """
    from playwright.async_api import async_playwright

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            try:
                page = await browser.new_page(
                    viewport={"width": 1280, "height": 900},
                )
                await page.goto(url, wait_until="networkidle", timeout=15000)
                html = await page.content()
                png = await page.screenshot(full_page=True)
            finally:
                await browser.close()
    except Exception as exc:
        msg = f"Failed to capture {url}: {exc}"
        raise RuntimeError(msg) from exc
    else:
        return html, png


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def run_evaluators(
    prompt: str,
    html: str,
    screenshot_b64: str,
) -> dict[str, int]:
    """Run all evaluators for a single player.

    Args:
        prompt: The original prompt given to the contestant.
        html: The HTML source of the contestant's website.
        screenshot_b64: Base64-encoded PNG screenshot.

    Returns:
        Mapping of feedback key to integer score (0 through 10).
        Failed evaluators default to 0 with a warning on stderr.
    """
    scores: dict[str, int] = {}
    for evaluator, key in zip(ALL_EVALUATORS, FEEDBACK_KEYS, strict=True):
        try:
            result = evaluator(
                inputs=prompt,
                outputs=html,
                screenshot_b64=screenshot_b64,
            )
            raw = float(result["score"])
            scores[key] = round(raw * 10)
        except Exception as exc:
            msg = f"Evaluator '{key}' failed: {exc}"
            print(f"  WARNING: {msg}", file=sys.stderr)
            scores[key] = 0
    return scores


_DISPLAY_NAMES: dict[str, str] = {
    "visual_design": "Visual Design",
    "content_completeness": "Content",
    "creativity": "Creativity",
    "prompt_adherence": "Prompt Adherence",
}


# ---------------------------------------------------------------------------
# Results printing
# ---------------------------------------------------------------------------


def print_results(
    round_num: int,
    prompt: str,
    p1_name: str,
    p1_scores: dict[str, int],
    p1_total: int,
    p2_name: str,
    p2_scores: dict[str, int],
    p2_total: int,
    winner: str,
) -> None:
    """Print the formatted results table to stdout.

    Args:
        round_num: Competition round number.
        prompt: The prompt given to contestants.
        p1_name: Player 1 display name.
        p1_scores: Player 1 scores by feedback key.
        p1_total: Player 1 total score.
        p2_name: Player 2 display name.
        p2_scores: Player 2 scores by feedback key.
        p2_total: Player 2 total score.
        winner: Name of the winning player.
    """
    max_total = len(FEEDBACK_KEYS) * 10
    bar = "\u2550" * 51
    thin = "\u2500" * 5

    print()
    print(bar)
    print(f"  VIBE CODING OLYMPICS \u2014 ROUND {round_num}")
    print(f'  Prompt: "{prompt}"')
    print(bar)
    print()

    col1 = max(len(p1_name), 5)
    col2 = max(len(p2_name), 5)
    header = f"{'':20s}{p1_name:>{col1}s}    {p2_name:>{col2}s}"
    print(header)

    for key in FEEDBACK_KEYS:
        label = _DISPLAY_NAMES.get(key, key)
        s1 = p1_scores.get(key, 0)
        s2 = p2_scores.get(key, 0)
        print(f"  {label:18s}{s1:>{col1}d}    {s2:>{col2}d}")

    print(f"{'':20s}{thin:>{col1}s}   {thin:>{col2}s}")
    print(
        f"  {'TOTAL':18s}{p1_total:>{col1}d}/{max_total}"
        f"  {p2_total:>{col2}d}/{max_total}"
    )
    print()
    print(f"  WINNER: {winner}")
    print(bar)
    print()


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------


def write_json(
    round_num: int,
    prompt: str,
    p1_name: str,
    p1_scores: dict[str, int],
    p1_total: int,
    p2_name: str,
    p2_scores: dict[str, int],
    p2_total: int,
    winner: str,
) -> Path:
    """Write results to `results/round-{round_num}.json`, creating the dir if needed.

    Args:
        round_num: Competition round number.
        prompt: The prompt given to contestants.
        p1_name: Player 1 display name.
        p1_scores: Player 1 scores by feedback key.
        p1_total: Player 1 total score.
        p2_name: Player 2 display name.
        p2_scores: Player 2 scores by feedback key.
        p2_total: Player 2 total score.
        winner: Name of the winning player.

    Returns:
        Path to the written JSON file.
    """
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    out = results_dir / f"round-{round_num}.json"
    data = {
        "round": round_num,
        "prompt": prompt,
        "player1": {
            "name": p1_name,
            "scores": p1_scores,
            "total": p1_total,
        },
        "player2": {
            "name": p2_name,
            "scores": p2_scores,
            "total": p2_total,
        },
        "winner": winner,
    }
    out.write_text(json.dumps(data, indent=2) + "\n")
    return out


# ---------------------------------------------------------------------------
# LangSmith logging (best-effort)
# ---------------------------------------------------------------------------


def log_to_langsmith(
    round_num: int,
    prompt: str,
    player_name: str,
    html: str,
    screenshot_b64: str,
    scores: dict[str, int],
) -> None:
    """Log one player's evaluation to LangSmith.

    Best-effort: prints to stderr and continues on failure.

    Args:
        round_num: Competition round number.
        prompt: The prompt given to contestants.
        player_name: Display name of the player being logged.
        html: HTML source of the player's website.
        screenshot_b64: Base64-encoded PNG screenshot.
        scores: Mapping of feedback key to integer score (0-10).
    """
    try:
        from langsmith import evaluate as ls_evaluate
    except ImportError:
        return

    try:

        def _target(inputs: dict[str, Any]) -> dict[str, Any]:
            return {
                "html": inputs["html"],
                "screenshot_b64": inputs["screenshot_b64"],
            }

        def _make_evaluator(key: str, score: int) -> Callable[..., dict[str, Any]]:
            def _eval(
                run: Any,  # noqa: ARG001
                example: Any,  # noqa: ARG001
            ) -> dict[str, Any]:
                return {"key": key, "score": score / 10.0}

            return _eval

        evaluators = list(starmap(_make_evaluator, scores.items()))

        ls_evaluate(
            target=_target,
            data=[
                {
                    "inputs": {
                        "prompt": prompt,
                        "html": html,
                        "screenshot_b64": screenshot_b64,
                    },
                }
            ],
            evaluators=evaluators,
            experiment_prefix=f"round-{round_num}-{player_name}",
            metadata={
                "event": "interrupt-2026",
                "round": round_num,
                "player": player_name,
                "prompt": prompt,
            },
        )
        print(f"  LangSmith: logged {player_name}")
    except Exception as exc:
        msg = f"LangSmith: failed for {player_name}: {exc}"
        print(f"  {msg}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def async_main(args: argparse.Namespace) -> None:
    """Run the full judging pipeline.

    Pipeline: capture -> evaluate -> print results -> write JSON -> log.

    Args:
        args: Parsed CLI arguments with prompt, URLs, names, and round number.
    """
    prompt = args.prompt
    round_num = args.round

    # 1. Capture screenshots in parallel
    print(f"Capturing screenshots for round {round_num}...")
    results = await asyncio.gather(
        capture(args.p1_url),
        capture(args.p2_url),
        return_exceptions=True,
    )
    errors = [r for r in results if isinstance(r, BaseException)]
    if errors:
        for err in errors:
            msg = str(err)
            print(msg, file=sys.stderr)
        sys.exit(1)
    (html1, png1), (html2, png2) = results  # type: ignore[misc]

    # Save screenshots for debugging
    try:
        ss_dir = Path("screenshots")
        ss_dir.mkdir(exist_ok=True)  # noqa: ASYNC240
        (ss_dir / f"round-{round_num}-p1.png").write_bytes(png1)
        (ss_dir / f"round-{round_num}-p2.png").write_bytes(png2)
    except OSError as exc:
        msg = f"Warning: could not save screenshots: {exc}"
        print(msg, file=sys.stderr)

    b64_1 = base64.b64encode(png1).decode()
    b64_2 = base64.b64encode(png2).decode()

    print("Running evaluators...")
    p1_scores = run_evaluators(prompt, html1, b64_1)
    p2_scores = run_evaluators(prompt, html2, b64_2)

    p1_total = sum(p1_scores.values())
    p2_total = sum(p2_scores.values())

    # P1 wins ties (arbitrary tiebreak for MVP)
    winner = args.p1_name if p1_total >= p2_total else args.p2_name

    # Print results first so they're visible even if file writes fail
    print_results(
        round_num,
        prompt,
        args.p1_name,
        p1_scores,
        p1_total,
        args.p2_name,
        p2_scores,
        p2_total,
        winner,
    )

    try:
        json_path = write_json(
            round_num,
            prompt,
            args.p1_name,
            p1_scores,
            p1_total,
            args.p2_name,
            p2_scores,
            p2_total,
            winner,
        )
        print(f"Results written to {json_path}")
    except OSError as exc:
        msg = f"Warning: could not write results JSON: {exc}"
        print(msg, file=sys.stderr)

    log_to_langsmith(round_num, prompt, args.p1_name, html1, b64_1, p1_scores)
    log_to_langsmith(round_num, prompt, args.p2_name, html2, b64_2, p2_scores)


def main() -> None:
    """Parse arguments and run the judge."""
    parser = argparse.ArgumentParser(
        description="Vibe Coding Olympics judge — screenshot and score two websites.",
    )
    parser.add_argument(
        "--prompt", required=True, help="The prompt given to contestants."
    )
    parser.add_argument("--p1-url", required=True, help="Player 1 website URL.")
    parser.add_argument("--p1-name", required=True, help="Player 1 display name.")
    parser.add_argument("--p2-url", required=True, help="Player 2 website URL.")
    parser.add_argument("--p2-name", required=True, help="Player 2 display name.")
    parser.add_argument("--round", type=int, default=1, help="Round number.")

    args = parser.parse_args()
    try:
        asyncio.run(async_main(args))
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
