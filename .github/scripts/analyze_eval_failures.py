"""Analyze eval failures with an LLM and surface explanations in CI.

Reads per-test failure data from evals_report.json (populated by the pytest
reporter plugin), sends each failure to an LLM for analysis in parallel, and
writes results to `GITHUB_STEP_SUMMARY` and a JSON artifact.

Usage:
    uv run python .github/scripts/analyze_eval_failures.py [evals_report.json]

Environment variables:
    ANALYSIS_MODEL  — LLM to use for analysis (default: anthropic:claude-haiku-4-5-20251001)
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

_ANALYSIS_PROMPT = """\
You are analyzing a failed evaluation for an AI coding agent.

## Test
Name: {test_name}
Category: {category}

## Failure details
{failure_message}

The failure details above contain the assertion that failed and the agent's
full trajectory (every step, tool call, and text output).

Analyze concisely:
1. What the agent was supposed to do
2. Where it went wrong (which step/decision)
3. Root cause — pick ONE: prompt issue | model capability gap | \
wrong tool selection | hallucination | eval too strict | non-deterministic
4. One-sentence summary"""

_DEFAULT_MODEL = "anthropic:claude-haiku-4-5-20251001"


async def analyze_one(model: BaseChatModel, failure: dict[str, str]) -> dict[str, str]:
    """Analyze a single failure and return a new dict with an `analysis` key.

    Args:
        model: A LangChain chat model instance.
        failure: Dict with `test_name`, `category`, and `failure_message` keys.

    Returns:
        A new dict combining the original failure fields with an
            `analysis` string.
    """
    prompt = _ANALYSIS_PROMPT.format(
        test_name=failure.get("test_name", "unknown"),
        category=failure.get("category", ""),
        failure_message=failure.get("failure_message", ""),
    )
    try:
        response = await model.ainvoke(prompt)
        return {**failure, "analysis": response.text}
    except Exception as exc:  # noqa: BLE001
        print(  # noqa: T201
            f"warning: analysis failed for {failure.get('test_name', '?')}: {exc!r}",
            file=sys.stderr,
        )
        return {**failure, "analysis": f"Analysis failed: {exc!r}"}


def _format_markdown(results: list[dict[str, str]]) -> str:
    """Format analysis results as a Markdown summary.

    Args:
        results: List of failure dicts each containing an `analysis` key.

    Returns:
        Markdown-formatted string.
    """
    lines = [
        f"## Failure analysis ({len(results)} failure{'s' if len(results) != 1 else ''})\n",
        "<details>",
        "<summary>(click to expand)</summary>",
        "",
    ]
    for result in results:
        lines.append(f"### `{result.get('test_name', 'unknown')}`")
        category = result.get("category")
        if category:
            lines.append(f"**Category:** {category}\n")
        lines.append(result.get("analysis", ""))
        lines.append("\n---\n")
    lines.append("</details>")
    return "\n".join(lines)


async def run(report_path: Path) -> None:
    """Load failures, analyze in parallel, and write outputs.

    Exits early when no failures are present. On success, writes a
    `failure_analysis.json` alongside the input report and appends a Markdown
    summary to `GITHUB_STEP_SUMMARY` (when set). Markdown is always printed
    to stdout.

    Args:
        report_path: Path to evals_report.json.
    """
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        msg = f"error: failed to read report {report_path}: {exc}"
        print(msg, file=sys.stderr)  # noqa: T201
        sys.exit(1)

    failures: list[dict[str, str]] = report.get("failures", [])
    if not failures:
        print("No failures to analyze.")  # noqa: T201
        return

    try:
        from langchain.chat_models import init_chat_model
    except ImportError:
        msg = "error: langchain is not installed; cannot analyze failures"
        print(msg, file=sys.stderr)  # noqa: T201
        sys.exit(1)

    model_name = os.environ.get("ANALYSIS_MODEL", _DEFAULT_MODEL)
    try:
        model = init_chat_model(model_name)
    except Exception as exc:  # noqa: BLE001
        msg = f"error: failed to initialize model {model_name!r}: {exc!r}"
        print(msg, file=sys.stderr)  # noqa: T201
        sys.exit(1)

    results = list(await asyncio.gather(*(analyze_one(model, f) for f in failures)))

    markdown = _format_markdown(results)

    # Write JSON artifact first so it is available even if summary write fails.
    output_path = report_path.parent / "failure_analysis.json"
    output_path.write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {output_path}")  # noqa: T201

    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_file:
        with Path(summary_file).open("a", encoding="utf-8") as fh:
            fh.write(markdown)
    print(markdown)  # noqa: T201


def main() -> None:
    """Entry point: resolve report path and run the async analysis.

    Exits with code 1 if the report file does not exist.
    """
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("evals_report.json")
    if not path.exists():
        print(f"error: report file not found: {path}", file=sys.stderr)  # noqa: T201
        sys.exit(1)
    try:
        asyncio.run(run(path))
    except Exception as exc:  # noqa: BLE001
        print(f"error: failure analysis failed: {exc!r}", file=sys.stderr)  # noqa: T201
        sys.exit(1)


if __name__ == "__main__":
    main()
