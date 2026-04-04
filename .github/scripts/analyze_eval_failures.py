"""Analyze eval failures with an LLM and surface explanations in CI.

Reads per-test failure data from evals_report.json (populated by the
pytest reporter plugin), sends each failure to an LLM for analysis in
parallel, and writes results to GITHUB_STEP_SUMMARY and a JSON artifact.

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


async def analyze_one(model: object, failure: dict[str, str]) -> dict[str, str]:
    """Analyze a single failure and return the failure dict with an `analysis` key.

    Args:
        model: A LangChain chat model instance supporting `ainvoke`.
        failure: Dict with `test_name`, `category`, and `failure_message` keys.

    Returns:
        The original failure dict extended with an `analysis` string.
    """
    prompt = _ANALYSIS_PROMPT.format(
        test_name=failure.get("test_name", "unknown"),
        category=failure.get("category", ""),
        failure_message=failure.get("failure_message", ""),
    )
    try:
        response = await model.ainvoke(prompt)  # type: ignore[union-attr]
        return {**failure, "analysis": response.content}
    except Exception as exc:  # noqa: BLE001
        return {**failure, "analysis": f"Analysis failed: {exc}"}


def _format_markdown(results: list[dict[str, str]]) -> str:
    """Format analysis results as a Markdown summary.

    Args:
        results: List of failure dicts each containing an `analysis` key.

    Returns:
        Markdown string suitable for GITHUB_STEP_SUMMARY.
    """
    lines = [f"## Failure analysis ({len(results)} failure{'s' if len(results) != 1 else ''})\n"]
    for result in results:
        lines.append(f"### `{result['test_name']}`")
        category = result.get("category")
        if category:
            lines.append(f"**Category:** {category}\n")
        lines.append(result.get("analysis", ""))
        lines.append("\n---\n")
    return "\n".join(lines)


async def run(report_path: Path) -> None:
    """Load failures, analyze in parallel, and write outputs.

    Args:
        report_path: Path to evals_report.json.
    """
    report = json.loads(report_path.read_text(encoding="utf-8"))
    failures: list[dict[str, str]] = report.get("failures", [])
    if not failures:
        print("No failures to analyze.")  # noqa: T201
        return

    from langchain.chat_models import init_chat_model

    model_name = os.environ.get("ANALYSIS_MODEL", _DEFAULT_MODEL)
    model = init_chat_model(model_name)

    results = await asyncio.gather(*(analyze_one(model, f) for f in failures))

    markdown = _format_markdown(list(results))

    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_file:
        with Path(summary_file).open("a", encoding="utf-8") as fh:
            fh.write(markdown)
    print(markdown)  # noqa: T201

    output_path = report_path.parent / "failure_analysis.json"
    output_path.write_text(
        json.dumps(list(results), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {output_path}")  # noqa: T201


def main() -> None:
    """Entry point: resolve report path and run the async analysis."""
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("evals_report.json")
    if not path.exists():
        print(f"Report file not found: {path}")  # noqa: T201
        return
    asyncio.run(run(path))


if __name__ == "__main__":
    main()
