#!/usr/bin/env python3
"""Render a fork-vs-handoff comparison from two fork_cache_demo.py --out-json files.

Purely a formatting step over numbers fork_cache_demo.py already computed
locally (via its callback handler) -- no LangSmith dependency.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def load(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def fmt_row(name: str, fork: dict[str, Any] | None, handoff: dict[str, Any] | None) -> str:
    def cell(bucket: dict[str, Any] | None, key: str) -> str:
        if bucket is None:
            return "-"
        value = bucket.get(key)
        return "-" if value is None else str(value)

    return (
        f"| {name} "
        f"| {cell(fork, 'llm_calls')} | {cell(fork, 'input_tokens')} | {cell(fork, 'output_tokens')} "
        f"| {cell(fork, 'cache_read')} | {cell(fork, 'first_call_cache_read')} | {cell(fork, 'tool_calls')} "
        f"| {cell(handoff, 'llm_calls')} | {cell(handoff, 'input_tokens')} | {cell(handoff, 'output_tokens')} "
        f"| {cell(handoff, 'cache_read')} | {cell(handoff, 'first_call_cache_read')} | {cell(handoff, 'tool_calls')} |"
    )


def total_tokens(per_agent: dict[str, dict[str, Any]]) -> int:
    return sum(b.get("input_tokens", 0) + b.get("output_tokens", 0) for b in per_agent.values())


def total_tool_calls(per_agent: dict[str, dict[str, Any]]) -> int:
    return sum(b.get("tool_calls", 0) for b in per_agent.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fork-json", required=True)
    parser.add_argument("--handoff-json", required=True)
    parser.add_argument(
        "--out-md", default=None, help="Optional path to also write the Markdown table"
    )
    args = parser.parse_args()

    fork = load(args.fork_json)
    handoff = load(args.handoff_json)

    names = sorted(set(fork["per_agent"]) | set(handoff["per_agent"]))

    lines = [
        f"## Fork vs handoff — PR #{fork.get('pr')} ({fork.get('repo')})",
        "",
        (
            "| agent | fork llm_calls | fork in_tok | fork out_tok | fork cache_read | fork 1st_hit | fork tools "
            "| handoff llm_calls | handoff in_tok | handoff out_tok | handoff cache_read | handoff 1st_hit | handoff tools |"
        ),
        "|---" * 13 + "|",
    ]
    for name in names:
        lines.append(fmt_row(name, fork["per_agent"].get(name), handoff["per_agent"].get(name)))

    fork_tokens = total_tokens(fork["per_agent"])
    handoff_tokens = total_tokens(handoff["per_agent"])
    fork_tools = total_tool_calls(fork["per_agent"])
    handoff_tools = total_tool_calls(handoff["per_agent"])
    token_delta_pct = (
        ((handoff_tokens - fork_tokens) / fork_tokens * 100) if fork_tokens else float("nan")
    )
    wall_delta_pct = (
        (handoff["total_wall_clock"] - fork["total_wall_clock"]) / fork["total_wall_clock"] * 100
        if fork.get("total_wall_clock")
        else float("nan")
    )

    lines += [
        "",
        "### Totals",
        "",
        "| metric | fork | handoff | handoff vs fork |",
        "|---|---|---|---|",
        f"| total tokens (in+out, all agents) | {fork_tokens} | {handoff_tokens} | {token_delta_pct:+.0f}% |",
        f"| total tool calls (all agents) | {fork_tools} | {handoff_tools} | {handoff_tools - fork_tools:+d} |",
        (
            f"| total wall clock (s) | {fork.get('total_wall_clock', 0):.1f} "
            f"| {handoff.get('total_wall_clock', 0):.1f} | {wall_delta_pct:+.0f}% |"
        ),
        "",
        (
            "`1st_hit` is the `cache_read` on each subagent's very first model call — "
            "a full hit for fork means it matched the parent's cached prefix exactly; "
            "0 for handoff is expected, since an isolated subagent starts cold."
        ),
        "",
        ("### Delegation directives"),
        "",
        "| lens | fork chars | handoff chars |",
        "|---|---|---|",
    ]
    for name in names:
        fork_desc = fork.get("directives", {}).get(name)
        handoff_desc = handoff.get("directives", {}).get(name)
        fork_len = len(fork_desc) if fork_desc is not None else "-"
        handoff_len = len(handoff_desc) if handoff_desc is not None else "-"
        lines.append(f"| {name} | {fork_len} | {handoff_len} |")

    text = "\n".join(lines) + "\n"
    print(text)

    if args.out_md:
        Path(args.out_md).write_text(text)

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a") as f:
            f.write(text)


if __name__ == "__main__":
    main()
