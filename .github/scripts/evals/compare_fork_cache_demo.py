#!/usr/bin/env python3
"""Render a fork-vs-isolated comparison from two fork_cache_demo.py --out-json files.

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


def fmt_row(name: str, fork: dict[str, Any] | None, isolated: dict[str, Any] | None) -> str:
    def cell(bucket: dict[str, Any] | None, key: str) -> str:
        if bucket is None:
            return "-"
        value = bucket.get(key)
        return "-" if value is None else str(value)

    return (
        f"| {name} "
        f"| {cell(fork, 'llm_calls')} | {cell(fork, 'input_tokens')} | {cell(fork, 'output_tokens')} "
        f"| {cell(fork, 'cache_read')} | {cell(fork, 'first_call_cache_read')} | {cell(fork, 'tool_calls')} "
        f"| {cell(isolated, 'llm_calls')} | {cell(isolated, 'input_tokens')} | {cell(isolated, 'output_tokens')} "
        f"| {cell(isolated, 'cache_read')} | {cell(isolated, 'first_call_cache_read')} | {cell(isolated, 'tool_calls')} |"
    )


def total_tokens(per_agent: dict[str, dict[str, Any]]) -> int:
    return sum(b.get("input_tokens", 0) + b.get("output_tokens", 0) for b in per_agent.values())


def total_tool_calls(per_agent: dict[str, dict[str, Any]]) -> int:
    return sum(b.get("tool_calls", 0) for b in per_agent.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fork-json", required=True)
    parser.add_argument("--isolated-json", required=True)
    parser.add_argument(
        "--out-md", default=None, help="Optional path to also write the Markdown table"
    )
    args = parser.parse_args()

    fork = load(args.fork_json)
    isolated = load(args.isolated_json)

    names = sorted(set(fork["per_agent"]) | set(isolated["per_agent"]))

    lines = [
        f"## Fork vs isolated — PR #{fork.get('pr')} ({fork.get('repo')})",
        "",
        (
            "| agent | fork llm_calls | fork in_tok | fork out_tok | fork cache_read | fork 1st_hit | fork tools "
            "| isolated llm_calls | isolated in_tok | isolated out_tok | isolated cache_read | isolated 1st_hit | isolated tools |"
        ),
        "|---" * 13 + "|",
    ]
    for name in names:
        lines.append(fmt_row(name, fork["per_agent"].get(name), isolated["per_agent"].get(name)))

    fork_tokens = total_tokens(fork["per_agent"])
    isolated_tokens = total_tokens(isolated["per_agent"])
    fork_tools = total_tool_calls(fork["per_agent"])
    isolated_tools = total_tool_calls(isolated["per_agent"])
    token_delta_pct = (
        ((isolated_tokens - fork_tokens) / fork_tokens * 100) if fork_tokens else float("nan")
    )
    wall_delta_pct = (
        (isolated["total_wall_clock"] - fork["total_wall_clock"]) / fork["total_wall_clock"] * 100
        if fork.get("total_wall_clock")
        else float("nan")
    )

    lines += [
        "",
        "### Totals",
        "",
        "| metric | fork | isolated | isolated vs fork |",
        "|---|---|---|---|",
        f"| total tokens (in+out, all agents) | {fork_tokens} | {isolated_tokens} | {token_delta_pct:+.0f}% |",
        f"| total tool calls (all agents) | {fork_tools} | {isolated_tools} | {isolated_tools - fork_tools:+d} |",
        (
            f"| total wall clock (s) | {fork.get('total_wall_clock', 0):.1f} "
            f"| {isolated.get('total_wall_clock', 0):.1f} | {wall_delta_pct:+.0f}% |"
        ),
        "",
        (
            "`1st_hit` is the `cache_read` on each subagent's very first model call — "
            "a full hit for fork means it matched the parent's cached prefix exactly; "
            "0 for isolated is expected, since an isolated subagent starts cold."
        ),
        "",
        ("### Delegation directives"),
        "",
        "| lens | fork chars | isolated chars |",
        "|---|---|---|",
    ]
    for name in names:
        fork_desc = fork.get("directives", {}).get(name)
        isolated_desc = isolated.get("directives", {}).get(name)
        fork_len = len(fork_desc) if fork_desc is not None else "-"
        isolated_len = len(isolated_desc) if isolated_desc is not None else "-"
        lines.append(f"| {name} | {fork_len} | {isolated_len} |")

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
