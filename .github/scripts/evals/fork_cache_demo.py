#!/usr/bin/env python3
"""Fork-vs-handoff cost/cache demo: PR review with specialized reviewers.

A parent agent fetches a real PR's diff (and any repo files it needs) once,
then delegates review to the general-purpose subagent N times -- one call per
lens (e.g. correctness/backcompat/tests/performance/api_design) -- using an
IDENTICAL short instruction regardless of branch:

    "Review this change specifically for {lens} issues."

This deliberately does NOT declare a custom subagent or pass a `mode=` flag.
The general-purpose subagent's own default (fork vs. handoff) comes entirely
from whichever `deepagents` checkout this script runs against -- compare a
branch that forces it into `mode: "fork"` by default (e.g.
`bengret/sug-agent-forking-evals`) against one that leaves it as the normal
isolated default (e.g. `bengret/feat-subagent-forking`). See
.github/workflows/fork_cache_demo.yml, which checks out a different
`deepagents` ref per leg rather than passing this script a mode flag.

Metrics are captured via a plain callback handler attached to the top-level
`invoke()` call -- subagent `.invoke()` calls run in the same call stack and
inherit callbacks/config automatically, so no LangSmith round-trip is needed
for the numbers themselves. LangSmith tracing (when LANGSMITH_API_KEY is set)
is enabled purely so a human can inspect the run visually afterward; it is
not load-bearing for anything this script prints or writes.

Each of the general-purpose subagent's five invocations shares the same
`lc_agent_name` ("general-purpose"), so per-lens attribution instead walks
each event's `parent_run_id` chain back to the specific `task` tool call that
spawned it, and labels that invocation by its position in the (deterministic,
instructed) delegation order.

Local usage (from libs/deepagents, with a real ANTHROPIC_API_KEY exported):

    uv run python ../../.github/scripts/evals/fork_cache_demo.py \
        --repo langchain-ai/deepagents --pr 5873 --branch-tag fork

CI usage: see .github/workflows/fork_cache_demo.yml, which runs this once per
branch (checking out a different `deepagents` ref each time) and compares the
resulting --out-json files.
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from deepagents import create_deep_agent

DEFAULT_LENSES = ["correctness", "backcompat", "tests", "performance", "api_design"]


def repo_root() -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(result.stdout.strip())


def fetch_pr_diff(repo: str, pr_number: str) -> str:
    result = subprocess.run(
        ["gh", "pr", "diff", pr_number, "--repo", repo],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def fetch_pr_meta(repo: str, pr_number: str) -> dict[str, str]:
    result = subprocess.run(
        ["gh", "pr", "view", pr_number, "--repo", repo, "--json", "title,body"],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def build_parent_instruction(
    repo: str, pr_number: str, title: str, body: str, lenses: list[str]
) -> str:
    trimmed_body = (body or "(no description)").strip()[:2000]
    lens_line = '  "Review this change specifically for {lens} issues."'
    return f"""You are reviewing pull request #{pr_number} in the {repo} repo: "{title}"

{trimmed_body}

First, use get_pr_diff to fetch the diff, and read_repo_file for any files you
need to understand the change in its surrounding context.

Then delegate the review to the general-purpose subagent, ONE CALL AT A TIME
(wait for each result before starting the next -- do not delegate in
parallel), using EXACTLY this instruction each time (only the lens name
changes):

{lens_line}

Use these lenses in this exact order: {", ".join(lenses)}.

Once all have reported back, summarize their findings in one paragraph.
"""


class MetricsHandler(BaseCallbackHandler):
    """Captures per-lens token/tool/wall-clock metrics locally.

    Every call to the general-purpose subagent shares the same
    `lc_agent_name`, so attribution instead walks each event's
    `parent_run_id` chain back to the enclosing `task` tool call and labels
    it by that call's position in the (instructed, deterministic) lens
    order -- not by `lc_agent_name`.
    """

    def __init__(self, lenses: list[str]) -> None:
        self._lenses = lenses
        self._task_seq = 0

        self._parent_of: dict[str, str | None] = {}
        self._run_label: dict[str, str] = {}
        self._task_run_label: dict[str, str] = {}
        self._run_start: dict[str, float] = {}

        self.per_agent: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "llm_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_read": 0,
                "first_call_cache_read": None,
                "tool_calls": 0,
                "wall_clock": 0.0,
            }
        )
        self.directives: dict[str, str] = {}

    def _label_for(self, run_id: Any, parent_run_id: Any) -> str:
        rid = str(run_id)
        pid = str(parent_run_id) if parent_run_id is not None else None
        self._parent_of[rid] = pid
        cursor = pid
        while cursor is not None:
            if cursor in self._task_run_label:
                return self._task_run_label[cursor]
            cursor = self._parent_of.get(cursor)
        return "main"

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: Any,
        parent_run_id: Any = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        label = self._label_for(run_id, parent_run_id)
        self._run_label[str(run_id)] = label
        self._run_start[str(run_id)] = time.monotonic()

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        key = str(run_id)
        agent_name = self._run_label.pop(key, "main")
        start = self._run_start.pop(key, None)
        elapsed = time.monotonic() - start if start is not None else 0.0

        bucket = self.per_agent[agent_name]
        bucket["llm_calls"] += 1
        bucket["wall_clock"] += elapsed

        try:
            message = response.generations[0][0].message
        except (AttributeError, IndexError):
            return
        usage = getattr(message, "usage_metadata", None) or {}
        input_tokens = usage.get("input_tokens", 0) or 0
        output_tokens = usage.get("output_tokens", 0) or 0
        details = usage.get("input_token_details", {}) or {}
        cache_read = details.get("cache_read", 0) or 0

        bucket["input_tokens"] += input_tokens
        bucket["output_tokens"] += output_tokens
        bucket["cache_read"] += cache_read
        if bucket["first_call_cache_read"] is None:
            bucket["first_call_cache_read"] = cache_read

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        label = self._label_for(run_id, parent_run_id)
        self.per_agent[label]["tool_calls"] += 1

        if serialized.get("name") != "task":
            return
        # on_tool_start's input_str is Python's dict repr (single quotes),
        # not JSON -- confirmed empirically, json.loads fails on it silently.
        try:
            parsed = ast.literal_eval(input_str)
        except (ValueError, SyntaxError):
            return
        if not isinstance(parsed, dict):
            return
        if parsed.get("subagent_type") != "general-purpose":
            return

        lens = (
            self._lenses[self._task_seq]
            if self._task_seq < len(self._lenses)
            else f"unexpected-lens-{self._task_seq}"
        )
        self._task_seq += 1
        self._task_run_label[str(run_id)] = lens
        self.directives[lens] = parsed.get("description", "")

    def as_json(self) -> dict[str, Any]:
        return {"per_agent": self.per_agent, "directives": self.directives}

    def print_report(self) -> None:
        print("\n=== Per-lens metrics ===")
        header = (
            f"{'lens':<14} {'llm_calls':>9} {'in_tok':>8} {'out_tok':>8} "
            f"{'cache_read':>10} {'1st_hit':>8} {'tool_calls':>10} {'wall_s':>7}"
        )
        print(header)
        for name, b in sorted(self.per_agent.items()):
            print(
                f"{name:<14} {b['llm_calls']:>9} {b['input_tokens']:>8} "
                f"{b['output_tokens']:>8} {b['cache_read']:>10} "
                f"{str(b['first_call_cache_read']):>8} {b['tool_calls']:>10} "
                f"{b['wall_clock']:>7.1f}"
            )

        print("\n=== Delegation directives ===")
        for lens, desc in sorted(self.directives.items()):
            print(f"  {lens}: {len(desc)} chars -- {desc!r}")


def make_tools(allowed_prefix: Path, diff_text: str):
    @tool
    def get_pr_diff() -> str:
        """Return the full diff for the pull request under review."""
        return diff_text

    @tool
    def read_repo_file(path: str) -> str:
        """Read a file from the repo. `path` must be relative to the repo
        root and stay under the allowed prefix.
        """
        resolved = (allowed_prefix.parent / path).resolve()
        if not str(resolved).startswith(str(allowed_prefix)):
            return f"Error: access denied for path outside {allowed_prefix.name}/: {path}"
        if not resolved.is_file():
            return f"Error: file not found: {path}"
        try:
            return resolved.read_text()
        except Exception as exc:  # noqa: BLE001
            return f"Error reading file: {exc}"

    return get_pr_diff, read_repo_file


def build_agent(model: str, tools: list):
    # No `subagents=` override: the general-purpose subagent this delegates
    # to is whichever default `create_deep_agent` auto-adds -- fork or
    # handoff is entirely a property of the `deepagents` checkout this runs
    # against, not anything this script configures.
    return create_deep_agent(model=model, tools=list(tools))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="owner/repo, e.g. langchain-ai/deepagents")
    parser.add_argument("--pr", required=True, help="PR number to use as the diff substrate")
    parser.add_argument(
        "--branch-tag",
        default="unknown",
        help="Label for this run's output only (e.g. 'fork'/'handoff') -- "
        "does not affect agent construction. The actual fork/handoff "
        "behavior comes from whichever deepagents checkout is on PYTHONPATH.",
    )
    parser.add_argument("--model", default="anthropic:claude-sonnet-4-6")
    parser.add_argument(
        "--lenses",
        default=",".join(DEFAULT_LENSES),
        help="Comma-separated review lenses, delegated to in this exact order",
    )
    parser.add_argument(
        "--allowed-prefix",
        default="libs",
        help="Repo-relative dir read_repo_file is scoped to",
    )
    parser.add_argument("--out-json", default=None, help="Optional path to write metrics as JSON")
    args = parser.parse_args()

    lenses = [lens.strip() for lens in args.lenses.split(",") if lens.strip()]
    allowed_prefix = (repo_root() / args.allowed_prefix).resolve()

    print(f"Fetching PR #{args.pr} from {args.repo}...", flush=True)
    diff_text = fetch_pr_diff(args.repo, args.pr)
    meta = fetch_pr_meta(args.repo, args.pr)
    instruction = build_parent_instruction(
        args.repo, args.pr, meta.get("title", ""), meta.get("body", ""), lenses
    )

    tools = make_tools(allowed_prefix, diff_text)
    handler = MetricsHandler(lenses)
    agent = build_agent(args.model, tools)

    print(f"=== Running branch_tag={args.branch_tag} model={args.model} ===", flush=True)
    start = time.monotonic()
    result = agent.invoke(
        {"messages": [HumanMessage(content=instruction)]},
        config={"callbacks": [handler], "recursion_limit": 50},
    )
    elapsed = time.monotonic() - start
    print(f"Total wall clock: {elapsed:.1f}s", flush=True)

    handler.print_report()

    final = result["messages"][-1]
    print("\n=== Parent's final summary ===")
    print(getattr(final, "text", None) or final.content)

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "branch_tag": args.branch_tag,
            "model": args.model,
            "repo": args.repo,
            "pr": args.pr,
            "lenses": lenses,
            "total_wall_clock": elapsed,
            **handler.as_json(),
        }
        out_path.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
