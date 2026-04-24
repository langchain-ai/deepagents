"""Build the proposer workspace: the directory the outer agent reads and edits."""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from better_harness.core import IterResult, SplitResult, Trace
from better_harness.traces import render_trace_md

# Max passing traces to include (enough to understand successful behavior without noise).
_MAX_PASSING_TRACES = 2

_TASK_INSTRUCTIONS = """\
# Better Agent Task

You are improving an AI agent harness. Edit `agent.py` so more eval cases pass.

## Workflow — follow this order strictly

**Step 1: Analyze (write to scratchpad.md before editing anything)**

Read `case_status.md` to understand which cases are tractable right now.
For each failing train case, read `traces/failing/<case-id>.md`.
Then read `traces/passing/` to see what the agent does right when it succeeds.

In `scratchpad.md`, fill in all three sections:

```
## Failure classification
[For each failing case: case_id → misbehavior class from the taxonomy below]

## Pattern across failures
[What do 2+ failing cases have in common? One root cause.]

## Hypothesis and prediction
[I believe X causes the failures. My change is Y.
I predict this will fix cases: [list].
Anti-overfit check: if these specific tasks disappeared,
would Y still be a worthwhile harness improvement? YES/NO + why.]
```

**Step 2: Edit (after scratchpad is complete)**

Edit `agent.py` (editable section above the FIXED ADAPTER BOUNDARY). Change exactly **one mechanism** — if you want to add "and also...", that is a second candidate for a later iteration.

**Step 3: Write proposal.md**
Brief summary: what you changed and why.

---

## Failure taxonomy (classify before editing)

| Class | What it looks like | Typical fix |
|---|---|---|
| **PROMPT** | Agent misunderstands the task goal or uses wrong strategy | Rewrite the relevant instruction in SYSTEM_PROMPT |
| **TOOL_MISSING** | Agent tries to do something but has no tool for it | Add a specialized tool to create_tools() |
| **TOOL_CONFUSED** | Agent calls a tool with wrong arguments or misunderstands what it does | Improve the tool's docstring/description |
| **ACTION_LOOP** | Agent repeats the same tool call with minor variants | Add loop detection, or restructure run_task() |
| **OVERCONFIDENT** | Agent says "done" when it hasn't actually finished | Add a verification step or a check tool |
| **CONFUSED_MODEL** | Agent has a wrong assumption about the environment state | Clarify the environment in SYSTEM_PROMPT |
| **CONTEXT_POLLUTION** | Earlier tool results misdirect later reasoning | Limit output length, add summarization |
| **CAPABILITY_GAP** | The model genuinely cannot do this; no prompt fix will work | Add a purpose-built tool, or accept the gap |
| **INFRA_FAILURE** | The tool or environment failed, not the agent's fault | Fix the tool or dependency |

---

## What you can change in agent.py

Everything above the `FIXED ADAPTER BOUNDARY` line:

- `SYSTEM_PROMPT` — the agent's base instructions
- `MODEL` — model spec (only change if explicitly authorized)
- `create_tools()` — add, remove, or modify tools
- `create_agent()` — agent construction, sub-agents, handoffs
- `run_task()` — orchestration logic, turn budget, verification

---

## Rules

- Do not edit anything at or below the `FIXED ADAPTER BOUNDARY` line.
- Do not add task-specific hacks. Overfitting test: "If this task disappeared,
  would this change still be worthwhile?" If no, don't do it.
- Focus on `case_status.md`'s "Sometimes Passing" cases — highest leverage.
  For "Never Passed" cases, assume capability gap unless you have a specific new tool.
- Simpler is better. Equal performance with shorter/cleaner code is a win.
- Read `history.md` before proposing — do not repeat failed approaches.

---

## Files

| File | Purpose |
|---|---|
| `agent.py` | **Edit this** — editable section above FIXED ADAPTER BOUNDARY |
| `scratchpad.md` | **Write analysis here first** — fill before editing agent.py |
| `trace_summaries.md` | **Read first** — one-liner per case for fast triage (50+ scale) |
| `case_status.md` | Case tier classification across all runs |
| `scores.json` | Current pass/fail per case |
| `traces/failing/` | Full traces for failing train cases |
| `traces/passing/` | 1-2 passing train cases for comparison |
| `history.md` | Prior iterations with per-case deltas |
| `proposal.md` | **Write brief summary here after editing** |

The harness-optimizer skill has the full analysis workflow including triage guidance for 50+ cases.
"""

_SCRATCHPAD_TEMPLATE = """\
# Analysis Scratchpad

Complete all three sections before editing agent.py.

## Failure classification
<!-- For each failing case, pick a class from the taxonomy in task.md:
     PROMPT | TOOL_MISSING | TOOL_CONFUSED | ACTION_LOOP | OVERCONFIDENT |
     CONFUSED_MODEL | CONTEXT_POLLUTION | CAPABILITY_GAP | INFRA_FAILURE -->


## Pattern across failures
<!-- What do 2+ failing cases have in common?
     Find the single root cause that, if fixed, would address the most cases. -->


## Hypothesis and prediction
<!-- Template:
I believe [specific failure pattern] is caused by [specific harness property].
My change: [exactly what I will edit and how].
I predict this will fix: [list of case IDs].
Anti-overfit check: if these specific tasks disappeared, would this change still
be a worthwhile harness improvement? YES/NO — [reason]. -->
"""


def build_workspace(  # noqa: PLR0913
    *,
    harness_path: Path,
    train_result: SplitResult,
    holdout_result: SplitResult,
    history: list[IterResult],
    baseline_train: SplitResult | None = None,
    workspace_dir: Path,
) -> Path:
    """Create the proposer workspace directory and populate it. Returns workspace_dir."""
    if workspace_dir.exists():
        shutil.rmtree(workspace_dir)
    workspace_dir.mkdir(parents=True)

    # The harness file — the outer agent edits this.
    shutil.copy2(harness_path, workspace_dir / "agent.py")

    # Scores overview.
    scores = {
        "train": {
            t.case_id: {"score": t.score, "passed": t.passed()}
            for t in train_result.traces
        },
        "holdout": {
            t.case_id: {"score": t.score, "passed": t.passed()}
            for t in holdout_result.traces
        },
        "summary": {
            "train": train_result.summary(),
            "holdout": holdout_result.summary(),
        },
    }
    (workspace_dir / "scores.json").write_text(json.dumps(scores, indent=2) + "\n")

    # Trace summaries — one-liner per case for fast triage at 50+ scale.
    prior_train = history[-1].train if history else baseline_train
    (workspace_dir / "trace_summaries.md").write_text(
        _render_trace_summaries(train_result, holdout_result, prior_train=prior_train)
    )

    # Case tier classification (across all runs seen so far).
    all_train = ([baseline_train] if baseline_train else []) + [r.train for r in history] + [train_result]
    (workspace_dir / "case_status.md").write_text(_render_case_tiers(all_train))

    # Traces — separate directories for failing and passing.
    traces_dir = workspace_dir / "traces"
    failing_dir = traces_dir / "failing"
    passing_dir = traces_dir / "passing"
    failing_dir.mkdir(parents=True)
    passing_dir.mkdir(parents=True)

    failing = train_result.failing()
    if not failing:
        (failing_dir / "ALL_PASSING.md").write_text("All train cases are currently passing.\n")
    else:
        for trace in failing:
            safe = _safe_name(trace.case_id)
            (failing_dir / f"{safe}.md").write_text(render_trace_md(trace))

    passing_traces = [t for t in train_result.traces if t.passed()][:_MAX_PASSING_TRACES]
    if not passing_traces:
        (passing_dir / "NONE_PASSING.md").write_text(
            "No train cases currently pass. No baseline comparison available.\n"
        )
    else:
        for trace in passing_traces:
            safe = _safe_name(trace.case_id)
            (passing_dir / f"{safe}.md").write_text(render_trace_md(trace))

    # History with per-case deltas.
    (workspace_dir / "history.md").write_text(_render_history(history))

    # Scratchpad — outer agent fills this in before editing.
    (workspace_dir / "scratchpad.md").write_text(_SCRATCHPAD_TEMPLATE)

    # Instructions.
    (workspace_dir / "task.md").write_text(_TASK_INSTRUCTIONS)

    # Proposal placeholder.
    (workspace_dir / "proposal.md").write_text(
        "# Proposal\n\n"
        "**What I changed:**\n\n"
        "**Why (tie to scratchpad hypothesis):**\n"
    )

    return workspace_dir


def read_proposal(workspace_dir: Path) -> str:
    """Read the proposal.md the outer agent wrote, stripped of the blank template."""
    path = workspace_dir / "proposal.md"
    if not path.exists():
        return ""
    text = path.read_text().strip()
    placeholder = (
        "# Proposal\n\n"
        "**What I changed:**\n\n"
        "**Why (tie to scratchpad hypothesis):**"
    )
    return "" if text == placeholder else text


def read_edited_harness(workspace_dir: Path) -> str:
    """Read back the agent.py the outer agent edited."""
    return (workspace_dir / "agent.py").read_text()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _render_trace_summaries(
    train_result: SplitResult,
    holdout_result: SplitResult,
    *,
    prior_train: SplitResult | None = None,
) -> str:
    """One-liner per case for fast triage — read before opening any full trace.

    Turn count flags are ONLY per-task across iterations (same task, different run).
    Cross-task comparison of turn counts is not done — different tasks have
    inherently different complexity and turn requirements.
    """
    if not train_result.traces and not holdout_result.traces:
        return "# Trace Summaries\n\nNo traces available.\n"

    # Prior iteration turn counts indexed by case_id — only valid basis for flagging.
    prior_turns: dict[str, int] = {}
    if prior_train is not None:
        prior_turns = {t.case_id: t.total_turns for t in prior_train.traces if t.total_turns > 0}

    def _summary_row(trace: Trace) -> str:
        status = "PASS" if trace.passed() else "FAIL"
        turns = trace.total_turns
        flags: list[str] = []

        # Only compare this task against ITSELF in prior runs — not against other tasks.
        prior = prior_turns.get(trace.case_id)
        if prior is not None and turns > 0:
            if turns >= prior * 2:
                flags.append(f"turns {prior}→{turns} (regressed)")
            elif prior > 0 and turns <= prior // 2:
                flags.append(f"turns {prior}→{turns} (improved)")

        # First run: flag very fast failures as potentially infra-level.
        # This is a weak signal — read the trace to confirm.
        if prior is None and turns > 0 and turns <= 2 and not trace.passed():
            flags.append("fast-fail — check infra/imports first")

        flag_str = f"  [{', '.join(flags)}]" if flags else ""
        final = (trace.final_output or trace.failure or "")[:80].replace("\n", " ")
        turn_str = str(turns) if turns > 0 else "?"
        return f"| `{trace.case_id}` | {status} | {turn_str} | {final}{flag_str} |"

    train_rows = [_summary_row(t) for t in train_result.traces]
    holdout_rows = [_summary_row(t) for t in holdout_result.traces]

    lines = [
        "# Trace Summaries",
        "",
        f"Train: {train_result.summary()}  |  Holdout: {holdout_result.summary()}",
        "",
        "**Read this file first for triage.** Turn count changes are flagged only when",
        "the SAME case changed significantly vs the prior iteration — cross-task",
        "comparison is not valid since different tasks have different complexity.",
        "",
        "## Train cases",
        "",
        "| Case | Result | Turns | Final statement / failure |",
        "|------|--------|-------|---------------------------|",
        *train_rows,
        "",
        "## Holdout cases",
        "",
        "| Case | Result | Turns | Final statement / failure |",
        "|------|--------|-------|---------------------------|",
        *holdout_rows,
        "",
    ]
    return "\n".join(lines)


def _render_case_tiers(all_train_results: list[SplitResult]) -> str:
    """Classify cases into always-passing / sometimes / never tiers across all runs."""
    if not all_train_results:
        return "# Case Status\n\nNo runs yet.\n"

    # Collect pass counts across all runs.
    pass_counts: dict[str, int] = {}
    for sr in all_train_results:
        for t in sr.traces:
            pass_counts[t.case_id] = pass_counts.get(t.case_id, 0) + (1 if t.passed() else 0)

    total_runs = len(all_train_results)
    always: list[str] = []
    sometimes: list[tuple[str, int]] = []
    never: list[str] = []

    for case_id in sorted(pass_counts):
        count = pass_counts[case_id]
        if count == total_runs:
            always.append(case_id)
        elif count > 0:
            sometimes.append((case_id, count))
        else:
            never.append(case_id)

    lines = [f"# Case Status  (across {total_runs} run(s))", ""]

    if always:
        lines.extend(["## Always Passing — floor (don't lose these)", ""])
        for c in always:
            lines.append(f"- `{c}`")
        lines.append("")

    if sometimes:
        lines.extend(["## Sometimes Passing — ⭐ target these first (highest leverage)", ""])
        for c, count in sometimes:
            lines.append(f"- `{c}` — passed {count}/{total_runs} runs")
        lines.append("")

    if never:
        lines.extend([
            "## Never Passed — likely capability gap (do not over-invest in prompt tweaks)", "",
        ])
        for c in never:
            lines.append(f"- `{c}` — 0/{total_runs} runs")
        lines.append("")

    lines.extend([
        "**Strategy:** fix 'Sometimes Passing' cases. For 'Never Passed' cases, assume a",
        "capability gap — prompt changes won't fix them unless you also add a new tool.",
    ])
    return "\n".join(lines)


def _render_history(history: list[IterResult]) -> str:
    if not history:
        return "# History\n\nNo iterations yet — this is the first proposal.\n"

    lines = ["# Optimization History", ""]
    for r in history:
        status = "ACCEPTED ✓" if r.accepted else "REJECTED ✗"
        lines.extend([
            f"## Iteration {r.iteration} — {status}",
            f"Train: {r.train.summary()}  |  Holdout: {r.holdout.summary()}",
            "",
        ])

        # Per-case delta — show exactly what changed.
        if r.prior_train is not None:
            delta = _case_delta(r.prior_train, r.train)
            lines.extend([delta, ""])

        lines.extend([
            r.proposal or "*(no proposal written)*",
            "",
        ])
    return "\n".join(lines)


def _case_delta(before: SplitResult, after: SplitResult) -> str:
    """Describe which cases newly passed or regressed."""
    before_passing = {t.case_id for t in before.traces if t.passed()}
    after_passing = {t.case_id for t in after.traces if t.passed()}
    gained = sorted(after_passing - before_passing)
    lost = sorted(before_passing - after_passing)
    lines = []
    for c in gained:
        lines.append(f"  ↑ `{c}`: newly PASSING")
    for c in lost:
        lines.append(f"  ↓ `{c}`: now FAILING (regression)")
    if not lines:
        lines.append("  No case-level changes.")
    return "\n".join(lines)


def _safe_name(case_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", case_id).strip("-") or "case"
