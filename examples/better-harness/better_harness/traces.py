"""Trace loading and markdown rendering for the outer proposer agent.

Traces are written by the harness (fixed adapter section) as trace.json
alongside Harbor's result.json.  This module reads them and renders them
as clean, scannable markdown so a language model can reason about what
the inner agent did, what tools it called, and where it went wrong.
"""
from __future__ import annotations

import json
from pathlib import Path

from better_harness.core import ToolCall, Trace, Turn

# Env var set by the runner so the harness knows where to write trace.json.
TRACE_ENV = "BETTER_HARNESS_TRACE_FILE"

_MAX_OUTPUT = 1500        # per tool output — increased so errors at end of long outputs are visible
_MAX_OUTPUT_HEAD = 600    # chars kept from the start of a long output
_MAX_OUTPUT_TAIL = 600    # chars kept from the end of a long output
_MAX_INPUT = 400
_MAX_AGENT_TEXT = 500


def load_trace(
    trace_path: Path,
    *,
    case_id: str,
    split: str,
    score: float,
    failure: str | None = None,
) -> Trace:
    """Load a trace.json written by the harness and patch in the score + failure.

    If trace.json doesn't exist (harness doesn't support trace capture yet),
    returns a minimal Trace so the rest of the pipeline still works.
    """
    if not trace_path.exists():
        return Trace(
            case_id=case_id,
            split=split,
            score=score,
            task="",
            turns=[],
            final_output="",
            failure=failure or "trace.json not found — harness may not support trace capture",
        )

    data = json.loads(trace_path.read_text())
    turns = [
        Turn(
            agent=t.get("agent", ""),
            calls=[
                ToolCall(
                    tool=c["tool"],
                    input=c.get("input", {}),
                    output=c.get("output"),
                    error=c.get("error"),
                )
                for c in t.get("calls", [])
            ],
        )
        for t in data.get("turns", [])
    ]
    return Trace(
        case_id=case_id,
        split=split,
        score=score,
        task=data.get("task", ""),
        turns=turns,
        final_output=data.get("final_output", ""),
        failure=failure or data.get("failure"),
        total_turns=int(data.get("total_turns", len(turns))),
    )


def render_trace_md(trace: Trace) -> str:
    """Render a Trace as clean markdown for the outer proposer agent.

    The format is designed for LLM readability:
    - Each agent turn shows reasoning text + tool calls inline
    - Tool call inputs are compact JSON
    - Errors are clearly flagged with ✗
    - The failure reason is shown at the end
    """
    status = "PASSED ✓" if trace.passed() else "FAILED ✗"
    turn_note = ""
    if trace.total_turns > 0:
        turn_note = f", {trace.total_turns} turns"
    lines = [
        f"# Case: {trace.case_id} — {status}  (score: {trace.score:.2g}{turn_note})",
        "",
        f"**Task:** {trace.task}" if trace.task else "**Task:** *(unknown)*",
        "",
    ]

    if not trace.turns:
        lines.extend(["## Execution", "", "*(no turns captured)*", ""])
    else:
        n = len(trace.turns)
        total_calls = sum(len(t.calls) for t in trace.turns)
        lines.append(f"## Execution  ({n} agent response{'s' if n != 1 else ''}, {total_calls} tool call{'s' if total_calls != 1 else ''})")
        lines.append("")

        for i, turn in enumerate(trace.turns, 1):
            text = turn.agent
            if len(text) > _MAX_AGENT_TEXT:
                text = text[:_MAX_AGENT_TEXT] + "…"

            if text:
                lines.append(f'**[{i}]** Agent: "{text}"')
            else:
                lines.append(f"**[{i}]** Agent: *(no text)*")

            for call in turn.calls:
                inp = _compact_json(call.input, _MAX_INPUT)
                lines.append(f"  → `{call.tool}({inp})`")
                if call.error is not None:
                    err = _trim_content(call.error)
                    lines.append(f"  ✗ **Error:** {err}")
                elif call.output is not None:
                    out = _trim_content(call.output)
                    lines.append(f"  ← `{out}`")
                else:
                    lines.append("  ← *(no output)*")
            lines.append("")

    if trace.final_output:
        out = trace.final_output[:500]
        lines.extend([f"**Final output:** {out}", ""])

    if not trace.passed():
        lines.append("## Why it failed")
        lines.append(trace.failure or "*(no failure message — check harbor result.json)*")
        lines.append("")

    return "\n".join(lines)


def _compact_json(obj: object, max_len: int) -> str:
    s = json.dumps(obj, separators=(",", ":"), ensure_ascii=False)
    if len(s) > max_len:
        return s[:max_len] + "…"
    return s


def _trim_content(text: str) -> str:
    """Keep head and tail of long tool outputs so errors at the end remain visible."""
    if len(text) <= _MAX_OUTPUT:
        return text
    mid = len(text) - _MAX_OUTPUT_HEAD - _MAX_OUTPUT_TAIL
    return f"{text[:_MAX_OUTPUT_HEAD]}\n[… {mid} chars trimmed …]\n{text[-_MAX_OUTPUT_TAIL:]}"
