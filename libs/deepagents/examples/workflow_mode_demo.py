#!/usr/bin/env python
"""Demo: a DeepAgent vs. a DeepAgent in workflow mode — actually building software.

The task is something anyone can picture: build a tiny Python utility library.
For each function, write the code, write a test, and RUN the test to prove it
works — then run the whole suite. It needs real files and a real shell.

We run the SAME task two ways and compare cost (tokens) and — importantly —
how the CONTEXT behaves:

  1. DEEPAGENT (no workflow): ONE agent does every step in a single conversation.
     Each tool call's output stays in the context, so by step N the model
     re-reads steps 1..N-1 on every turn. Input tokens climb each turn, and a
     context stuffed with stale prior-step detail raises the risk of
     context-degradation / hallucination.
  2. DEEPAGENT + WORKFLOW: each step runs in a FRESH, isolated sub-agent. Step 2
     never carries step 1's raw context — context stays small and focused per
     step, so the orchestrator's cost stays flat as the work grows.

The per-turn input-token numbers in the output make the difference visible.

Reads `OPENAI_API_KEY` / `OPENAI_BASE_URL` from the env. Model via `DEMO_MODEL`
(default `gpt-5.5`); number of functions via `DEMO_FUNCS` (default 5). Runs real
`python3` in a throwaway temp directory.

    cd libs/deepagents
    uv run --group test python examples/workflow_mode_demo.py
"""

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from langchain_core.messages import AIMessage, ToolMessage
from langchain_openai import ChatOpenAI

from deepagents import (
    GeneralPurposeSubagentProfile,
    HarnessProfile,
    create_deep_agent,
    register_harness_profile,
)
from deepagents.backends import LocalShellBackend

MODEL = os.environ.get("DEMO_MODEL", "gpt-5.5")
WORKFLOW_MODEL = os.environ.get("DEMO_WORKFLOW_MODEL")  # optional cheaper model for workflow workers
NUM_FUNCS = max(2, min(int(os.environ.get("DEMO_FUNCS", "5")), 8))
_SUPPORTS_TEMPERATURE = MODEL.startswith(("gpt-4", "gpt-3"))
MAX_DEPS_SHOWN = 3

# Small, obviously-correct functions so the build is fast and deterministic.
_FUNCS = [
    ("add", "add(a, b) returns the sum of two numbers"),
    ("is_even", "is_even(n) returns True if n is even, else False"),
    ("reverse_string", "reverse_string(s) returns the string reversed"),
    ("factorial", "factorial(n) returns n! (n factorial); factorial(0) == 1"),
    ("celsius_to_fahrenheit", "celsius_to_fahrenheit(c) returns c*9/5 + 32"),
    ("gcd", "gcd(a, b) returns the greatest common divisor of a and b"),
    ("clamp", "clamp(x, lo, hi) returns x bounded to the range [lo, hi]"),
    ("word_count", "word_count(s) returns the number of whitespace-separated words in s"),
]
FUNCS = _FUNCS[:NUM_FUNCS]
SPEC_LINES = "\n".join(f"- {name}.py: {spec}" for name, spec in FUNCS)

SYSTEM_PROMPT = f"""You are a coding agent working in the current directory. You have a shell (the `execute` tool) and filesystem tools.

Build a small Python utility library. Implement these functions, each in its OWN file:
{SPEC_LINES}

For EVERY function:
1. Write <name>.py containing just that function.
2. Write test_<name>.py: a script of `assert` statements that exercise the function and `print("OK")` at the end.
3. Run `python3 test_<name>.py` and confirm it prints OK (fix the code if it doesn't).

Finally, run every test once more and report which passed.

If a `workflow` tool is available, you MUST do this as ONE workflow, two phases:
- Phase 1 "Build": one step PER function (ids b1, b2, ...), subagent_type "general-purpose", each with a short `description` like "Build add". Each step writes <name>.py + test_<name>.py and runs the test.
- Phase 2 "Verify": one step (description "Run all tests") that depends_on ALL build steps, runs every test_*.py, and reports the results.
Emit the whole plan in a single workflow call.

If no `workflow` tool is available, do it yourself, one function at a time."""

TASK = "Build the utility library described in your instructions and run all the tests. Begin."

# --------------------------------------------------------------------------- #
# Tiny ANSI + formatting helpers
# --------------------------------------------------------------------------- #
_COLORS = {"dim": "2", "bold": "1", "cyan": "36", "green": "32", "yellow": "33", "magenta": "35", "red": "31"}


def col(text: str, name: str) -> str:
    return text if not sys.stdout.isatty() else f"\033[{_COLORS[name]}m{text}\033[0m"


def trunc(text: str, n: int = 130) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[:n] + "…"


_THOUSAND = 1000
_MAX_TURNS_SHOWN = 6


def kfmt(n: int) -> str:
    """Format a token count like 12,400 -> '12.4k'."""
    return f"{n / _THOUSAND:.1f}k" if n >= _THOUSAND else str(n)


def turn_sequence(turns: list[int]) -> str:
    """Render a per-turn token sequence compactly (first/last when long)."""
    parts = [kfmt(t) for t in turns]
    if len(parts) > _MAX_TURNS_SHOWN:
        parts = [*parts[:2], "…", *parts[-2:]]
    return " → ".join(parts)


# --------------------------------------------------------------------------- #
# Streaming logger — model turns, the workflow plan preview, step progress, tokens
# --------------------------------------------------------------------------- #
class RunLog:
    def __init__(self) -> None:
        self.start = time.perf_counter()
        self.tool_calls: dict[str, int] = {}
        self.substeps = 0
        self.turn_inputs: list[int] = []  # input tokens per main-loop turn (shows context growth)
        self.tok_out = 0
        self.names: dict[str, str] = {}
        self.final = ""

    @property
    def turns(self) -> int:
        return len(self.turn_inputs)

    @property
    def tok_in(self) -> int:
        return sum(self.turn_inputs)

    @property
    def calls(self) -> int:
        return sum(self.tool_calls.values())

    def context_note(self, *, workflow_mode: bool) -> str:
        if workflow_mode:
            return "← orchestrator only; per-step detail stays in the sub-agents"
        return "← climbs: step 1 is still in context when it reaches the last step"

    def t(self) -> str:
        return col(f"[{time.perf_counter() - self.start:5.1f}s]", "dim")

    def handle(self, mode: str, chunk: object) -> None:
        if mode == "custom" and isinstance(chunk, dict) and "workflow" in chunk:
            self._workflow(chunk["workflow"])
        elif mode == "updates" and isinstance(chunk, dict):
            for update in chunk.values():
                if isinstance(update, dict):
                    for msg in update.get("messages", []):
                        self._message(msg)

    def _message(self, msg: object) -> None:
        if isinstance(msg, AIMessage):
            usage = msg.usage_metadata or {}
            self.turn_inputs.append(usage.get("input_tokens", 0))
            self.tok_out += usage.get("output_tokens", 0)
            for call in msg.tool_calls:
                name = call["name"]
                self.tool_calls[name] = self.tool_calls.get(name, 0) + 1
                self.names[call.get("id", "")] = name
                arg = next((str(v) for k, v in (call.get("args") or {}).items() if k in ("file_path", "command", "description")), "")
                print(f"{self.t()} {col('🧠 →', 'yellow')} {col(name, 'cyan')} {trunc(arg, 55)}")
            if not msg.tool_calls and msg.text.strip():
                self.final = msg.text.strip()
                print(f"{self.t()} {col('🧠 final', 'green')} ({len(self.final)} chars)")
        elif isinstance(msg, ToolMessage):
            print(f"{self.t()}   {col('🔧 ' + self.names.get(msg.tool_call_id, 'tool'), 'cyan')}: {trunc(msg.content)}")

    def _workflow(self, ev: dict) -> None:
        kind = ev.get("event")
        if kind == "plan":
            head = f"{ev['phase_count']} phases / {ev['step_count']} steps"
            print(f"{self.t()} {col('📋 plan', 'magenta')} {col(head, 'bold')}")
            for ph in ev["phases"]:
                steps = ph["steps"]
                shape = "parallel" if len(steps) > 1 else "single"
                title = f"Phase {ph['index']} · {ph['title']}"
                print(f"          {col(title, 'bold')} {col(f'({len(steps)} {shape})', 'dim')}")
                for s in steps:
                    deps = s["depends_on"]
                    overflow = "…" if len(deps) > MAX_DEPS_SHOWN else ""
                    dep = col(f" ⇐ {','.join(deps[:MAX_DEPS_SHOWN])}{overflow}", "dim") if deps else ""
                    sub = col(f"[{s['subagent_type']}]", "cyan")
                    print(f"            {col('•', 'green')} {s['id']} {sub}{dep}  {s.get('description') or ''}")
        elif kind == "phase_start":
            print(f"{self.t()} {col('⚙ phase', 'magenta')} #{ev['index']} {col(ev['title'], 'bold')}")
        elif kind == "step_done":
            self.substeps += 1
            print(f"{self.t()}     {col('✓', 'green')} {ev['id']}")
        elif kind == "step_error":
            print(f"{self.t()}     {col('✗ ' + str(ev['id']), 'red')}: {trunc(ev.get('error', ''), 80)}")


# --------------------------------------------------------------------------- #
# Objective verification: run the test files the agent produced
# --------------------------------------------------------------------------- #
def verify(workdir: str) -> tuple[int, int, int]:
    """Return (python files written, tests passing, total tests) on disk."""
    py_files = list(Path(workdir).glob("*.py"))
    tests = sorted(p for p in py_files if p.name.startswith("test_"))

    def passes(test: Path) -> bool:
        try:
            res = subprocess.run([sys.executable, test.name], cwd=workdir, capture_output=True, timeout=30, check=False)  # noqa: S603
        except (subprocess.SubprocessError, OSError):
            return False
        return res.returncode == 0

    return len(py_files), sum(passes(t) for t in tests), len(tests)


def run_agent(label: str, *, workflow_mode: bool) -> tuple[RunLog, tuple[int, int, int]]:
    workdir = tempfile.mkdtemp(prefix="deepagent_demo_")
    print("\n" + col("=" * 84, "dim"))
    print(col(f"  {label}  (model={MODEL}, functions={NUM_FUNCS}, workflow_mode={workflow_mode})", "bold"))
    print(col(f"  workdir: {workdir}", "dim"))
    print(col("=" * 84, "dim"))

    # Baseline is a plain agent (no sub-agents) so it does the work in its own
    # loop; workflow mode re-enables the general-purpose worker that backs steps.
    register_harness_profile("openai", HarnessProfile(general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=workflow_mode)))

    model = ChatOpenAI(model=MODEL, stream_usage=True, **({"temperature": 0} if _SUPPORTS_TEMPERATURE else {}))
    worker_model = ChatOpenAI(model=WORKFLOW_MODEL, stream_usage=True) if (workflow_mode and WORKFLOW_MODEL) else None
    agent = create_deep_agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        backend=LocalShellBackend(root_dir=workdir, inherit_env=True),
        workflow_mode=workflow_mode,
        workflow_model=worker_model,
    )

    log = RunLog()
    for mode, chunk in agent.stream({"messages": [{"role": "user", "content": TASK}]}, stream_mode=["updates", "custom"]):
        log.handle(mode, chunk)

    files, passed, total = verify(workdir)
    elapsed = time.perf_counter() - log.start
    tools = ", ".join(f"{k}×{v}" for k, v in sorted(log.tool_calls.items())) or "none"
    extra = f" · {log.substeps} sub-agent steps" if log.substeps else ""
    print(col("-" * 84, "dim"))
    print(f"{col('time', 'bold')} {elapsed:5.1f}s · {log.turns} main-loop turns, {log.calls} tool calls{extra} · {tools}")
    print(f"{col('built', 'bold')}: {files} files · {col(f'tests {passed}/{total} passing', 'green' if passed == total and total else 'red')}")
    print(
        f"{col('input tokens / main-loop turn', 'bold')}: {turn_sequence(log.turn_inputs)}  {col(log.context_note(workflow_mode=workflow_mode), 'dim')}"
    )
    print(f"{col('main-loop tokens', 'bold')}: in={log.tok_in:,}  out={log.tok_out:,}")
    return log, (files, passed, total)


def main() -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set.", file=sys.stderr)
        return 1
    print(col("Deep Agents — workflow mode demo (build & test a small library)", "bold"))
    print(f"endpoint: {os.environ.get('OPENAI_BASE_URL', '<default>')}")
    print(f"task    : implement {NUM_FUNCS} functions, write a test for each, and run them")

    plain, plain_art = run_agent("DEEPAGENT (no workflow)", workflow_mode=False)
    flow, flow_art = run_agent("DEEPAGENT + WORKFLOW", workflow_mode=True)

    print("\n" + col("=" * 84, "dim"))
    print(col("  COMPARISON — same task, same result; what differs is the context", "bold"))
    print(col("=" * 84, "dim"))
    print(f"  {'':<30}{'DeepAgent':>16}{'+Workflow':>16}")

    def row(name: str, a: object, b: object) -> None:
        print(f"  {name:<30}{a!s:>16}{b!s:>16}")

    row("main-loop turns", plain.turns, flow.turns)
    row("main-loop input tokens", f"{plain.tok_in:,}", f"{flow.tok_in:,}")
    row("main-loop output tokens", f"{plain.tok_out:,}", f"{flow.tok_out:,}")
    row("per-step context", "1 shared, grows", f"{flow.substeps} isolated")
    row("files written", plain_art[0], flow_art[0])
    row("tests passing", f"{plain_art[1]}/{plain_art[2]}", f"{flow_art[1]}/{flow_art[2]}")

    print()
    print(col("  Why this matters — it's about the context, not only the bill:", "bold"))
    print(f"  • Without workflow, ONE agent runs the whole job in ONE conversation ({plain.turns} model calls).")
    print("    Every tool call's output stays in that context, so by the last step the model is still")
    print("    carrying step 1, and it re-sends the growing history on every turn (the per-turn numbers")
    print("    above climb). A long, mixed-purpose context costs more tokens and gives the model more")
    print("    room to drift or hallucinate.")
    print(f"  • Workflow mode runs each step in a FRESH, isolated sub-agent ({flow.substeps} of them). A step only")
    print("    sees its own task plus any results explicitly passed in — never step 1's raw history. The")
    print(f"    orchestrator stayed at {flow.turns} turns / {flow.tok_in:,} input tokens; the per-step work is spread")
    print("    across small, focused contexts instead of piled into one. (Those sub-agents cost tokens")
    print("    too — the win is that no single context carries everything.)")

    if flow.final:
        print("\n" + col("final answer — DeepAgent + workflow (after really running the tests):", "bold"))
        print(trunc(flow.final, 400))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
