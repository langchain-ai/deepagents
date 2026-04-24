"""Outer Deep Agent invocation."""
from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from better_harness.core import Experiment

_SYSTEM_PROMPT = """\
You are Better Agent, an outer-loop Deep Agent that improves an AI agent harness.

Your job: analyze execution traces and edit agent.py so more eval cases pass.

## Required workflow

1. Read task.md for the full instructions.
2. Read case_status.md — focus only on "Sometimes Passing" cases. Never-passed cases
   are likely capability gaps; prompt changes won't fix them.
3. Read traces/failing/ for failing train cases. Read traces/passing/ for passing
   cases. Compare — the gap between them shows you what the agent is doing differently.
4. Fill in ALL THREE sections of scratchpad.md BEFORE editing agent.py:
   - Failure classification (misbehavior class for each failing case)
   - Pattern across failures (the single root cause)
   - Hypothesis + prediction (what you'll change and which cases it will fix)
5. Run the anti-overfit test: "If this task disappeared, would this change still be
   a worthwhile harness improvement?" If no, don't make it.
6. Make exactly ONE mechanism change in agent.py. If you want to add "and also..." —
   stop. That's a second candidate for a later iteration.
7. Write a brief summary to proposal.md.

## What makes a good change

Good: changes that improve a general capability (better tool for a class of actions,
clearer prompt instruction that applies to many task types, orchestration fix for
a common failure mode like action loops).

Bad: case-specific hacks, hardcoded task knowledge, keyword matching, changes that
only help because the test set contains that exact pattern.

## What makes a bad failure diagnosis

Calling something "PROMPT" when you haven't read the reasoning content of the trace.
Calling something "CAPABILITY_GAP" without checking if a purpose-built tool would help.
Diagnosing a single case without looking for the pattern across multiple failures.

## Simplicity criterion

If your change achieves the same pass count with shorter, cleaner code — that's a
real improvement. Keep it.
"""


def run_outer_agent(*, experiment: Experiment, workspace_dir: Path) -> str | None:
    """Run the outer Deep Agent against the workspace. Returns its final message."""
    deepagents_root = _resolve_root(experiment.better_agent_deepagents_root)
    if deepagents_root is not None:
        return _run_subprocess(
            experiment=experiment,
            workspace_dir=workspace_dir,
            deepagents_root=deepagents_root,
        )
    return _run_import(experiment=experiment, workspace_dir=workspace_dir)


# ---------------------------------------------------------------------------
# Invocation paths
# ---------------------------------------------------------------------------


def _run_import(*, experiment: Experiment, workspace_dir: Path) -> str | None:
    """Invoke the outer agent in-process (deepagents already on sys.path)."""
    try:
        backends = importlib.import_module("deepagents.backends")
        graph = importlib.import_module("deepagents.graph")
        lc_msgs = importlib.import_module("langchain_core.messages")
    except ImportError as exc:
        msg = (
            "deepagents not installed. "
            "Set better_agent.deepagents_root in config or install deepagents."
        )
        raise RuntimeError(msg) from exc

    backend = backends.FilesystemBackend(root_dir=str(workspace_dir), virtual_mode=True)
    agent = graph.create_deep_agent(
        model=experiment.better_agent_model,
        system_prompt=_SYSTEM_PROMPT,
        backend=backend,
    )
    for attempt in range(3):
        try:
            result = agent.invoke(
                {
                    "messages": [
                        lc_msgs.HumanMessage(
                            content=(
                                "Read task.md first, then read the failing traces in traces/, "
                                "diagnose the failures, edit agent.py, and write your summary "
                                "to proposal.md."
                            )
                        )
                    ]
                },
                config={"recursion_limit": experiment.better_agent_max_turns},
            )
            return _last_ai_text(result)
        except Exception as exc:
            if attempt == 2 or not _is_transient(str(exc)):
                raise
            time.sleep(2 * (attempt + 1))
    return None  # pragma: no cover


def _run_subprocess(
    *,
    experiment: Experiment,
    workspace_dir: Path,
    deepagents_root: Path,
) -> str | None:
    """Invoke the outer agent in a uv subprocess (deepagents in a separate project)."""
    project_root = deepagents_root / "libs" / "deepagents"
    if not project_root.exists():
        project_root = deepagents_root

    repo_root = Path(__file__).resolve().parents[1]
    request = {
        "workspace_root": str(workspace_dir),
        "model": experiment.better_agent_model,
        "max_turns": experiment.better_agent_max_turns,
        "system_prompt": _SYSTEM_PROMPT,
    }
    request_path = workspace_dir / "_agent_request.json"
    result_path = workspace_dir / "_agent_result.json"
    request_path.write_text(json.dumps(request, indent=2) + "\n")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    for attempt in range(3):
        completed = subprocess.run(
            [
                "uv", "run", "--project", str(project_root),
                "python", "-m", "better_harness.agent",
                str(request_path), str(result_path),
            ],
            cwd=project_root,
            env=env,
            capture_output=True,
            check=False,
            text=True,
        )
        (workspace_dir / "_agent_stdout.log").write_text(completed.stdout)
        (workspace_dir / "_agent_stderr.log").write_text(completed.stderr)
        if completed.returncode == 0:
            payload = json.loads(result_path.read_text())
            return payload.get("final_message")
        error = completed.stderr.strip() or completed.stdout.strip() or "subprocess failed"
        if attempt == 2 or not _is_transient(error):
            raise RuntimeError(f"outer agent subprocess failed: {error}")
        time.sleep(2 * (attempt + 1))
    return None  # pragma: no cover


# ---------------------------------------------------------------------------
# Subprocess entry point  (python -m better_harness.agent request.json result.json)
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        msg = "usage: python -m better_harness.agent <request.json> <result.json>"
        raise SystemExit(msg)

    request_path = Path(args[0]).resolve()
    result_path = Path(args[1]).resolve()
    payload = json.loads(request_path.read_text())

    backends = importlib.import_module("deepagents.backends")
    graph = importlib.import_module("deepagents.graph")
    lc_msgs = importlib.import_module("langchain_core.messages")

    backend = backends.FilesystemBackend(root_dir=str(payload["workspace_root"]), virtual_mode=True)
    agent = graph.create_deep_agent(
        model=str(payload["model"]),
        system_prompt=str(payload["system_prompt"]),
        backend=backend,
    )
    result = agent.invoke(
        {
            "messages": [
                lc_msgs.HumanMessage(
                    content=(
                        "Read task.md first, then read the failing traces in traces/, "
                        "diagnose the failures, edit agent.py, and write your summary "
                        "to proposal.md."
                    )
                )
            ]
        },
        config={"recursion_limit": int(payload["max_turns"])},
    )
    final_message = _last_ai_text(result)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps({"final_message": final_message}, indent=2) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _last_ai_text(result: dict[str, Any]) -> str | None:
    for msg in reversed(result.get("messages", [])):
        if getattr(msg, "type", None) == "ai":
            content = getattr(msg, "content", None)
            if isinstance(content, str) and content.strip():
                return content
            if isinstance(content, list):
                text = "\n".join(
                    block.get("text", "") if isinstance(block, dict) else str(block)
                    for block in content
                    if not (isinstance(block, dict) and block.get("type") == "tool_use")
                ).strip()
                if text:
                    return text
    return None


def _is_transient(message: str) -> bool:
    lowered = message.lower()
    return any(t in lowered for t in ("overloaded", "rate limit", "timeout", "529 "))


def _resolve_root(root: Path | None) -> Path | None:
    if root is not None:
        return root
    sibling = Path(__file__).resolve().parents[2] / "deepagents"
    return sibling if sibling.exists() else None
