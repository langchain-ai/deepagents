"""Score the `write-file-simple` Harbor task as a multi-key reward.

Containerized counterpart of the pytest eval
``tests/evals/test_file_operations.py::test_write_file_simple``. It demonstrates
that the trajectory / efficiency metrics the pytest suite asserts on
(``TrajectoryScorer``) can be reproduced inside a Harbor trial and emitted as a
multi-dimensional ``reward.json``.

Harbor runs ``tests/test.sh`` after the agent finishes; that calls this module.
It reads two sources:

* ``/logs/agent/trajectory.json`` — the ATIF trajectory written by
  ``DeepAgentsWrapper._save_trajectory`` (agent steps, tool calls, final text).
* ``/app/name.txt`` — the container filesystem end-state the agent produced.

and writes ``/logs/verifier/reward.json`` with one key per metric. Harbor reads
that dict into ``VerifierResult.rewards``. The ``correctness`` key is the gated
success metric (see ``[min_reward]`` in ``task.toml``), mirroring the pytest
``.success(...)`` tier; the rest are diagnostics analogous to ``.expect(...)``.

Pure standard library so it runs in a vanilla ``python:3.12-slim`` container
with no extra install. The scoring logic lives in :func:`compute_rewards` so it
can be unit-tested directly (see
``tests/unit_tests/test_harbor_task_reward.py``).

Step / tool-call counting matches the pytest definition: each ``source:
"agent"`` ATIF step corresponds to one ``AIMessage`` (one agent step), and
``tool_call_requests`` is the total number of tool calls across those steps.
"""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path
from typing import Any

TRAJECTORY_PATH = Path("/logs/agent/trajectory.json")
REWARD_PATH = Path("/logs/verifier/reward.json")
NAME_FILE = "/app/name.txt"

EXPECTED_NAME = "Foo Bar"
EXPECTED_STEPS = 2
EXPECTED_TOOL_CALLS = 1


def _agent_steps(trajectory: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the ``source: "agent"`` steps from an ATIF trajectory dict."""
    steps = trajectory.get("steps")
    if not isinstance(steps, list):
        return []
    return [s for s in steps if isinstance(s, dict) and s.get("source") == "agent"]


def _message_text(message: object) -> str:
    """Flatten an ATIF step ``message`` (str or list of content parts) to text."""
    if isinstance(message, str):
        return message
    if isinstance(message, list):
        return "".join(
            part["text"]
            for part in message
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        )
    return ""


def _final_text(agent_steps: list[dict[str, Any]]) -> str:
    """Return the text of the last agent step (the agent's final answer)."""
    if not agent_steps:
        return ""
    return _message_text(agent_steps[-1].get("message", ""))


def _tool_calls(agent_steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return every tool call across all agent steps."""
    return [
        tc for step in agent_steps for tc in (step.get("tool_calls") or []) if isinstance(tc, dict)
    ]


def compute_rewards(
    trajectory: object,
    files: object,
    *,
    expected_name: str = EXPECTED_NAME,
    expected_steps: int = EXPECTED_STEPS,
    expected_tool_calls: int = EXPECTED_TOOL_CALLS,
) -> dict[str, float]:
    """Compute the multi-key reward dict for the write-file-simple task.

    Args:
        trajectory: Parsed ATIF trajectory (``dict``); anything else is treated
            as an empty trajectory so a malformed/missing file scores 0 rather
            than crashing the verifier.
        files: Mapping of container path -> file contents. Only ``/app/name.txt``
            is consulted here.
        expected_name: The name the agent was told to write and report.
        expected_steps: Expected agent-step count (efficiency baseline).
        expected_tool_calls: Expected tool-call count (efficiency baseline).

    Returns:
        A ``dict[str, float]`` suitable for Harbor's ``reward.json``. ``reward``
        and ``correctness`` are the gated success metric; the rest are
        diagnostics (counts and efficiency ratios).
    """
    if not isinstance(trajectory, dict):
        trajectory = {}
    if not isinstance(files, dict):
        files = {}

    agent_steps = _agent_steps(trajectory)
    n_steps = len(agent_steps)
    tool_calls = _tool_calls(agent_steps)
    n_tool_calls = len(tool_calls)

    final_text = _final_text(agent_steps)
    name_file_contents = files.get(NAME_FILE, "")

    file_ok = expected_name in name_file_contents
    final_ok = expected_name in final_text
    wrote_name_file = any(
        tc.get("function_name") == "write_file" and "name.txt" in str(tc.get("arguments", ""))
        for tc in tool_calls
    )

    # Correctness mirrors the pytest .success() tier: the file must contain the
    # name AND the agent must report it back.
    correctness = 1.0 if (file_ok and final_ok) else 0.0

    rewards: dict[str, float] = {
        # Canonical scalar: keeps the existing `harbor_reward` feedback and
        # Harbor's default metric working unchanged.
        "reward": correctness,
        # Gated success metric (see [min_reward] in task.toml).
        "correctness": correctness,
        # Component success checks (diagnostics).
        "file_name_contains": 1.0 if file_ok else 0.0,
        "final_text_contains": 1.0 if final_ok else 0.0,
        "wrote_name_file": 1.0 if wrote_name_file else 0.0,
        # Raw trajectory shape (diagnostics).
        "agent_steps": float(n_steps),
        "tool_call_requests": float(n_tool_calls),
    }

    # Efficiency ratios mirror the pytest reporter (actual / expected). Only
    # emitted when the baseline is positive, to avoid a divide-by-zero.
    if expected_steps > 0:
        rewards["step_ratio"] = n_steps / expected_steps
    if expected_tool_calls > 0:
        rewards["tool_call_ratio"] = n_tool_calls / expected_tool_calls

    return rewards


def _load_trajectory() -> dict[str, Any]:
    """Read and parse the ATIF trajectory, degrading to ``{}`` on any error."""
    try:
        parsed = json.loads(TRAJECTORY_PATH.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"warning: could not read trajectory at {TRAJECTORY_PATH}: {exc}", file=sys.stderr)
        return {}
    if not isinstance(parsed, dict):
        print(f"warning: trajectory at {TRAJECTORY_PATH} is not a JSON object", file=sys.stderr)
        return {}
    return parsed


def _load_files() -> dict[str, str]:
    """Read the filesystem end-state the verifier inspects."""
    files: dict[str, str] = {}
    # Missing file is a valid (failing) outcome, not an error.
    with contextlib.suppress(OSError):
        files[NAME_FILE] = Path(NAME_FILE).read_text()
    return files


def main() -> int:
    """Compute rewards and write ``/logs/verifier/reward.json``."""
    rewards = compute_rewards(_load_trajectory(), _load_files())
    REWARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    REWARD_PATH.write_text(json.dumps(rewards, indent=2))
    print(json.dumps(rewards, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
