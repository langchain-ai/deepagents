#!/usr/bin/env python3
"""Patch DeepSWE task.toml files for Harbor compatibility.

DeepSWE tasks ship two Pier-convention artifacts that Harbor doesn't
auto-discover:

1. ``pre_artifacts.sh`` — a script that captures the agent's committed work
   as ``/logs/artifacts/model.patch`` (a ``git diff`` between the task's
   base commit and the agent's final HEAD). Pier runs this automatically;
   Harbor's equivalent is a ``[[verifier.collect]]`` hook that runs a shell
   command in the agent container after the agent phase ends.

2. ``allow_internet = false`` on ``[environment]`` and
   ``[verifier.environment]`` — air-gaps the sandbox. The LangGraph agent
   runs *inside* the sandbox and must call the model provider, so the agent
   phase needs network. Harbor supports per-phase overrides via
   ``[agent] network_mode = "public"`` (the verifier env stays air-gapped,
   preserving grading integrity).

This script reads each ``tasks/*/task.toml``, extracts
``[metadata] base_commit_hash``, and injects both fixes idempotently.

Usage::

    python .github/scripts/patch_deepswe_tasks.py <tasks_dir>

    # e.g.
    python .github/scripts/patch_deepswe_tasks.py deep-swe/tasks
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import tomllib


def _collect_command(base_commit: str) -> str:
    """Shell command that reproduces pre_artifacts.sh as a collect hook."""
    return (
        "cd /app || exit 0; "
        "mkdir -p /logs/artifacts; "
        "git config --global --add safe.directory /app 2>/dev/null || true; "
        f"git diff --binary {base_commit} HEAD > /logs/artifacts/model.patch "
        "2>/dev/null || true; "
        'echo "[pre_artifacts] captured $(wc -c < /logs/artifacts/model.patch) bytes"'
    )


def _has_network_mode(content: str) -> bool:
    """True if [agent] already declares network_mode."""
    # Match network_mode inside the [agent] section (before [verifier] or EOF).
    agent_section = re.search(
        r"\[agent\]\n(.*?)(?=\n\[|\Z)", content, re.DOTALL
    )
    if agent_section is None:
        return False
    return "network_mode" in agent_section.group(1)


def _has_collect_hook(data: dict) -> bool:
    """True if [verifier] already has a [[verifier.collect]] entry."""
    verifier = data.get("verifier", {})
    collect = verifier.get("collect", [])
    return bool(collect)


def patch_task_toml(path: Path) -> str:
    """Patch a single task.toml. Returns a status string."""
    content = path.read_text()
    data = tomllib.loads(content)

    metadata = data.get("metadata", {})
    base_commit = metadata.get("base_commit_hash")
    if not base_commit:
        return f"SKIP (no base_commit_hash): {path.name}"

    changed = False

    # 1. Inject [agent] network_mode = "public" if not present.
    if not _has_network_mode(content):
        agent_match = re.search(r"\[agent\]\n", content)
        if agent_match:
            # [agent] section exists — insert right after the header.
            content = content.replace(
                agent_match.group(0),
                agent_match.group(0) + 'network_mode = "public"\n',
                1,
            )
        else:
            # No [agent] section — create one before [verifier] or at end.
            agent_block = '[agent]\nnetwork_mode = "public"\n'
            verifier_match = re.search(r"\[verifier\]", content)
            if verifier_match:
                content = content.replace(
                    verifier_match.group(0),
                    agent_block + "\n" + verifier_match.group(0),
                    1,
                )
            else:
                content += "\n" + agent_block
        changed = True

    # 2. Inject [[verifier.collect]] hook if not present.
    if not _has_collect_hook(data):
        collect_block = (
            f"\n[[verifier.collect]]\n"
            f"command = '''{_collect_command(base_commit)}'''\n"
        )
        # Ensure file ends with newline before appending.
        if not content.endswith("\n"):
            content += "\n"
        content += collect_block
        changed = True

    if changed:
        path.write_text(content)
        return f"PATCHED: {path.name}"
    return f"OK (already patched): {path.name}"


def main() -> int:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <tasks_dir>", file=sys.stderr)
        return 2

    tasks_dir = Path(sys.argv[1])
    if not tasks_dir.is_dir():
        print(f"Error: {tasks_dir} is not a directory", file=sys.stderr)
        return 1

    task_tomls = sorted(tasks_dir.glob("*/task.toml"))
    if not task_tomls:
        print(f"Error: no task.toml files found under {tasks_dir}", file=sys.stderr)
        return 1

    patched = 0
    skipped = 0
    for toml_path in task_tomls:
        status = patch_task_toml(toml_path)
        print(status)
        if status.startswith("PATCHED"):
            patched += 1
        else:
            skipped += 1

    print(f"\nDone: {patched} patched, {skipped} skipped (of {len(task_tomls)} total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
