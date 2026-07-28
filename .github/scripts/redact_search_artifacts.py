#!/usr/bin/env python3
"""Strip benchmark plaintext from a Harbor job tree before it is uploaded.

LoHoSearch questions and answers are decrypted at runtime and its canary forbids
republishing them. This repo is public, so shard artifacts are world-readable and
anything carrying plaintext must not reach one.

The redaction is a **key-level allowlist**, not a file filter. An earlier
file-filter version kept every file named ``result.json`` and leaked
``<trial>/agent/result.json`` -- the full message transcript, which shares that
basename with the trial record the aggregator needs. Filtering by name or by
directory only holds until Harbor adds a new path, so instead each surviving
trial record is rebuilt from a fixed set of fields and everything else is
deleted. A field Harbor adds later is dropped by default rather than published.

Kept per trial record (exactly what ``aggregate_shards.py`` reads):
  task_name, verifier_result.rewards.reward, config.agent.model_name,
  config.job_id, and ``exception_info.exception_type``.

``exception_info`` is reduced to its type: the message and traceback can quote
task content, and the aggregator only tests the field for presence.

Usage:
    python3 redact_search_artifacts.py <job-root>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Marker files the aggregator counts to distinguish an empty shard from a
# missing one. Names only, no content.
_EMPTY_SHARD_PREFIX = "empty-shard-"


def safe_record(result: dict[str, Any]) -> dict[str, Any]:
    """Rebuild one trial record from the fields the aggregator reads.

    Args:
        result: A parsed Harbor `result.json`.

    Returns:
        A new dict containing only allowlisted fields. Unknown fields, the agent
        transcript, and exception messages/tracebacks are dropped.
    """
    rewards = (result.get("verifier_result") or {}).get("rewards") or {}
    config = result.get("config") or {}
    agent = config.get("agent") or {}

    safe: dict[str, Any] = {}
    if result.get("task_name") is not None:
        safe["task_name"] = result["task_name"]
    if "reward" in rewards:
        safe["verifier_result"] = {"rewards": {"reward": rewards["reward"]}}
    if exception_info := result.get("exception_info"):
        # Presence is what `trial_errored` tests; the type aids triage. The
        # message and traceback are dropped -- they can quote task content.
        exception_type = (
            exception_info.get("exception_type")
            if isinstance(exception_info, dict)
            else None
        )
        safe["exception_info"] = {"exception_type": exception_type or "unknown"}

    safe_config: dict[str, Any] = {}
    if job_id := config.get("job_id"):
        safe_config["job_id"] = job_id
    if model_name := agent.get("model_name"):
        safe_config["agent"] = {"model_name": model_name}
    if safe_config:
        safe["config"] = safe_config
    return safe


def redact(root: Path) -> tuple[int, int]:
    """Rewrite trial records in place and delete everything else under `root`.

    Args:
        root: The Harbor job directory that is about to be uploaded.

    Returns:
        A `(rewritten, deleted)` count.
    """
    if not root.is_dir():
        return (0, 0)

    rewritten = deleted = 0
    for path in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if path.is_dir():
            continue
        if path.name.startswith(_EMPTY_SHARD_PREFIX):
            continue
        if path.name == "result.json":
            try:
                parsed = json.loads(path.read_text())
            except (OSError, ValueError):
                parsed = None
            # A record with no task_name is the job-level summary or, in the
            # agent directory, the transcript. Neither is needed, and only the
            # latter is dangerous -- drop both rather than distinguish them.
            if isinstance(parsed, dict) and parsed.get("task_name"):
                path.write_text(json.dumps(safe_record(parsed), indent=2) + "\n")
                rewritten += 1
                continue
        path.unlink(missing_ok=True)
        deleted += 1

    for directory in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if directory.is_dir() and not any(directory.iterdir()):
            directory.rmdir()
    return (rewritten, deleted)


def main(argv: list[str] | None = None) -> int:
    """Redact a Harbor job tree in place.

    Args:
        argv: Command-line arguments, excluding the program name.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Harbor job directory to redact")
    args = parser.parse_args(argv)

    rewritten, deleted = redact(args.root)
    print(f"Redacted {args.root}: rewrote {rewritten} trial record(s), deleted {deleted} file(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
