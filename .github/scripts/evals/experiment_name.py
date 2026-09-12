#!/usr/bin/env python3
"""Canonical LangSmith experiment (project) name for one eval leaf.

Single source of truth shared by three call sites so the name can be computed
*up front* without carrying it back through shard artifacts:

- ``_harbor_run.yml`` — sets ``HARBOR_LANGSMITH_EXPERIMENT`` (the project each
  leaf's rollouts are logged to).
- ``unified_prep.py`` — lists the experiment names the usage collector queries.
- ``aggregate_unified.py`` — maps each leaderboard row's usage back to its leaf.

Because all three call ``experiment_name`` with the same leaf identity, the
logged name and the queried name cannot drift. Mirrors the sanitization the
workflow previously did in shell (``tr -c '[:alnum:]._-' '-'`` + ``sha256`` of
the raw branch), so names are stable across the migration.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re

_DISALLOWED = re.compile(r"[^A-Za-z0-9._-]")


def _sanitize(value: str) -> str:
    """Replace each character outside ``[A-Za-z0-9._-]`` with ``-`` (no collapsing)."""
    return _DISALLOWED.sub("-", value)


def experiment_name(
    *,
    model: str,
    branch: str,
    config: str,
    category: str | None,
    run_id: str,
    run_attempt: str,
) -> str:
    """Return the deterministic experiment name for one leaf.

    Shape:
    ``deepagents-harbor-{branch}-{sha8}-{config}-{model}[-{category}]-{run_id}-{run_attempt}``
    where branch/config/model/category are sanitized and ``sha8`` is the first 8
    hex chars of ``sha256(branch)`` (of the raw, unsanitized branch).
    """
    branch_slug = f"{_sanitize(branch)}-{hashlib.sha256(branch.encode('utf-8')).hexdigest()[:8]}"
    name = f"deepagents-harbor-{branch_slug}-{_sanitize(config)}-{_sanitize(model)}"
    if category:
        name += f"-{_sanitize(category)}"
    return f"{name}-{run_id}-{run_attempt}"


def main(argv: list[str] | None = None) -> int:
    """CLI: read the leaf identity from HARBOR_*/GITHUB_* env, print the name.

    Reading from the environment (rather than interpolating values into a shell
    command) keeps the derivation injection-safe. Env values may be overridden by
    flags for testing.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.getenv("HARBOR_MODEL", ""))
    parser.add_argument("--branch", default=os.getenv("HARBOR_BRANCH", ""))
    parser.add_argument("--config", default=os.getenv("HARBOR_AGENT_IMPL", ""))
    parser.add_argument("--category", default=os.getenv("HARBOR_CATEGORY", ""))
    parser.add_argument("--run-id", default=os.getenv("GITHUB_RUN_ID", ""))
    parser.add_argument("--run-attempt", default=os.getenv("GITHUB_RUN_ATTEMPT", ""))
    args = parser.parse_args(argv)

    print(
        experiment_name(
            model=args.model,
            branch=args.branch,
            config=args.config,
            category=args.category,
            run_id=args.run_id,
            run_attempt=args.run_attempt,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
