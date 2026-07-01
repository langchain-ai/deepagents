#!/usr/bin/env python3
"""Branch-local workaround: fix Harbor's LangGraph agent venv-setup guard.

Harbor installs the LangGraph agent into each task sandbox by creating a venv
(`python3 -m venv`). It guards the `python3-venv` install with
`python3 -m venv --help`, which *passes* on minimal / emulation base images even
when venv *creation* fails because `ensurepip` is not bundled. So it skips the
`apt-get install python3-venv` / `apk add` fallback, then real venv creation
crashes and the agent never runs. On Terminal-Bench 2.1 this kills ~17 tasks at
setup (qemu-*, mailman, install-windows-3.11, extract-elf, db-wal-recovery,
path-tracing, ...) before the model does anything.

This patches the guard to test an *actual* throwaway venv creation, so on the
failing images it falls through to the existing apt/apk install branch. See
IDEAS.md "Iter-VENV". The proper fix belongs upstream in harbor; this keeps our
branch's CI runs unblocked. Idempotent and a no-op if harbor's internals change.
"""

from __future__ import annotations

import glob
import sys

OLD = "if python3 -m venv --help >/dev/null 2>&1; then "
NEW = (
    "if python3 -m venv /tmp/__harbor_venv_probe >/dev/null 2>&1; then "
    "rm -rf /tmp/__harbor_venv_probe; "
)

# Search common locations relative to the CI working directory (libs/evals).
_GLOBS = (
    ".venv/**/harbor/agents/installed/langgraph.py",
    "**/site-packages/harbor/agents/installed/langgraph.py",
)


def main() -> int:
    matches: list[str] = []
    for pattern in _GLOBS:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            break
    if not matches:
        print("patch_harbor_venv: harbor langgraph.py not found — skipping (no-op)")
        return 0

    path = matches[0]
    with open(path, encoding="utf-8") as handle:
        src = handle.read()

    if NEW in src:
        print(f"patch_harbor_venv: already patched ({path})")
        return 0
    if OLD not in src:
        print(
            "patch_harbor_venv: guard string not found "
            f"(harbor internals changed?) — skipping ({path})"
        )
        return 0

    with open(path, "w", encoding="utf-8") as handle:
        handle.write(src.replace(OLD, NEW))
    print(f"patch_harbor_venv: patched {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
