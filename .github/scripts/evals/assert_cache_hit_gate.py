#!/usr/bin/env python3
"""Fail if the fork leg didn't show a real prompt-cache advantage over isolated.

Sums each leg's per-lens first-call `cache_read` (excluding the parent's own
"main" bucket) and asserts fork's total is both non-zero and strictly greater
than isolated's -- the concrete claim this whole demo exists to support.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def total_first_hit(payload: dict[str, Any]) -> int:
    return sum(
        (bucket.get("first_call_cache_read") or 0)
        for name, bucket in payload.get("per_agent", {}).items()
        if name != "main"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fork-json", required=True)
    parser.add_argument("--isolated-json", required=True)
    args = parser.parse_args()

    fork = json.loads(Path(args.fork_json).read_text())
    isolated = json.loads(Path(args.isolated_json).read_text())

    fork_total = total_first_hit(fork)
    isolated_total = total_first_hit(isolated)

    print(f"fork first-call cache_read total: {fork_total}")
    print(f"isolated first-call cache_read total: {isolated_total}")

    if fork_total <= 0:
        print(
            "::error::fork leg showed zero cache_read on any lens's first call -- "
            "the fork mechanism isn't reusing the parent's cached prefix."
        )
        sys.exit(1)
    if fork_total <= isolated_total:
        print(
            f"::error::fork leg's cache_read ({fork_total}) did not exceed "
            f"isolated's ({isolated_total}) -- no demonstrated cache advantage."
        )
        sys.exit(1)

    print("Cache-hit gate passed: fork showed a real, non-zero cache advantage over isolated.")


if __name__ == "__main__":
    main()
