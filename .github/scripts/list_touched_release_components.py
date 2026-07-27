"""List release-please components touched by a PR's changed files.

Used by advisory workflows (e.g. release fan-out bypass warnings) that need
the component list without running the full fan-out gate.

Stdout JSON:

```json
{"components": ["deepagents-cli"], "bump_worthy": true}
```
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from check_lockfile_release_scope import (
    DEFAULT_CONFIG,
    is_bump_worthy,
    touched_components,
)


def main(title: str, changed: list[str], config_path: Path = DEFAULT_CONFIG) -> int:
    """Print `{"components": [...], "bump_worthy": bool}` to stdout.

    Returns:
        `0` on success, `2` on config error.
    """
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        print(f"::error::Could not read release-please config {config_path}: {e}", file=sys.stderr)
        return 2
    if not isinstance(config.get("packages"), dict) or not config["packages"]:
        print(f"::error::{config_path} has no non-empty 'packages' map", file=sys.stderr)
        return 2
    payload = {
        "components": touched_components(changed, config),
        "bump_worthy": is_bump_worthy(title, config),
    }
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "usage: list_touched_release_components.py <pr-title>  (changed files on stdin)",
            file=sys.stderr,
        )
        raise SystemExit(2)
    raise SystemExit(
        main(sys.argv[1], [line.strip() for line in sys.stdin if line.strip()])
    )
