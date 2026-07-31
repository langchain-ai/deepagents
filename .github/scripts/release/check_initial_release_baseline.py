"""Flag new release-please packages with a `0.0.1` manifest baseline.

`.release-please-manifest.json` stores the latest released version baseline. If a
new package is added there as `0.0.1`, release-please treats `0.0.1` as already
released and opens the first release PR for `0.0.2` after a qualifying commit.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

BAD_INITIAL_BASELINE = "0.0.1"
RECOMMENDED_INITIAL_BASELINE = "0.0.0"


def _load_json(path: Path) -> dict[str, object]:
    """Read a JSON object from `path`.

    Args:
        path: File path to read.

    Returns:
        Parsed JSON object.

    Raises:
        TypeError: If the file does not contain a JSON object.
        ValueError: If the file cannot be read or is invalid JSON.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        msg = f"Could not read JSON object from {path}: {exc}"
        raise ValueError(msg) from exc
    if not isinstance(data, dict):
        msg = f"Expected JSON object in {path}"
        raise TypeError(msg)
    return data


def _packages(config: dict[str, object]) -> dict[str, object]:
    """Return the release-please `packages` map from `config`."""
    packages = config.get("packages", {})
    if not isinstance(packages, dict):
        msg = "release-please configs must contain a 'packages' object"
        raise TypeError(msg)
    return {str(path): meta for path, meta in packages.items()}


def new_bad_baseline_packages(
    *,
    base_config: dict[str, object],
    base_manifest: dict[str, object],
    head_config: dict[str, object],
    head_manifest: dict[str, object],
) -> list[str]:
    """Return new components whose manifest baseline is `0.0.1`.

    Args:
        base_config: Base branch `release-please-config.json`.
        base_manifest: Base branch `.release-please-manifest.json`.
        head_config: PR head `release-please-config.json`.
        head_manifest: PR head `.release-please-manifest.json`.

    Returns:
        Sorted component names whose package path is newly present in the config
        or manifest and whose head manifest version is `0.0.1`.
    """
    base_packages = _packages(base_config)
    head_packages = _packages(head_config)

    new_paths = {
        path
        for path in set(head_packages) | set(head_manifest)
        if path not in base_packages or path not in base_manifest
    }

    components: list[str] = []
    for path in new_paths:
        if head_manifest.get(path) != BAD_INITIAL_BASELINE:
            continue
        meta = head_packages.get(path, {})
        component = meta.get("component", path) if isinstance(meta, dict) else path
        components.append(str(component))
    return sorted(components)


def main(
    base_config_path: Path,
    base_manifest_path: Path,
    head_config_path: Path,
    head_manifest_path: Path,
) -> int:
    """Print offending components as a JSON array.

    Returns:
        `0` on successful analysis. Offenders are emitted as a JSON array on
            stdout (with a human-readable note on stderr) and the calling
            workflow decides whether to fail.
        `2` on invalid inputs.
    """
    try:
        offenders = new_bad_baseline_packages(
            base_config=_load_json(base_config_path),
            base_manifest=_load_json(base_manifest_path),
            head_config=_load_json(head_config_path),
            head_manifest=_load_json(head_manifest_path),
        )
    except (TypeError, ValueError) as exc:
        print(f"::error::{exc}", file=sys.stderr)
        return 2

    if offenders:
        print(
            f"New package manifest baseline must not be {BAD_INITIAL_BASELINE}: "
            + ", ".join(offenders),
            file=sys.stderr,
        )
    print(json.dumps(offenders))
    return 0


if __name__ == "__main__":
    expected_args = 5
    if len(sys.argv) != expected_args:
        print(
            "usage: check_initial_release_baseline.py <base-config> "
            "<base-manifest> <head-config> <head-manifest>",
            file=sys.stderr,
        )
        raise SystemExit(2)
    raise SystemExit(main(*(Path(arg) for arg in sys.argv[1:])))
