#!/usr/bin/env python3
"""Measure LangSmith sandbox snapshot build time, standalone (no Harbor trial).

Builds a verifier image snapshot from a deep-swe task's ``tests/`` Dockerfile
twice — once with a larger builder (``vcpus``), once with the SDK default —
and reports wall-clock time for each. This isolates the snapshot-build cost
that causes ``VerifierTimeoutError`` in the full pipeline, so we can validate
whether a bigger builder cuts the slow BuildKit layer extraction *without*
running an agent/trial.

Auth mirrors ``harbor.environments.langsmith.LangSmithEnvironment``: a
``SandboxClient`` keyed on ``LANGSMITH_SANDBOX_API_KEY`` with the workspace
headers from a ``langsmith.Client``. Secrets are read from the environment and
passed to the SDK only — never printed.

Usage::

    python snapshot_build_timing.py <task_tests_dir> <vcpus>
"""

from __future__ import annotations

import os
import sys
import time
import uuid
from pathlib import Path

from langsmith import Client
from langsmith.sandbox import SandboxClient

# Harbor's default snapshot filesystem capacity (32 GiB).
_FS_CAPACITY_BYTES = 32 * 1024 * 1024 * 1024
# Generous per-build cap so a slow default build isn't cut before we learn its
# cost; still bounded so the job can't hang forever.
_BUILD_TIMEOUT_SEC = 3000


def _make_client() -> tuple[SandboxClient, dict]:
    """Build a SandboxClient the way Harbor's LangSmithEnvironment does."""
    api_key = os.environ.get("LANGSMITH_SANDBOX_API_KEY") or os.environ.get(
        "LANGSMITH_API_KEY"
    )
    if not api_key:
        raise SystemExit("LANGSMITH_SANDBOX_API_KEY (or LANGSMITH_API_KEY) is required")

    endpoint = os.environ.get("LANGSMITH_ENDPOINT")
    ls_client = Client(api_key=api_key, api_url=endpoint) if endpoint else Client(
        api_key=api_key
    )
    ensure_profile_auth = getattr(ls_client, "_ensure_profile_auth", None)
    if callable(ensure_profile_auth):
        try:
            ensure_profile_auth()
        except Exception:
            pass
    raw_headers = getattr(ls_client, "_headers", None)
    headers = dict(raw_headers) if isinstance(raw_headers, dict) else {}

    client = SandboxClient(
        api_endpoint=os.environ.get("LANGSMITH_SANDBOX_API_URL"),
        api_key=api_key,
        headers=headers,
    )
    return client, headers


def _build_once(
    client: SandboxClient,
    headers: dict,
    *,
    tests_dir: Path,
    dockerfile: Path,
    label: str,
    vcpus: int | None,
) -> float:
    """Build a uniquely-named snapshot and return elapsed seconds. Deletes it."""
    name = f"snap-timing-{uuid.uuid4().hex[:12]}"
    print(f"\n=== build [{label}] (vcpus={vcpus}) name={name} ===", flush=True)

    def on_log(line: str) -> None:
        line = line.rstrip()
        if line:
            print(f"[{label}] {line}", flush=True)

    started = time.monotonic()
    try:
        client.create_snapshot_from_dockerfile(
            name,
            str(dockerfile),
            _FS_CAPACITY_BYTES,
            context=str(tests_dir),
            on_build_log=on_log,
            vcpus=vcpus,
            timeout=_BUILD_TIMEOUT_SEC,
            headers=headers,
        )
        elapsed = time.monotonic() - started
        print(f"=== [{label}] BUILT in {elapsed:.1f}s ===", flush=True)
        return elapsed
    finally:
        try:
            client.delete_snapshot(name, headers=headers)
            print(f"=== [{label}] cleaned up snapshot {name} ===", flush=True)
        except Exception as exc:  # best-effort cleanup
            print(f"=== [{label}] cleanup failed for {name}: {exc} ===", flush=True)


def main() -> int:
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <task_tests_dir> <vcpus>", file=sys.stderr)
        return 2

    tests_dir = Path(sys.argv[1]).resolve()
    try:
        vcpus = int(sys.argv[2])
    except ValueError:
        print(f"vcpus must be an integer, got: {sys.argv[2]!r}", file=sys.stderr)
        return 2
    if vcpus < 1:
        print(f"vcpus must be >= 1, got: {vcpus}", file=sys.stderr)
        return 2

    dockerfile = tests_dir / "Dockerfile"
    if not dockerfile.is_file():
        print(f"No Dockerfile at {dockerfile}", file=sys.stderr)
        return 1

    client, headers = _make_client()

    results: dict[str, float | None] = {}
    # Larger builder first so we get the key data point even if the default is slow.
    for label, n in ((f"vcpus={vcpus}", vcpus), ("default", None)):
        try:
            results[label] = _build_once(
                client,
                headers,
                tests_dir=tests_dir,
                dockerfile=dockerfile,
                label=label,
                vcpus=n,
            )
        except Exception as exc:
            results[label] = None
            print(f"=== [{label}] BUILD FAILED: {exc} ===", flush=True)

    print("\n================ SUMMARY ================", flush=True)
    print(f"task tests dir: {tests_dir}", flush=True)
    for label, elapsed in results.items():
        shown = f"{elapsed:.1f}s" if elapsed is not None else "FAILED/timeout"
        print(f"  {label:12s} -> {shown}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
