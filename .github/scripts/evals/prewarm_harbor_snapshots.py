"""Pre-create LangSmith sandbox snapshots for a Harbor registry dataset.

Trials resolve snapshots by name, so building each one once before the shards
start keeps parallel trials from building the same snapshot concurrently.
Failures are reported but never fail the job: a trial still builds a missing
snapshot itself.

Selects tasks exactly as the run step does (manifest order, include globs, then
the n_tasks cap), and prepares each task's environment and separate verifier
environment. Compose environments run without a snapshot and are skipped.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import enumerate_tasks  # noqa: E402
import shard_matrix  # noqa: E402

if TYPE_CHECKING:
    from harbor.environments.langsmith import LangSmithEnvironment

ATTEMPTS = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--include-tasks", default="")
    parser.add_argument("--n-tasks", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout-sec", type=int, default=3600)
    args = parser.parse_args()
    if args.concurrency < 1 or args.timeout_sec < 1 or args.n_tasks < 0:
        parser.error("--concurrency and --timeout-sec must be >= 1, --n-tasks >= 0")
    return args


def select_task_names(args: argparse.Namespace) -> list[str]:
    names = enumerate_tasks.registry_task_names(args.dataset)
    globs = args.include_tasks.split()
    return shard_matrix.select_shard_tasks(names, globs, args.n_tasks, 1, 0)


def download_tasks(dataset: str, selected: list[str], out: Path) -> list[Path]:
    subprocess.run(["harbor", "download", dataset, "-o", str(out)], check=True)
    from harbor.models.task.task import Task

    wanted = set(selected)
    dirs = []
    for toml in sorted(out.rglob("task.toml")):
        task = Task(toml.parent)
        name = task.config.task.name if task.config.task else toml.parent.name
        if name in wanted:
            dirs.append(toml.parent)
    return dirs


def environments(
    task_dir: Path, scratch: Path
) -> list[tuple[str, LangSmithEnvironment]]:
    from harbor.environments.langsmith import LangSmithEnvironment
    from harbor.models.task.task import Task
    from harbor.models.task.verifier_mode import (
        resolve_verifier_environment_definition,
    )
    from harbor.models.trial.paths import TrialPaths

    task = Task(task_dir)
    specs = [("env", task.paths.environment_dir, task.config.environment)]
    if task.config.verifier.environment_mode == "separate":
        definition = resolve_verifier_environment_definition(
            task.config, task.paths, validate_inputs=False
        )
        if definition is not None:
            specs.append(("verifier", definition.directory, definition.config))

    envs = []
    for kind, env_dir, config in specs:
        paths = TrialPaths(scratch / f"{task.short_name}-{kind}")
        paths.mkdir()
        env = LangSmithEnvironment(
            environment_dir=env_dir,
            environment_name=task.short_name,
            session_id="prewarm",
            trial_paths=paths,
            task_env_config=config,
        )
        envs.append((f"{task.short_name}/{kind}", env))
    return envs


async def prepare(
    label: str, env: LangSmithEnvironment, sem: asyncio.Semaphore, timeout_sec: int
) -> str:
    async with sem:
        for attempt in range(1, ATTEMPTS + 1):
            start = time.monotonic()
            try:
                async with asyncio.timeout(timeout_sec):
                    name = await env.prepare_snapshot()
            except Exception as e:  # noqa: BLE001 - any failure falls back to trial-time build
                print(f"ERROR {label} attempt {attempt}: {type(e).__name__}: {e}")
                await asyncio.sleep(10 * attempt)
                continue
            if name is None:
                print(f"SKIP  {label} (compose)")
                return "skipped"
            print(f"READY {label} {name} ({time.monotonic() - start:.0f}s)")
            return "ready"
        return "failed"


def write_summary(results: dict[str, str]) -> None:
    failed = sorted(label for label, r in results.items() if r == "failed")
    counts = {
        r: list(results.values()).count(r) for r in ("ready", "skipped", "failed")
    }
    line = ", ".join(f"{k}={v}" for k, v in counts.items())
    print(f"DONE {line}")
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(summary, "a") as f:
            f.write(f"### Snapshot prewarm\n\n{line}\n")
            if failed:
                f.write(
                    "\nFailed (built at trial time instead): "
                    + ", ".join(failed)
                    + "\n"
                )


async def prepare_all(
    envs: list[tuple[str, LangSmithEnvironment]], concurrency: int, timeout_sec: int
) -> list[str]:
    sem = asyncio.Semaphore(concurrency)
    return await asyncio.gather(
        *(prepare(label, env, sem, timeout_sec) for label, env in envs)
    )


def main() -> None:
    args = parse_args()
    from harbor.environments.langsmith import LangSmithEnvironment

    if not hasattr(LangSmithEnvironment, "prepare_snapshot"):
        print("::notice::Installed Harbor lacks prepare_snapshot; skipping prewarm.")
        return
    selected = select_task_names(args)
    print(f"{len(selected)} tasks selected from {args.dataset}")
    with tempfile.TemporaryDirectory(prefix="prewarm-") as tmp:
        root = Path(tmp)
        task_dirs = download_tasks(args.dataset, selected, root / "tasks")
        envs = [e for d in task_dirs for e in environments(d, root / "trials")]
        outcomes = asyncio.run(prepare_all(envs, args.concurrency, args.timeout_sec))
    write_summary(dict(zip((label for label, _ in envs), outcomes)))


if __name__ == "__main__":
    main()
