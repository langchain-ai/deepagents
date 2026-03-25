#!/usr/bin/env python3
"""Harbor workflow runner — single source of truth for CI and local runs.

Wraps the full lifecycle: credential verification, LangSmith experiment
creation, ``harbor run``, reward feedback push, and summary output.
When running in GitHub Actions, writes outputs to ``$GITHUB_OUTPUT`` and a
markdown summary to ``$GITHUB_STEP_SUMMARY``.

Usage:
    uv run python scripts/run_harbor.py --model anthropic:claude-sonnet-4-6
    uv run python scripts/run_harbor.py --model openai:gpt-4.1 --env daytona -n 10 --n-tasks 5
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Credential mappings (mirrors CI's verification step)
# ---------------------------------------------------------------------------

PROVIDER_API_KEYS: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "baseten": "BASETEN_API_KEY",
    "fireworks": "FIREWORKS_API_KEY",
    "google_genai": "GOOGLE_API_KEY",
    "groq": "GROQ_API_KEY",
    "nvidia": "NVIDIA_API_KEY",
    "ollama": "OLLAMA_API_KEY",
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "xai": "XAI_API_KEY",
}

SANDBOX_CREDENTIALS: dict[str, list[str]] = {
    "docker": [],
    "daytona": ["DAYTONA_API_KEY"],
    "langsmith": [],
    "modal": ["MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"],
    "runloop": ["RUNLOOP_API_KEY"],
}

DATASET_NAME = "terminal-bench"
DATASET_VERSION = "2.0"
JOBS_DIR = Path("jobs/terminal-bench")


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


def verify_credentials(model: str, sandbox_env: str) -> None:
    """Check that required env vars are set for the provider and sandbox.

    Raises:
        SystemExit: If any required credentials are missing.
    """
    missing: list[str] = []

    provider = model.split(":")[0]
    env_var = PROVIDER_API_KEYS.get(provider)
    if env_var and not os.environ.get(env_var):
        missing.append(env_var)

    for var in SANDBOX_CREDENTIALS.get(sandbox_env, []):
        if not os.environ.get(var):
            missing.append(var)

    if missing:
        print(
            f"Missing credentials for {provider}/{sandbox_env}: {', '.join(missing)}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    print(f"Credentials verified for {provider}/{sandbox_env}")


def create_langsmith_experiment(model: str) -> str | None:
    """Create a LangSmith experiment session if `LANGSMITH_API_KEY` is set.

    Returns:
        The experiment name, or ``None`` if LangSmith is not configured.
    """
    if not os.environ.get("LANGSMITH_API_KEY"):
        print("LANGSMITH_API_KEY not set — skipping experiment creation")
        return None

    from deepagents_harbor.langsmith import create_experiment

    name = create_experiment(dataset_name=DATASET_NAME, model=model)
    print(f"LangSmith experiment: {name}")
    return name


def suppress_harbor_tips() -> None:
    """Write the Harbor notifications cache to suppress first-run hints."""
    cache_dir = Path.home() / ".cache" / "harbor"
    cache_dir.mkdir(parents=True, exist_ok=True)
    notifications = cache_dir / "notifications.json"
    notifications.write_text(json.dumps({"seen": ["registry-datasets-hint"]}))


def run_harbor(
    model: str,
    sandbox_env: str,
    concurrency: int,
    n_tasks: int,
    agent_mode: str,
    experiment_name: str | None,
) -> int:
    """Execute ``harbor run`` as a subprocess, streaming output.

    Returns:
        The subprocess return code.
    """
    cmd: list[str] = [
        "uv",
        "run",
        "harbor",
        "run",
        "--agent-import-path",
        "deepagents_harbor:DeepAgentsWrapper",
        "--dataset",
        f"{DATASET_NAME}@{DATASET_VERSION}",
        "-n",
        str(concurrency),
        "--jobs-dir",
        str(JOBS_DIR),
        "--env",
        sandbox_env,
        "--model",
        model,
        "--agent-kwarg",
        f"use_cli_agent={'true' if agent_mode == 'cli' else 'false'}",
    ]
    if n_tasks > 0:
        cmd.extend(["--n-tasks", str(n_tasks)])

    env = os.environ.copy()
    if experiment_name:
        env["LANGSMITH_EXPERIMENT"] = experiment_name

    print(f"\n{'=' * 60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'=' * 60}\n")

    result = subprocess.run(cmd, env=env, check=False)
    return result.returncode


def find_latest_job() -> Path | None:
    """Return the most recently created job directory, or ``None``."""
    if not JOBS_DIR.exists():
        return None
    dirs = sorted(p for p in JOBS_DIR.iterdir() if p.is_dir())
    return dirs[-1] if dirs else None


def push_feedback(job_dir: Path, experiment_name: str) -> None:
    """Push Harbor reward scores to LangSmith."""
    from deepagents_harbor.langsmith import add_feedback

    print(f"\nPushing feedback from {job_dir} to experiment {experiment_name}")
    add_feedback(job_folder=job_dir, project_name=experiment_name)


def write_github_output(experiment_name: str | None, job_dir: Path | None) -> None:
    """Write step outputs for downstream CI steps (artifact upload, etc.).

    Only writes when ``$GITHUB_OUTPUT`` is set (i.e. inside GitHub Actions).
    """
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        return
    with open(path, "a") as fh:  # noqa: PTH123
        if experiment_name:
            fh.write(f"experiment_name={experiment_name}\n")
        if job_dir:
            fh.write(f"job_dir={job_dir}\n")


def write_summary(
    model: str,
    sandbox_env: str,
    concurrency: int,
    n_tasks: int,
    agent_mode: str,
    experiment_name: str | None,
    job_dir: Path | None,
    harbor_rc: int,
) -> None:
    """Print a run summary to stdout and, in CI, to ``$GITHUB_STEP_SUMMARY``."""
    tasks_label = "all" if n_tasks == 0 else str(n_tasks)
    status = "SUCCESS" if harbor_rc == 0 else f"FAILED (exit {harbor_rc})"

    # -- stdout (always) --
    print(f"\n{'=' * 60}")
    print("Harbor run summary")
    print(f"{'=' * 60}")
    print(f"  Model:        {model}")
    print(f"  Dataset:      {DATASET_NAME}@{DATASET_VERSION}")
    print(f"  Sandbox:      {sandbox_env}")
    print(f"  Concurrency:  {concurrency}")
    print(f"  Max tasks:    {tasks_label}")
    print(f"  Agent mode:   {agent_mode}")
    if experiment_name:
        print(f"  Experiment:   {experiment_name}")
    if job_dir:
        print(f"  Job dir:      {job_dir}")
    print(f"  Status:       {status}")
    print(f"{'=' * 60}")

    # -- $GITHUB_STEP_SUMMARY (CI only) --
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return
    lines = [
        "## Harbor run",
        "",
        f"- Model: {model}",
        f"- Dataset: {DATASET_NAME}@{DATASET_VERSION}",
        f"- Sandbox: {sandbox_env}",
        f"- Concurrency: {concurrency}",
        f"- Max tasks: {tasks_label}",
        f"- Agent mode: {agent_mode}",
        f"- LangSmith experiment: {experiment_name or 'N/A'}",
    ]
    if job_dir:
        lines.append(f"- Harbor job dir: {job_dir}")
    lines.append("")
    with open(summary_path, "a") as fh:  # noqa: PTH123
        fh.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the full Harbor workflow locally (mirrors CI).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s --model anthropic:claude-sonnet-4-6\n"
            "  %(prog)s --model openai:gpt-4.1 --env daytona --concurrency 10\n"
            "  %(prog)s --model anthropic:claude-opus-4-6 --n-tasks 5\n"
        ),
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model to evaluate (e.g. 'anthropic:claude-sonnet-4-6')",
    )
    parser.add_argument(
        "--env",
        default="docker",
        choices=sorted(SANDBOX_CREDENTIALS),
        help="Harbor sandbox environment (default: docker)",
    )
    parser.add_argument(
        "--concurrency",
        "-n",
        type=int,
        default=1,
        help="Number of concurrent trials / parallel sandbox slots (default: 1)",
    )
    parser.add_argument(
        "--n-tasks",
        type=int,
        default=0,
        help="Maximum number of tasks to run; 0 = all (default: 0)",
    )
    parser.add_argument(
        "--agent-mode",
        default="cli",
        choices=["cli", "sdk"],
        help="Agent implementation to use (default: cli)",
    )
    return parser.parse_args()


def main() -> int:
    """Orchestrate the full Harbor workflow."""
    args = parse_args()

    # 1. Verify credentials
    verify_credentials(args.model, args.env)

    # 2. Create LangSmith experiment (optional)
    experiment_name = create_langsmith_experiment(args.model)

    # 3. Suppress first-run tips
    suppress_harbor_tips()

    # 4. Run Harbor
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    harbor_rc = run_harbor(
        model=args.model,
        sandbox_env=args.env,
        concurrency=args.concurrency,
        n_tasks=args.n_tasks,
        agent_mode=args.agent_mode,
        experiment_name=experiment_name,
    )

    # 5. Find latest job dir
    job_dir = find_latest_job()

    # 6. Push feedback (even if harbor failed — matches CI behavior)
    if experiment_name and job_dir:
        try:
            push_feedback(job_dir, experiment_name)
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: failed to push feedback: {exc}", file=sys.stderr)

    # 7. CI outputs (no-op locally)
    write_github_output(experiment_name, job_dir)

    # 8. Summary (stdout always, $GITHUB_STEP_SUMMARY in CI)
    write_summary(
        model=args.model,
        sandbox_env=args.env,
        concurrency=args.concurrency,
        n_tasks=args.n_tasks,
        agent_mode=args.agent_mode,
        experiment_name=experiment_name,
        job_dir=job_dir,
        harbor_rc=harbor_rc,
    )

    return harbor_rc


if __name__ == "__main__":
    sys.exit(main())
