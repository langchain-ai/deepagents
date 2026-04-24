"""Optimization loop and CLI entry point."""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from better_harness.agent import run_outer_agent
from better_harness.core import Experiment, IterResult, SplitResult, load_experiment
from better_harness.runner import run_split
from better_harness.workspace import build_workspace, read_edited_harness, read_proposal


def run_optimization(
    experiment: Experiment,
    *,
    output_dir: Path,
    max_iterations: int | None = None,
    reuse_existing: bool = False,
) -> dict:
    """Run the optimization loop.

    Baseline then (propose, eval, accept/reject) x N, then final report.

    Acceptance criterion: new train passed count > current train passed count.
    Holdout is reported but never used for acceptance (prevents overfitting).

    Returns a summary dict (also written to report.json and report.md).
    """
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    iteration_limit = max_iterations if max_iterations is not None else experiment.max_iterations
    harness_path = experiment.harness_path

    # --- Baseline ---
    _log(f"baseline: {len(experiment.train_cases())} train + {len(experiment.holdout_cases())} holdout")
    baseline_train = run_split(
        experiment=experiment,
        harness_path=harness_path,
        split="train",
        output_dir=output_dir / "baseline",
        reuse_existing=reuse_existing,
    )
    baseline_holdout = run_split(
        experiment=experiment,
        harness_path=harness_path,
        split="holdout",
        output_dir=output_dir / "baseline",
        reuse_existing=reuse_existing,
    )
    _log(f"baseline  train {baseline_train.summary()}  holdout {baseline_holdout.summary()}")

    current_harness = harness_path
    current_train = baseline_train
    current_holdout = baseline_holdout
    history: list[IterResult] = []

    # --- Iteration loop ---
    for i in range(1, iteration_limit + 1):
        if current_train.passed == current_train.total:
            _log(f"iter {i}: all train cases passing — stopping early")
            break

        iter_dir = output_dir / f"iter-{i:03d}"
        workspace_dir = iter_dir / "workspace"

        _log(f"\niter {i}: building workspace ({current_train.passed}/{current_train.total} train passing)…")
        build_workspace(
            harness_path=current_harness,
            train_result=current_train,
            holdout_result=current_holdout,
            history=history,
            baseline_train=baseline_train,
            workspace_dir=workspace_dir,
        )

        _log(f"iter {i}: running outer agent…")
        run_outer_agent(experiment=experiment, workspace_dir=workspace_dir)

        # Read back the edited harness.
        edited = read_edited_harness(workspace_dir)
        candidate_path = iter_dir / "agent.py"
        candidate_path.write_text(edited)

        # If the outer agent made no changes, stop.
        if edited.strip() == current_harness.read_text().strip():
            _log(f"iter {i}: no changes made — stopping")
            break

        _log(f"iter {i}: evaluating candidate…")
        candidate_train, candidate_holdout, mean_train_passed = _eval_with_replays(
            experiment=experiment,
            candidate_path=candidate_path,
            iter_dir=iter_dir,
        )

        # Accept if mean train pass count improved (handles min_replays >= 1).
        prior_train = current_train
        accepted = mean_train_passed > current_train.passed

        # Simplicity tiebreaker (EvoForge): equal pass count + simpler code = accept.
        if not accepted and mean_train_passed == current_train.passed and len(edited.strip()) < len(current_harness.read_text().strip()):
            accepted = True
            _log(f"iter {i}: simplicity tiebreaker — equal performance, candidate is shorter")

        proposal = read_proposal(workspace_dir)
        iter_result = IterResult(
            iteration=i,
            accepted=accepted,
            train=candidate_train,
            holdout=candidate_holdout,
            proposal=proposal,
            prior_train=prior_train,
        )
        history.append(iter_result)
        _write_decision(iter_dir, iter_result)

        verdict = "ACCEPTED ✓" if accepted else "REJECTED ✗"
        _log(
            f"iter {i}: {verdict}  "
            f"train {candidate_train.summary()}  holdout {candidate_holdout.summary()}"
        )

        if accepted:
            current_harness = candidate_path
            current_train = candidate_train
            current_holdout = candidate_holdout

    # --- Final report ---
    summary = {
        "created_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "experiment": experiment.name,
        "baseline": {
            "train": baseline_train.summary(),
            "holdout": baseline_holdout.summary(),
        },
        "final": {
            "train": current_train.summary(),
            "holdout": current_holdout.summary(),
        },
        "iterations_run": len(history),
        "iterations_accepted": sum(1 for r in history if r.accepted),
        "history": [
            {
                "iteration": r.iteration,
                "accepted": r.accepted,
                "train": r.train.summary(),
                "holdout": r.holdout.summary(),
                "proposal": r.proposal[:400] if r.proposal else "",
            }
            for r in history
        ],
    }
    (output_dir / "report.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output_dir / "report.md").write_text(_render_report(summary))

    _log(f"\n{'='*50}")
    _log(f"baseline  train {summary['baseline']['train']}  holdout {summary['baseline']['holdout']}")
    _log(f"final     train {summary['final']['train']}  holdout {summary['final']['holdout']}")
    _log(f"report    {output_dir / 'report.md'}")

    return summary


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _eval_with_replays(
    *,
    experiment: Experiment,
    candidate_path: Path,
    iter_dir: Path,
) -> tuple[SplitResult, SplitResult, float]:
    """Run eval once (or N times if min_replays > 1).

    Returns (representative_train, representative_holdout, mean_train_passed).
    The representative results are from the median-performing replica.
    Mean is used for the acceptance decision; representative results go into history.
    """
    all_train: list[SplitResult] = []
    all_holdout: list[SplitResult] = []

    for rep in range(experiment.min_replays):
        suffix = "" if rep == 0 else f"-r{rep + 1:02d}"
        eval_dir = iter_dir / f"eval{suffix}"
        all_train.append(run_split(
            experiment=experiment,
            harness_path=candidate_path,
            split="train",
            output_dir=eval_dir,
        ))
        all_holdout.append(run_split(
            experiment=experiment,
            harness_path=candidate_path,
            split="holdout",
            output_dir=eval_dir,
        ))

    mean_train_passed = sum(r.passed for r in all_train) / len(all_train)

    # Pick the replica closest to the mean for display.
    best_idx = min(
        range(len(all_train)),
        key=lambda k: abs(all_train[k].passed - mean_train_passed),
    )
    return all_train[best_idx], all_holdout[best_idx], mean_train_passed


def _write_decision(iter_dir: Path, result: IterResult) -> None:
    data = {
        "iteration": result.iteration,
        "accepted": result.accepted,
        "train": result.train.summary(),
        "holdout": result.holdout.summary(),
        "proposal": result.proposal,
    }
    (iter_dir / "decision.json").write_text(json.dumps(data, indent=2) + "\n")


def _render_report(summary: dict) -> str:
    lines = [
        "# better-harness report",
        "",
        f"- Experiment: `{summary['experiment']}`",
        f"- Baseline:   train {summary['baseline']['train']}  |  holdout {summary['baseline']['holdout']}",
        f"- Final:      train {summary['final']['train']}  |  holdout {summary['final']['holdout']}",
        f"- Iterations: {summary['iterations_run']} run, {summary['iterations_accepted']} accepted",
        "",
        "## History",
        "",
    ]
    for r in summary["history"]:
        status = "✓ ACCEPTED" if r["accepted"] else "✗ REJECTED"
        lines.extend([
            f"### Iteration {r['iteration']} — {status}",
            f"Train: {r['train']}  |  Holdout: {r['holdout']}",
            "",
            r["proposal"] or "*(no proposal written)*",
            "",
        ])
    return "\n".join(lines)


def _log(message: str) -> None:
    print(message, flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Optimize an agent harness with an outer Deep Agent loop"
    )
    parser.add_argument("config", type=Path, help="Experiment TOML config path")
    parser.add_argument("--model", help="Override better_agent.model")
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args(argv)

    experiment = load_experiment(args.config, model_override=args.model)
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or Path("runs") / f"{experiment.name}-{timestamp}"
    run_optimization(
        experiment,
        output_dir=output_dir,
        max_iterations=args.max_iterations,
        reuse_existing=args.reuse_existing,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
