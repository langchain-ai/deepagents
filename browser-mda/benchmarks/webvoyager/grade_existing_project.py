"""Grade an experiment project whose runner died before it posted feedback.

`experiment_native.py` posts every grade in one pass *after* `asyncio.gather`
returns, so a run interrupted at 80% leaves a project full of finished, paid-for
agent traces carrying no `correct`, `blocked`, `steps` or `seconds` at all. The
agent work survives -- `runs.create` fires on the deployment and the server
finishes it whether or not the local poller is alive -- and the trace is already
bound to its example. Only the grading is lost, and grading is cheap.

So re-grade from the traces rather than re-running the agent:

    uv run --with "langsmith>=0.14.7" --with httpx python grade_existing_project.py \
        --project webvoyager-repl-native-1791635157

Idempotent: a run that already carries a `correct` feedback is skipped, so this
is safe to re-run and safe to point at a project the original runner did finish.
"""
from __future__ import annotations

import argparse
import os

from langsmith import Client

from experiment_native import grade


def answer_of(run) -> tuple[str, int]:
    """Final assistant text and message count, read off the trace."""
    msgs = (run.outputs or {}).get("messages") or []
    content = (msgs[-1] if msgs else {}).get("content")
    if isinstance(content, list):
        content = " ".join(b.get("text", "") for b in content if isinstance(b, dict))
    return (content or "").strip(), len(msgs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True, help="experiment project name")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    ls = Client()
    session = ls.read_project(project_name=a.project)
    runs = [r for r in ls.list_runs(project_id=session.id, is_root=True)]
    graded = {
        str(f.run_id)
        for f in ls.list_feedback(run_ids=[r.id for r in runs])
        if f.key == "correct"
    }
    examples = {str(e.id): e for e in ls.list_examples(dataset_id=session.reference_dataset_id)}

    posted = skipped = failed = 0
    for run in runs:
        if str(run.id) in graded:
            skipped += 1
            continue
        example = examples.get(str(run.reference_example_id))
        if run.status != "success" or example is None:
            failed += 1
            continue
        answer, steps = answer_of(run)
        # Wall-clock from the trace, not from the dead runner's timer.
        seconds = round((run.end_time - run.start_time).total_seconds(), 1) if run.end_time else 0.0
        try:
            scores = grade(example.inputs["ques"], example.inputs["web"], answer, example.outputs)
        except Exception as err:  # noqa: BLE001 - one bad grade must not stop the pass
            print(f"  judge error on {example.inputs.get('task_id')}: {str(err)[:90]}", flush=True)
            failed += 1
            continue
        if scores is None:
            failed += 1
            continue
        verdict = ("BLOCKED" if scores["blocked"] >= 0.5 else
                   "CORRECT" if scores["correct"] >= 0.5 else "INCORRECT")
        print(f"[{verdict:9}] {str(example.inputs.get('task_id')):26} steps={steps}", flush=True)
        if not a.dry_run:
            # Same four keys, same binary-verdict-plus-confidence shape as the
            # live runner, so a salvaged project reads identically to a clean one.
            ls.create_feedback(run.id, key="correct", score=float(scores["correct"] >= 0.5),
                               comment=verdict)
            ls.create_feedback(run.id, key="blocked", score=float(scores["blocked"] >= 0.5))
            ls.create_feedback(run.id, key="correct_confidence", score=round(scores["correct"], 3))
            ls.create_feedback(run.id, key="blocked_confidence", score=round(scores["blocked"], 3))
            ls.create_feedback(run.id, key="steps", score=steps)
            ls.create_feedback(run.id, key="seconds", score=seconds)
        posted += 1

    verb = "would grade" if a.dry_run else "graded"
    print(f"\n{verb} {posted}; skipped {skipped} already graded; {failed} ungradeable")


if __name__ == "__main__":
    main()
