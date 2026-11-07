"""Grade rows from a local suite that was cut short.

`bench_local.py` grades and posts feedback after the last task, so a suite
killed partway leaves finished rows with native Cost and Tokens and no scores.
The answers are already in each run's outputs, so recovering them costs a Jev
call and nothing else -- no browser, no model, no re-run.

Idempotent: a row that already carries a `correct` score is skipped.
"""

from __future__ import annotations

import argparse
import os
import sys

from langsmith import Client

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiment_native import grade  # noqa: E402


def answer_of(run) -> tuple[str, int]:
    messages = (run.outputs or {}).get("messages") or []
    if not messages:
        return "", 0
    content = messages[-1].get("content") if isinstance(messages[-1], dict) else None
    if isinstance(content, list):
        content = " ".join(b.get("text", "") for b in content if isinstance(b, dict))
    return (content or "").strip(), len(messages)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    args = ap.parse_args()

    ls = Client()
    session = ls.read_project(project_name=args.project)
    rows = list(ls.list_runs(project_id=session.id, is_root=True))
    print(f"{len(rows)} rows in {args.project}", flush=True)

    examples: dict[str, object] = {}
    posted = skipped = ungraded = 0
    for run in rows:
        if (run.feedback_stats or {}).get("correct"):
            skipped += 1
            continue
        if run.error or not run.reference_example_id:
            ungraded += 1
            continue

        key = str(run.reference_example_id)
        if key not in examples:
            examples[key] = ls.read_example(run.reference_example_id)
        example = examples[key]

        answer, steps = answer_of(run)
        seconds = round((run.end_time - run.start_time).total_seconds(), 1) if run.end_time else 0
        if not answer:
            ungraded += 1
            continue

        try:
            scores = grade(example.inputs["ques"], example.inputs["web"], answer, example.outputs)
        except Exception as err:  # noqa: BLE001 - one bad grade must not stop the backfill
            print(f"  judge error on {example.inputs.get('task_id')}: {str(err)[:90]}", flush=True)
            ungraded += 1
            continue

        verdict = (
            "BLOCKED" if scores["blocked"] >= 0.5
            else "CORRECT" if scores["correct"] >= 0.5 else "INCORRECT"
        )
        ls.create_feedback(run.id, key="correct",
                           score=float(scores["correct"] >= 0.5), comment=verdict)
        ls.create_feedback(run.id, key="blocked", score=float(scores["blocked"] >= 0.5))
        ls.create_feedback(run.id, key="correct_confidence", score=round(scores["correct"], 3))
        ls.create_feedback(run.id, key="blocked_confidence", score=round(scores["blocked"], 3))
        ls.create_feedback(run.id, key="steps", score=steps)
        ls.create_feedback(run.id, key="seconds", score=seconds)
        posted += 1
        print(f"[{verdict:9}] {str(example.inputs.get('task_id')):26} "
              f"{seconds:6.0f}s steps={steps}", flush=True)

    print(f"\nposted {posted}, already graded {skipped}, ungradeable {ungraded}", flush=True)
    print(f"project id: {session.id}", flush=True)


if __name__ == "__main__":
    main()
