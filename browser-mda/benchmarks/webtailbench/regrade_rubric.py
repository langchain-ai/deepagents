"""Re-score an existing WebTailBench experiment with the current grader.

The first facade smoke was scored with an asymmetric prompt -- the `false`
side carried a disqualifier the `true` side did not -- and every criterion
came back 0.16-0.49, hedging just under the threshold. A row whose answer met
all five criteria scored 0/5. Those scores are wrong, not merely noisy, so
they are replaced rather than left alongside the corrected ones.

Costs Jev calls only: the answers are already in LangSmith.
"""
from __future__ import annotations

import argparse
import os
import sys

from langsmith import Client

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grade_rubric import grade_rubric  # noqa: E402


def answer_of(run) -> str:
    messages = (run.outputs or {}).get("messages") or []
    if not messages:
        return ""
    content = messages[-1].get("content")
    if isinstance(content, list):
        content = " ".join(b.get("text", "") for b in content if isinstance(b, dict))
    return (content or "").strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    a = ap.parse_args()

    ls = Client()
    session = ls.read_project(project_name=a.project)
    rows = [r for r in ls.list_runs(project_id=session.id, is_root=True) if r.end_time]
    print(f"{len(rows)} completed rows in {a.project}", flush=True)

    changed = same = skipped = 0
    for run in rows:
        if not run.reference_example_id:
            skipped += 1
            continue
        example = ls.read_example(run.reference_example_id)
        answer = answer_of(run)
        rubric = (example.outputs or {}).get("rubric")
        if not answer or not rubric:
            skipped += 1
            continue

        scores = grade_rubric(example.inputs["ques"], answer, rubric)
        if scores is None:
            skipped += 1
            continue

        before = (run.feedback_stats or {}).get("rubric_score", {}).get("avg")
        for feedback in ls.list_feedback(run_ids=[run.id]):
            if feedback.key in ("rubric_score", "blocked"):
                ls.delete_feedback(feedback.id)
        ls.create_feedback(run.id, key="rubric_score", score=round(scores["score"], 3),
                           comment=f"{scores['met']}/{scores['total']}")
        ls.create_feedback(run.id, key="blocked", score=float(scores["blocked"] >= 0.5))

        moved = before is None or abs(before - scores["score"]) > 1e-6
        changed += moved
        same += not moved
        print(f"  {example.inputs['task_id']:30} {before} -> "
              f"{scores['met']}/{scores['total']}" + ("  CHANGED" if moved else ""), flush=True)

    print(f"\nchanged {changed}, unchanged {same}, skipped {skipped}", flush=True)


if __name__ == "__main__":
    main()
