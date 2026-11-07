"""Rewrite raw Jev probabilities on an experiment into binary verdicts.

Early runs posted Jev's confidence straight into `correct` and `blocked`, so a
project average mixed pass rate with judge certainty -- a row the judge scored
0.56 counted as a little over half a success. The experiment_native.py grader
now posts the verdict (0 or 1) under those keys and keeps the probability under
`*_confidence`; this brings already-finished projects to the same shape.

Idempotent: a project whose `correct` scores are all 0/1 and that already has
`correct_confidence` is left alone.

    uv run --with langsmith python backfill_binary_grades.py <project-id> ...
    uv run --with langsmith python backfill_binary_grades.py --apply <project-id>
"""
from __future__ import annotations

import argparse

from langsmith import Client

KEYS = ("correct", "blocked")
THRESHOLD = 0.5


def backfill(ls: Client, project_id: str, apply: bool) -> None:
    project = ls.read_project(project_id=project_id)
    run_ids = [r.id for r in ls.list_runs(project_id=project.id, is_root=True)]
    if not run_ids:
        print(f"{project.name}: no runs")
        return
    feedback = list(ls.list_feedback(run_ids=run_ids))
    have_conf = {(f.run_id, f.key[: -len("_confidence")])
                 for f in feedback if f.key.endswith("_confidence")}

    changed = 0
    for f in feedback:
        if f.key not in KEYS or f.score is None:
            continue
        binary = float(f.score >= THRESHOLD)
        needs_conf = (f.run_id, f.key) not in have_conf
        if binary == f.score and not needs_conf:
            continue
        print(f"  {str(f.run_id)[:8]} {f.key}: {f.score} -> {binary}"
              f"{'  +confidence' if needs_conf else ''}")
        if apply:
            # The probability moves to its own key BEFORE the verdict overwrites
            # it, so an interrupted pass never loses a score it has not copied.
            if needs_conf:
                ls.create_feedback(f.run_id, key=f"{f.key}_confidence",
                                   score=round(float(f.score), 3))
            if binary != f.score:
                ls.update_feedback(f.id, score=binary)
        changed += 1
    verb = "updated" if apply else "would update"
    print(f"{project.name}: {verb} {changed} feedback rows "
          f"across {len(run_ids)} runs")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("project_ids", nargs="+", help="LangSmith experiment/session ids")
    ap.add_argument("--apply", action="store_true", help="write; default is a dry run")
    a = ap.parse_args()
    ls = Client()
    for pid in a.project_ids:
        backfill(ls, pid, a.apply)


if __name__ == "__main__":
    main()
