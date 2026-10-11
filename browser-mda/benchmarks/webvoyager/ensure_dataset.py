"""Create the eval dataset in the current workspace if it is not already there.

The suite used to assume the dataset already existed in whatever workspace the
`LANGSMITH_API_KEY` belonged to. It did not: every CI run of this workflow has
died in three seconds on `Dataset webvoyager-100 not found` (and before that,
`webvoyager-30`), because the key in the `evals` GitHub environment belongs to a
different workspace than the one the dataset was built in.

The rows are committed next to this file, so the dependency is unnecessary --
seed the dataset from the JSONL instead of requiring someone to share it across
workspaces. Idempotent: an existing dataset with the right example count is left
alone, so this is safe to run before every suite.

    python ensure_dataset.py --dataset webvoyager-100 --file webvoyager_100.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from langsmith import Client
from langsmith.utils import LangSmithNotFoundError


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--file", required=True, help="JSONL with `inputs`/`outputs` per row")
    a = ap.parse_args()

    rows = [json.loads(line) for line in Path(a.file).read_text().splitlines() if line.strip()]
    ls = Client()
    try:
        dataset = ls.read_dataset(dataset_name=a.dataset)
        have = sum(1 for _ in ls.list_examples(dataset_id=dataset.id))
        if have == len(rows):
            print(f"{a.dataset}: present with {have} examples, nothing to do")
            return
        # Never silently reconcile: a count mismatch means the local file and the
        # remote set have diverged, and quietly topping it up would make two runs
        # of "the same" dataset not the same.
        raise SystemExit(
            f"{a.dataset} exists with {have} examples but {a.file} has {len(rows)}. "
            "Refusing to modify it -- rename the dataset or reconcile by hand."
        )
    except LangSmithNotFoundError:
        pass

    dataset = ls.create_dataset(
        dataset_name=a.dataset,
        description=f"Seeded from {Path(a.file).name} ({len(rows)} vetted WebVoyager tasks).",
    )
    ls.create_examples(
        dataset_id=dataset.id,
        inputs=[r["inputs"] for r in rows],
        outputs=[r.get("outputs", {}) for r in rows],
    )
    print(f"{a.dataset}: created with {len(rows)} examples (id {dataset.id})")


if __name__ == "__main__":
    main()
