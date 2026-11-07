"""Submit the runs an interrupted experiment still owes, into the same project.

`experiment_native.py` holds every example behind one semaphore and posts all
grades only after the last one lands, so a suite that is 85% done still trickles
its tail out at the original concurrency. This submits the shortfall directly --
same project, same example binding, so the rows are indistinguishable from the
runner's own -- and leaves grading to `grade_existing_project.py`.

    uv run --with "langsmith>=0.14.7" --with langgraph-sdk python top_up_project.py \
        --project webvoyager-repl-native-1791635157 --variant repl --reps 3 --dry-run

**Kill the original runner first.** Both processes work from the same shortfall,
so leaving it alive double-submits the tail and pays for every duplicate.

Concurrency is bounded by Browserbase sessions, not by the model: the project
caps concurrent sessions (25 at the time of writing) and sessions outlive the
run that opened them, so the ceiling here is deliberately below the headroom.
"""
from __future__ import annotations

import argparse
import asyncio
import time
from collections import Counter

from langgraph_sdk import get_client
from langsmith import Client

from experiment_native import PROMPT, SUBMIT_TIMEOUT, VARIANTS
import os


async def submit(client, aid, project, example, rep, sem, results):
    async with sem:
        task = example.inputs.get("task_id")
        t0 = time.time()
        try:
            thread = await asyncio.wait_for(client.threads.create(), SUBMIT_TIMEOUT)
            run = await asyncio.wait_for(client.runs.create(
                thread["thread_id"], aid,
                input={"messages": [{"role": "user",
                                     "content": PROMPT.format(**example.inputs)}]},
                config={"recursion_limit": 1000},
                langsmith_tracing={"project_name": project, "example_id": str(example.id)},
            ), SUBMIT_TIMEOUT)
        except Exception as err:  # noqa: BLE001 - report and keep going
            print(f"  submit failed {task}: {str(err)[:90]}", flush=True)
            results.append((task, "SUBMIT_FAILED"))
            return
        print(f"  submitted {task} (rep {rep})", flush=True)
        # Poll only to report progress; the deployment finishes regardless.
        while time.time() - t0 < 900:
            await asyncio.sleep(15)
            try:
                status = (await client.runs.get(thread["thread_id"], run["run_id"]))["status"]
            except Exception:
                continue
            if status not in ("pending", "running"):
                break
        results.append((task, "DONE"))


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--variant", default="repl", choices=list(VARIANTS))
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    ls = Client()
    session = ls.read_project(project_name=a.project)
    examples = list(ls.list_examples(dataset_id=session.reference_dataset_id))
    have = Counter(str(r.reference_example_id)
                   for r in ls.list_runs(project_id=session.id, is_root=True))

    shortfall = [(ex, rep)
                 for ex in examples
                 for rep in range(have.get(str(ex.id), 0), a.reps)]
    print(f"{a.project}: {sum(have.values())} runs present, "
          f"{len(shortfall)} outstanding of {len(examples) * a.reps}")
    for ex, rep in shortfall:
        print(f"   {ex.inputs.get('task_id'):28} rep {rep + 1}")
    if a.dry_run or not shortfall:
        print("\ndry run -- nothing submitted" if a.dry_run else "\nnothing to do")
        return

    url, graph = VARIANTS[a.variant]
    client = get_client(url=url, api_key=os.environ["LANGSMITH_API_KEY"])
    aid = (await client.assistants.search(graph_id=graph, limit=1))[0]["assistant_id"]
    sem = asyncio.Semaphore(a.concurrency)
    results: list[tuple[str, str]] = []
    await asyncio.gather(*[submit(client, aid, a.project, ex, rep, sem, results)
                           for ex, rep in shortfall])
    print(f"\n{len(results)} submitted. Now grade the project:")
    print(f"  python grade_existing_project.py --project {a.project}")


if __name__ == "__main__":
    asyncio.run(main())
