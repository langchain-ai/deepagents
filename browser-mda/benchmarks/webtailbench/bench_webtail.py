"""WebTailBench smoke across harnesses, scored against each row's rubric.

Same shape as the WebVoyager runners -- a LangSmith experiment per arm, rows
bound to examples, feedback posted per row -- but the judge differs because
the dataset does. WebVoyager ships a reference answer and asks two yes/no
questions; WebTailBench ships a 3-6 item rubric, and the useful score is the
fraction met. A task where the agent found the flights but not the fare
difference is not simply "wrong".

    uv run bench_webtail.py --arm facade --limit 11
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time

from langgraph_sdk import get_client
from langsmith import Client

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grade_rubric import grade_rubric  # noqa: E402

DATASET = os.environ.get("WEBTAIL_DATASET", "webtailbench-smoke")

# Deployments, so the browser is Browserbase or a sandbox rather than the
# machine running this.
ARMS = {
    "facade": ("https://browser-agent-facade-c02954d868c45753911471d1251e2739.us.langgraph.app",
               "browser-agent-facade"),
    # agent-browser used to be CI-only, driving a Chrome installed on the
    # runner. It now has a sandbox deployment of its own, so it runs the same
    # way every other arm does and in the same kind of environment.
    "agent-browser": (
        "https://browser-agent-agent-browser-ba5da2114a285fc380481d9442818f9f.us.langgraph.app",
        "browser-agent-agent-browser",
    ),
}

PROMPT = """{ques}

{web_line}Answer the question directly and concisely, citing what the page actually
showed. If the site blocks you or the information is not available, say exactly
why. Do not guess."""

MAX_SECONDS = 900


async def run_one(client, aid, project, ex, sem, out):
    async with sem:
        inp = ex.inputs
        web = inp.get("web") or ""
        prompt = PROMPT.format(ques=inp["ques"],
                               web_line=f"Start at {web}\n" if web else "")
        t0 = time.time()
        answer, steps, status = "", 0, "ok"
        try:
            th = await asyncio.wait_for(client.threads.create(), 120)
            run = await asyncio.wait_for(client.runs.create(
                th["thread_id"], aid,
                input={"messages": [{"role": "user", "content": prompt}]},
                config={"recursion_limit": 1000},
                langsmith_tracing={"project_name": project, "example_id": str(ex.id)},
            ), 120)
            while time.time() - t0 < MAX_SECONDS:
                await asyncio.sleep(10)
                try:
                    status = (await client.runs.get(th["thread_id"], run["run_id"]))["status"]
                except Exception:  # noqa: BLE001 - a dropped poll is not a failed run
                    continue
                if status not in ("pending", "running"):
                    break
            state = await asyncio.wait_for(client.threads.get_state(th["thread_id"]), 120)
            msgs = state["values"].get("messages", [])
            steps = len(msgs)
            content = (msgs[-1] if msgs else {}).get("content")
            if isinstance(content, list):
                content = " ".join(b.get("text", "") for b in content if isinstance(b, dict))
            answer = (content or "").strip()
        except Exception as err:  # noqa: BLE001 - one task must not stop the suite
            status = f"{type(err).__name__}: {str(err)[:80]}"
            th = None

        secs = round(time.time() - t0, 1)
        scores = None
        if status == "success" and answer:
            try:
                scores = grade_rubric(inp["ques"], answer, ex.outputs.get("rubric"))
            except Exception as err:  # noqa: BLE001
                print(f"    judge error on {inp.get('task_id')}: {str(err)[:90]}", flush=True)

        label = ("ERROR" if status not in ("success", "ok") and not answer
                 else "UNGRADED" if scores is None
                 else f"{scores['met']}/{scores['total']}")
        out.append({"task": inp.get("task_id"), "category": inp.get("category"),
                    "label": label, "secs": secs, "steps": steps})
        print(f"[{label:9}] {str(inp.get('task_id')):22} {inp.get('category'):22} "
              f"{secs:6.0f}s steps={steps}", flush=True)
        return (th or {}).get("thread_id"), scores, steps, secs


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--dataset", default=DATASET)
    # Prefer the id in CI: name lookup is not read-after-write consistent and
    # a dataset seconds old still reads as not found.
    ap.add_argument("--dataset-id", default=os.environ.get("DATASET_ID") or None)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    url, graph = ARMS[a.arm]
    ls = Client()
    ds = (ls.read_dataset(dataset_id=a.dataset_id) if a.dataset_id
          else ls.read_dataset(dataset_name=a.dataset))
    examples = list(ls.list_examples(dataset_id=ds.id))
    if a.limit:
        examples = examples[: a.limit]

    project = f"{a.dataset}-{a.arm}-{int(time.time())}"
    try:
        ls.create_project(project_name=project, reference_dataset_id=ds.id,
                          num_examples=len(examples), num_repetitions=1,
                          description=f"{a.arm} on {a.dataset}, rubric-scored.")
    except TypeError:
        ls.create_project(project_name=project, reference_dataset_id=ds.id)
    print(f"experiment project: {project}  ({len(examples)} tasks)", flush=True)

    client = get_client(url=url, api_key=os.environ["LANGSMITH_API_KEY"])
    aid = (await client.assistants.search(graph_id=graph, limit=1))[0]["assistant_id"]
    sem = asyncio.Semaphore(a.concurrency)
    out: list = []
    results = await asyncio.gather(*[run_one(client, aid, project, ex, sem, out)
                                     for ex in examples])

    await asyncio.sleep(25)
    sess = ls.read_project(project_name=project)
    by_thread = {}
    for r in ls.list_runs(project_id=sess.id, is_root=True):
        md = ((r.extra or {}).get("metadata") or {}) if r.extra else {}
        if tid := md.get("thread_id"):
            by_thread[str(tid)] = r.id

    posted = 0
    for thread_id, scores, steps, secs in results:
        target = by_thread.get(str(thread_id))
        if not target:
            continue
        if scores:
            ls.create_feedback(target, key="rubric_score", score=round(scores["score"], 3),
                               comment=f"{scores['met']}/{scores['total']}")
            ls.create_feedback(target, key="blocked", score=float(scores["blocked"] >= 0.5))
        ls.create_feedback(target, key="steps", score=steps)
        ls.create_feedback(target, key="seconds", score=secs)
        posted += 1

    scored = [r for r in out if "/" in r["label"]]
    print(f"\nfeedback posted on {posted}/{len(results)} rows")
    print(f"scored {len(scored)}/{len(out)}")
    print(f"project: {project}  id: {sess.id}")


if __name__ == "__main__":
    asyncio.run(main())
