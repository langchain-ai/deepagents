"""Run a WebVoyager suite so the DEPLOYMENT run is the experiment row.

`aevaluate` makes its target the row. When the target just calls a deployment
over HTTP it holds no LLM spans, so the row prices at zero and the real numbers
sit on a second run. Routing the deployment trace into the experiment session
with `langsmith_tracing` fixes the location but not the shape: you get two root
rows per example, `AsyncTarget` with the grades and `browser-agent` with the
cost.

An experiment is just a project carrying `reference_dataset_id`. So create that
project directly, route each deployment run into it with the example id, and
attach grades as feedback on the run itself. One row per example, native Cost
and Tokens, no wrapper run.
"""
from __future__ import annotations

import argparse, asyncio, json, os, random, time
import httpx
from langsmith import Client
from langgraph_sdk import get_client

DATASET = "webvoyager-30"
GW = "https://gateway.smith.langchain.com/anthropic/v1/messages"
# A stronger judge than the agent it grades. Sonnet flipped the same answer
# between SUCCESS and FAILURE across identical runs; grading is a short
# reasoning task where the model matters more than the token budget, and at ~8
# output tokens per call the upgrade is negligible against a ~$0.08 task.
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "claude-opus-5-5")
VARIANTS = {
    "cli": ("https://browser-agent-aed6b8f018df58ef8920406eda94dc76.us.langgraph.app",
            "browser-agent"),
    "mcp": ("https://browser-agent-mcp-efdbb25a710b5ffbbf964d3d435ec125.us.langgraph.app",
            "browser-agent-mcp"),
    "repl": ("https://browser-agent-repl-f60393f1fadc5b898e56d4f190de4f2d.us.langgraph.app",
             "browser-agent-repl"),
}
PROMPT = """{ques}

Start at {web}
Answer the question directly and concisely. If after a genuine effort the site
blocks you or the information is not available, say exactly why. Do not guess."""

JEV_URL = "https://gateway.smith.langchain.com/v1/systemone"
JEV_MODEL = os.environ.get("JEV_MODEL", "typesafe/jev-latest")

# Two questions, two feedback keys. `blocked` stays separate from `correct`
# because a bot wall says nothing about agent quality -- folding it in is what
# made a correctly-reported Allrecipes block read as a failure.
#
# Jev over an LLM judge: the answer is typed, so there is nothing to parse and no
# fallback to get wrong. The previous free-text judge truncated mid-reasoning,
# returned an empty string, and a "default to FAILURE" parser recorded that as a
# failed task -- 106 of 180 grades. It is also ~8x faster and ~55x cheaper.
JEV_QUESTIONS = {
    # Keep the two sides balanced in length. An earlier version stacked four
    # disqualifiers into `false` against one sentence of `true`, and Jev dragged
    # clear successes down with it -- an answer matching the reference exactly
    # scored 0.56.
    "correct": {"type": "noul",
                "instructions": "Does the answer give what the task asked for, taken from the site?",
                "criteria": {
                    "true": "The answer supplies what was asked. Current values that differ from "
                            "the reference are fine: the reference records one acceptable answer "
                            "against data that changes. Reporting, from the site, that the thing "
                            "does not exist also counts.",
                    "false": "Something asked for is missing or hedged, or a value was guessed, "
                             "recalled, or worked out without visiting the site."}},
    "blocked": {"type": "noul",
                "instructions": "Did the site stop the agent reaching the content?",
                "criteria": {"true": "A bot wall, captcha, login, 403 or stripped parameters "
                                     "blocked it.",
                             "false": "The agent reached the content it needed."}},
}


def grade(ques, web, answer, outputs=None):
    """Return {"correct": p, "blocked": p} from Jev, or None when ungradeable."""
    if not answer:
        return None
    outputs = outputs or {}
    state = (f"TASK\n{ques}\n\n"
             f"REFERENCE ANSWER (type: {outputs.get('answer_type')})\n"
             f"{outputs.get('reference_answer')}\n\n"
             f"AGENT ANSWER\n{answer[:6000]}")
    r = httpx.post(JEV_URL, timeout=90,
                   headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"],
                            "Authorization": f"Bearer {os.environ['ANTHROPIC_API_KEY']}",
                            "Content-Type": "application/json"},
                   json={"model": JEV_MODEL, "state": state, "questions": JEV_QUESTIONS})
    r.raise_for_status()
    a = r.json()["answers"]
    return {k: float(a[k]["noul"]) for k in ("correct", "blocked")}


MAX_TRANSPORT_RETRIES = 2


async def run_one(client, aid, ls, project, ex, sem, out, rep=0, attempt=0):
    async with sem:
        inp = ex.inputs
        t0 = time.time()
        th = await client.threads.create()
        run = await client.runs.create(
            th["thread_id"], aid,
            input={"messages": [{"role": "user", "content": PROMPT.format(**inp)}]},
            config={"recursion_limit": 1000},
            # Routes this run into the experiment project AND binds it to the
            # example, which is what makes it the row.
            langsmith_tracing={"project_name": project, "example_id": str(ex.id)},
        )
        status = "pending"
        while time.time() - t0 < 900:
            await asyncio.sleep(10)
            try:
                status = (await client.runs.get(th["thread_id"], run["run_id"]))["status"]
            except Exception:
                continue
            if status not in ("pending", "running"):
                break
        secs = round(time.time() - t0, 1)
        answer, steps = "", 0
        try:
            st = await client.threads.get_state(th["thread_id"])
            msgs = st["values"].get("messages", [])
            steps = len(msgs)
            c = (msgs[-1] if msgs else {}).get("content")
            if isinstance(c, list):
                c = " ".join(b.get("text", "") for b in c if isinstance(b, dict))
            answer = (c or "").strip()
        except Exception:
            pass
        # A dropped MCP stream is infrastructure, not an agent outcome. Retry it
        # rather than scoring it: 19 of 90 MCP rows died on
        # MCPError(-32000, 'SSE stream ended without a response'), which would
        # otherwise read as the agent failing a fifth of its tasks.
        if status != "success" and attempt < MAX_TRANSPORT_RETRIES:
            detail = ""
            try:
                detail = str((await client.runs.get(th["thread_id"], run["run_id"])).get("kwargs", ""))
            except Exception:
                pass
            if True:  # retry any non-success once; transport errors dominate
                print(f"    retry {attempt + 1} for {inp.get('task_id')} (status={status})", flush=True)
                return await run_one(client, aid, ls, project, ex, sem, out, rep, attempt + 1)

        scores = None
        if status == "success":
            try:
                scores = grade(inp["ques"], inp["web"], answer, ex.outputs)
            except Exception as err:
                print(f"    judge error: {str(err)[:110]}", flush=True)
        g = ("ERROR" if status != "success" else
             "UNGRADED" if scores is None else
             "BLOCKED" if scores["blocked"] >= 0.5 else
             "CORRECT" if scores["correct"] >= 0.5 else "INCORRECT")
        out.append({"task": inp.get("task_id"), "grade": g, "steps": steps, "seconds": secs})
        rec_scores = scores
        tag = f" r{rep+1}" if rep else ""
        print(f"[{g:8}] {str(inp.get('task_id')):26}{tag} {secs:6.0f}s steps={steps}", flush=True)
        return th["thread_id"], ex.id, g, steps, secs, rec_scores


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="cli", choices=list(VARIANTS))
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--url", default=None)
    ap.add_argument("--limit", type=int, default=0, help="smoke-test on the first N examples")
    ap.add_argument("--repetitions", type=int, default=1,
                    help="runs per example; >1 gives a variance estimate")
    a = ap.parse_args()
    url, graph = VARIANTS[a.variant]
    url = a.url or url

    ls = Client()
    ds = ls.read_dataset(dataset_name=DATASET)
    examples = list(ls.list_examples(dataset_id=ds.id))
    if a.limit:
        examples = examples[: a.limit]
    project = f"webvoyager-{a.variant}-native-{int(time.time())}"
    # num_examples/num_repetitions give the UI its denominator, so progress shows
    # as "n/30" rather than a bare run count. Requires langsmith >= 0.14.x --
    # older releases reject both kwargs, so fall back rather than failing a run.
    project_kwargs = dict(
        project_name=project,
        reference_dataset_id=ds.id,
        metadata={"examples": len(examples), "repetitions": a.repetitions},
        description="Deployment runs ARE the experiment rows.",
    )
    try:
        ls.create_project(num_examples=len(examples), num_repetitions=a.repetitions,
                          **project_kwargs)
    except TypeError:
        print("  note: langsmith too old for num_examples; progress bar will lack a total",
              flush=True)
        ls.create_project(**project_kwargs)
    print(f"experiment project: {project}  "
          f"({len(examples)} examples x {a.repetitions} reps = "
          f"{len(examples) * a.repetitions} runs)", flush=True)

    client = get_client(url=url, api_key=os.environ["LANGSMITH_API_KEY"])
    aid = (await client.assistants.search(graph_id=graph, limit=1))[0]["assistant_id"]
    sem = asyncio.Semaphore(a.concurrency)
    out = []
    results = await asyncio.gather(*[
        run_one(client, aid, ls, project, ex, sem, out, rep)
        for rep in range(a.repetitions) for ex in examples])

    # Match feedback to runs by thread_id from the run's metadata. The LangGraph
    # run id is NOT the LangSmith run id -- posting to it makes create_feedback
    # succeed silently against nothing, which is how an earlier version reported
    # "feedback posted" onto rows that stayed empty. thread_id is unique per run
    # and survives repetitions, where keying on example_id would collapse them.
    await asyncio.sleep(25)
    sess = ls.read_project(project_name=project)
    by_thread = {}
    for r in ls.list_runs(project_id=sess.id, is_root=True):
        md = ((r.extra or {}).get("metadata") or {}) if r.extra else {}
        tid = md.get("thread_id")
        if tid:
            by_thread[str(tid)] = r.id
    posted = 0
    for thread_id, _ex_id, g, steps, secs, scores in results:
        target = by_thread.get(str(thread_id))
        if not target:
            print(f"  no run found for thread {thread_id}", flush=True)
            continue
        if scores:
            ls.create_feedback(target, key="correct", score=round(scores["correct"], 3), comment=g)
            ls.create_feedback(target, key="blocked", score=round(scores["blocked"], 3))
        ls.create_feedback(target, key="steps", score=steps)
        ls.create_feedback(target, key="seconds", score=secs)
        posted += 1
    succ = sum(1 for r in out if r["grade"] == "CORRECT")
    print(f"\nfeedback posted on {posted}/{len(results)} rows")
    print(f"SUCCESS {succ}/{len(out)}")
    print(f"project: {project}  id: {sess.id}")


if __name__ == "__main__":
    asyncio.run(main())
