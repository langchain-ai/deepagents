"""Run a stratified WebVoyager sample against the deployed browser MDA."""
import argparse, asyncio, json, os, random, time
from langgraph_sdk import get_client

URL = os.environ["MDA_URL"]; KEY = os.environ["LANGSMITH_API_KEY"]
GRAPH = "browser-agent"

def sample(path, per_site, seed=7):
    by = {}
    for line in open(path):
        line = line.strip()
        if not line: continue
        t = json.loads(line); by.setdefault(t["web_name"], []).append(t)
    rnd = random.Random(seed); out = []
    for site in sorted(by):
        out.extend(rnd.sample(by[site], min(per_site, len(by[site]))))
    return out

PROMPT = """{ques}

Start at {web}
Answer the question directly and concisely. If after a genuine effort the site
blocks you or the information is not available, say exactly why. Do not guess."""

async def one(client, aid, task, sem, cap, timeout_s):
    """Create the run, then POLL for completion.

    Deliberately not client.runs.wait(): that holds one HTTP join connection open
    for the whole run, and on long browsing tasks it dies with "Too little data
    for declared Content-Length" -- recorded as an agent error when it is really
    a transport failure. Polling also makes the wall-clock bound real, which
    wait_for around wait() did not (runs there overshot the timeout 3x).
    """
    async with sem:
        t0 = time.time()
        rec = {"id": task["id"], "site": task["web_name"], "ques": task["ques"]}
        try:
            th = await client.threads.create()
            rec["thread_id"] = th["thread_id"]
            run = await client.runs.create(
                th["thread_id"], aid,
                input={"messages": [{"role": "user",
                                     "content": PROMPT.format(**task)}]},
                config={"recursion_limit": cap},
            )
            rec["run_id"] = run["run_id"]
            status = "pending"
            while time.time() - t0 < timeout_s:
                await asyncio.sleep(10)
                try:
                    status = (await client.runs.get(
                        th["thread_id"], run["run_id"]))["status"]
                except Exception:
                    continue          # transient poll failure is not a run failure
                if status not in ("pending", "running"):
                    break
            if status in ("pending", "running"):
                rec["status"] = "timeout"
                try:
                    await client.runs.cancel(th["thread_id"], run["run_id"])
                except Exception:
                    pass
                rec["answer"] = ""
            else:
                st = await client.threads.get_state(th["thread_id"])
                msgs = st["values"].get("messages", [])
                rec["steps"] = len(msgs)
                rec["tool_calls"] = sum(len(m.get("tool_calls") or []) for m in msgs
                                        if m.get("type") == "ai")
                last = msgs[-1] if msgs else {}
                c = last.get("content")
                if isinstance(c, list):
                    c = " ".join(b.get("text", "") for b in c if isinstance(b, dict))
                rec["answer"] = (c or "").strip()
                rec["run_status"] = status
                rec["status"] = "ok" if (status == "success" and rec["answer"]) else status
        except Exception as e:
            rec["status"] = "harness_error"; rec["answer"] = ""
            rec["error"] = f"{type(e).__name__}: {e}"[:400]
        rec["seconds"] = round(time.time() - t0, 1)
        print(f"[{rec['status']:13}] {rec['id']:28} {rec['seconds']:6.1f}s "
              f"steps={rec.get('steps','-')} :: {rec['answer'][:80]}", flush=True)
        return rec

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-site", type=int, default=1)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--recursion-limit", type=int, default=1000)  # effectively uncapped; wall-clock timeout is the real guard
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--out", default="results.jsonl")
    a = ap.parse_args()

    tasks = sample("WebVoyager_data.jsonl", a.per_site)
    print(f"running {len(tasks)} tasks, concurrency={a.concurrency}, "
          f"recursion_limit={a.recursion_limit}", flush=True)
    client = get_client(url=URL, api_key=KEY)
    aid = (await client.assistants.search(graph_id=GRAPH, limit=1))[0]["assistant_id"]
    sem = asyncio.Semaphore(a.concurrency)
    recs = await asyncio.gather(*[
        one(client, aid, t, sem, a.recursion_limit, a.timeout) for t in tasks])
    with open(a.out, "w") as f:
        for r in recs: f.write(json.dumps(r) + "\n")
    print(f"\nwrote {a.out}")

asyncio.run(main())
