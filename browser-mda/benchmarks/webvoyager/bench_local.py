"""The same WebVoyager experiment, for browser surfaces that run on this machine.

`experiment_native.py` drives deployed Managed Deep Agents. Playwright's and
WebdriverIO's MCP servers are stdio, so the agent, the server and the browser
all live in this process tree -- but the experiment has to come out identical
or the arms are not comparable: same dataset, same Jev judge, same thresholds,
same feedback keys, same project shape.

Two differences follow from running locally, and both are simplifications. The
row is bound with `tracing_v2_enabled(project_name=..., example_id=...)` rather
than the deployment's `langsmith_tracing=` kwarg, and the run id comes straight
off the tracer instead of being matched back through `thread_id`. The model
runs in process, so Cost and Tokens populate natively.

    uv run bench_local.py --arm playwright --limit 100 --concurrency 4
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

from langchain.agents import create_agent
from langchain_core.tracers.context import tracing_v2_enabled
from langchain_mcp_adapters.client import MultiServerMCPClient
from langsmith import Client

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiment_native import DATASET, PROMPT, grade  # noqa: E402

# Every arm runs headless, by a different mechanism each time:
#   playwright    --headless flag
#   browser-use   BROWSER_USE_HEADLESS env var
#   wdio          a `start_session` tool argument -- no CLI switch exists, so
#                 it has to come from the prompt and is worth checking in the
#                 traces rather than assuming
#   agent-browser headless by default; `--headed` is the opt-in
ARMS: dict[str, dict] = {
    "playwright": {
        "server": {
            "command": "npx",
            "args": ["-y", "@playwright/mcp@latest", "--headless", "--isolated"],
            "transport": "stdio",
        },
        "extra": "",
    },
    # Ships its own agent loop as `retry_with_browser_use_agent`; the system
    # prompt forbids it, so this measures browser-use's tool surface rather
    # than browser-use's agent.
    "browser-use": {
        "server": {
            "command": "uv",
            # --no-project, and VIRTUAL_ENV stripped. The runner starts this
            # server from inside the repo while bench_local.py is itself under
            # a `uv run`, so the child inherited VIRTUAL_ENV, collided with the
            # repo's project environment, and resolved an interpreter that was
            # not the one provisioning had configured:
            #   warning: `VIRTUAL_ENV=...` does not match the project
            #   environment path `.../deepagents/bro...`
            # Chrome then died downstream, which is all the agent could report.
            "args": [
                "run", "--no-project", "--with", "browser-use",
                "python", "-m", "browser_use.mcp.server",
            ],
            "transport": "stdio",
            "env": {
                **{k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"},
                "BROWSER_USE_HEADLESS": "true",
            },
        },
        "extra": (
            "\n\nDo not call retry_with_browser_use_agent: it delegates the whole task "
            "to a second agent, which is not what is being measured here."
        ),
    },
    # BLOCKED, left here as a record rather than deleted. `provider: external`
    # wants a W3C WebDriver endpoint on localhost -- "Unable to connect to
    # http://127.0.0.1:4445/" -- and Browserbase speaks CDP, so there is no URL
    # that satisfies it. Its real remote options are the named providers
    # (browserstack, saucelabs, testingbot), which need their own accounts.
    "wdio": {
        "server": {
            "command": "npx",
            "args": ["-y", "@wdio/mcp@latest"],
            "transport": "stdio",
        },
        # `external` plus a webdriverConfig pointed at the session: the
        # provider enum is local/browserstack/saucelabs/testmu/testingbot/
        # digitalai/external, and only the last one takes an arbitrary endpoint.
        "extra": (
            "\n\nStart with start_session using exactly:\n"
            '{{"provider": "external", "browser": "chrome", "headless": true, '
            '"webdriverConfig": {{"capabilities": {{"goog:chromeOptions": '
            '{{"debuggerAddress": "{cdp_url}"}}}}}}}}\n'
            "That endpoint is a hosted browser; do not try to launch a local "
            "one. Call close_session at the end."
        ),
    },
}

SHELL_ARMS = {
    # Vercel Labs' agent-browser is a CLI, so the agent drives it through
    # `execute` -- the same shape as the sandbox `browse` CLI arm, and the
    # reason this runner needs two kinds of arm rather than one.
    "agent-browser": (
        "You browse by running the `agent-browser` CLI through `execute`.\n\n"
        "Run `agent-browser skills get core --full` first: it ships the command\n"
        "reference and the ref/selector patterns, and is version-matched to the\n"
        "CLI. Then use `agent-browser open <url>`, `read`, `click`, and friends.\n"
        "Chain commands with && and trim output with `head -c` so you pull back\n"
        "only what you need."
    ),
}

BROWSERBASE_SESSIONS = "https://api.browserbase.com/v1/sessions"


def _bb_headers() -> dict[str, str]:
    return {"X-BB-API-Key": os.environ["BROWSERBASE_API_KEY"],
            "Content-Type": "application/json"}


def browserbase_session() -> tuple[str, str]:
    """Mint a Browserbase session; return its id and CDP endpoint.

    The hosted-MCP arm never had a browser to provision, which is why it never
    hit any of the Chrome-on-a-runner problems these arms did. Same trick:
    hand them a remote browser instead of installing one.

    `keepAlive` is required, not a nicety. Without it the session ends on the
    first disconnect -- which happens during MCP tool discovery, long before
    the task runs -- and the real connection then gets `HTTP 410`.
    """
    response = httpx.post(
        BROWSERBASE_SESSIONS, timeout=60, headers=_bb_headers(),
        json={"projectId": os.environ["BROWSERBASE_PROJECT_ID"],
              "keepAlive": True,
              "timeout": SESSION_TIMEOUT_S},
    )
    response.raise_for_status()
    body = response.json()
    # connectUrl is only on the create response, never on a later read.
    return body["id"], body["connectUrl"]


def release_browserbase(session_id: str) -> None:
    """End a session. `keepAlive` means nothing else will.

    One suite leaked its way to the project's full 75-slot concurrency cap
    this week and started failing every later task; this is the other half of
    that lesson.
    """
    try:
        httpx.post(f"{BROWSERBASE_SESSIONS}/{session_id}", timeout=30,
                   headers=_bb_headers(),
                   json={"projectId": os.environ["BROWSERBASE_PROJECT_ID"],
                         "status": "REQUEST_RELEASE"})
    except Exception:  # noqa: BLE001 - teardown must not fail a finished task
        logging.getLogger(__name__).debug("releasing %s failed", session_id, exc_info=True)


def browser_use_config(cdp_url: str) -> str:
    """Write a browser-use config pinned to `cdp_url`, return its path.

    One file per task, because `BROWSER_USE_CONFIG_PATH` is read per server
    process and every task spawns its own. The file is minted by browser-use
    and then patched: a hand-written one is treated as old-format and silently
    replaced with a fresh default.
    """
    directory = Path(tempfile.mkdtemp(prefix="browseruse-"))
    path = directory / "config.json"
    env = {**os.environ, "BROWSER_USE_CONFIG_PATH": str(path)}
    subprocess.run(
        ["uv", "run", "--no-project", "--with", "browser-use", "python", "-c",
         "from browser_use.config import load_browser_use_config; load_browser_use_config()"],
        env=env, check=True, capture_output=True,
    )
    config = json.loads(path.read_text())
    for entry in config["browser_profile"].values():
        if entry.get("default"):
            entry.update(headless=True, cdp_url=cdp_url)
    path.write_text(json.dumps(config, indent=2))
    return str(path)


ANSWER_RULES = """Answer directly and concisely, from what the page actually showed.
If the site blocks you or the information is not there, say exactly that and
why. Never guess a value you did not read."""

REMOTE_BROWSER_ARMS = {"browser-use", "wdio"}

INSTRUCTIONS = "Browse with the tools you have been given to answer the question.\n\n" + ANSWER_RULES

MAX_SECONDS = 600
SESSION_TIMEOUT_S = 900  # a task cap of 600s plus room for setup

# Every arm in this benchmark runs the same agent model, deployed or local.
# An earlier revision of this file typed "claude-sonnet-5" here while the five
# deployed arms ran "claude-sonnet-5-5", which quietly made a cost comparison
# across them meaningless.
MODEL = "anthropic:claude-sonnet-5-5"


async def run_one(arm, ls, project, ex, sem, out, rep=0):
    async with sem:
        inp = ex.inputs
        t0 = time.time()
        answer, steps, run_id, status = "", 0, None, "ok"
        session_id = None
        try:
            if arm in SHELL_ARMS:
                from deepagents import create_deep_agent
                from deepagents.backends import LocalShellBackend

                agent = create_deep_agent(
                    model=MODEL,
                    backend=LocalShellBackend(),
                    system_prompt=SHELL_ARMS[arm] + "\n\n" + ANSWER_RULES,
                )
            else:
                server = dict(ARMS[arm]["server"])
                extra = ARMS[arm]["extra"]
                if arm in REMOTE_BROWSER_ARMS:
                    # A Browserbase session per task, so neither arm needs a
                    # browser installed where it runs.
                    session_id, cdp = browserbase_session()
                    if arm == "browser-use":
                        server["env"] = {**server.get("env", os.environ),
                                         "BROWSER_USE_CONFIG_PATH": browser_use_config(cdp)}
                    else:
                        extra = extra.format(cdp_url=cdp)
                client = MultiServerMCPClient({arm: server})
                agent = create_agent(
                    model=MODEL,
                    tools=await client.get_tools(),
                    system_prompt=INSTRUCTIONS + extra,
                )
            # The local counterpart of the deployment's `langsmith_tracing=`:
            # it roots the run in the experiment project and binds it to the
            # example, which is what makes it a row.
            with tracing_v2_enabled(project_name=project, example_id=ex.id) as tracer:
                state = await asyncio.wait_for(
                    agent.ainvoke(
                        {"messages": [{"role": "user", "content": PROMPT.format(**inp)}]},
                        config={"recursion_limit": 200},
                    ),
                    timeout=MAX_SECONDS,
                )
                run_id = getattr(tracer.latest_run, "id", None)
            msgs = state["messages"]
            steps = len(msgs)
            c = msgs[-1].content
            if isinstance(c, list):
                c = " ".join(b.get("text", "") for b in c if isinstance(b, dict))
            answer = (c or "").strip()
        except Exception as err:  # noqa: BLE001 - one task must not stop the suite
            status = f"{type(err).__name__}: {str(err)[:90]}"
        finally:
            if session_id:
                release_browserbase(session_id)

        secs = round(time.time() - t0, 1)
        scores = None
        if status == "ok" and answer:
            try:
                scores = grade(inp["ques"], inp["web"], answer, ex.outputs)
            except Exception as err:  # noqa: BLE001
                print(f"    judge error: {str(err)[:110]}", flush=True)

        g = ("ERROR" if status != "ok" else
             "UNGRADED" if scores is None else
             "BLOCKED" if scores["blocked"] >= 0.5 else
             "CORRECT" if scores["correct"] >= 0.5 else "INCORRECT")
        out.append({"task": inp.get("task_id"), "grade": g, "steps": steps, "seconds": secs})
        tag = f" r{rep + 1}" if rep else ""
        print(f"[{g:8}] {str(inp.get('task_id')):26}{tag} {secs:6.0f}s steps={steps}"
              f"{'' if status == 'ok' else '  ' + status}", flush=True)
        return run_id, g, steps, secs, scores


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=[*ARMS, *SHELL_ARMS])
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--dataset-id", default=os.environ.get("DATASET_ID") or None)
    ap.add_argument("--limit", type=int, default=0, help="smoke-test on the first N examples")
    ap.add_argument("--only", default=None,
                    help="file of example ids, one per line: resume a suite that was cut short")
    ap.add_argument("--project", default=None,
                    help="write into an existing experiment instead of creating one")
    ap.add_argument("--repetitions", type=int, default=1)
    a = ap.parse_args()

    ls = Client()
    ds = (ls.read_dataset(dataset_id=a.dataset_id) if a.dataset_id
          else ls.read_dataset(dataset_name=a.dataset))
    examples = list(ls.list_examples(dataset_id=ds.id))
    if a.only:
        wanted = {line.strip() for line in open(a.only) if line.strip()}
        examples = [e for e in examples if str(e.id) in wanted]
    if a.limit:
        examples = examples[: a.limit]

    if a.project:
        # Resuming: the rows already there keep their native Cost and Tokens,
        # so only the gap is paid for again.
        project = a.project
        print(f"appending to {project}  ({len(examples)} remaining)", flush=True)
        ls.read_project(project_name=project)
        sem = asyncio.Semaphore(a.concurrency)
        out: list = []
        results = await asyncio.gather(*[
            run_one(a.arm, ls, project, ex, sem, out) for ex in examples])
        await post_feedback(ls, results, out, project)
        return

    project = f"{a.dataset}-{a.arm}-{int(time.time())}"
    # num_examples/num_repetitions give the UI its denominator, so progress
    # shows as "n/100" rather than a bare run count.
    project_kwargs = dict(
        project_name=project,
        reference_dataset_id=ds.id,
        metadata={"examples": len(examples), "repetitions": a.repetitions, "arm": a.arm},
        description=f"{a.arm} over stdio MCP, headless browser on the runner.",
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

    sem = asyncio.Semaphore(a.concurrency)
    out = []
    results = await asyncio.gather(*[
        run_one(a.arm, ls, project, ex, sem, out, rep)
        for rep in range(a.repetitions) for ex in examples])

    await post_feedback(ls, results, out, project)


async def post_feedback(ls, results, out, project):
    posted = 0
    for run_id, g, steps, secs, scores in results:
        if run_id is None:
            continue
        if scores:
            ls.create_feedback(run_id, key="correct",
                               score=float(scores["correct"] >= 0.5), comment=g)
            ls.create_feedback(run_id, key="blocked", score=float(scores["blocked"] >= 0.5))
            ls.create_feedback(run_id, key="correct_confidence",
                               score=round(scores["correct"], 3))
            ls.create_feedback(run_id, key="blocked_confidence",
                               score=round(scores["blocked"], 3))
        ls.create_feedback(run_id, key="steps", score=steps)
        ls.create_feedback(run_id, key="seconds", score=secs)
        posted += 1

    succ = sum(1 for r in out if r["grade"] == "CORRECT")
    sess = ls.read_project(project_name=project)
    print(f"\nfeedback posted on {posted}/{len(results)} rows")
    print(f"SUCCESS {succ}/{len(out)}")
    print(f"project: {project}  id: {sess.id}")


if __name__ == "__main__":
    asyncio.run(main())
