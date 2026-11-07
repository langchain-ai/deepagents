"""Where the browsing agent spends its steps.

Reads a results/graded jsonl from run_eval.py, pulls each thread's full message
list back off the deployment, and pulls the matching run's token usage from
LangSmith. Then it answers the only question that matters for tuning the
harness: of the steps the agent took, which ones bought nothing?

    export MDA_URL=https://<deployment>.us.langgraph.app
    export LANGSMITH_API_KEY=...
    uv run --with langgraph-sdk --with httpx python analyze_traces.py graded.jsonl report.md

The thread state is the ground truth for *what* the agent did (every `browse`
command verbatim); the LangSmith run is the ground truth for *what it cost*.
Neither alone is enough -- a cheap-looking 20-step run that re-read a 60k-token
page on every step is the expensive one.
"""

import asyncio
import collections
import json
import os
import re
import sys

import httpx
from langgraph_sdk import get_client

URL = os.environ["MDA_URL"]
KEY = os.environ["LANGSMITH_API_KEY"]
LS = "https://api.smith.langchain.com/api/v1/runs"

MAX_THREADS = 200  # upper bound on work regardless of input size
TIMEOUT = 60.0


def text_of(content) -> str:
    """Message content is a string or a list of blocks; flatten to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            b.get("text", "") for b in content if isinstance(b, dict)
        )
    return ""


def commands_in(tool_call: dict) -> list[str]:
    """Shell commands inside one tool call, however the tool spells them."""
    args = tool_call.get("args")
    if not isinstance(args, dict):
        return []
    for key in ("command", "commands", "cmd", "script"):
        v = args.get(key)
        if isinstance(v, str):
            return [v]
        if isinstance(v, list):
            return [x for x in v if isinstance(x, str)]
    return []


TWO_WORD = ("get", "wait", "tab", "mouse")


def browse_verbs(cmd: str) -> list[str]:
    """Every browse verb in one shell string.

    The agent chains with `&&` and `|`, so a single `execute` call routinely
    holds three or four browse commands. Counting only the first would credit a
    chained run with a third of the work it actually did -- and chaining is the
    behaviour we most want to measure, since it is what makes the shell variant
    cheaper per model round-trip than one-action-per-tool-call.
    """
    verbs: list[str] = []
    for m in re.finditer(r"\bbrowse\s+([a-z]+)(?:\s+([a-z]+))?", cmd):
        head, second = m.group(1), m.group(2)
        verbs.append(f"{head} {second}" if head in TWO_WORD and second else head)
    verbs += ["web-open"] * len(re.findall(r"\bweb-open\b", cmd))
    return verbs


async def thread_detail(client, rec: dict) -> dict | None:
    """Replay one thread into a per-step record of calls, sizes, and repeats."""
    tid = rec.get("thread_id")
    if not tid:
        return None
    try:
        state = await client.threads.get_state(tid)
    except Exception as e:  # a deleted or unreadable thread is not fatal
        return {"id": rec.get("id"), "error": f"{type(e).__name__}: {e}"[:200]}

    msgs = state.get("values", {}).get("messages", [])
    calls: list[dict] = []          # one per tool call, in order
    tool_out_chars: dict[str, int] = collections.defaultdict(int)
    out_by_index: list[int] = []

    pending: list[dict] = []
    for m in msgs:
        if m.get("type") == "ai":
            for tc in m.get("tool_calls") or []:
                cmds = commands_in(tc)
                entry = {
                    "tool": tc.get("name"),
                    "cmds": cmds,
                    "verbs": [v for c in cmds for v in browse_verbs(c)],
                }
                calls.append(entry)
                pending.append(entry)
        elif m.get("type") == "tool":
            body = text_of(m.get("content"))
            if pending:
                e = pending.pop(0)
                e["out_chars"] = len(body)
                tool_out_chars[e["tool"] or "?"] += len(body)
                out_by_index.append(len(body))

    verbs = [v for c in calls for v in c["verbs"]]
    all_cmds = [c for call in calls for c in call["cmds"]]

    # A command issued more than once with no intervening navigation is the
    # clearest signal of a loop the agent could not see it was in.
    cmd_counts = collections.Counter(c.strip() for c in all_cmds)
    repeated = {c: n for c, n in cmd_counts.items() if n > 2}

    # Consecutive identical verbs: snapshot, snapshot, snapshot...
    longest_repeat, cur, prev = 0, 0, None
    for v in verbs:
        cur = cur + 1 if v == prev else 1
        prev = v
        longest_repeat = max(longest_repeat, cur)

    big_reads = sum(1 for n in out_by_index if n > 20_000)

    # Model round-trips are the expensive unit, not browse commands. A run that
    # chains four verbs per `execute` pays a quarter the round-trips of one that
    # sends them one at a time, for identical browsing.
    shell_calls = sum(1 for c in calls if c["verbs"])
    chain_ratio = round(len(verbs) / shell_calls, 2) if shell_calls else 0.0

    return {
        "id": rec.get("id"),
        "site": rec.get("site"),
        "grade": rec.get("grade"),
        "status": rec.get("status"),
        "seconds": rec.get("seconds"),
        "run_id": rec.get("run_id"),
        "messages": len(msgs),
        "tool_calls": len(calls),
        "shell_calls": shell_calls,
        "chain_ratio": chain_ratio,
        "verbs": collections.Counter(verbs),
        "tool_names": collections.Counter(c["tool"] for c in calls),
        "tool_out_chars": dict(tool_out_chars),
        "total_out_chars": sum(out_by_index),
        "big_reads": big_reads,
        "longest_verb_repeat": longest_repeat,
        "repeated_commands": repeated,
        "answer_chars": len(rec.get("answer") or ""),
        "answer": (rec.get("answer") or "")[:600],
        "commands": all_cmds[:400],
    }


async def run_usage(http: httpx.AsyncClient, run_id: str) -> dict:
    """Token totals for one run. GET-by-id, not /query -- far cheaper on the limiter."""
    if not run_id:
        return {}
    for attempt in range(4):
        try:
            r = await http.get(f"{LS}/{run_id}", headers={"x-api-key": KEY})
            if r.status_code == 429:
                await asyncio.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            d = r.json()
            if not isinstance(d, dict):
                return {}
            return {
                "total_tokens": d.get("total_tokens") or 0,
                "prompt_tokens": d.get("prompt_tokens") or 0,
                "completion_tokens": d.get("completion_tokens") or 0,
                "total_cost": float(d.get("total_cost") or 0),
            }
        except Exception:
            await asyncio.sleep(2 ** attempt)
    return {}


def report(rows: list[dict]) -> str:
    ok = [r for r in rows if not r.get("error")]
    out: list[str] = ["# Trace analysis\n"]

    verbs = collections.Counter()
    for r in ok:
        verbs.update(r.get("verbs") or {})
    total_verbs = sum(verbs.values()) or 1

    tools = collections.Counter()
    for r in ok:
        tools.update(r.get("tool_names") or {})
    out.append("## Tools\n")
    out.append("| tool | calls |")
    out.append("| --- | ---: |")
    for t, n in tools.most_common():
        out.append(f"| `{t}` | {n} |")

    shell = sum(r.get("shell_calls", 0) for r in ok)
    if shell:
        n_verbs = sum(verbs.values())
        out.append(
            f"\nChaining: {n_verbs} browse verbs across {shell} shell calls "
            f"= {n_verbs / shell:.2f} verbs per model round-trip."
        )

    out.append("\n## Where the steps go\n")
    out.append("| browse verb | calls | share |")
    out.append("| --- | ---: | ---: |")
    for v, n in verbs.most_common(20):
        out.append(f"| `{v}` | {n} | {100 * n / total_verbs:.1f}% |")

    out.append("\n## Per task\n")
    out.append("| task | grade | calls | chain | out chars | big reads | max repeat | tokens | $ | s |")
    out.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in sorted(ok, key=lambda x: -(x.get("tool_calls") or 0)):
        u = r.get("usage") or {}
        out.append(
            f"| {r['id']} | {r.get('grade','-')} | {r.get('tool_calls',0)} | "
            f"{r.get('chain_ratio',0)} | "
            f"{r.get('total_out_chars',0):,} | {r.get('big_reads',0)} | "
            f"{r.get('longest_verb_repeat',0)} | {u.get('total_tokens',0):,} | "
            f"{u.get('total_cost',0):.3f} | {r.get('seconds','-')} |"
        )

    # Token spend per task is superlinear in steps: every step re-sends the
    # whole transcript, page reads included. Showing both columns is what makes
    # the "one big read early costs you on every later step" effect visible.
    tok = [(r.get("usage") or {}).get("total_tokens", 0) for r in ok]
    calls_ = [r.get("tool_calls", 0) for r in ok]
    if tok and sum(tok):
        n = len(tok)
        out.append(
            f"\n**Totals.** {n} tasks, {sum(calls_)} tool calls, "
            f"{sum(tok):,} tokens, "
            f"${sum((r.get('usage') or {}).get('total_cost', 0) for r in ok):.2f}. "
            f"Median {sorted(calls_)[n // 2]} calls / "
            f"{sorted(tok)[n // 2]:,} tokens per task."
        )

    out.append("\n## Repeated commands (issued 3+ times in one thread)\n")
    any_rep = False
    for r in ok:
        rep = r.get("repeated_commands") or {}
        if rep:
            any_rep = True
            out.append(f"\n**{r['id']}** ({r.get('grade','-')})")
            for c, n in sorted(rep.items(), key=lambda kv: -kv[1])[:8]:
                out.append(f"- {n}x `{c[:160]}`")
    if not any_rep:
        out.append("_none_")

    return "\n".join(out) + "\n"


async def main() -> None:
    src, dst = sys.argv[1], sys.argv[2]
    recs = [json.loads(l) for l in open(src) if l.strip()][:MAX_THREADS]
    client = get_client(url=URL, api_key=KEY)

    rows = [r for r in await asyncio.gather(
        *(thread_detail(client, rec) for rec in recs)) if r]

    async with httpx.AsyncClient(timeout=TIMEOUT) as http:
        # Serial on purpose: the LangSmith limiter punishes a burst of these,
        # and it stays 429'd long enough to poison later requests.
        for r in rows:
            if not r.get("error"):
                r["usage"] = await run_usage(http, r.get("run_id"))

    with open(dst, "w") as f:
        f.write(report(rows))
    with open(dst.replace(".md", ".json"), "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"wrote {dst}")


asyncio.run(main())
