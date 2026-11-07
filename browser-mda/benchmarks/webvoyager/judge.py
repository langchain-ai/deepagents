"""LLM-as-judge over WebVoyager results.

NOTE: the official WebVoyager protocol judges with a vision model over the
agent's screenshots. This judges the final text answer only, so it can catch a
wrong/evasive/non-specific answer but cannot catch a confident answer that the
page did not actually support. Treat scores as an upper bound.
"""
import json, os, sys, httpx

GW = "https://gateway.smith.langchain.com/anthropic/v1/messages"
KEY = os.environ["ANTHROPIC_API_KEY"]
RUBRIC = """You are grading a web-browsing agent on a WebVoyager task.

TASK: {ques}
START SITE: {site}
AGENT ANSWER: {answer}

Grade one of:

SUCCESS  - the answer directly and specifically answers the task with concrete
           details (names, numbers, titles) that a correct completion would
           contain. Also SUCCESS when the agent establishes from the site that
           the requested thing does not exist, and says so plainly -- a correct
           negative is a correct answer, not a failure.
BLOCKED  - the agent reports that the site blocked it (bot wall, captcha, login
           required, 403) and did not fabricate an answer. Judge this on whether
           it was blocked, not on how politely it said so.
FAILURE  - the answer is vague, non-specific, self-contradictory, does not
           address what was asked, or asserts a result the agent did not
           actually obtain from a page.

A verbose answer that nonetheless contains the correct specifics is SUCCESS;
judge substance, not tidiness. An answer that is mostly correct but carries
honest caveats about what could not be verified is still SUCCESS.

Reply with exactly one word: SUCCESS, BLOCKED, or FAILURE."""

def grade(rec):
    if rec["status"] != "ok" or not rec.get("answer"):
        return rec["status"].upper()
    body = {"model": "claude-sonnet-4-6", "max_tokens": 8,
            "messages": [{"role": "user", "content": RUBRIC.format(
                ques=rec["ques"], site=rec["site"], answer=rec["answer"][:4000])}]}
    r = httpx.post(GW, json=body, timeout=90,
                   headers={"x-api-key": KEY, "anthropic-version": "2023-06-01"})
    r.raise_for_status()
    txt = "".join(b.get("text", "") for b in r.json()["content"]).strip().upper()
    for v in ("SUCCESS", "BLOCKED", "FAILURE"):
        if v in txt: return v
    return "FAILURE"

recs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
out = []
for rec in recs:
    try: rec["grade"] = grade(rec)
    except Exception as e: rec["grade"] = "JUDGE_ERROR"; rec["judge_error"] = str(e)[:200]
    out.append(rec)
    print(f"{rec['grade']:12} {rec['id']}")
with open(sys.argv[2], "w") as f:
    for r in out: f.write(json.dumps(r) + "\n")
