"""Score a WebTailBench answer against the rubric the dataset ships.

WebVoyager rows carry a reference answer, so the Jev judge there asks two
yes/no questions. WebTailBench rows carry a `precomputed_rubric` of 3-6
criteria instead, and the useful score is the fraction met -- a task where the
agent found the flights but not the fare difference is not simply "wrong".

Each criterion is asked separately rather than all at once: a single prompt
holding six criteria reliably produced one blended judgement, which is the
same miscalibration that dragged an exact-match answer to 0.56 earlier in this
benchmark's history.
"""
from __future__ import annotations

import os

import httpx

JEV_URL = "https://gateway.smith.langchain.com/v1/systemone"
JEV_MODEL = os.environ.get("JEV_MODEL", "typesafe/jev-latest")


def _ask(state: str, questions: dict) -> dict:
    response = httpx.post(
        JEV_URL,
        timeout=90,
        headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"],
                 "Authorization": f"Bearer {os.environ['ANTHROPIC_API_KEY']}",
                 "Content-Type": "application/json"},
        json={"model": JEV_MODEL, "state": state, "questions": questions},
    )
    response.raise_for_status()
    return response.json()["answers"]


def grade_rubric(ques: str, answer: str, rubric: dict) -> dict | None:
    """Return {"score": fraction met, "met": n, "total": n, "blocked": 0/1}."""
    if not answer:
        return None
    items = (rubric or {}).get("items") or []
    if not items:
        return None

    met = 0
    for item in items:
        criterion = item.get("criterion", "")
        detail = item.get("description", "")
        state = (f"TASK\n{ques}\n\nCRITERION\n{criterion}\n\n"
                 f"HOW THIS CRITERION IS SCORED\n{detail}\n\n"
                 f"AGENT ANSWER\n{answer[:6000]}")
        # Defer to the item's own full-credit rule rather than inventing one.
        # Each WebTailBench item spells out what earns full, partial and no
        # credit -- including, often, that a correctly reported blocker is
        # full credit. Two earlier versions got this wrong in opposite
        # directions: an asymmetric `false` clause floored every criterion to
        # 0.16-0.49, and the fix for it rewarded merely *mentioning* a
        # criterion, giving 5/5 to an answer that never opened a job listing.
        answers = _ask(state, {"met": {
            "type": "noul",
            "instructions": (
                "Using the scoring rule above, does the answer earn FULL credit "
                "for this criterion?"
            ),
            "criteria": {
                "true": "The answer does what the rule requires for full credit.",
                "false": "The answer falls short of what the rule requires for full credit.",
            },
        }})
        met += float(answers["met"]["noul"]) >= 0.5

    blocked = _ask(
        f"TASK\n{ques}\n\nAGENT ANSWER\n{answer[:6000]}",
        {"blocked": {
            "type": "noul",
            "instructions": "Did the site block the agent from completing the task?",
            "criteria": {"true": "The agent reports being blocked, walled, or denied access.",
                         "false": "The agent completed the task or failed for another reason."},
        }},
    )
    return {"score": met / len(items), "met": met, "total": len(items),
            "blocked": float(blocked["blocked"]["noul"])}
