"""In-sandbox dual-judge grader for one LoHoSearch task.

Reproduces the benchmark's published scoring: each response is graded twice --
once with the BrowseComp grading prompt and once with the SimpleQA grading
prompt, using two independent judge models -- and the reward is the mean of the
two binary verdicts, so a task scores 0.0, 0.5, or 1.0. Averaging two
complementary judges is what the LoHoSearch authors do to avoid the
over-strictness or over-leniency of any single setup.

Both grader prompts are verbatim from openai/simple-evals. Judge selection and
credentials come from the verifier environment the harness injects
(`LOHO_JUDGE_A_*`, `LOHO_JUDGE_B_*`, falling back to `OPENAI_API_KEY` /
`OPENROUTER_API_KEY`); nothing is hardcoded and keys are never printed.

A judge that answers "incorrect" scores 0 for its half -- that is a real grade.
A judge that cannot be *reached* is different: it is not evidence about the
answer, so it fails the trial rather than scoring 0, because a silently halved
ceiling still produces a number that looks publishable.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

_CASE_PATH = Path("/tests/case.json")
_SUBMISSION_PATH = Path("/app/answer.txt")
_REWARD_PATH = Path("/logs/verifier/reward.txt")
# Rich detail, including judge rationales. Rationales quote the answer, so this
# file stays in the sandbox and is never downloaded into an artifact.
_JUDGES_PATH = Path("/logs/verifier/judges.json")
# Plaintext-free by construction: model, whether the call succeeded, and the
# verdict token. Nothing derived from the question, answer, or submission. This
# is the one verifier file the workflow downloads, so a 0.0 can be told apart
# from a judge that errored and fell back to 0.
_STATUS_PATH = Path("/logs/verifier/judge_status.json")

_BROWSECOMP_PROMPT_PATH = Path("/tests/browsecomp_grader.txt")
_SIMPLEQA_PROMPT_PATH = Path("/tests/simpleqa_grader.txt")

_DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
# Judge B substitutes Qwen2.5-72B for the paper's Qwen2.5-32B: the 32B is not
# served by any provider configured for these runs (Fireworks 404s on it,
# OpenRouter carries 72B/7B/coder-32B/VL-72B, Groq only Qwen3). Same family and
# generation, one size up, so grading behavior should be the closer match than
# jumping a generation would be. Scores are therefore LoHoSearch-derived rather
# than directly comparable to the published 34.74%.
_DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

_MAX_ATTEMPTS = 5
_TIMEOUT_SECONDS = 120

# Substituted in a single pass so a value containing a placeholder string cannot
# cascade into another slot.
_PLACEHOLDER_RE = re.compile(r"\{(question|response|correct_answer|target|predicted_answer)\}")

# Scrub anything token-shaped out of API error bodies before they reach the logs.
_SECRET_RE = re.compile(r"\b(?:sk|fw)-[A-Za-z0-9_-]{8,}")


def _render(template, **fields):
    return _PLACEHOLDER_RE.sub(lambda m: fields.get(m.group(1), m.group(0)), template)


def _temperature(model):
    """Reasoning judges (o1/o3/gpt-5) reject temperature 0.0 at the API."""
    if model.startswith(("o1", "o3")) or "gpt-5" in model.lower():
        return 1.0
    return 0.0


class _JudgeHTTPError(RuntimeError):
    """An HTTP failure from a judge endpoint, carrying its status code.

    The status is the single most useful triage signal (401 wrong key, 404
    unknown model, 429 rate limit) and is not sensitive, so it survives into the
    downloadable status file while the response body does not.
    """

    def __init__(self, status, detail):
        super().__init__(f"HTTP {status} from judge: {detail}")
        self.status = status


def _call(base_url, api_key, model, prompt):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": _temperature(model),
    }
    request = urllib.request.Request(  # noqa: S310 - OpenAI-compatible host from env
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response:  # noqa: S310
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        # Surface the API's own reason (e.g. an unsupported-temperature 400)
        # rather than a bare status, with anything token-shaped scrubbed first.
        detail = _SECRET_RE.sub("<redacted>", exc.read().decode("utf-8", "replace")[:200])
        raise _JudgeHTTPError(exc.code, detail) from exc
    return payload["choices"][0]["message"]["content"]


def _judge(name, prompt, default_model, default_base_url, fallback_key_var):
    """Call one judge and return (correct, detail). Any failure scores 0."""
    model = os.environ.get(f"LOHO_JUDGE_{name}_MODEL") or default_model
    base_url = os.environ.get(f"LOHO_JUDGE_{name}_BASE_URL") or default_base_url
    api_key = os.environ.get(f"LOHO_JUDGE_{name}_API_KEY") or os.environ.get(fallback_key_var)
    if not api_key:
        return 0, {"model": model, "error": f"no API key (LOHO_JUDGE_{name}_API_KEY)"}

    last_error = ""
    last_status = None
    for _ in range(_MAX_ATTEMPTS):
        try:
            return None, {"model": model, "raw": _call(base_url, api_key, model, prompt)}
        except Exception as exc:  # noqa: BLE001 - report, retry, then give up
            last_error = f"{type(exc).__name__}: {exc}"
            last_status = getattr(exc, "status", None)
    return 0, {"model": model, "error": last_error, "http_status": last_status}


def _grade_browsecomp(question, ground_truth, submission):
    prompt = _render(
        _BROWSECOMP_PROMPT_PATH.read_text(encoding="utf-8"),
        question=question,
        response=submission,
        correct_answer=ground_truth,
    )
    score, detail = _judge("A", prompt, "gpt-4.1", _DEFAULT_OPENAI_BASE_URL, "OPENAI_API_KEY")
    if score is not None:
        return score, detail
    # Upstream parses the structured verdict block for `correct: yes|no`.
    match = re.search(r"correct\s*:\s*(yes|no)", detail["raw"], re.IGNORECASE)
    if match is None:
        return 0, {"model": detail["model"], "error": "unparseable verdict", "verdict": None}
    correct = int(match.group(1).lower() == "yes")
    return correct, {"model": detail["model"], "verdict": match.group(1).lower()}


def _grade_simpleqa(question, ground_truth, submission):
    prompt = _render(
        _SIMPLEQA_PROMPT_PATH.read_text(encoding="utf-8"),
        question=question,
        target=ground_truth,
        predicted_answer=submission,
    )
    score, detail = _judge(
        "B",
        prompt,
        "qwen/qwen-2.5-72b-instruct",
        _DEFAULT_OPENROUTER_BASE_URL,
        "OPENROUTER_API_KEY",
    )
    if score is not None:
        return score, detail
    # Upstream takes the first A/B/C and defaults to C (NOT_ATTEMPTED).
    match = re.search(r"(A|B|C)", detail["raw"])
    grade = match.group(0) if match else "C"
    return int(grade == "A"), {"model": detail["model"], "verdict": grade}


def _status_entry(correct, detail):
    """Reduce a judge result to fields that cannot contain task content."""
    return {
        "model": detail.get("model"),
        "called": "error" not in detail,
        "verdict": detail.get("verdict"),
        "correct": correct,
        # Class name only -- an exception message can echo the prompt.
        "error_type": (detail.get("error") or "").split(":")[0] or None,
        # Not sensitive, and the fastest triage signal: 401 key, 404 model, 429 rate.
        "http_status": detail.get("http_status"),
    }


def _write_status(payload):
    _STATUS_PATH.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def main():
    _REWARD_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not _SUBMISSION_PATH.is_file():
        print("no /app/answer.txt; scoring 0.0")
        _REWARD_PATH.write_text("0.0\n", encoding="utf-8")
        _JUDGES_PATH.write_text(json.dumps({"error": "no submission"}) + "\n", encoding="utf-8")
        _write_status({"reward": 0.0, "graded": False, "reason": "no submission"})
        return

    case = json.loads(_CASE_PATH.read_text(encoding="utf-8"))
    question = str(case.get("question", ""))
    ground_truth = str(case.get("ground_truth", ""))
    submission = _SUBMISSION_PATH.read_text(encoding="utf-8", errors="replace")

    browsecomp, browsecomp_detail = _grade_browsecomp(question, ground_truth, submission)
    simpleqa, simpleqa_detail = _grade_simpleqa(question, ground_truth, submission)
    reward = (browsecomp + simpleqa) / 2

    # A judge that could not be reached is not evidence the answer was wrong.
    # Scoring it 0 silently halves the ceiling for every trial and reads as a
    # legitimate result, so an unreachable judge fails the trial instead: no
    # reward file, non-zero exit, and the aggregator counts it as errored rather
    # than folding a transport failure into the score. A judge that answered and
    # said "incorrect" still scores 0 -- that is a real grade.
    unreachable = [
        name
        for name, detail in (("browsecomp", browsecomp_detail), ("simpleqa", simpleqa_detail))
        if "error" in detail
    ]
    if unreachable:
        _write_status(
            {
                "reward": None,
                "graded": False,
                "unreachable": unreachable,
                "browsecomp": _status_entry(browsecomp, browsecomp_detail),
                "simpleqa": _status_entry(simpleqa, simpleqa_detail),
            }
        )
        print(f"judge(s) unreachable: {unreachable}; failing the trial instead of scoring 0")
        raise SystemExit(1)

    _REWARD_PATH.write_text(f"{reward}\n", encoding="utf-8")
    _JUDGES_PATH.write_text(
        json.dumps(
            {
                "reward": reward,
                "browsecomp": {"correct": browsecomp, **browsecomp_detail},
                "simpleqa": {"correct": simpleqa, **simpleqa_detail},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_status(
        {
            "reward": reward,
            "graded": True,
            "browsecomp": _status_entry(browsecomp, browsecomp_detail),
            "simpleqa": _status_entry(simpleqa, simpleqa_detail),
        }
    )
    print(f"reward={reward} browsecomp={browsecomp} simpleqa={simpleqa}")


if __name__ == "__main__":
    main()
