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
import time
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

# Statuses that will never succeed on retry: wrong key, no access, unknown
# model, unprocessable request. Everything else is retried, including a plain
# 400 -- OpenRouter surfaces transient upstream-provider failures as 400, and a
# task observed 400ing on one run graded normally on the next. Treating 4xx as
# uniformly deterministic (as an OpenAI-style API would be) turns a flaky judge
# into a spuriously errored trial.
_NON_RETRYABLE_STATUSES = frozenset({401, 403, 404, 422})
_BACKOFF_CAP_SECONDS = 8

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
    """An HTTP failure from a judge endpoint, carrying triage fields.

    The status code (401 wrong key, 404 unknown model, 429 rate limit) and the
    provider's machine-readable `error.code` / `error.type` are not sensitive, so
    they survive into the downloadable status file. The free-text `error.message`
    does not: providers echo request content into it, which for this benchmark
    means the question.
    """

    def __init__(self, status, body):
        super().__init__(f"HTTP {status} from judge: {body[:200]}")
        self.status = status
        self.code, self.type, self.message = _error_fields(body)


def _error_fields(body):
    """Pull `error.code` / `error.type` / `error.message` from a JSON error body."""
    try:
        error = (json.loads(body) or {}).get("error") or {}
    except ValueError:
        return (None, None, None)
    if not isinstance(error, dict):
        return (None, None, None)
    code, kind, message = error.get("code"), error.get("type"), error.get("message")
    return (
        code if isinstance(code, (str, int)) else None,
        kind if isinstance(kind, str) else None,
        message if isinstance(message, str) else None,
    )


# Longest run of prompt text allowed to appear in a message before it is withheld.
# Short enough that an echoed phrase trips it, long enough that shared boilerplate
# ("Question:", "the model") does not.
_ECHO_WINDOW = 24


def safe_message(message, prompt):
    """Return a provider error message only if it does not echo the request.

    Most 400s carry a template message ("maximum context length is N tokens")
    that names the cause and nothing else, and that is exactly what is needed to
    diagnose a judge failure. Some providers instead quote the offending input
    back, which for this benchmark is the decrypted question. Rather than choose
    between a useless diagnostic and a leak, the message is emitted only when no
    window of the prompt appears in it.

    Args:
        message: The provider's free-text error message, or None.
        prompt: The request text that must not be echoed.

    Returns:
        The message, a withheld marker, or None when there was no message.
    """
    if not message:
        return None
    haystack = message.casefold()
    prompt_cf = prompt.casefold()
    for start in range(0, max(len(prompt_cf) - _ECHO_WINDOW, 0) + 1, _ECHO_WINDOW // 2):
        if prompt_cf[start : start + _ECHO_WINDOW] in haystack:
            return "<withheld: echoes request content>"
    return message[:300]


def _call(base_url, api_key, model, prompt):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": _temperature(model),
    }
    if "openrouter.ai" in base_url:
        # OpenRouter fronts several upstream providers per model and load-balances
        # across them (qwen-2.5-72b has two). When one is unhealthy it returns a
        # bare 400, and plain retries kept landing on the same one -- a task
        # failed all five attempts in one run and graded fine in the next two.
        # Asking OpenRouter to fail over between upstreams fixes that without
        # changing the judge model, which a different-provider fallback would.
        body["provider"] = {"allow_fallbacks": True}
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
        body = _SECRET_RE.sub("<redacted>", exc.read().decode("utf-8", "replace")[:2000])
        raise _JudgeHTTPError(exc.code, body) from exc
    return payload["choices"][0]["message"]["content"]


def _endpoints(name, default_model, default_base_url, default_key_var):
    """Resolve this judge's endpoints, primary first, from the verifier env.

    A judge may declare a fallback on an unrelated provider
    (`LOHO_JUDGE_<n>_FALLBACK_*`). It is only reached once the primary has
    exhausted its retries, so the common case still grades every task with the
    same model; the fallback exists so one provider's outage costs a trial's
    grade rather than the trial.
    """
    endpoints = []
    primary_key = os.environ.get(f"LOHO_JUDGE_{name}_API_KEY") or os.environ.get(default_key_var)
    if primary_key:
        endpoints.append(
            (
                os.environ.get(f"LOHO_JUDGE_{name}_MODEL") or default_model,
                os.environ.get(f"LOHO_JUDGE_{name}_BASE_URL") or default_base_url,
                primary_key,
            )
        )
    fallback_model = os.environ.get(f"LOHO_JUDGE_{name}_FALLBACK_MODEL")
    fallback_key = os.environ.get(f"LOHO_JUDGE_{name}_FALLBACK_API_KEY")
    fallback_base = os.environ.get(f"LOHO_JUDGE_{name}_FALLBACK_BASE_URL")
    if fallback_model and fallback_key and fallback_base:
        endpoints.append((fallback_model, fallback_base, fallback_key))
    return endpoints


def _call_with_retries(model, base_url, api_key, prompt):
    """Try one endpoint up to `_MAX_ATTEMPTS` times. Returns (raw, detail).

    Failure detail is kept separate from the returned success detail on purpose.
    Merging them let a recovered retry return an `error` key alongside a real
    verdict, which downstream reads as "judge unreachable" and fails the trial —
    turning the retry that saved the grade into the thing that discarded it.
    """
    base = {"model": model, "prompt_chars": len(prompt)}
    failure = {}
    for attempt in range(_MAX_ATTEMPTS):
        try:
            raw = _call(base_url, api_key, model, prompt)
        except Exception as exc:  # noqa: BLE001 - report, maybe retry, then give up
            status = getattr(exc, "status", None)
            failure = {
                "error": f"{type(exc).__name__}: {exc}",
                "http_status": status,
                "error_code": getattr(exc, "code", None),
                "error_kind": getattr(exc, "type", None),
                "error_message": safe_message(getattr(exc, "message", None), prompt),
                "attempts": attempt + 1,
            }
            if status in _NON_RETRYABLE_STATUSES:
                break
            if attempt + 1 < _MAX_ATTEMPTS:
                # Backoff, so a rate limit or a brief upstream outage gets a
                # chance to clear instead of burning all five attempts at once.
                time.sleep(min(2**attempt, _BACKOFF_CAP_SECONDS))
        else:
            detail = {**base, "attempts": attempt + 1}
            if failure:
                # Surface that a retry was needed without implying it failed.
                detail["recovered_after"] = attempt
            return raw, detail
    return None, {**base, **failure}


def _judge(name, prompt, default_model, default_base_url, fallback_key_var):
    """Call one judge, trying its fallback endpoint if the primary is exhausted.

    Returns `(None, detail_with_raw)` when a judge answered, or `(0, detail)`
    when none could be reached -- the caller turns the latter into a failed
    trial rather than a zero score.
    """
    endpoints = _endpoints(name, default_model, default_base_url, fallback_key_var)
    if not endpoints:
        return 0, {
            "model": default_model,
            "error": f"no API key (LOHO_JUDGE_{name}_API_KEY)",
        }

    last = {}
    for index, (model, base_url, api_key) in enumerate(endpoints):
        raw, detail = _call_with_retries(model, base_url, api_key, prompt)
        if raw is not None:
            # Record when a fallback graded: the judge model then differs from
            # the one other trials used, which matters when reading the scores.
            return None, {**detail, "raw": raw, "used_fallback": index > 0}
        last = detail
    return 0, last


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
        return 0, {**detail, "error": "unparseable verdict", "verdict": None}
    correct = int(match.group(1).lower() == "yes")
    return correct, {**detail, "verdict": match.group(1).lower()}


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
    return int(grade == "A"), {**detail, "verdict": grade}


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
        # Machine-readable provider fields, never the free-text message (which
        # echoes request content). `prompt_chars` distinguishes a length limit
        # from a content rejection when the status alone is ambiguous.
        "error_code": detail.get("error_code"),
        "error_kind": detail.get("error_kind"),
        "prompt_chars": detail.get("prompt_chars"),
        "error_message": detail.get("error_message"),
        "attempts": detail.get("attempts"),
        "used_fallback": detail.get("used_fallback"),
        "recovered_after": detail.get("recovered_after"),
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
