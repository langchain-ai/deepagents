"""In-sandbox reimplementation of Letta letta-evals' ``RubricGrader``.

Reproduces the upstream ``model_judge`` grader
(`letta_evals/graders/rubric.py`, OpenAI provider) for one Context-Bench task,
so our score matches upstream's grading rather than a string comparison:

* the judge prompt is the upstream rubric (``/tests/rubric.txt``) with
  ``{input}`` / ``{ground_truth}`` / ``{submission}`` substituted via
  ``string.Formatter().vformat`` -- no system prompt, no wrapping;
* the judge is called through Chat Completions with the upstream
  ``json_schema`` response format ``{score: float in [0, 1], rationale}``;
* upstream's temperature rule is preserved: a judge model matching ``o1`` /
  ``o3`` / ``gpt-5`` is called at temperature 1.0, any other model at 0.0. These
  are reasoning models that reject temperature 0.0 at the API (a 0.0 call 400s),
  which is exactly why upstream bumps them; ``gpt-5.6-luna`` is one of them.
* ``score = clamp(float(score), 0.0, 1.0)``; any error scores 0.0, exactly as
  the upstream grader returns 0.0 on exception.

Two deliberate deviations from strict upstream, both set by the deepagents
harness rather than this file:

* the judge model comes from ``JUDGE_MODELS`` (the harness grader, e.g.
  ``gpt-5.6-luna``) instead of upstream's ``gpt-5-mini``;
* the submission is ``/app/answer.txt`` (the harness answer channel) instead of
  the agent's last assistant message.

Credentials and judge selection come from the verifier environment the harness
injects (``OPENAI_API_KEY``, ``OPENAI_BASE_URL``, ``JUDGE_MODELS``,
``JUDGE_PROVIDER``); nothing is hardcoded and the key is never printed.
"""

from __future__ import annotations

import json
import os
import re
import string
import urllib.error
import urllib.request
from pathlib import Path

_CASE_PATH = Path("/tests/case.json")
_RUBRIC_PATH = Path("/tests/rubric.txt")
_SUBMISSION_PATH = Path("/app/answer.txt")
_REWARD_PATH = Path("/logs/verifier/reward.txt")

# Mirrors letta_evals.graders.rubric._JudgeResponse.model_json_schema(); the
# numeric bounds are advisory (re-applied by the clamp below) because not every
# provider honors them, matching the upstream note.
_RESPONSE_SCHEMA = {
    "properties": {
        "score": {
            "description": "Score between 0.0 and 1.0",
            "maximum": 1.0,
            "minimum": 0.0,
            "title": "Score",
            "type": "number",
        },
        "rationale": {
            "description": "Explanation of the grading decision",
            "title": "Rationale",
            "type": "string",
        },
    },
    "required": ["score", "rationale"],
    "title": "JudgeResponse",
    "type": "object",
}


def _judge_model() -> str:
    """First model in the harness-injected ``JUDGE_MODELS`` (or a fallback)."""
    raw = os.environ.get("JUDGE_MODELS") or os.environ.get("JUDGE_MODEL") or "gpt-5.6-luna"
    for token in re.split(r"[\s,]+", raw.strip()):
        if token:
            return token
    return "gpt-5.6-luna"


def _temperature(model: str) -> float:
    """Upstream rule: reasoning judges (o1/o3/gpt-5) reject 0.0, so use 1.0."""
    if model.startswith(("o1", "o3")) or "gpt-5" in model.lower():
        return 1.0
    return 0.0


def _judge_prompt(submission: str) -> str:
    case = json.loads(_CASE_PATH.read_text(encoding="utf-8"))
    rubric = _RUBRIC_PATH.read_text(encoding="utf-8")
    substitutions = {
        "input": str(case.get("input", "")),
        "ground_truth": str(case.get("ground_truth", "")),
        "submission": submission,
    }
    return string.Formatter().vformat(rubric, (), substitutions)


def _call_judge(prompt: str, model: str) -> float:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        msg = "OPENAI_API_KEY not set"
        raise RuntimeError(msg)
    base_url = (os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": _temperature(model),
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "JudgeResponse", "schema": _RESPONSE_SCHEMA},
        },
    }
    request = urllib.request.Request(  # noqa: S310 - fixed OpenAI-compatible host from env
        f"{base_url}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        # Surface the API's reason (e.g. an unsupported-temperature 400) rather
        # than a bare "HTTP Error 400"; the body carries no secrets.
        detail = exc.read().decode("utf-8", "replace")[:500]
        msg = f"HTTP {exc.code} from judge: {detail}"
        raise RuntimeError(msg) from exc
    content = payload["choices"][0]["message"]["content"]
    result = json.loads(content)
    score = float(result["score"])
    return max(0.0, min(1.0, score))


def _grade() -> float:
    if not _SUBMISSION_PATH.is_file():
        print("no /app/answer.txt; scoring 0.0")
        return 0.0
    submission = _SUBMISSION_PATH.read_text(encoding="utf-8", errors="replace")
    model = _judge_model()
    prompt = _judge_prompt(submission)
    last_error = ""
    for _ in range(5):
        try:
            return _call_judge(prompt, model)
        except Exception as exc:  # noqa: BLE001 - report, retry, then score 0.0 like upstream
            last_error = f"{type(exc).__name__}: {exc}"
    print(f"judge failed after retries: {last_error}")
    return 0.0


def main() -> None:
    score = _grade()
    _REWARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    _REWARD_PATH.write_text(f"{score}\n", encoding="utf-8")
    print(f"reward={score}")


if __name__ == "__main__":
    main()
