"""Official OOLONG-synth scorer, vendored from the reference implementation.

Source: ``src/eval/eval_helpers.py`` in https://github.com/abertsch72/oolong
(commit on ``main``), MIT License, Copyright (c) 2025 Amanda Bertsch.

Only the two ``synth_*`` functions used for the synthetic-aggregation split are
vendored here, verbatim in behavior (whitespace normalized to satisfy our
linter; the parsing/scoring logic is unchanged). Keeping the official functions
— rather than a paraphrase — guarantees our ``score`` matches the numbers the
paper and the north-star LangSmith dataset report. Do not "improve" these; if
upstream changes, re-vendor.

`synth_attempt_answer_parse` extracts the candidate answer from a model
response. `synth_process_response` grades it against the dataset gold and
returns a record whose ``score`` is the authoritative correctness signal
(1.0 exact / comparison match, fractional partial credit for NUMERIC, dateutil
equality for DATE).
"""

from __future__ import annotations

import ast
from datetime import datetime
from typing import Any

import dateutil.parser

# Vendored verbatim from upstream (MIT) — excluded from ruff + ty in pyproject.toml
# to preserve the official code's exact behavior (broad excepts, magic numbers,
# naive datetime, split() semantics, loose typing). Do not "fix" these.


def synth_attempt_answer_parse(answer: str) -> tuple[str, str]:
    """Parse a candidate answer + confidence out of a model response.

    Verbatim from the official OOLONG harness: split on ``:``, take the last
    segment, strip markdown decorators, and normalize comparison phrases.
    """
    parse_confidence = "low"
    if ":" not in answer:  # bad start
        if len(answer) < 20:  # it's short, return the whole thing
            return answer, parse_confidence
        return answer.split()[-1], parse_confidence
    candidate_answer = answer.split(":")[-1].strip()
    candidate_answer = candidate_answer.replace("*", "")  # OpenAI models like bolding
    candidate_answer = candidate_answer.replace("[", "")
    candidate_answer = candidate_answer.replace("]", "")  # Anthropic models use []
    parse_confidence = "med"
    if "User:" in answer or "Answer:" in answer or "Date:" in answer or "Label" in answer:
        parse_confidence = "high"
    if len(candidate_answer) < 20:
        parse_confidence = "vhigh"
    elif "more common" in candidate_answer:
        candidate_answer = "more common"
    elif "less common" in candidate_answer:
        candidate_answer = "less common"
    elif "same frequency" in candidate_answer:
        candidate_answer = "same frequency"

    return candidate_answer, parse_confidence


def synth_process_response(datapoint: dict[str, Any], output: str, model: str) -> dict[str, Any]:
    """Grade ``output`` against ``datapoint`` gold; verbatim official scoring.

    ``datapoint`` must carry ``answer`` (raw stringified gold), ``answer_type``,
    ``id``, ``context_window_id`` and ``dataset``. Returns a record whose
    ``score`` is the authoritative correctness value.
    """
    score: float = 0
    gold = (
        ast.literal_eval(datapoint["answer"])[0]
        if "datetime" not in datapoint["answer"]
        else datetime.strptime(datapoint["answer"], "[datetime.date(%Y, %m, %d)]")
    )

    trimmed_output, parse_confidence = synth_attempt_answer_parse(output)
    if str(trimmed_output) == str(gold):
        score = 1
    elif str(trimmed_output) in ["more common", "less common", "same frequency"]:
        # account for these being slightly different wordings
        if str(trimmed_output) in str(gold):
            score = 1
    elif datapoint["answer_type"] == "ANSWER_TYPE.NUMERIC":  # partial credit for numbers
        try:
            trimmed_output = int(trimmed_output)
            gold = int(gold)
            score = 0.75 ** (abs(gold - trimmed_output))
        except Exception:
            parse_confidence = "low"  # didn't parse as a number, that's a bad sign
    elif datapoint["answer_type"] == "ANSWER_TYPE.DATE":
        try:
            trimmed_output = dateutil.parser.parse(trimmed_output)
            score = trimmed_output == gold
        except Exception:
            parse_confidence = "low"  # didn't parse as a date, that's a bad sign

    return {
        "id": datapoint["id"],
        "context_window_id": datapoint["context_window_id"],
        "dataset": datapoint["dataset"],
        "model": model,
        "attempted_parse": str(trimmed_output),
        "parse_confidence": parse_confidence,
        "full_answer": output,
        "score": score,
        "answer": str(gold),
    }
