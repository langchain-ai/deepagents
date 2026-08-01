"""In-sandbox reimplementation of DRBench's ``insights_recall`` metric.

Reproduces upstream ServiceNow/drbench scoring for one task so our reward matches the
benchmark's own number rather than a bespoke rubric:

* the report is first split into atomic claim/citation pairs by the prompt in
  ``drbench.agents.utils.break_report_to_insights``, truncating the report at 60,000
  characters exactly as ``drbench.score_report`` does;
* recall is then ``drbench.metrics.qa_similarity_v2.QASimilarityV2``: one judge call
  per ground-truth insight, asking whether the report's claims contain enough to
  derive it, scored 1.0 for ``yes`` and 0.0 for anything else, using the upstream
  ``prompts/eval_metrics/insight_scoring.txt`` prompt verbatim;
* ``reward = mean(per-insight scores)``, and 0.0 when there are no insights;
* both stages retry three times on unparseable output, matching upstream's
  ``max_retries``; a claim split that never parses yields no claims, which scores 0.0
  the same way upstream does.

Only ``qa_type == "insight"`` ground truth reaches ``case.json`` (the adapter filters
it), which is the same subset upstream's ``compute`` scores; distractors are excluded.

Deviations from strict upstream, both set by the deepagents harness rather than this
file: the judge model comes from ``JUDGE_MODELS`` instead of upstream's configured
evaluation model, and the report is read from ``/app/report.md`` (the harness answer
channel). Credentials and judge selection come from the verifier environment the
harness injects (``OPENAI_API_KEY``, ``OPENAI_BASE_URL``, ``JUDGE_MODELS``); nothing
is hardcoded and the key is never printed.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

_CASE_PATH = Path("/tests/case.json")
_REPORT_PATH = Path("/app/report.md")
_REWARD_PATH = Path("/logs/verifier/reward.txt")
_BREAKDOWN_PATH = Path("/logs/verifier/insights_recall.json")

# drbench.score_report.MAX_REPORT_LENGTH
MAX_REPORT_LENGTH = 60_000
# QASimilarityV2(max_retries=3) and break_report_to_insights(max_retries=3)
MAX_RETRIES = 3

# Verbatim from drbench.agents.utils.break_report_to_insights.
_CLAIM_SPLIT_PROMPT = """
    Please break down the following report text into insight claims. Each insight claim should be:
    1. A single insight, that might include multiple statements and claims
    2. Independent and self-contained
    3. Each claim can have more than one sentence, but should be focused on a single insight
    4. Support each insight with citations from the report text following these specific rules:
       - Usually at the end of the report there is a list of citations with numbers
       - In the report text, citations are referenced with numbers in square brackets like [1], [2], [^1], etc.
       - When providing citations, write the actual name of the cited documents (file names or URLs), NOT the reference numbers
       - Do NOT find citations based on your general knowledge - only use citations that are explicitly presented in the report
       - For each insight you extract, look for citation markers in that specific text section and find the corresponding citation names from the reference list
       - If no citations are found for a specific insight, leave citations empty
    5. Citations should be in one of these formats (various formats will be automatically normalized):
       - Valid URLs: "https://www.example.com/article", "https://techcrunch.com/2023/report.html"
       - File names: "quarterly_report.pdf", "market_analysis.docx", "shared/file.pdf"
       - MatterMost chats: "MatterMost-Channel-Team-User" or natural descriptions like "Mattermost Message - Enterprise Chat (User: john.doe, Team: Compliance, Channel: General)"
       - Email messages: "RoundCube-from@email-to@emails-Subject" or the full reference line like "**Re: Budget Review** - Email from jane.doe@company.com on 15 Jan 2024"
       - IMPORTANT: for email citations always include the sender address and date so the email can be identified
       - NOTE: Various citation formats are supported and will be automatically normalized during evaluation
    6. Do not include general summaries, opinions, or claims that lack citation, just the sentences that are facts.
    7. Each claim should be a concise but complete sentence.

    ## Report text:
    <START OF REPORT>
    {report_text}
    <END OF REPORT>

    ## Output format:
    Please return the insight claims as a JSON array. For example:
    [
        {{
            "claim": "The company's revenue increased by 15% in Q3 2023",
            "citations": ["https://techfundingnews.com/salesforce-acquires-convergence-how-it-competes-with-openai-anthropic-and-googles-duet-ai/"]
        }},
        {{
            "claim": "The new product launch contributed to the growth",
            "citations": ["product_launch.pdf", "quarterly_report.pdf"]
        }},
        {{
            "claim": "Compliance team discussed FSMA requirements",
            "citations": ["MatterMost-fsma_compliance-compliance_team-john.doe"]
        }},
        {{
            "claim": "Budget review meeting was scheduled for next quarter",
            "citations": ["**Q2 Budget Review and Planning** - Email from jason.kim@leesmarket.com on 10 Mar 2024"]
        }},
        {{
            "claim": "This claim has no supporting citations",
            "citations": []
        }}
    ]

    Return only valid JSON, no additional text.
"""

# Verbatim from drbench/prompts/eval_metrics/insight_scoring.txt.
_INSIGHT_SCORING_PROMPT = """Your goal is to check if one of the Predicted Insights extracted from a report is a Golden Insight. You must be STRICT and pay attention to every small detail.

Instructions:
* Evaluate if the Predicted Insights contain sufficient information to derive a Golden Insight.
* Select the insight that most closely matches the Golden Insight. Select one and only one insight.
* Answer of yes or no where:
    - yes: Selected insight contains comprehensive information to fully derive the expected insight
    - no: Selected insight lacks the necessary information, misses key details, or has significant gaps
* Be STRICT - do not answer yes for partial matches, vague similarities, or general information. However, no exact wording is required and paraphrasing is acceptable.
* IMPORTANT: Only consider details given in the Golden Insight when answering yes or no. Don't expect anything more than what is given in the Golden Insight.
* Focus on factual accuracy, completeness, and specificity.

Predicted Insights:
{claims_text}

Golden Insight:
{gold_insight}

Return a valid json dictionary with the following structure:
{{
    "answer": "the answer yes or no as described above",
    "confidence": "how confident you are that your answer is correct",
    "justification": "a detailed justification explaining exactly what information given in the golden insight is present or missing in the selected insight",
    "selected_insight": "the claim text of the selected insight that most closely matches the golden insight, but not the number of the selected insight",
    "golden_insight": {gold_insight}
}}

Ensure only a json dictionary is returned, and return nothing else."""  # noqa: W291


def _judge_model() -> str:
    """First model in the harness-injected ``JUDGE_MODELS`` (or a fallback)."""
    raw = os.environ.get("JUDGE_MODELS") or os.environ.get("JUDGE_MODEL") or "gpt-5.6-luna"
    for token in re.split(r"[\s,]+", raw.strip()):
        if token:
            return token
    return "gpt-5.6-luna"


def _temperature(model: str) -> float:
    """Reasoning judges (o1/o3/gpt-5) reject temperature 0.0 at the API, so use 1.0."""
    if model.startswith(("o1", "o3")) or "gpt-5" in model.lower():
        return 1.0
    return 0.0


def _call_judge(prompt: str, model: str) -> str:
    """POST one prompt to the OpenAI-compatible judge and return its message text."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        msg = "OPENAI_API_KEY not set"
        raise RuntimeError(msg)
    base_url = (os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": _temperature(model),
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
        with urllib.request.urlopen(request, timeout=180) as response:  # noqa: S310
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        # Surface the API's reason (e.g. an unsupported-temperature 400) rather than a
        # bare "HTTP Error 400"; the body carries no credentials.
        detail = exc.read().decode("utf-8", "replace")[:500]
        msg = f"HTTP {exc.code} from judge: {detail}"
        raise RuntimeError(msg) from exc
    return str(payload["choices"][0]["message"]["content"])


def _extract_json(response: str) -> Any:
    """Parse the first JSON array or object in a judge response.

    Mirrors upstream `qa_similarity_v2.extract_json_from_response`.
    """
    match = re.search(r"(\[.*\]|\{.*\})", response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(response.strip())
    except json.JSONDecodeError as exc:
        msg = "Could not extract valid JSON from response"
        raise ValueError(msg) from exc


def _split_report(report_text: str, model: str) -> list[dict[str, Any]]:
    """Break the report into atomic claim/citation pairs."""
    prompt = _CLAIM_SPLIT_PROMPT.format(report_text=report_text)
    last_error = ""
    for _ in range(MAX_RETRIES):
        try:
            parsed = _extract_json(_call_judge(prompt, model))
            if not isinstance(parsed, list):
                msg = "claim split did not return a JSON array"
                raise ValueError(msg)  # noqa: TRY301 - retried below like upstream
            return [item for item in parsed if isinstance(item, dict)]
        except Exception as exc:  # noqa: BLE001 - retry, then fall through to no claims
            last_error = f"{type(exc).__name__}: {exc}"
    print(f"claim split failed after {MAX_RETRIES} attempts: {last_error}")
    return []


def _format_claims(claims: list[dict[str, Any]]) -> str:
    """Render claims as the numbered list the scoring prompt expects.

    Mirrors upstream `QASimilarityV2._format_claims_as_text`.
    """
    if not claims:
        return "No claims found in the report."
    return "\n".join(
        f"Insight {index}: {claim.get('claim', '')}" for index, claim in enumerate(claims, 1)
    ).strip()


def _score_insight(claims_text: str, gold_insight: str, model: str) -> dict[str, Any]:
    """Judge whether the report's claims cover one gold insight."""
    prompt = _INSIGHT_SCORING_PROMPT.format(
        claims_text=claims_text,
        gold_insight=gold_insight,
    )
    last_error = ""
    for _ in range(MAX_RETRIES):
        try:
            parsed = _extract_json(_call_judge(prompt, model))
            if not isinstance(parsed, dict):
                msg = "insight scoring did not return a JSON object"
                raise ValueError(msg)  # noqa: TRY301 - retried below like upstream
            answer = str(parsed.get("answer", "")).strip().lower()
            return {
                "score": 1.0 if answer == "yes" else 0.0,
                "answer": answer,
                "justification": str(parsed.get("justification", "")),
                "selected_insight": str(parsed.get("selected_insight", "")),
            }
        except Exception as exc:  # noqa: BLE001 - retry, then score 0.0 like upstream
            last_error = f"{type(exc).__name__}: {exc}"
    return {
        "score": 0.0,
        "answer": "no",
        "justification": f"Failed to parse model response after {MAX_RETRIES} retries: {last_error}",
        "selected_insight": "",
    }


def _grade() -> tuple[float, dict[str, Any]]:
    """Return the insights-recall reward and a per-insight breakdown."""
    case = json.loads(_CASE_PATH.read_text(encoding="utf-8"))
    insights = [
        insight
        for insight in case.get("insights", [])
        if isinstance(insight, dict) and str(insight.get("answer", "")).strip()
    ]
    breakdown: dict[str, Any] = {
        "task_id": case.get("task_id", ""),
        "insight_count": len(insights),
        "per_insight": [],
    }
    if not insights:
        print("case.json declares no gold insights; scoring 0.0")
        return 0.0, breakdown
    if not _REPORT_PATH.is_file():
        print(f"no {_REPORT_PATH}; scoring 0.0")
        breakdown["error"] = "report missing"
        return 0.0, breakdown

    report_text = _REPORT_PATH.read_text(encoding="utf-8", errors="replace")
    if len(report_text) > MAX_REPORT_LENGTH:
        print(
            f"report is {len(report_text)} characters; using the first {MAX_REPORT_LENGTH}"
        )
        report_text = report_text[:MAX_REPORT_LENGTH]

    model = _judge_model()
    claims = _split_report(report_text, model)
    claims_text = _format_claims(claims)
    breakdown["claim_count"] = len(claims)

    scores = []
    for insight in insights:
        result = _score_insight(claims_text, str(insight["answer"]), model)
        scores.append(result["score"])
        breakdown["per_insight"].append(
            {
                "id": insight.get("id", ""),
                "type": insight.get("type", ""),
                "gold_insight": insight["answer"],
                **result,
            }
        )

    reward = sum(scores) / len(scores)
    breakdown["insights_recall"] = reward
    return reward, breakdown


def main() -> None:
    """Score the report and write Harbor's reward plus a per-insight breakdown."""
    try:
        reward, breakdown = _grade()
    except Exception as exc:  # noqa: BLE001 - a verifier crash must still write a reward
        print(f"grading failed: {type(exc).__name__}: {exc}")
        reward, breakdown = 0.0, {"error": f"{type(exc).__name__}: {exc}"}

    reward = max(0.0, min(1.0, reward))
    _REWARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    _REWARD_PATH.write_text(f"{reward}\n", encoding="utf-8")
    try:
        _BREAKDOWN_PATH.write_text(
            json.dumps(breakdown, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    except OSError as exc:
        print(f"could not write breakdown: {exc}")
    print(f"reward={reward}")


if __name__ == "__main__":
    main()
