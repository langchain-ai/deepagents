"""In-sandbox reimplementation of DRBench's report metrics.

Reproduces upstream ServiceNow/drbench scoring for one task so our numbers match the
benchmark's own rather than a bespoke rubric. Four metrics, then one composite:

* ``insights_recall`` — ``drbench.metrics.qa_similarity_v2.QASimilarityV2``. The report is
  first split into atomic claim/citation pairs by the prompt in
  ``drbench.agents.utils.break_report_to_insights`` (truncating at 60,000 characters, as
  ``drbench.score_report`` does), then one judge call per gold insight asks whether the
  claims contain enough to derive it. 1.0 for ``yes``, 0.0 otherwise.
* ``distractor_recall`` — the same loop against the planted distractors
  (``DistractorRecall`` literally subclasses ``QASimilarityV2`` with
  ``insight_key="distractor"``). **Higher is worse**: it means the report swallowed
  material it should have rejected.
* ``report_quality`` — ``drbench.metrics.report_quality.ReportQuality``. One call scoring
  five criteria 1-10, averaged and divided by 10.
* ``factuality`` — ``drbench.metrics.factuality_v2.CitationFactuality``. Each cited claim
  is checked against the source it cites. Sources are re-fetched from the running app
  stack over WebDAV (or from the open web, for URL citations), chunked, ranked by
  embedding similarity, and judged.

The composite is a harmonic mean over ``insights_recall``, ``1 - distractor_recall``,
``factuality``, and ``report_quality``, each floored at ``EPSILON`` so a single zero
craters the score without erasing all ranking signal. **Upstream defines no combined
score**, so this number is ours, not DRBench's, and is not comparable to the paper. Every
component is emitted alongside it for that reason.

Deviations from strict upstream, all set by the deepagents harness rather than this file:
the judge model comes from ``JUDGE_MODELS``; the report is read from ``/app/report.md``;
and chunk ranking computes cosine similarity here rather than via faiss (upstream's
``IndexFlatIP`` over L2-normalized vectors is exactly cosine similarity).

Credentials and judge selection come from the verifier environment the harness injects
(``OPENAI_API_KEY``, ``OPENAI_BASE_URL``, ``JUDGE_MODELS``). Nothing is hardcoded, and no
key is ever printed or written to a reward or breakdown file.
"""

from __future__ import annotations

import base64
import ipaddress
import json
import math
import os
import re
import socket
import subprocess
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

_CASE_PATH = Path("/tests/case.json")
_REPORT_PATH = Path("/app/report.md")
_REWARD_JSON_PATH = Path("/logs/verifier/reward.json")
_BREAKDOWN_PATH = Path("/logs/verifier/drbench_metrics.json")

# drbench.score_report.MAX_REPORT_LENGTH
MAX_REPORT_LENGTH = 60_000
# QASimilarityV2(max_retries=3), break_report_to_insights(max_retries=3)
MAX_RETRIES = 3
# get_most_relevant_chunks(top_k=5, chunk_size=2048, max_chunks=200)
CHUNK_SIZE = 2048
TOP_K = 5
MAX_CHUNKS = 200
# Bound on one fetched document, so a large file cannot dominate a judge prompt.
MAX_DOCUMENT_BYTES = 200_000
# Floor per component in the composite, so one zero does not erase all signal.
EPSILON = 0.01

_URL_RE = re.compile(r"^https?://", re.IGNORECASE)

_METRIC_NAMES = (
    "insights_recall",
    "distractor_recall",
    "distractor_avoidance",
    "factuality",
    "report_quality",
)


def _judge_model() -> str:
    """First model in the harness-injected ``JUDGE_MODELS`` (or a fallback)."""
    raw = os.environ.get("JUDGE_MODELS") or os.environ.get("JUDGE_MODEL") or "gpt-5.6-luna"
    for token in re.split(r"[\s,]+", raw.strip()):
        if token:
            return token
    return "gpt-5.6-luna"


def _embedding_model() -> str:
    """Embedding model for chunk ranking; upstream defaults to text-embedding-3-small."""
    return os.environ.get("JUDGE_EMBEDDING_MODEL") or "text-embedding-3-small"


def _temperature(model: str) -> float:
    """Reasoning judges (o1/o3/gpt-5) reject temperature 0.0 at the API, so use 1.0."""
    if model.startswith(("o1", "o3")) or "gpt-5" in model.lower():
        return 1.0
    return 0.0


def _api_base() -> str:
    """Base URL of the OpenAI-compatible judge endpoint."""
    return (os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")


def _post_json(path: str, body: dict[str, Any], *, timeout: int = 180) -> dict[str, Any]:
    """POST to the judge endpoint and return the parsed response."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        msg = "OPENAI_API_KEY not set"
        raise RuntimeError(msg)
    request = urllib.request.Request(  # noqa: S310 - fixed OpenAI-compatible host from env
        f"{_api_base()}{path}",
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            return json.load(response)
    except urllib.error.HTTPError as exc:
        # Surface the API's reason (e.g. an unsupported-temperature 400) rather than a
        # bare "HTTP Error 400". The body carries no credentials; headers are never read.
        detail = exc.read().decode("utf-8", "replace")[:500]
        msg = f"HTTP {exc.code} from {path}: {detail}"
        raise RuntimeError(msg) from exc


def _call_judge(prompt: str, model: str) -> str:
    """Send one prompt to the judge and return its message text."""
    payload = _post_json(
        "/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": _temperature(model),
        },
    )
    return str(payload["choices"][0]["message"]["content"])


def _embed(texts: list[str]) -> list[list[float]]:
    """Return one embedding vector per input text."""
    payload = _post_json("/embeddings", {"model": _embedding_model(), "input": texts})
    return [item["embedding"] for item in payload["data"]]


def _cosine(left: list[float], right: list[float]) -> float:
    """Cosine similarity — upstream's faiss IndexFlatIP over L2-normalized vectors."""
    norm_left = math.sqrt(sum(value * value for value in left))
    norm_right = math.sqrt(sum(value * value for value in right))
    if norm_left == 0.0 or norm_right == 0.0:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (norm_left * norm_right)


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


# ---------------------------------------------------------------------------------------
# Prompts, verbatim from upstream so our scores match the benchmark's own. The report and
# the fetched sources are untrusted (a model wrote the report, and it chose the sources),
# so both stay inside explicit delimiters rather than being concatenated loose.
# ---------------------------------------------------------------------------------------

# drbench.agents.utils.break_report_to_insights
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
            "claim": "This claim has no supporting citations",
            "citations": []
        }}
    ]

    Return only valid JSON, no additional text.
"""

# drbench/prompts/eval_metrics/insight_scoring.txt
_QA_SCORING_PROMPT = """Your goal is to check if one of the Predicted Insights extracted from a report is a Golden Insight. You must be STRICT and pay attention to every small detail.

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

Ensure only a json dictionary is returned, and return nothing else."""

# drbench.metrics.report_quality.ReportQuality
_QUALITY_CRITERIA = (
    "depth_quality",
    "relevance_to_question",
    "persona_consistency",
    "coherence_conciseness",
    "contradictions",
)

_QUALITY_PROMPT = """You are a Deep Research Evaluator.

You are given:
1. A research report.
2. A deep research (DR) question that the report attempts to answer.
3. A persona that represents the intended audience for the report.

------------------
<persona>
{persona}
</persona>

<dr_question>
{question}
</dr_question>

<report>
{report_text}
</report>
------------------

## Instructions:

**ANALYZE THOROUGHLY**: Examine the report in detail and identify any issues, even small ones.

Evaluate the report according to the five criteria listed below. For **each criterion**, give a **score between 1 and 10** (an integer) using the scale below, plus a **detailed justification** (2-3 sentences) in simple plain English.

### Scoring Scale (1-10, integers only):
- **1-2** = Very poor, major deficiencies, completely inadequate
- **3-4** = Poor, significant problems, below expectations
- **5-6** = Average, meets basic requirements but has notable issues
- **7-8** = Good, meets expectations with minor issues
- **9-10** = Excellent, exceeds expectations with minimal or no issues

### Criteria:
1. **Depth & Quality of Analysis**: How far the report delves into the details of the question, explores multiple factors, and provides a comprehensive and nuanced understanding.
2. **Relevance To DR Question**: How directly the report addresses the stated question, and whether it offers actionable recommendations that address it.
3. **Persona Consistency**: How well the report aligns with the persona's values, goals, expertise, and expected tone.
4. **Coherence & Conciseness**: Whether information is presented in a logical flow with clear connections and without unnecessary jargon.
5. **Degree of Contradictions**: Whether the report contains internal inconsistencies, logical contradictions, or conflicting statements.

------------------

## Output format:

<evaluation>
<depth_quality><score>1-10</score><justification>...</justification></depth_quality>
<relevance_to_question><score>1-10</score><justification>...</justification></relevance_to_question>
<persona_consistency><score>1-10</score><justification>...</justification></persona_consistency>
<coherence_conciseness><score>1-10</score><justification>...</justification></coherence_conciseness>
<contradictions><score>1-10</score><justification>...</justification></contradictions>
</evaluation>"""

# drbench.agents.utils.get_factuality_verdict_multi
_FACTUALITY_PROMPT = """
    Given the following relevant source context from multiple sources and an insight, determine if the insight is factually supported by the sources.

    Relevant Source Materials (from multiple sources):
    <START OF SOURCES>
    {context}
    <END OF SOURCES>

    Atomic Claim: {insight}

    EVALUATION CRITERIA:
    The claim is factual if the core factual content is supported by the sources. You should be strict about important details but flexible about exact wording:

    REQUIRED for TRUE:
    1. All key factual details (numbers, dates, names, percentages, specific facts) must be present in at least one source
    2. The main substance and meaning of the claim must be supported by the source contexts
    3. No part of the claim should contradict the information in any of the sources

    Return a valid json dictionary with the following structure:
    {{
        "is_factual": true or false,
        "explanation": "a short explanation of which source supports the claim, or what is missing"
    }}

    Ensure only a json dictionary is returned, and return nothing else.
"""


def _format_claims(claims: list[dict[str, Any]]) -> str:
    """Render claims as the numbered list the scoring prompt expects.

    Mirrors upstream `QASimilarityV2._format_claims_as_text`.
    """
    if not claims:
        return "No claims found in the report."
    return "\n".join(
        f"Insight {index}: {claim.get('claim', '')}" for index, claim in enumerate(claims, 1)
    ).strip()


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
            claims = []
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                citations = item.get("citations")
                claims.append(
                    {
                        "claim": str(item.get("claim", "")),
                        "citations": [str(c) for c in citations]
                        if isinstance(citations, list)
                        else [],
                    }
                )
            return claims
        except Exception as exc:  # noqa: BLE001 - retry, then fall through to no claims
            last_error = f"{type(exc).__name__}: {exc}"
    print(f"claim split failed after {MAX_RETRIES} attempts: {last_error}")
    return []


def _score_one_entry(claims_text: str, gold: str, model: str) -> dict[str, Any]:
    """Judge whether the report's claims cover one ground-truth entry."""
    prompt = _QA_SCORING_PROMPT.format(claims_text=claims_text, gold_insight=gold)
    last_error = ""
    for _ in range(MAX_RETRIES):
        try:
            parsed = _extract_json(_call_judge(prompt, model))
            if not isinstance(parsed, dict):
                msg = "scoring did not return a JSON object"
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
        "justification": (
            f"Failed to parse model response after {MAX_RETRIES} retries: {last_error}"
        ),
        "selected_insight": "",
    }


def _score_qa_set(
    claims_text: str, entries: list[dict[str, Any]], model: str
) -> tuple[float, list[dict[str, Any]]]:
    """Score one class of ground truth (insights or distractors).

    Returns:
        The mean score and a per-entry breakdown. An empty entry list scores 0.0, which
        is what upstream's `overall_score` does.
    """
    details = []
    for entry in entries:
        result = _score_one_entry(claims_text, str(entry["answer"]), model)
        details.append(
            {
                "id": entry.get("id", ""),
                "type": entry.get("type", ""),
                "gold": entry["answer"],
                **result,
            }
        )
    scores = [detail["score"] for detail in details]
    return (sum(scores) / len(scores) if scores else 0.0), details


def _report_quality(
    report_text: str, question: str, persona: dict[str, Any], model: str
) -> tuple[float, dict[str, Any]]:
    """Score the report against upstream's five quality criteria."""
    prompt = _QUALITY_PROMPT.format(
        persona=json.dumps(persona, indent=2), question=question, report_text=report_text
    )
    last_error = ""
    for _ in range(MAX_RETRIES):
        try:
            response = _call_judge(prompt, model)
            per_criterion: dict[str, float] = {}
            for criterion in _QUALITY_CRITERIA:
                match = re.search(
                    rf"<{criterion}>.*?<score>\s*(.*?)\s*</score>",
                    response,
                    re.DOTALL | re.IGNORECASE,
                )
                if match is None:
                    continue
                try:
                    raw = int(float(match.group(1).strip()))
                except ValueError:
                    continue
                # Upstream captures the score loosely then clamps to 1..10 and /10, so an
                # out-of-range or negative score becomes the floor rather than a parse miss.
                per_criterion[criterion] = max(1, min(10, raw)) / 10.0
            if len(per_criterion) != len(_QUALITY_CRITERIA):
                msg = f"parsed only {len(per_criterion)}/{len(_QUALITY_CRITERIA)} criteria"
                raise ValueError(msg)  # noqa: TRY301 - retried below like upstream
            return sum(per_criterion.values()) / len(_QUALITY_CRITERIA), {
                "per_criterion": per_criterion
            }
        except Exception as exc:  # noqa: BLE001 - retry, then score 0.0 like upstream
            last_error = f"{type(exc).__name__}: {exc}"
    return 0.0, {"error": f"report_quality failed after {MAX_RETRIES} attempts: {last_error}"}


# ---------------------------------------------------------------------------------------
# Factuality: re-fetch each cited source, then judge the claim against it.
# ---------------------------------------------------------------------------------------


def _is_public_host(host: str) -> bool:
    """True when a hostname resolves only to public addresses.

    Citations come from the agent's report, so a cited URL is untrusted input. Without
    this, a citation could point the verifier at cloud instance metadata or another
    service on the runner. The app stack is reached by its own service name through a
    separate path, so it is unaffected by this check.
    """
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return False
    for info in infos:
        address = ipaddress.ip_address(info[4][0])
        if (
            address.is_private
            or address.is_loopback
            or address.is_link_local
            or address.is_reserved
            or address.is_multicast
        ):
            return False
    return bool(infos)


def _basic_auth(credentials: dict[str, str]) -> str:
    """Build a Basic auth header value. Never logged."""
    token = base64.b64encode(
        f"{credentials['username']}:{credentials['password']}".encode()
    ).decode()
    return f"Basic {token}"


def _http_get(
    url: str, *, credentials: dict[str, str] | None = None, timeout: int = 60
) -> bytes:
    """GET a URL, returning at most MAX_DOCUMENT_BYTES."""
    request = urllib.request.Request(url, method="GET")  # noqa: S310 - scheme checked by callers
    if credentials is not None:
        request.add_header("Authorization", _basic_auth(credentials))
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return response.read(MAX_DOCUMENT_BYTES)


def _webdav_index(endpoint: str, credentials: dict[str, str]) -> dict[str, str]:
    """Map each document's lowercased base name to its full WebDAV URL.

    Raises:
        RuntimeError: If the app stack cannot be listed. Failing loudly matters here — an
            empty index would mark every cited claim unfactual, which reads as a bad
            report rather than a broken environment.
    """
    user = credentials["username"]
    root = f"{endpoint}/remote.php/dav/files/{urllib.parse.quote(user)}/"
    request = urllib.request.Request(root, method="PROPFIND")  # noqa: S310 - fixed service host
    request.add_header("Authorization", _basic_auth(credentials))
    request.add_header("Depth", "infinity")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
            body = response.read().decode("utf-8", "replace")
    except Exception as exc:
        msg = f"could not list {root}: {type(exc).__name__}: {exc}"
        raise RuntimeError(msg) from exc

    parsed_endpoint = urllib.parse.urlsplit(endpoint)
    index: dict[str, str] = {}
    for href in re.findall(r"<[^>]*href>([^<]+)</[^>]*href>", body, re.IGNORECASE):
        raw = href.strip()
        if raw.endswith("/"):
            continue
        name = urllib.parse.unquote(raw).rsplit("/", 1)[-1]
        if name:
            index.setdefault(
                name.lower(), f"{parsed_endpoint.scheme}://{parsed_endpoint.netloc}{raw}"
            )
    return index


def _extract_text(payload: bytes, name: str) -> str:
    """Convert a downloaded document to text via the image's `extract-text`."""
    suffix = Path(name).suffix or ".bin"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        handle.write(payload)
        path = handle.name
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["extract-text", path],  # noqa: S607 - resolved from the image's PATH
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if completed.returncode == 0 and completed.stdout.strip():
            return completed.stdout
    except (OSError, subprocess.SubprocessError):
        pass
    finally:
        Path(path).unlink(missing_ok=True)
    return payload.decode("utf-8", "replace")


def _resolve_citation(
    citation: str, index: dict[str, str], credentials: dict[str, str] | None
) -> str | None:
    """Return the text of the source a citation names, or None if unresolvable."""
    citation = citation.strip()
    if not citation:
        return None

    if _URL_RE.match(citation):
        host = urllib.parse.urlsplit(citation).hostname or ""
        if not _is_public_host(host):
            print(f"  skipping cited URL that is not publicly routable: {citation[:120]}")
            return None
        try:
            return _http_get(citation).decode("utf-8", "replace")
        except Exception as exc:  # noqa: BLE001 - an unreachable citation is not support
            print(f"  could not fetch {citation[:120]}: {type(exc).__name__}")
            return None

    # Otherwise treat it as a document name and look it up in the app stack.
    name = citation.rstrip("/").rsplit("/", 1)[-1].lower()
    url = index.get(name)
    if url is None:
        return None
    try:
        payload = _http_get(url, credentials=credentials)
    except Exception as exc:  # noqa: BLE001
        print(f"  could not download {name}: {type(exc).__name__}")
        return None
    return _extract_text(payload, name)


def _relevant_chunks(query: str, content: str) -> list[str]:
    """Return the chunks of `content` most relevant to `query`.

    Mirrors upstream `get_most_relevant_chunks`: fixed-size character chunks, evenly
    sampled down to MAX_CHUNKS, returned whole when there are at most TOP_K of them
    (which skips embeddings entirely), otherwise ranked by cosine similarity.
    """
    chunks = [
        content[start : start + CHUNK_SIZE].strip()
        for start in range(0, len(content), CHUNK_SIZE)
    ]
    chunks = [chunk for chunk in chunks if chunk]
    if not chunks:
        return []
    if len(chunks) > MAX_CHUNKS:
        step = len(chunks) / MAX_CHUNKS
        chunks = [chunks[int(index * step)] for index in range(MAX_CHUNKS)]
    if len(chunks) <= TOP_K:
        return chunks
    try:
        vectors = _embed(chunks)
        query_vector = _embed([query])[0]
    except Exception as exc:  # noqa: BLE001 - fall back to a prefix rather than failing
        print(f"  embedding failed ({type(exc).__name__}); using the leading chunks")
        return chunks[:TOP_K]
    ranked = sorted(
        range(len(chunks)), key=lambda i: _cosine(query_vector, vectors[i]), reverse=True
    )
    return [chunks[i] for i in ranked[:TOP_K]]


def _factuality_verdict(claim: str, context: str, model: str) -> dict[str, Any]:
    """Ask the judge whether one claim is supported by its retrieved context."""
    prompt = _FACTUALITY_PROMPT.format(context=context, insight=claim)
    last_error = ""
    for _ in range(MAX_RETRIES):
        try:
            parsed = _extract_json(_call_judge(prompt, model))
            if not isinstance(parsed, dict):
                msg = "factuality did not return a JSON object"
                raise ValueError(msg)  # noqa: TRY301 - retried below
            return {
                "is_factual": bool(parsed.get("is_factual")),
                "explanation": str(parsed.get("explanation", "")),
            }
        except Exception as exc:  # noqa: BLE001 - retry, then treat as unsupported
            last_error = f"{type(exc).__name__}: {exc}"
    return {"is_factual": False, "explanation": f"judge failed: {last_error}"}


def _factuality(
    claims: list[dict[str, Any]], case: dict[str, Any], model: str
) -> tuple[float, dict[str, Any]]:
    """Score what fraction of the report's claims their citations actually support.

    Claims with no citations count as unsupported, exactly as upstream does.
    """
    if not claims:
        return 0.0, {"error": "no claims extracted from the report"}

    endpoints = case.get("endpoints") or {}
    credentials = (case.get("credentials") or {}).get("nextcloud")
    index: dict[str, str] = {}
    if endpoints.get("nextcloud") and credentials:
        # Deliberately not caught: an unreachable app stack must not masquerade as a bad
        # report. `main`'s caller turns this into a loud verifier failure.
        index = _webdav_index(endpoints["nextcloud"], credentials)

    details = []
    for claim in claims:
        text = claim.get("claim", "")
        citations = claim.get("citations") or []
        if not citations:
            details.append(
                {
                    "claim": text,
                    "citations": [],
                    "is_factual": False,
                    "explanation": "No citations provided to verify the claim",
                }
            )
            continue

        sources = []
        for citation in citations:
            content = _resolve_citation(citation, index, credentials)
            if content:
                sources.append(f"Source: {citation}\n{content}")
        if not sources:
            details.append(
                {
                    "claim": text,
                    "citations": citations,
                    "is_factual": False,
                    "explanation": "No content could be retrieved from any cited source",
                }
            )
            continue

        context = "\n".join(_relevant_chunks(text, "\n\n".join(sources)))
        details.append(
            {"claim": text, "citations": citations, **_factuality_verdict(text, context, model)}
        )

    factual = sum(1 for detail in details if detail["is_factual"])
    return factual / len(details), {"per_claim": details, "claim_count": len(details)}


def composite(components: dict[str, float]) -> float:
    """Harmonic mean of the scored components, each floored at EPSILON.

    Upstream defines no combined score, so this one is ours. The floor keeps a single
    zero from erasing all ranking signal while still driving the headline to near zero.
    """
    values = [max(value, EPSILON) for value in components.values()]
    if not values:
        return 0.0
    return len(values) / sum(1.0 / value for value in values)


def _zero_rewards() -> dict[str, float]:
    """Reward mapping for a run that produced nothing to score."""
    return dict.fromkeys(("reward", *_METRIC_NAMES), 0.0)


def _grade() -> tuple[dict[str, float], dict[str, Any]]:
    """Return the reward mapping and a per-metric breakdown."""
    case = json.loads(_CASE_PATH.read_text(encoding="utf-8"))
    insights = [
        entry
        for entry in case.get("insights", [])
        if isinstance(entry, dict) and str(entry.get("answer", "")).strip()
    ]
    distractors = [
        entry
        for entry in case.get("distractors", [])
        if isinstance(entry, dict) and str(entry.get("answer", "")).strip()
    ]
    breakdown: dict[str, Any] = {
        "task_id": case.get("task_id", ""),
        "insight_count": len(insights),
        "distractor_count": len(distractors),
    }

    if not _REPORT_PATH.is_file():
        print(f"no {_REPORT_PATH}; scoring 0.0")
        breakdown["error"] = "report missing"
        return _zero_rewards(), breakdown

    report_text = _REPORT_PATH.read_text(encoding="utf-8", errors="replace")
    if len(report_text) > MAX_REPORT_LENGTH:
        print(f"report is {len(report_text)} characters; using the first {MAX_REPORT_LENGTH}")
        report_text = report_text[:MAX_REPORT_LENGTH]

    model = _judge_model()
    claims = _split_report(report_text, model)
    claims_text = _format_claims(claims)
    breakdown["claim_count"] = len(claims)

    recall, recall_details = _score_qa_set(claims_text, insights, model)
    distractor_recall, distractor_details = _score_qa_set(claims_text, distractors, model)
    quality, quality_details = _report_quality(
        report_text, str(case.get("question", "")), case.get("persona") or {}, model
    )
    factuality, factuality_details = _factuality(claims, case, model)

    # Inverted for the composite: recalling a planted distractor is a failure, so
    # avoidance is what belongs in a "higher is better" aggregate. Both are reported.
    components = {
        "insights_recall": recall,
        "distractor_avoidance": 1.0 - distractor_recall,
        "factuality": factuality,
        "report_quality": quality,
    }
    rewards = {
        "reward": composite(components),
        "insights_recall": recall,
        "distractor_recall": distractor_recall,
        "distractor_avoidance": components["distractor_avoidance"],
        "factuality": factuality,
        "report_quality": quality,
    }
    breakdown.update(
        {
            "components": components,
            "composite": rewards["reward"],
            "insights_recall": {"score": recall, "per_insight": recall_details},
            "distractor_recall": {
                "score": distractor_recall,
                "per_distractor": distractor_details,
            },
            "report_quality": {"score": quality, **quality_details},
            "factuality": {"score": factuality, **factuality_details},
        }
    )
    return rewards, breakdown


def main() -> None:
    """Score the report and write Harbor's rewards plus a per-metric breakdown."""
    try:
        rewards, breakdown = _grade()
    except Exception as exc:  # noqa: BLE001 - a verifier crash must still write a reward
        print(f"grading failed: {type(exc).__name__}: {exc}")
        rewards = _zero_rewards()
        breakdown = {"error": f"{type(exc).__name__}: {exc}"}

    rewards = {name: max(0.0, min(1.0, value)) for name, value in rewards.items()}
    _REWARD_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    # reward.json takes precedence over reward.txt in Harbor, and `reward` is the key the
    # deepagents aggregation reads for pass@k / avg@k.
    _REWARD_JSON_PATH.write_text(json.dumps(rewards, indent=2) + "\n", encoding="utf-8")
    try:
        _BREAKDOWN_PATH.write_text(
            json.dumps(breakdown, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    except OSError as exc:
        print(f"could not write breakdown: {exc}")
    print("rewards=" + json.dumps(rewards))


if __name__ == "__main__":
    main()
