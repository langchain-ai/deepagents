#!/usr/bin/env python3
"""Score a DRBench report with upstream DRBench's own metrics.

Runs in the SEPARATE verifier environment built from this task's ``tests/`` directory,
where ``drbench`` is pip-installed (see ``Dockerfile``). Upstream supplies the metrics,
their prompts, the ground truth, and the document corpus, so this file only has to:

  1. read the report Harbor re-materialized at ``/app/report.md``,
  2. call ``drbench.score_report.score_report`` for the four metrics,
  3. combine them and write Harbor's ``reward.json``.

Everything else -- claim extraction, citation normalization, chunk retrieval,
insight/distractor judging, the report-quality rubric, and resolving a citation back to a
Nextcloud document, a mailbox export, a chat log, or a live URL -- is upstream's code.
Reimplementing it here previously cost ~900 lines and, because that version resolved
documents over WebDAV against the live app stack instead of the corpus shipped with the
package, could not resolve email, chat, or file-browser citations at all.

The four metrics and the harmonic mean are the paper's own (arXiv 2510.00172, Table 2:
Insight Recall, Factuality, Distractor Avoidance, Report Quality, Harmonic Mean), which
also defines distractor avoidance as ``1 - distractor recall``. Upstream's released code
computes the four but not the mean, so the combination happens here. EPSILON below is the
only deviation from the paper.

Judge selection and credentials come from the verifier environment the harness injects
(``JUDGE_MODELS``, ``OPENAI_API_KEY``, ``OPENAI_BASE_URL``, and ``OPENROUTER_API_KEY`` for
an ``openrouter/`` judge). Nothing is hardcoded, and no key is ever printed or written to a
reward or breakdown file.
"""

from __future__ import annotations

import ipaddress
import json
import os
import re
import socket
import traceback
import urllib.parse
from pathlib import Path
from typing import Any

_CASE_PATH = Path("/tests/case.json")
_REPORT_PATH = Path("/app/report.md")
_REWARD_JSON_PATH = Path("/logs/verifier/reward.json")
_BREAKDOWN_PATH = Path("/logs/verifier/drbench_metrics.json")

# Packages whose version can change a score, recorded in every breakdown. `drbench`
# itself is pinned by commit in the Dockerfile, but it declares only version ranges for
# these, so the image resolves them fresh from PyPI on each build: scikit-learn and
# faiss-cpu drive factuality's embedding and retrieval, tiktoken drives chunking, and
# openai is the judge transport.
_SCORING_PACKAGES = (
    "drbench",
    "faiss-cpu",
    "openai",
    "scikit-learn",
    "tiktoken",
)

# The metric names upstream's `get_metric` accepts, in report order.
_UPSTREAM_METRICS = (
    "insights_recall",
    "distractor_recall",
    "factuality",
    "report_quality",
)

# Every key written to reward.json besides `reward` itself.
_METRIC_NAMES = (
    "insights_recall",
    "distractor_recall",
    "distractor_avoidance",
    "factuality",
    "report_quality",
)

# Floor per component in the harmonic mean, so one zero does not erase all ranking
# signal. The paper specifies no floor; this is our only deviation from it.
EPSILON = 0.01

# drbench.score_report.MAX_REPORT_LENGTH. Checked here too so the truncation is visible
# in this log rather than only in upstream's stdout.
MAX_REPORT_LENGTH = 60_000

# The unified eval workflow uses one judge for every category. DRBench predates this model
# and hardcodes three routing allowlists, so `_install_default_judge_compatibility` adds it
# to those lists before upstream selects a transport.
_DEFAULT_JUDGE_MODEL = "gpt-5.6-luna"

# Upstream routes an `openrouter/<vendor>/<model>` slug to OpenRouter's OpenAI-compatible
# endpoint in both transports, resolving it before either model registry is consulted.
_OPENROUTER_SLUG_RE = re.compile(
    r"^openrouter/[A-Za-z0-9._-]+/[A-Za-z0-9._-]+(?::[A-Za-z0-9._-]+)?$"
)

# Output cap for an OpenRouter judge; `AIAgentManager`'s 1000-token default truncates a
# reasoning model's verdict. The direct-OpenAI path keeps upstream's default, both because
# 1000 tokens suffice there and because it is what earlier runs scored with.
_OPENROUTER_JUDGE_MAX_TOKENS = 32_000

# Bounds on the captured judge verdicts, which are model-written free text serialized into
# a CI artifact: enough to read every gold insight's verdict, not enough for a long or
# adversarial report to inflate the breakdown file.
_MAX_CAPTURED_VERDICTS = 32
_MAX_CAPTURED_CHARS = 600
_CAPTURED_FIELDS = (
    "expected_insight",
    "predicted_insight",
    "score",
    "justification",
    "confidence",
)


def _requested_judge_model() -> str:
    """The judge selected for DRBench, then the suite-wide judge as a fallback.

    The harness sets ``DRBENCH_JUDGE_MODEL`` when an unsupported suite-wide judge has to
    resolve to DRBench's compatible default. Both variables otherwise carry the same
    value, so one ``judge_models`` input controls every category.
    """
    raw = (
        os.environ.get("DRBENCH_JUDGE_MODEL")
        or os.environ.get("JUDGE_MODELS")
        or os.environ.get("JUDGE_MODEL")
        or ""
    )
    for token in re.split(r"[\s,]+", raw.strip()):
        if token:
            return token
    return _DEFAULT_JUDGE_MODEL


def _supported_judge_models() -> set[str]:
    """Models the installed DRBench can actually drive, read from upstream itself.

    Upstream gates the judge model in two independent places that do not agree:
    ``agents.utils.OPENAI_MODELS`` (used by `prompt_llm`) and
    ``gen_agent.SERVICE_TO_MODELS["openai"]` (used by `AIAgentManager`, which
    `QASimilarityV2` constructs). A model in only one of them fails partway through
    scoring, so the usable set is the intersection after registering the one pinned
    compatibility model. Native models remain derived at runtime, so a future upstream
    bump is picked up automatically.
    """
    from drbench import gen_agent  # noqa: PLC0415 - installed only in the sandbox
    from drbench.agents import utils  # noqa: PLC0415

    _install_default_judge_compatibility()

    return set(utils.OPENAI_MODELS) & set(gen_agent.SERVICE_TO_MODELS.get("openai", []))


def _install_default_judge_compatibility() -> None:
    """Teach pinned upstream DRBench to route the unified default through OpenAI.

    Upstream checks the model in three independent mutable allowlists. Two choose the
    OpenAI transport, while `AVAILABLE_MODELS` prevents `AIAgentManager.prompt_llm` from
    silently replacing the requested model with its own default. Registering exactly the
    workflow's pinned default keeps this compatibility shim narrow; other models still
    need native upstream support or an explicit `openrouter/<vendor>/<model>` route.
    """
    from drbench import gen_agent  # noqa: PLC0415 - installed only in the sandbox
    from drbench.agents import utils  # noqa: PLC0415

    registries = (
        utils.OPENAI_MODELS,
        gen_agent.SERVICE_TO_MODELS.setdefault("openai", []),
        gen_agent.AVAILABLE_MODELS,
    )
    for registry in registries:
        if _DEFAULT_JUDGE_MODEL not in registry:
            registry.append(_DEFAULT_JUDGE_MODEL)


def _judge_model() -> str:
    """Return the judge to score with, or raise when upstream cannot drive it.

    `gpt-5.6-luna` is the unified default and is registered with upstream's OpenAI
    transport by `_install_default_judge_compatibility`. Any other bare name is honored
    only if upstream natively drives it; an ``openrouter/<vendor>/<model>`` slug is honored
    as-is because upstream resolves the prefix ahead of its registries.

    An unusable name raises instead of falling back. Quietly substituting a different judge
    publishes numbers under the wrong model's name, and a run of 30 tasks is a poor place
    to discover a typo.

    Note that DRBench's prompts and thresholds were calibrated with a GPT-4o-class judge,
    so a different judge produces scores that are not comparable to the paper's or to
    earlier runs.
    """
    requested = _requested_judge_model()
    if _OPENROUTER_SLUG_RE.match(requested):
        return requested
    supported = _supported_judge_models()
    if requested in supported:
        return requested
    msg = (
        f"judge model {requested!r} is not one upstream DRBench can drive "
        f"({sorted(supported) or 'none'}). Reach any other model through OpenRouter as "
        f"'openrouter/<vendor>/<model>', e.g. 'openrouter/openai/{requested}'."
    )
    raise ValueError(msg)


def _embedding_model() -> str | None:
    """Embedding model for factuality chunk ranking, or None for upstream's default.

    Upstream's code defaults to ``text-embedding-3-small`` while the paper reports
    ``text-embedding-3-large``. Returning None keeps whatever the installed version
    chose, so the metric matches the code being reused rather than our guess.
    """
    return os.environ.get("JUDGE_EMBEDDING_MODEL") or None


# Ceiling for one embeddings request, under the API's 300k-token limit with headroom for
# the tokenizer estimate being approximate.
_EMBED_TOKEN_BUDGET = 200_000

# Upstream fetches cited URLs with no timeout at all, so one slow host could otherwise
# consume the whole verifier budget.
_URL_FETCH_TIMEOUT = 30.0
# Bound on redirect hops we will follow while re-validating each one.
_MAX_REDIRECTS = 5


def _is_public_host(host: str) -> bool:
    """True when a hostname resolves only to public addresses.

    Citations come from the agent's report, so a cited URL is untrusted input. Without
    this, a citation could point the verifier at cloud instance metadata or another
    service on the runner. Upstream performs no such check -- it fetches whatever the
    citation says -- so this restores the guard our own earlier verifier had.
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


class _BlockedHostError(Exception):
    """A cited URL resolved to an address the verifier must not fetch."""


class _MissingReportError(FileNotFoundError):
    """The agent produced no report.

    Distinct from every other failure so `main` can score it zero -- a task the agent did
    not finish -- while still failing the verifier on infrastructure errors. A bare
    `FileNotFoundError` would conflate it with, say, a missing `case.json`, which is a
    broken task rather than a failed attempt.
    """


def _install_url_fetch_guard(failures: list[dict[str, str]]) -> None:
    """Make cited-URL fetching safe, bounded, and non-fatal.

    `get_content` resolves an `http...` citation through
    `drbench.agents.utils.SourceReader.parse_website`, whose live body is a bare
    `session.get(url)` -- no host validation, no timeout, no exception handling. Three
    consequences, all observed or reachable:

      1. A network error propagates out of `score_report`, so ONE unreachable citation
         zeroes all four metrics for the task, discarding metrics that already computed.
         This cost 1 of 15 tasks (~0.03 on the dataset mean) in a real run.
      2. Nothing stops a citation naming a private or link-local address, which would aim
         a server-side request from inside the runner at, say, cloud metadata.
      3. With no timeout, one slow host can burn the entire verifier budget.

    So this installs three wrappers:

      * `Session.send` validates the host of every outbound request before it is sent,
         which is what catches a public host that 302s to a private address: redirect
         hops go through `send`, not back through `request`.
      * `Session.request` repeats the check for a caller that bypasses `send`, and
         supplies a default timeout.
      * `Session.resolve_redirects` caps the hop count.
      * `parse_website` catches, records, and returns None -- exactly what `get_content`'s
         own contract already expects (`return result if result is not None else None`),
         so an unfetchable citation becomes an unsupported claim rather than a dead task.

    Failures are appended to `failures` and surface in the breakdown, so a fetch that was
    skipped is never silent -- the alternative, swallowing them, would make a low
    factuality score indistinguishable from an unreachable corpus.

    Nothing here changes how a resolved citation is judged.
    """
    import requests  # noqa: PLC0415 - a drbench dependency

    from drbench.agents import utils  # noqa: PLC0415

    def _check(url: str) -> None:
        host = urllib.parse.urlsplit(url).hostname or ""
        if not _is_public_host(host):
            msg = f"refusing to fetch non-public host {host!r}"
            raise _BlockedHostError(msg)

    # `send` is the only chokepoint every request passes through before any network I/O,
    # including each redirect hop: `resolve_redirects` calls `self.send()` per hop rather
    # than going back through `request`. Validating a hop's *response* URL, as an earlier
    # version did, ran after that hop had already been fetched -- so a cited public URL
    # answering `302 Location: http://169.254.169.254/...` reached the metadata endpoint
    # before the guard fired.
    original_send = requests.Session.send
    if not getattr(original_send, "_deepagents_guarded", False):

        def send(self: Any, request: Any, **kwargs: Any) -> Any:
            _check(request.url)
            kwargs.setdefault("timeout", _URL_FETCH_TIMEOUT)
            return original_send(self, request, **kwargs)

        send._deepagents_guarded = True  # type: ignore[attr-defined]
        requests.Session.send = send  # type: ignore[method-assign]

    original_request = requests.Session.request
    if not getattr(original_request, "_deepagents_guarded", False):

        def request(self: Any, method: str, url: str, **kwargs: Any) -> Any:
            # Redundant with `send` above, kept so a caller that bypasses `send` is still
            # checked, and so the failure names the cited URL rather than a redirect target.
            _check(url)
            kwargs.setdefault("timeout", _URL_FETCH_TIMEOUT)
            return original_request(self, method, url, **kwargs)

        request._deepagents_guarded = True  # type: ignore[attr-defined]
        requests.Session.request = request  # type: ignore[method-assign]

    original_resolve = requests.Session.resolve_redirects
    if not getattr(original_resolve, "_deepagents_guarded", False):

        def resolve_redirects(self: Any, resp: Any, req: Any, **kwargs: Any) -> Any:
            # Hop cap only. Host validation happens in `send`, before each hop is sent.
            kwargs.setdefault("timeout", _URL_FETCH_TIMEOUT)
            for hop, response in enumerate(original_resolve(self, resp, req, **kwargs)):
                if hop >= _MAX_REDIRECTS:
                    return
                yield response

        resolve_redirects._deepagents_guarded = True  # type: ignore[attr-defined]
        requests.Session.resolve_redirects = resolve_redirects  # type: ignore[method-assign]

    original_parse = utils.SourceReader.parse_website
    if not getattr(original_parse, "_deepagents_guarded", False):

        def parse_website(self: Any, url: str) -> Any:
            try:
                return original_parse(self, url)
            except Exception as exc:  # noqa: BLE001 - a bad citation must not end the run
                failures.append({"url": url[:300], "error": f"{type(exc).__name__}: {exc}"[:300]})
                print(f"  could not fetch cited URL {url[:120]}: {type(exc).__name__}")
                return None

        parse_website._deepagents_guarded = True  # type: ignore[attr-defined]
        utils.SourceReader.parse_website = parse_website  # type: ignore[method-assign]


def _install_embedding_batching() -> None:
    """Split upstream's embedding requests so one large source cannot exceed the API limit.

    `get_most_relevant_chunks` embeds up to 200 chunks of 2048 characters in a single
    request. At a typical 4 characters per token that is ~100k tokens, but content that
    tokenizes badly -- a web page or PDF that parsed into near-binary text -- approaches
    one token per character and blows past the 300k-token request ceiling, failing
    factuality with a `BadRequestError`. Upstream already batches in
    `metrics/utils/semantic_retriever.py`; this path simply does not.

    This changes no scores. The same texts are embedded, in the same order, producing the
    same vectors -- only the request framing differs. It is applied here rather than
    upstream because the chunk cap is not reachable through `score_report`.
    """
    from drbench.agents import utils  # noqa: PLC0415 - installed only in the sandbox

    original = utils.get_embeddings
    if getattr(original, "_deepagents_batched", False):
        return

    def batched(texts: list[str], *args: Any, **kwargs: Any) -> Any:
        items = list(texts)
        if len(items) <= 1:
            return original(items, *args, **kwargs)

        try:
            import tiktoken  # noqa: PLC0415 - a drbench dependency

            encoder = tiktoken.get_encoding("cl100k_base")
            sizes = [len(encoder.encode(text)) for text in items]
        except Exception:  # noqa: BLE001 - fall back to a character estimate
            sizes = [max(1, len(text) // 3) for text in items]

        groups: list[list[str]] = []
        batch: list[str] = []
        budget = 0
        for text, size in zip(items, sizes, strict=True):
            if batch and budget + size > _EMBED_TOKEN_BUDGET:
                groups.append(batch)
                batch, budget = [], 0
            batch.append(text)
            budget += size
        if batch:
            groups.append(batch)

        results = [original(group, *args, **kwargs) for group in groups]
        if len(results) == 1:
            return results[0]

        # The result must stay a C-contiguous float32 ndarray, not a list: the caller reads
        # `.shape[1]` and hands it to `faiss.normalize_L2`, which mutates in place and
        # requires that exact layout. Returning a list here fails with
        # "'list' object has no attribute 'shape'".
        try:
            import numpy as np  # noqa: PLC0415 - a drbench dependency

            joined = np.concatenate(results, axis=0)
            combined: Any = np.ascontiguousarray(joined, dtype=results[0].dtype)
        except (ImportError, AttributeError, ValueError):
            # A non-array embedding backend; preserve sequence semantics instead.
            combined = [vector for result in results for vector in result]

        if len(combined) != len(items):
            msg = f"embedding batching returned {len(combined)} vectors for {len(items)} texts"
            raise RuntimeError(msg)
        return combined

    batched._deepagents_batched = True  # type: ignore[attr-defined]
    utils.get_embeddings = batched


def _install_judge_sampling(max_tokens: int = _OPENROUTER_JUDGE_MAX_TOKENS) -> None:
    """Raise `AIAgentManager`'s output cap so a reasoning judge's verdict is not truncated.

    `QASimilarityV2` -- the insight-recall metric, and the only metric on this transport --
    constructs the manager with no arguments (`metrics/qa_similarity_v2.py`), so it takes
    upstream's 1000-token default. That is ample for a GPT-4o-class verdict but not for a
    reasoning model, whose thinking tokens count against the same cap.

    Only the default is replaced, so an explicit caller value still wins, and nothing about
    the prompts, parsing, or thresholds changes.
    """
    from drbench import gen_agent  # noqa: PLC0415 - installed only in the sandbox

    original = gen_agent.AIAgentManager.__init__
    if getattr(original, "_deepagents_sampling", False):
        return

    def patched(self: Any, *args: Any, **kwargs: Any) -> None:
        # `max_tokens` is the fourth positional parameter after `self`; injecting the
        # keyword as well when it was passed positionally would be a TypeError.
        if len(args) <= 3:
            kwargs.setdefault("max_tokens", max_tokens)
        original(self, *args, **kwargs)

    patched._deepagents_sampling = True  # type: ignore[attr-defined]
    gen_agent.AIAgentManager.__init__ = patched  # type: ignore[method-assign]


def _trim_captured(result: object) -> list[dict[str, Any]]:
    """Reduce a metric result to its per-gold-insight verdicts, bounded in count and length.

    The fields are judge-written free text that ends up in a CI artifact, so only an
    allowlist is kept and every string is truncated: a long report must not be able to
    inflate the breakdown file.
    """
    if not isinstance(result, dict):
        return []
    rows = result.get("per_question_results")
    if not isinstance(rows, list):
        return []

    trimmed: list[dict[str, Any]] = []
    for row in rows[:_MAX_CAPTURED_VERDICTS]:
        if not isinstance(row, dict):
            continue
        entry: dict[str, Any] = {}
        for field in _CAPTURED_FIELDS:
            value = row.get(field)
            if isinstance(value, str):
                entry[field] = value[:_MAX_CAPTURED_CHARS]
            elif value is None or isinstance(value, bool | int | float):
                entry[field] = value
        trimmed.append(entry)
    return trimmed


def _install_metric_detail_capture(sink: dict[str, list[dict[str, Any]]]) -> None:
    """Record the per-insight verdicts `score_report` computes and then discards.

    Upstream keeps each `metric.compute` result in a local list and returns only the four
    scores, so the judge's per-gold-insight `justification`, `predicted_insight`, and
    `confidence` are lost -- the very fields that distinguish a real zero from a parse
    failure. Capturing them costs no extra model call, unlike `include_per_insight_scores`.

    The patch target is `score_report`'s own `get_metric`: it binds the function with
    `from drbench.metrics import get_metric`, so rebinding `drbench.metrics.get_metric`
    would leave that already-bound reference untouched.
    """
    from drbench import score_report as upstream  # noqa: PLC0415 - installed only in the sandbox

    original = upstream.get_metric
    if getattr(original, "_deepagents_captured", False):
        return

    def capturing(name: str, *args: Any, **kwargs: Any) -> Any:
        metric = original(name, *args, **kwargs)
        inner = metric.compute

        def compute(*inner_args: Any, **inner_kwargs: Any) -> Any:
            result = inner(*inner_args, **inner_kwargs)
            captured = _trim_captured(result)
            if captured:
                sink[name] = captured
            return result

        metric.compute = compute
        return metric

    capturing._deepagents_captured = True  # type: ignore[attr-defined]
    upstream.get_metric = capturing


def composite(components: dict[str, float]) -> float:
    """Harmonic mean of the scored components, each floored at EPSILON.

    This is the paper's aggregate (Table 2, "Harmonic Mean") over insight recall,
    factuality, distractor avoidance, and report quality. The floor is ours: it keeps a
    single zero from erasing all ranking signal while still driving the headline to near
    zero.
    """
    values = [max(value, EPSILON) for value in components.values()]
    if not values:
        return 0.0
    return len(values) / sum(1.0 / value for value in values)


def _zero_rewards() -> dict[str, float]:
    """Reward mapping for a run that produced nothing to score.

    `distractor_avoidance` is 1.0, not 0.0: it is *defined* as `1 - distractor_recall`, and
    a report that does not exist recalled no distractors. Zeroing it alongside the others
    broke that identity, and because the aggregation sums each component over expected
    trials, the two stopped summing to 1.0 -- which read as a metric bug rather than as
    trials that scored nothing. With the identity restored, `distractor_avoidance +
    distractor_recall` equals the fraction of expected trials that produced a reward at
    all, which is a useful thing to be able to read off the scorecard.

    `reward` stays 0.0. An unearned composite is what the aggregate is meant to reflect,
    and it is deliberately not the harmonic mean of the components below for this one case.
    """
    rewards = dict.fromkeys(("reward", *_METRIC_NAMES), 0.0)
    rewards["distractor_avoidance"] = 1.0
    return rewards


def _read_report() -> str:
    """Return the report text, or raise after logging what was actually delivered.

    Harbor re-materializes each collected artifact at its ORIGINAL path, so the report
    declared as ``artifacts = ["/app/report.md"]`` lands back at ``/app/report.md`` in
    this environment. When it is absent the useful signal is what *is* present, because a
    silent zero here is indistinguishable from a genuinely empty report.
    """
    if _REPORT_PATH.is_file():
        return _REPORT_PATH.read_text(encoding="utf-8", errors="replace")

    print(f"no report at {_REPORT_PATH}; listing candidate locations")
    for probe in (_REPORT_PATH.parent, Path("/logs/artifacts")):
        try:
            listing = sorted(str(path) for path in probe.rglob("*"))
        except OSError as exc:
            print(f"  {probe}: unreadable ({exc})")
            continue
        print(f"  {probe}: {listing[:40] or 'empty'}")
    msg = f"no report at {_REPORT_PATH}"
    raise _MissingReportError(msg)


def _grade() -> tuple[dict[str, float], dict[str, Any]]:
    """Return the reward mapping and a per-metric breakdown."""
    from drbench import task_loader  # noqa: PLC0415 - installed only in the sandbox
    from drbench.score_report import score_report  # noqa: PLC0415

    case = json.loads(_CASE_PATH.read_text(encoding="utf-8"))
    task_id = str(case.get("task_id", "")).strip()
    if not task_id:
        msg = f"{_CASE_PATH} has no task_id"
        raise ValueError(msg)

    breakdown: dict[str, Any] = {"task_id": task_id}
    report_text = _read_report()
    if len(report_text) > MAX_REPORT_LENGTH:
        print(
            f"report is {len(report_text)} characters; "
            f"upstream will score the first {MAX_REPORT_LENGTH}"
        )

    model = _judge_model()
    embedding_model = _embedding_model()
    breakdown["judge_model"] = model
    # Recorded so a score is never silently attributed to the judge the harness asked for
    # when a different one actually ran.
    breakdown["requested_judge_model"] = _requested_judge_model()
    breakdown["embedding_model"] = embedding_model

    # `task` supplies both the task config and the ground truth from the installed
    # package, which is also where `CitationFactuality` resolves cited documents from --
    # so email, chat, file-browser, and Nextcloud sources all resolve as plain files.
    _install_embedding_batching()
    if model.startswith("openrouter/"):
        _install_judge_sampling()
    url_failures: list[dict[str, str]] = []
    _install_url_fetch_guard(url_failures)
    metric_detail: dict[str, list[dict[str, Any]]] = {}
    _install_metric_detail_capture(metric_detail)
    task = task_loader.get_task_from_id(task_id)
    scores = score_report(
        predicted_report_text=report_text,
        task=task,
        metrics=list(_UPSTREAM_METRICS),
        model=model,
        embedding_model=embedding_model,
        # Off because the results are only written to `savedir`, which we do not pass, so
        # they are absent from the return value: the pass costs an extra judge call per
        # insight plus up to five retries and yields nothing we can read.
        include_per_insight_scores=False,
        verbose=True,
    )
    if not isinstance(scores, dict):
        msg = f"score_report returned {type(scores).__name__}, expected a dict"
        raise TypeError(msg)
    missing = [name for name in _UPSTREAM_METRICS if name not in scores]
    if missing:
        msg = f"score_report omitted {missing}; got {sorted(scores)}"
        raise ValueError(msg)

    recall = float(scores["insights_recall"])
    distractor_recall = float(scores["distractor_recall"])
    factuality = float(scores["factuality"])
    quality = float(scores["report_quality"])

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
            "upstream_scores": {name: scores[name] for name in _UPSTREAM_METRICS},
            "components": components,
            "composite": rewards["reward"],
            # Cited URLs the verifier could not read. Each one cost the report a claim it
            # might otherwise have supported, so a depressed `factuality` is only
            # interpretable next to this list.
            "unfetchable_citations": url_failures,
            # The judge's own verdict per gold insight and per distractor. Without this a
            # zero is ambiguous: a report that missed everything and a judge whose replies
            # failed to parse both score 0.0.
            "metric_detail": metric_detail,
        }
    )
    if url_failures:
        print(f"{len(url_failures)} cited URL(s) could not be fetched; see the breakdown")
    return rewards, breakdown


def _scoring_package_versions() -> dict[str, str]:
    """Return the installed version of each package that can move a score.

    Upstream declares version *ranges*, not pins, so `pip install -e /opt/drbench`
    resolves these fresh from PyPI on every image build. Recording what actually got
    installed makes a score shift between two runs diagnosable by diffing their
    breakdowns, instead of leaving the resolver as an untracked variable.

    Only names and versions of an explicit allowlist are read, so nothing about the
    environment beyond the scoring surface is ever written out.
    """
    from importlib.metadata import PackageNotFoundError, version

    versions: dict[str, str] = {}
    for name in _SCORING_PACKAGES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = "absent"
    return versions


def _write_breakdown(breakdown: dict[str, Any]) -> None:
    """Write the diagnostic file, which has to survive a run that writes no reward."""
    if "scoring_package_versions" not in breakdown:
        # Recorded here rather than in `_grade` so an error breakdown carries it too --
        # a resolver change is exactly the kind of thing that turns a working verifier
        # into a crashing one.
        try:
            breakdown["scoring_package_versions"] = _scoring_package_versions()
        except Exception as exc:  # noqa: BLE001 - diagnostics must never fail the verifier
            breakdown["scoring_package_versions"] = {"error": f"{type(exc).__name__}: {exc}"}
    try:
        _BREAKDOWN_PATH.parent.mkdir(parents=True, exist_ok=True)
        _BREAKDOWN_PATH.write_text(
            json.dumps(breakdown, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    except OSError as exc:
        print(f"could not write breakdown: {exc}")


def main() -> None:
    """Score the report and write Harbor's rewards plus a per-metric breakdown."""
    # Resolved outside the guard below, so a judge upstream cannot drive exits non-zero and
    # Harbor records a failed verifier. Inside it the same error would become a 0.0 reward,
    # which is indistinguishable from a report that genuinely scored nothing.
    #
    # Only that one check is fatal. Anything else this touches -- a missing `drbench`, a
    # broken image -- stays on the guarded path, which still writes a reward and a
    # traceback rather than leaving Harbor with no reward file at all.
    try:
        _judge_model()
    except ValueError:
        raise
    except Exception as exc:  # noqa: BLE001 - deferred to the guarded path below
        print(f"could not pre-check the judge model ({type(exc).__name__}: {exc})")
    try:
        rewards, breakdown = _grade()
    except _MissingReportError as exc:
        # The agent finished without writing a report. That is a failed attempt, not a
        # broken harness, so it is scored zero like any other unearned metric.
        formatted = traceback.format_exc()
        print(f"no report to score: {type(exc).__name__}: {exc}")
        rewards = _zero_rewards()
        breakdown = {"error": f"{type(exc).__name__}: {exc}", "traceback": formatted}
    except Exception as exc:
        # Everything else is infrastructure: a judge outage, an unloadable task, a response
        # that would not parse. A 0.0 here would enter the model's average as though the
        # report had earned nothing, so write the diagnostic and let the verifier fail --
        # the harness then counts the trial as errored rather than scored.
        #
        # The traceback, not just the message: an earlier failure here reported only
        # "ConnectionError: ..." and locating the frame it came from took far longer than
        # reading one would have.
        formatted = traceback.format_exc()
        print(f"grading failed: {type(exc).__name__}: {exc}")
        print(formatted)
        _write_breakdown({"error": f"{type(exc).__name__}: {exc}", "traceback": formatted})
        raise

    rewards = {name: max(0.0, min(1.0, value)) for name, value in rewards.items()}
    _REWARD_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    # reward.json takes precedence over reward.txt in Harbor, and `reward` is the key the
    # deepagents aggregation reads for its dataset-level metrics.
    _REWARD_JSON_PATH.write_text(json.dumps(rewards, indent=2) + "\n", encoding="utf-8")
    _write_breakdown(breakdown)
    print("rewards=" + json.dumps(rewards))


if __name__ == "__main__":
    main()
