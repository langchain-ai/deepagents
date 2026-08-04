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
(``JUDGE_MODELS``, ``OPENAI_API_KEY``, ``OPENAI_BASE_URL``). Nothing is hardcoded, and no
key is ever printed or written to a reward or breakdown file.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

_CASE_PATH = Path("/tests/case.json")
_REPORT_PATH = Path("/app/report.md")
_REWARD_JSON_PATH = Path("/logs/verifier/reward.json")
_BREAKDOWN_PATH = Path("/logs/verifier/drbench_metrics.json")

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

# `gpt-4o` is in both of upstream's model registries (see `_supported_judge_models`), is
# what `get_metric` itself defaults to, and accepts `temperature=0`, which upstream passes
# on every judge call.
_DEFAULT_JUDGE_MODEL = "gpt-4o"


def _requested_judge_model() -> str:
    """First model in the harness-injected ``JUDGE_MODELS``, or the default."""
    raw = os.environ.get("JUDGE_MODELS") or os.environ.get("JUDGE_MODEL") or ""
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
    scoring, so the usable set is the intersection. Derived at runtime rather than
    hardcoded, so a future upstream bump is picked up for free.
    """
    from drbench import gen_agent  # noqa: PLC0415 - installed only in the sandbox
    from drbench.agents import utils  # noqa: PLC0415

    return set(utils.OPENAI_MODELS) & set(gen_agent.SERVICE_TO_MODELS.get("openai", []))


def _judge_model() -> str:
    """Return a judge model the installed DRBench supports.

    The harness picks one judge for the whole eval suite via ``JUDGE_MODELS``, but DRBench
    only drives a fixed set, and its default is a reasoning model that upstream cannot use:
    it is absent from both registries and rejects the ``temperature=0`` upstream sends on
    every call. Rather than monkeypatch three separate upstream internals to force it
    through, fall back to a supported model and say so.

    That is also the more faithful choice. DRBench's prompts and thresholds were calibrated
    with a GPT-4o-class judge, so scoring with one keeps our numbers comparable to the
    paper -- which is the whole reason for using upstream's metrics.
    """
    requested = _requested_judge_model()
    supported = _supported_judge_models()
    if not supported:
        print(f"upstream exposes no usable judge model; trying {requested!r} anyway")
        return requested
    if requested in supported:
        return requested
    fallback = _DEFAULT_JUDGE_MODEL if _DEFAULT_JUDGE_MODEL in supported else min(supported)
    print(
        f"judge model {requested!r} is not one upstream DRBench can drive "
        f"({sorted(supported)}); scoring with {fallback!r} instead"
    )
    return fallback


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
    """Reward mapping for a run that produced nothing to score."""
    return dict.fromkeys(("reward", *_METRIC_NAMES), 0.0)


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
    raise FileNotFoundError(msg)


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
    # deepagents aggregation reads for its dataset-level metrics.
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
