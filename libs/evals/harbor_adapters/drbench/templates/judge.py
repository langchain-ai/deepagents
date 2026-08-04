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

_DEFAULT_JUDGE_MODEL = "gpt-4.1"


def _judge_model() -> str:
    """First model in the harness-injected ``JUDGE_MODELS`` (or a supported default)."""
    raw = os.environ.get("JUDGE_MODELS") or os.environ.get("JUDGE_MODEL") or ""
    for token in re.split(r"[\s,]+", raw.strip()):
        if token:
            return token
    return _DEFAULT_JUDGE_MODEL


def _embedding_model() -> str | None:
    """Embedding model for factuality chunk ranking, or None for upstream's default.

    Upstream's code defaults to ``text-embedding-3-small`` while the paper reports
    ``text-embedding-3-large``. Returning None keeps whatever the installed version
    chose, so the metric matches the code being reused rather than our guess.
    """
    return os.environ.get("JUDGE_EMBEDDING_MODEL") or None


def _allow_judge_model(model: str) -> None:
    """Let upstream route `model` through its plain OpenAI-compatible client.

    ``drbench.agents.utils.prompt_llm`` dispatches on an allowlist -- currently
    ``["gpt-4o", "gpt-4o-mini", "gpt-4.1"]`` -- and anything outside it falls through to
    an OpenRouter client keyed on ``OPENROUTER_API_KEY``, which this environment does not
    set, so every judge call would fail. The harness picks one judge for the whole eval
    suite via ``JUDGE_MODELS`` and forwards ``OPENAI_BASE_URL`` next to ``OPENAI_API_KEY``
    for exactly this, so extending the allowlist points the model at the configured
    OpenAI-compatible endpoint. The alternative -- hardcoding one of the three names --
    would silently judge this category with a different model than every other category
    in the same run.
    """
    from drbench.agents import utils  # noqa: PLC0415 - installed only in the sandbox

    if model not in utils.OPENAI_MODELS:
        utils.OPENAI_MODELS.append(model)
        print(f"routing judge model {model!r} through the OpenAI-compatible client")


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
    _allow_judge_model(model)
    embedding_model = _embedding_model()
    breakdown["judge_model"] = model
    breakdown["embedding_model"] = embedding_model

    # `task` supplies both the task config and the ground truth from the installed
    # package, which is also where `CitationFactuality` resolves cited documents from --
    # so email, chat, file-browser, and Nextcloud sources all resolve as plain files.
    task = task_loader.get_task_from_id(task_id)
    scores = score_report(
        predicted_report_text=report_text,
        task=task,
        metrics=list(_UPSTREAM_METRICS),
        model=model,
        embedding_model=embedding_model,
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
