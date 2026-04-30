"""eval_search.py — Compare baseline vs. query-expansion search pipelines.

Usage
-----
    python eval_search.py                          # embedding-based relevance
    python eval_search.py --k 1 3 5 10             # override k values
    python eval_search.py --max-results 5          # results per query
    python eval_search.py --threshold 0.4          # cosine similarity cutoff

Metrics
-------
Recall@k  : fraction of relevant URLs found in the top-k ranked list.
MRR       : mean reciprocal rank of the first relevant URL.

Relevance (priority order)
--------------------------
1. GROUND_TRUTH — labelled URL lists; highest fidelity when available.
2. Embedding proxy — cosine similarity between the query embedding and each
   result's title+content snippet (all-MiniLM-L6-v2). Results whose
   similarity >= --threshold are treated as relevant. This is model-agnostic
   and avoids the bias of pooled relevance toward the system that retrieves
   more documents.

    GROUND_TRUTH = {
        "python async best practices": [
            "https://docs.python.org/3/library/asyncio.html",
        ],
    }
"""

from __future__ import annotations

import argparse
import asyncio
from typing import Any

import numpy as np

from tools import (
    _get_embedding_model,
    _text_from_result,
    _url_from_result,
    deduplicate_results,
    expand_query,
    parallel_search,
    rank_results,
    tavily_client,
)

# ── Eval configuration ────────────────────────────────────────────────────────

EVAL_QUERIES: list[str] = [
    "Python async best practices",
    "transformer architecture explained",
    "retrieval augmented generation overview",
]

# Map query → list of known-relevant URLs.
# Leave empty to fall back to embedding-based proxy relevance.
GROUND_TRUTH: dict[str, list[str]] = {}

DEFAULT_K_VALUES: list[int] = [1, 3, 5, 10]
DEFAULT_MAX_RESULTS: int = 5
DEFAULT_EMBED_THRESHOLD: float = 0.35  # cosine similarity cutoff for proxy relevance
TOPIC: str = "general"

# ── Retrieval ─────────────────────────────────────────────────────────────────


async def run_baseline(query: str, max_results: int) -> list[dict]:
    """Single Tavily call. Returns result dicts in Tavily's scored order."""
    result = await asyncio.to_thread(
        tavily_client.search, query, max_results=max_results, topic=TOPIC
    )
    return [item for item in result.get("results", []) if item.get("url")]


async def run_enhanced(query: str, max_results: int) -> list[dict]:
    """expand_query → parallel_search → deduplicate → rank. Returns ranked result dicts."""

    def _search(q: str) -> dict[str, Any]:
        return tavily_client.search(q, max_results=max_results, topic=TOPIC)

    queries = await expand_query(query)
    raw = await parallel_search(queries, _search)
    deduped = deduplicate_results(raw)
    ranked = await rank_results(deduped, query, raw_results=raw)
    return [r for r in ranked if _url_from_result(r)]


def _ordered_urls(items: list[dict]) -> list[str]:
    return [_url_from_result(r) for r in items if _url_from_result(r)]


# ── Embedding-based proxy relevance ───────────────────────────────────────────


async def _embed_relevance(
    query: str,
    items: list[dict],
    threshold: float,
) -> set[str]:
    """Return URLs whose title+content embedding is within `threshold` of the query.

    All items from both pipelines are deduplicated by URL before encoding so
    the relevance judgement is independent of which system retrieved them.
    """
    # Deduplicate by URL to avoid double-scoring the same page
    seen: set[str] = set()
    unique: list[dict] = []
    for item in items:
        url = _url_from_result(item)
        if url and url not in seen:
            seen.add(url)
            unique.append(item)

    if not unique:
        return set()

    model = _get_embedding_model()
    texts = [query] + [_text_from_result(item) for item in unique]
    embeddings: np.ndarray = await asyncio.to_thread(
        model.encode, texts, normalize_embeddings=True, show_progress_bar=False
    )

    query_emb: np.ndarray = embeddings[0]
    result_embs: np.ndarray = embeddings[1:]
    sims: np.ndarray = result_embs @ query_emb  # cosine sim via dot product on L2-normed vecs

    return {
        _url_from_result(item)
        for item, sim in zip(unique, sims)
        if float(sim) >= threshold
    }


# ── Metrics ───────────────────────────────────────────────────────────────────


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant URLs found in the top-k positions."""
    if not relevant:
        return 0.0
    hits = sum(1 for u in ranked[:k] if u in relevant)
    return hits / len(relevant)


def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    """1 / rank of the first relevant URL; 0.0 if none found."""
    for rank, url in enumerate(ranked, start=1):
        if url in relevant:
            return 1.0 / rank
    return 0.0


# ── Formatting ────────────────────────────────────────────────────────────────

_COL = {"metric": 15, "val": 10}
_SEP = "=" * 60


def _row(label: str, baseline: float, enhanced: float) -> str:
    delta = enhanced - baseline
    sign = "+" if delta >= 0 else ""
    return (
        f"{label:<{_COL['metric']}}"
        f"{baseline:>{_COL['val']}.3f}"
        f"{enhanced:>{_COL['val']}.3f}"
        f"{sign}{delta:>{_COL['val'] - 1}.3f}"
    )


def _header() -> str:
    return (
        f"{'Metric':<{_COL['metric']}}"
        f"{'Baseline':>{_COL['val']}}"
        f"{'Enhanced':>{_COL['val']}}"
        f"{'Delta':>{_COL['val']}}"
    )


# ── Evaluation loop ───────────────────────────────────────────────────────────


async def evaluate(k_values: list[int], max_results: int, threshold: float) -> None:
    per_query: list[dict[str, Any]] = []

    for query in EVAL_QUERIES:
        print(f"\n{_SEP}")
        print(f"Query: {query}")
        print(_SEP)

        baseline_items, enhanced_items = await asyncio.gather(
            run_baseline(query, max_results),
            run_enhanced(query, max_results),
        )

        baseline_urls = _ordered_urls(baseline_items)
        enhanced_urls = _ordered_urls(enhanced_items)

        if query in GROUND_TRUTH:
            relevant = set(GROUND_TRUTH[query])
            rel_source = "ground truth"
        else:
            relevant = await _embed_relevance(
                query, baseline_items + enhanced_items, threshold
            )
            rel_source = f"embedding (threshold={threshold})"

        print(f"Relevant ({rel_source}): {len(relevant)}")
        print(f"Baseline retrieved : {len(baseline_urls)}")
        print(f"Enhanced retrieved : {len(enhanced_urls)}")

        b_recalls = {k: recall_at_k(baseline_urls, relevant, k) for k in k_values}
        e_recalls = {k: recall_at_k(enhanced_urls, relevant, k) for k in k_values}
        b_mrr = reciprocal_rank(baseline_urls, relevant)
        e_mrr = reciprocal_rank(enhanced_urls, relevant)

        per_query.append(
            {
                "query": query,
                "b_recalls": b_recalls,
                "e_recalls": e_recalls,
                "b_mrr": b_mrr,
                "e_mrr": e_mrr,
            }
        )

        header = _header()
        print(f"\n{header}")
        print("-" * len(header))
        for k in k_values:
            print(_row(f"Recall@{k}", b_recalls[k], e_recalls[k]))
        print(_row("MRR", b_mrr, e_mrr))

        # Side-by-side top-5 URL comparison; * marks relevant hits
        print(f"\n{'Rank':<6} {'Baseline URL':<55} {'Enhanced URL'}")
        print("-" * 120)
        for i in range(max(len(baseline_urls[:5]), len(enhanced_urls[:5]))):
            b_url = baseline_urls[i] if i < len(baseline_urls) else ""
            e_url = enhanced_urls[i] if i < len(enhanced_urls) else ""
            b_mark = " *" if b_url in relevant else ""
            e_mark = " *" if e_url in relevant else ""
            print(f"{i+1:<6} {(b_url + b_mark):<55} {e_url + e_mark}")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    n = len(per_query)
    print(f"\n{_SEP}")
    print(f"AGGREGATE  (macro-average, n={n} queries)")
    print(_SEP)

    avg_b_recalls = {k: sum(r["b_recalls"][k] for r in per_query) / n for k in k_values}
    avg_e_recalls = {k: sum(r["e_recalls"][k] for r in per_query) / n for k in k_values}
    avg_b_mrr = sum(r["b_mrr"] for r in per_query) / n
    avg_e_mrr = sum(r["e_mrr"] for r in per_query) / n

    header = _header()
    print(f"\n{header}")
    print("-" * len(header))
    for k in k_values:
        print(_row(f"Recall@{k}", avg_b_recalls[k], avg_e_recalls[k]))
    print(_row("MRR", avg_b_mrr, avg_e_mrr))
    print()


# ── Entry point ───────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Eval baseline vs. enhanced search")
    parser.add_argument(
        "--k",
        nargs="+",
        type=int,
        default=DEFAULT_K_VALUES,
        metavar="K",
        help="k values for Recall@k (default: 1 3 5 10)",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=DEFAULT_MAX_RESULTS,
        dest="max_results",
        help="results per Tavily call (default: 5)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_EMBED_THRESHOLD,
        help="cosine similarity cutoff for embedding-based relevance (default: 0.35)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    asyncio.run(
        evaluate(
            k_values=sorted(args.k),
            max_results=args.max_results,
            threshold=args.threshold,
        )
    )
