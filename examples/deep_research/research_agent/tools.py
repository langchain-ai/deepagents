"""Research Tools.

This module provides search and content processing utilities for the research agent,
using Tavily for URL discovery and fetching full webpage content.
"""

import asyncio
import inspect
import re
from collections import Counter
from typing import Any, Callable

import numpy as np
from sentence_transformers import SentenceTransformer

import anthropic
import httpx
from langchain_core.tools import InjectedToolArg, tool
from markdownify import markdownify
from pydantic import BaseModel
from tavily import TavilyClient
from typing_extensions import Annotated, Literal

tavily_client = TavilyClient()

# Set to True to run expand_query → parallel_search → deduplicate → rank
# before fetching page content. Falls back to a single Tavily call when False.
USE_QUERY_EXPANSION: bool = False

_async_anthropic_client: anthropic.AsyncAnthropic | None = None


def _get_async_anthropic_client() -> anthropic.AsyncAnthropic:
    global _async_anthropic_client
    if _async_anthropic_client is None:
        _async_anthropic_client = anthropic.AsyncAnthropic()
    return _async_anthropic_client


_EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
_embedding_model: SentenceTransformer | None = None


def _get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(_EMBEDDING_MODEL_NAME)
    return _embedding_model


class _ExpandedQueries(BaseModel):
    paraphrase: str
    broader: str
    narrower: str
    synonym_variation: str
    related_concept: str


_EXPAND_QUERY_SYSTEM_PROMPT = """\
You are a search query expansion expert. Given a user query, produce exactly one \
query for each of the following five roles:

- paraphrase: reword the query while preserving its exact intent
- broader: widen the scope to the parent topic or domain
- narrower: drill into a specific aspect, subtopic, or use-case
- synonym_variation: replace key terms with meaningful synonyms or alternate phrasing
- related_concept: a query on a closely related concept that would yield complementary results

Return only the five fields — no explanations."""


async def expand_query(query: str) -> list[str]:
    """Generate 6 diverse search queries (original + 5 structured variations) using Claude.

    The five generated variants cover: paraphrase, broader scope, narrower scope,
    synonym variation, and related concept. The original query is prepended so callers
    always have the full set.

    Args:
        query: The original search query to expand.

    Returns:
        A list of 6 queries: [original, paraphrase, broader, narrower,
        synonym_variation, related_concept].
    """
    client = _get_async_anthropic_client()

    response = await client.messages.parse(
        model="claude-opus-4-7",
        max_tokens=512,
        system=[
            {
                "type": "text",
                "text": _EXPAND_QUERY_SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[
            {
                "role": "user",
                "content": f"Expand this query:\n\n{query}",
            }
        ],
        output_format=_ExpandedQueries,
    )

    variants = response.parsed_output
    return [
        query,
        variants.paraphrase,
        variants.broader,
        variants.narrower,
        variants.synonym_variation,
        variants.related_concept,
    ]


async def parallel_search(
    queries: list[str],
    search_fn: Callable[[str], Any],
    timeout: float = 10.0,
) -> list[Any]:
    """Run multiple search queries concurrently and return combined results.

    Accepts both sync and async search functions. Sync functions are offloaded
    to a thread pool so they don't block the event loop. Each query has an
    individual timeout; timed-out or failed queries are dropped so one slow
    result never blocks the rest.

    Args:
        queries: List of search queries to execute.
        search_fn: Callable that accepts a query string and returns results.
                   May be sync or async.
        timeout: Per-query timeout in seconds. Defaults to 10.

    Returns:
        Flat list of results in the same order as the input queries.
        Timed-out or failed queries are omitted from the output.
    """
    if inspect.iscoroutinefunction(search_fn):
        async def _call(q: str) -> Any:
            return await asyncio.wait_for(search_fn(q), timeout=timeout)
    else:
        async def _call(q: str) -> Any:
            return await asyncio.wait_for(
                asyncio.to_thread(search_fn, q), timeout=timeout
            )

    results = await asyncio.gather(*[_call(q) for q in queries], return_exceptions=True)

    return [r for r in results if not isinstance(r, BaseException)]


# Compiled once at module level — used by both string and dict paths
_URL_PATTERN = re.compile(r"\*\*URL:\*\*\s+(https?://\S+)")
_TITLE_PATTERN = re.compile(r"^##\s+(.+)", re.MULTILINE)


def _title_vector(title: str) -> Counter:
    """Bag-of-words term-frequency vector for a title string."""
    return Counter(re.findall(r"\w+", title.lower()))


def _cosine_similarity(a: Counter, b: Counter) -> float:
    """Cosine similarity between two term-frequency Counters."""
    if not a or not b:
        return 0.0
    dot = sum(a[k] * b[k] for k in a if k in b)
    mag_a = sum(v * v for v in a.values()) ** 0.5
    mag_b = sum(v * v for v in b.values()) ** 0.5
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


def _is_near_duplicate_title(
    title: str,
    kept_vectors: list[Counter],
    threshold: float,
) -> bool:
    """Return True if title is too similar to any already-kept title."""
    vec = _title_vector(title)
    return any(_cosine_similarity(vec, kv) >= threshold for kv in kept_vectors)


def deduplicate_results(
    results: list[Any],
    title_threshold: float = 0.85,
) -> list[Any]:
    """Remove duplicate and near-duplicate search results.

    Two deduplication passes are applied in order:

    1. **Exact URL dedup** — results sharing a URL are collapsed to the first
       occurrence.
    2. **Near-duplicate title dedup** — after URL dedup, results whose titles
       exceed ``title_threshold`` cosine similarity (bag-of-words TF vectors)
       with any already-kept title are dropped.

    Handles two formats produced by search functions in this module:

    - **String** (``tavily_search``): each item is a multi-result markdown
      block. Sections are split on ``---``; titles are extracted from ``## …``
      headers; URLs from ``**URL:** …`` lines.
    - **Dict** (raw Tavily / ``web_search``): each item is a response dict
      with a ``"results"`` list of ``{"url": …, "title": …, …}`` records.

    In both cases the first occurrence of each URL is kept, insertion order is
    preserved, and the output is a flat list of individual result items.

    Args:
        results: Output of ``parallel_search`` — a list of strings or dicts.
        title_threshold: Cosine similarity cutoff for near-duplicate titles.
            0.85 drops paraphrased headlines; lower values are more aggressive.

    Returns:
        Flat, deduplicated list of individual result items.
    """
    if not results:
        return []

    seen_urls: set[str] = set()
    kept_title_vectors: list[Counter] = []

    def _keep(url: str, title: str) -> bool:
        """Return True and register if this result should be kept."""
        if url and url in seen_urls:
            return False
        if title and _is_near_duplicate_title(title, kept_title_vectors, title_threshold):
            return False
        if url:
            seen_urls.add(url)
        if title:
            kept_title_vectors.append(_title_vector(title))
        return True

    deduped: list[Any] = []

    if isinstance(results[0], str):
        for response in results:
            for block in response.split("---"):
                block = block.strip()
                if not block:
                    continue
                url_match = _URL_PATTERN.search(block)
                title_match = _TITLE_PATTERN.search(block)
                url = url_match.group(1) if url_match else ""
                title = title_match.group(1).strip() if title_match else ""
                if _keep(url or block, title):
                    deduped.append(block)
        return deduped

    # Dict format: {"results": [{"url": ..., "title": ..., ...}], "query": "..."}
    for response in results:
        for item in response.get("results", []):
            url = item.get("url", "")
            title = item.get("title", "")
            if _keep(url, title):
                deduped.append(item)
    return deduped


def _url_from_result(result: Any) -> str:
    """Extract URL from a result regardless of format."""
    if isinstance(result, dict):
        return result.get("url", "")
    match = _URL_PATTERN.search(result)
    return match.group(1) if match else ""


def _text_from_result(result: Any) -> str:
    """Extract a bag-of-words-friendly text representation of a result."""
    if isinstance(result, dict):
        return f"{result.get('title', '')} {result.get('content', '')}"
    # String format: strip markdown symbols for cleaner tokenisation
    text = re.sub(r"[#*`>-]", " ", result)
    return text[:2000]  # cap to avoid over-weighting long content


def _url_frequency_map(raw_results: list[Any]) -> dict[str, int]:
    """Count how many query responses each URL appeared in."""
    counts: dict[str, int] = {}
    for response in raw_results:
        if isinstance(response, dict):
            urls = {item.get("url", "") for item in response.get("results", [])}
        else:
            urls = set(_URL_PATTERN.findall(response))
        for url in urls:
            if url:
                counts[url] = counts.get(url, 0) + 1
    return counts


async def rank_results(
    results: list[Any],
    original_query: str,
    raw_results: list[Any] | None = None,
    query_weight: float = 0.5,
    frequency_weight: float = 0.5,
) -> list[Any]:
    """Rank search results by semantic similarity to the original query and
    cross-query frequency.

    Each result receives a combined score:

    .. code-block:: text

        score = query_weight    * query_similarity
              + frequency_weight * frequency_score

    - **query_similarity**: cosine similarity between the sentence embedding of
      the original query and the embedding of each result's title + content
      snippet. All texts are encoded in a single batched call to
      ``all-MiniLM-L6-v2`` (run in a thread pool so the event loop is not
      blocked).
    - **frequency_score**: how many of the expanded queries returned this URL,
      normalised to [0, 1] by the maximum observed count. Requires
      ``raw_results`` (the pre-deduplication output of ``parallel_search``).
      When omitted, the full weight falls on ``query_similarity``.

    Args:
        results: Deduplicated results from ``deduplicate_results``.
        original_query: The original (unexpanded) query string.
        raw_results: Pre-deduplication output of ``parallel_search``. Used to
            count how many queries each URL appeared in. Optional — when
            ``None``, frequency scoring is skipped.
        query_weight: Weight for semantic similarity component (default 0.5).
        frequency_weight: Weight for frequency component (default 0.5).
            Ignored when ``raw_results`` is ``None``.

    Returns:
        Results sorted by descending score. Original list is not mutated.
    """
    if not results:
        return []

    # Build frequency map; normalise by the highest count seen
    freq_map: dict[str, int] = _url_frequency_map(raw_results) if raw_results else {}
    max_freq = max(freq_map.values(), default=1)

    # Effective weights — collapse to query-only when no frequency data
    eff_query_w = 1.0 if not freq_map else query_weight
    eff_freq_w = 0.0 if not freq_map else frequency_weight

    # Batch-encode query + all result texts in one call to avoid per-item overhead.
    # SentenceTransformer.encode is CPU-bound; run in a thread pool.
    texts = [original_query] + [_text_from_result(r) for r in results]
    model = _get_embedding_model()
    embeddings: np.ndarray = await asyncio.to_thread(
        model.encode, texts, normalize_embeddings=True, show_progress_bar=False
    )

    # With L2-normalised vectors cosine similarity reduces to a dot product.
    query_emb = embeddings[0]           # shape: (dim,)
    result_embs = embeddings[1:]        # shape: (n_results, dim)
    query_sims: np.ndarray = result_embs @ query_emb  # shape: (n_results,)

    def _score(idx: int) -> float:
        url = _url_from_result(results[idx])
        freq_score = freq_map.get(url, 0) / max_freq if freq_map else 0.0
        return float(eff_query_w * query_sims[idx] + eff_freq_w * freq_score)

    indices = sorted(range(len(results)), key=_score, reverse=True)
    return [results[i] for i in indices]


def fetch_webpage_content(url: str, timeout: float = 10.0) -> str:
    """Fetch and convert webpage content to markdown.

    Args:
        url: URL to fetch
        timeout: Request timeout in seconds

    Returns:
        Webpage content as markdown
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        response = httpx.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        return markdownify(response.text)
    except Exception as e:
        return f"Error fetching content from {url}: {str(e)}"


@tool(parse_docstring=True)
async def tavily_search(
    query: str,
    max_results: Annotated[int, InjectedToolArg] = 1,
    topic: Annotated[
        Literal["general", "news", "finance"], InjectedToolArg
    ] = "general",
) -> str:
    """Search the web for information on a given query.

    Uses Tavily to discover relevant URLs, then fetches and returns full webpage content as markdown.
    When USE_QUERY_EXPANSION is enabled, the query is expanded into multiple variants and results
    are deduplicated and ranked before page content is fetched.

    Args:
        query: Search query to execute
        max_results: Maximum number of results to return (default: 1)
        topic: Topic filter - 'general', 'news', or 'finance' (default: 'general')

    Returns:
        Formatted search results with full webpage content
    """
    def _search(q: str) -> dict:
        return tavily_client.search(q, max_results=max_results, topic=topic)

    if USE_QUERY_EXPANSION:
        queries = await expand_query(query)
        raw_results = await parallel_search(queries, _search)
        deduped = deduplicate_results(raw_results)
        ranked = await rank_results(deduped, query, raw_results=raw_results)
        result_items = ranked
    else:
        search_results = await asyncio.to_thread(_search, query)
        result_items = search_results.get("results", [])

    result_texts = []
    for item in result_items:
        url = item.get("url", "")
        title = item.get("title", "")
        content = await asyncio.to_thread(fetch_webpage_content, url)
        result_texts.append(f"## {title}\n**URL:** {url}\n\n{content}\n\n---\n")

    return f"🔍 Found {len(result_texts)} result(s) for '{query}':\n\n{chr(10).join(result_texts)}"


@tool(parse_docstring=True)
def think_tool(reflection: str) -> str:
    """Tool for strategic reflection on research progress and decision-making.

    Use this tool after each search to analyze results and plan next steps systematically.
    This creates a deliberate pause in the research workflow for quality decision-making.

    When to use:
    - After receiving search results: What key information did I find?
    - Before deciding next steps: Do I have enough to answer comprehensively?
    - When assessing research gaps: What specific information am I still missing?
    - Before concluding research: Can I provide a complete answer now?

    Reflection should address:
    1. Analysis of current findings - What concrete information have I gathered?
    2. Gap assessment - What crucial information is still missing?
    3. Quality evaluation - Do I have sufficient evidence/examples for a good answer?
    4. Strategic decision - Should I continue searching or provide my answer?

    Args:
        reflection: Your detailed reflection on research progress, findings, gaps, and next steps

    Returns:
        Confirmation that reflection was recorded for decision-making
    """
    return f"Reflection recorded: {reflection}"
