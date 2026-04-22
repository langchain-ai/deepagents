# ImproveQuery Implementation Plan

## Overview

Replace the single-provider `web_search` tool in `content_writer.py` with two new tools:
- `web_search_auto` — query expansion via LLM + multi-provider fan-out
- `web_search_multi` — lower-level multi-provider search (reused internally)

---

## 1. Query Expansion (`_expand_query`)

**Goal:** Turn one user query into 3 diverse, complementary sub-queries using `gpt-4o-mini`.

**Approach:**
- Call the OpenAI chat completions API (via `langchain_openai.ChatOpenAI` already available in the deepagents dependency tree, or directly via `openai`) with model `gpt-4o-mini`.
- System prompt instructs the model to return a JSON array of exactly 3 strings — no Markdown fences, no extra keys.
- Parse the JSON response; fall back to the original query only if parsing fails.

**System prompt template:**
```
You are a search query expander. Given a topic, return a JSON array of exactly 3
search queries that are diverse and complementary to the original. Each query
should target a different angle (e.g., definition/overview, recent developments,
practical examples). Output only a raw JSON array of strings — no markdown, no commentary.
```

**Environment variable required:** `OPENAI_API_KEY`

**Fallback:** if the LLM call fails or the response is not valid JSON, return `[query]` (single-element list) so the rest of the pipeline still works.

---

## 2. Provider Adapters

Three async helper functions, each normalising output to a common dict shape:

```python
{"url": str, "title": str, "content": str, "source": str}
```

### 2a. Tavily (`_search_tavily`)

- **Package:** `tavily-python` (already in `pyproject.toml`)
- **Key env var:** `TAVILY_API_KEY`
- **Sync client** → wrap with `asyncio.to_thread`

```python
async def _search_tavily(query: str, max_results: int, topic: str) -> list[dict]:
    from tavily import TavilyClient
    client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    raw = await asyncio.to_thread(
        client.search, query, max_results=max_results, topic=topic
    )
    return [
        {"url": r["url"], "title": r.get("title", ""), "content": r.get("content", ""), "source": "tavily"}
        for r in raw.get("results", [])
    ]
```

### 2b. Exa (`_search_exa`)

- **Package:** `exa-py>=2.0.0` (new dependency)
- **Key env var:** `EXA_API_KEY`
- **Async client available** (`AsyncExa`)

```python
async def _search_exa(query: str, max_results: int) -> list[dict]:
    from exa_py import AsyncExa
    exa = AsyncExa(api_key=os.environ["EXA_API_KEY"])
    response = await exa.search(
        query,
        num_results=max_results,
        type="auto",
        contents={"highlights": True},
    )
    results = []
    for r in response.results:
        highlights = " ".join(r.highlights or []) if hasattr(r, "highlights") else ""
        results.append({
            "url": r.url,
            "title": r.title or "",
            "content": highlights or getattr(r, "text", ""),
            "source": "exa",
        })
    return results
```

### 2c. DuckDuckGo (`_search_ddg`)

- **Package:** `duckduckgo-search>=8.0.0` (new dependency; note: package is being renamed to `ddgs` but `duckduckgo-search` still installs fine as of 8.1.1)
- **Key env var:** none — no API key required
- **Sync client** → wrap with `asyncio.to_thread`
- Result keys: `href` (URL), `title`, `body`

```python
async def _search_ddg(query: str, max_results: int) -> list[dict]:
    from duckduckgo_search import DDGS
    raw = await asyncio.to_thread(
        DDGS().text, query, max_results=max_results
    )
    return [
        {"url": r["href"], "title": r.get("title", ""), "content": r.get("body", ""), "source": "duckduckgo"}
        for r in (raw or [])
    ]
```

---

## 3. Provider Auto-Selection (`_auto_select_providers`)

Returns the top 2 available providers (in priority order: Tavily → Exa → DuckDuckGo).
DuckDuckGo requires no key and acts as the unconditional fallback.

```python
def _auto_select_providers() -> list[str]:
    available = []
    if os.environ.get("TAVILY_API_KEY"):
        available.append("tavily")
    if os.environ.get("EXA_API_KEY"):
        available.append("exa")
    available.append("duckduckgo")   # always last-resort
    return available[:2]
```

---

## 4. `web_search_multi` (fan-out + merge)

A lower-level tool that accepts an explicit `providers` list and a list of queries.

```python
@tool
async def web_search_multi(
    queries: list[str],
    providers: list[str] | None = None,
    max_results: int = 5,
    topic: Literal["general", "news"] = "general",
) -> dict:
    """Search with multiple providers concurrently and merge deduplicated results."""
    if providers is None:
        providers = _auto_select_providers()

    dispatch = {"tavily": _search_tavily, "exa": _search_exa, "duckduckgo": _search_ddg}

    async def _run(query: str, provider: str) -> list[dict]:
        try:
            fn = dispatch[provider]
            if provider == "tavily":
                return await fn(query, max_results, topic)
            return await fn(query, max_results)
        except Exception:
            return []

    tasks = [_run(q, p) for q in queries for p in providers]
    batches = await asyncio.gather(*tasks)

    seen_urls: set[str] = set()
    merged: list[dict] = []
    for batch in batches:
        for result in batch:
            url = result.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                merged.append(result)

    return {"results": merged, "query_count": len(queries), "provider_count": len(providers)}
```

---

## 5. `web_search_auto` (top-level tool)

Wraps `_expand_query` + `web_search_multi` into one @tool callable.

```python
@tool
async def web_search_auto(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news"] = "general",
) -> dict:
    """Search the web with automatic query expansion and multi-provider fan-out.

    Args:
        query: The original search query.
        max_results: Results per (query, provider) pair.
        topic: "general" for most queries, "news" for current events.

    Returns:
        Merged, deduplicated results plus the expanded sub-queries used.
    """
    sub_queries = await _expand_query(query)           # [q1, q2, q3]
    all_queries = [query] + sub_queries                 # original + 3 expansions

    result = await web_search_multi.ainvoke({
        "queries": all_queries,
        "max_results": max_results,
        "topic": topic,
    })
    result["original_query"] = query
    result["expanded_queries"] = sub_queries
    return result
```

---

## 6. Wire-up changes in `content_writer.py`

### 6a. Replace `web_search` with `web_search_auto`

- In `create_content_writer()`, replace `tools=[generate_cover, generate_social_image]` — the search tool is passed to subagents via `subagents.yaml`, not the top-level tool list, so update `load_subagents` to map `"web_search"` → `web_search_auto`.
- Alternatively, keep the old `web_search` name in `subagents.yaml` by aliasing: `available_tools = {"web_search": web_search_auto, ...}`.

### 6b. `AgentDisplay.print_message` update (minor)

The tool name in `tool_calls` will remain `web_search_auto` — update the display branch:
```python
elif name in ("web_search", "web_search_auto", "web_search_multi"):
    query = args.get("query", args.get("queries", [""])[0])
    ...
```

---

## 7. `pyproject.toml` dependency additions

```toml
dependencies = [
    ...
    "exa-py>=2.0.0",
    "duckduckgo-search>=8.0.0",
    "openai>=1.0.0",   # only if not already transitively available via deepagents
]
```

> `openai` is almost certainly already a transitive dependency of `deepagents`, but listing it explicitly makes the intent clear.

---

## 8. Environment variables summary

| Variable | Provider | Required? |
|---|---|---|
| `OPENAI_API_KEY` | Query expansion (gpt-4o-mini) | Yes (expansion falls back gracefully) |
| `TAVILY_API_KEY` | Tavily search | Optional |
| `EXA_API_KEY` | Exa search | Optional |
| *(none)* | DuckDuckGo | Always available |

---

## 9. File change summary

| File | Change |
|---|---|
| `content_writer.py` | Add `_expand_query`, `_search_tavily`, `_search_exa`, `_search_ddg`, `_auto_select_providers`; add `web_search_multi` and `web_search_auto` @tools; update `load_subagents` alias map; update `AgentDisplay` tool name check |
| `pyproject.toml` | Add `exa-py>=2.0.0`, `duckduckgo-search>=8.0.0` (and optionally `openai>=1.0.0`) |
| `subagents.yaml` | No change required — `web_search` key in `tools:` maps to `web_search_auto` via the alias in `load_subagents` |

---

## 10. Key design decisions

- **`asyncio.to_thread` for sync clients (Tavily, DDGS):** avoids blocking the event loop; no new threads created beyond what `asyncio.gather` manages.
- **Deduplication by URL only:** title/content may differ across providers; URL is the canonical identity of a result.
- **Original query always included:** the 3 LLM-generated sub-queries are *additions*, not replacements — the original query is always searched.
- **Graceful degradation:** each provider helper catches all exceptions and returns `[]`; `_expand_query` falls back to a single-query list; the overall tool always returns a valid dict.
- **No breaking change to `subagents.yaml`:** the `"web_search"` key in the YAML maps to whichever callable `load_subagents` wires it to — updating the alias dict is the only required change.
