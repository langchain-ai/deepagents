# ImproveQuery — Task List

Source plan: `ImproveQueryPlan.md`
Files to modify: `content_writer.py`, `pyproject.toml`

---

## Tasks

| # | Task | File | Depends on |
|---|---|---|---|
| T1 | Add `exa-py>=2.0.0` and `duckduckgo-search>=8.0.0` to `dependencies` | `pyproject.toml` | — |
| T2 | Implement `_expand_query(query) -> list[str]` — calls `gpt-4o-mini`, returns 3 sub-queries, falls back to `[query]` | `content_writer.py` | — |
| T3 | Implement `_search_tavily(query, max_results, topic) -> list[dict]` — wraps existing `TavilyClient` in `asyncio.to_thread`, normalises to `{url, title, content, source}` | `content_writer.py` | — |
| T4 | Implement `_search_exa(query, max_results) -> list[dict]` — uses `AsyncExa`, normalises highlights/text | `content_writer.py` | T1 |
| T5 | Implement `_search_ddg(query, max_results) -> list[dict]` — wraps `DDGS().text` in `asyncio.to_thread`, maps `href`→`url`, `body`→`content` | `content_writer.py` | T1 |
| T6 | Implement `_auto_select_providers() -> list[str]` — priority: Tavily → Exa → DuckDuckGo, returns top 2, based on env vars | `content_writer.py` | — |
| T7 | Implement `web_search_multi` `@tool` — fans out all `(query, provider)` pairs with `asyncio.gather`, merges and deduplicates by URL | `content_writer.py` | T3, T4, T5, T6 |
| T8 | Implement `web_search_auto` `@tool` — calls `_expand_query` then `web_search_multi`, attaches `original_query` and `expanded_queries` to result | `content_writer.py` | T2, T7 |
| T9 | Wire up: alias `"web_search"` → `web_search_auto` in `load_subagents`; update `AgentDisplay.print_message` to match tool names `web_search_auto`/`web_search_multi`; remove old `web_search` `@tool` | `content_writer.py` | T8 |

---

## Parallelism

- **Start immediately (parallel):** T1, T2, T3, T6
- **After T1:** T4, T5 (parallel with each other)
- **After T3 + T4 + T5 + T6:** T7
- **After T2 + T7:** T8
- **After T8:** T9


## Notes

- `subagents.yaml` requires no changes — the alias in `load_subagents` maps `"web_search"` → `web_search_auto` transparently.
- DuckDuckGo needs no API key and is the unconditional fallback provider.
- All provider adapters share the same output shape: `{"url": str, "title": str, "content": str, "source": str}`.
- Each provider adapter must catch all exceptions and return `[]` — the pipeline must never hard-fail on a single provider outage.
