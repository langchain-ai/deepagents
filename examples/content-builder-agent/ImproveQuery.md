1. One query in → LLM generates 3 related queries
web_search_auto(query) in content_writer.py sends the single query to an LLM (openai:gpt-4o-mini by default) with a system prompt that returns a JSON array of 3 diverse, complementary sub-queries.

2. Two search providers instead of only Tavily
web_search_multi fans the queries out across the providers list (default: top 2 auto-selected from Tavily, Exa, DuckDuckGo). Every (query, provider) pair runs concurrently via asyncio.gather.

3. Merge results, remove duplicates
After gathering, results are grouped by query and deduplicated by URL — if the same URL appears from two providers it is kept only once.

