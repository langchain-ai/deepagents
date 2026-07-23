---
name: langsmith-link-preservation
description: Batch-check, refresh, or preserve LangSmith trace/share links, run IDs, and workspace endpoints. Use when asked to check, verify, refresh, renew, or preserve LangSmith share links or expiring trace links, or to audit the reachability of many LangSmith runs/endpoints at once.
license: MIT
compatibility: designed for deepagents-code
---

# LangSmith Link Preservation

Checking many LangSmith share links, run IDs, or workspace endpoints is a batch job. Do it in ONE script, not one `execute` call per link.

## Workflow

1. **Collect the full worklist up front.** Gather every link / run-id / endpoint to check into one list before running anything, instead of discovering and checking them one at a time.
2. **Write ONE script that loads `.env` once and checks concurrently.** Load `.env` a single time, then issue the HTTP checks in parallel (a `ThreadPoolExecutor` or `asyncio`), collecting results into one consolidated status table (link → HTTP status / ok / expired / error).
3. **Run it with a single `execute` call.** One script per turn covers the whole worklist. Re-run only to retry failures, not per link.

## Anti-pattern to avoid

Do NOT re-run a per-link snippet like `from dotenv import load_dotenv; load_dotenv(); import os, requests; requests.get(one_link)` once for each link. Repeating the `load_dotenv`/`requests` preamble across dozens of `execute` calls explodes trajectory length and token cost for no benefit — the same checks belong in one concurrent script.

## Example

```python
import os
from concurrent.futures import ThreadPoolExecutor

import requests
from dotenv import load_dotenv

load_dotenv()
headers = {"x-api-key": os.environ["LANGSMITH_API_KEY"]}

links = [...]  # full worklist collected up front

def check(url):
    try:
        r = requests.get(url, headers=headers, timeout=15)
        return url, r.status_code
    except requests.RequestException as e:
        return url, f"error: {e}"

with ThreadPoolExecutor(max_workers=16) as pool:
    for url, status in pool.map(check, links):
        print(f"{status}\t{url}")
```

## Safety

Never `grep`, `cat`, or `print` raw `.env` contents while doing this. Read secrets through `os.environ` after `load_dotenv()`; never echo the API key or other credentials into output.
