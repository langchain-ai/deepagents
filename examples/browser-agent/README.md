# Browsing deep agent

One agent, two browsers. The filesystem and `execute` tool come from a sandbox
backend in both cases; `BrowserMiddleware` supplies `snapshot`, `run` and
`screenshot` on top.

```bash
# hosted Browserbase
export BROWSERBASE_API_KEY=bb_live_...
uv run --with "deepagents[browser]" python agent.py "what is the top story on Hacker News?"

# Chrome on this machine
BROWSER=local uv run --with "deepagents[browser]" python agent.py "..."
```

Swapping the provider is the only change:

```python
provider = LocalChrome() if os.environ.get("BROWSER") == "local" else Browserbase()
create_deep_agent(backend=LocalShellBackend(), middleware=[BrowserMiddleware(provider)])
```

`CDP(url=...)` attaches to a browser you already have. It does not close it --
only a provider that opened a browser closes it.

## The tools

| tool | what it does |
|---|---|
| `snapshot` | accessibility tree, bracketed ID on every interactive element |
| `run` | `code` (JavaScript in the page) **or** `actions` (ops against snapshot IDs) |
| `screenshot` | JPEG of the page, for charts and layout |

Ops: `goto`, `click`, `hover`, `fill`, `type`, `press`, `select`.

Set `BrowserMiddleware(provider, allow_javascript=False)` to drop the `code`
path, matching the default on Anthropic's own browser toolset.
