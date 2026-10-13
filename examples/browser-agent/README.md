# Browsing deep agent

One agent, two browsers. The filesystem and `execute` tool come from a sandbox
backend in both cases; `BrowserMiddleware` supplies `snapshot`, `run` and
`screenshot` on top.

```bash
# hosted Browserbase
export BROWSERBASE_API_KEY=bb_live_...
uv run --with "deepagents[browser]" python agent.py "what is the top story on Hacker News?"

# Chrome on this machine -- no Browserbase key needed
BROWSER=local uv run --with "deepagents[browser]" python agent.py "..."

# Chrome inside the agent's sandbox, attached over CDP -- no key either
BROWSER=sandbox uv run --with "deepagents[browser]" python agent.py "..."
```

Swapping the provider is the only change:

```python
provider = LocalChrome() if os.environ.get("BROWSER") == "local" else Browserbase()
create_deep_agent(backend=LocalShellBackend(), middleware=[BrowserMiddleware(provider)])
```

`CDP(url=...)` attaches to a browser you already have. It does not close it --
only a provider that opened a browser closes it.

## Chrome in the sandbox

`sandbox_chrome.py` is a provider in ~60 lines: it starts Chrome through the
agent's own backend, reads the CDP endpoint back out, and hands it to
`local_browser.connect`. Nothing about it is special-cased in the middleware.

Two things decide whether it works for you:

* **Reachability.** The tools run in the agent process, so the CDP URL the
  sandbox prints has to be routable from there. True when the sandbox shares a
  host with the agent (`LocalShellBackend`, a dev container, a VM with the port
  published). Not true for a Managed Deep Agent, whose per-thread sandbox
  offers shell and filesystem but no inbound ports.
* **Chrome's flags.** Stagehand drives a browser extension, so the browser must
  be started with `--enable-unsafe-extension-debugging`, `--remote-allow-origins=*`
  and friends, and must be full Chrome -- `chrome-headless-shell` has no
  extension support. `LocalChrome()` sets these for you; a browser you launch
  yourself does not.

## The tools

| tool | what it does |
|---|---|
| `snapshot` | accessibility tree, bracketed ID on every interactive element |
| `run` | `code` (JavaScript in the page) **or** `actions` (ops against snapshot IDs) |
| `screenshot` | JPEG of the page, for charts and layout |

Ops: `goto`, `click`, `hover`, `fill`, `type`, `press`, `select`.

Set `BrowserMiddleware(provider, allow_javascript=False)` to drop the `code`
path, matching the default on Anthropic's own browser toolset.
