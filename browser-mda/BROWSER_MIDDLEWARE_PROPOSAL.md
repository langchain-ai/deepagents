# Proposal: `BrowserMiddleware` for deepagents

A mock-up. Ships the Stagehand facade tools -- `snapshot`, `run`, `screenshot`
-- as one middleware, over either a hosted Browserbase session or any CDP URL.

## The whole API

```python
from deepagents.middleware.browser import BrowserMiddleware

# Hosted Browserbase. Reads BROWSERBASE_API_KEY.
agent = create_deep_agent(middleware=[BrowserMiddleware()])

# Any CDP endpoint: a Chrome you launched, one in a sandbox, a remote grid.
agent = create_deep_agent(middleware=[
    BrowserMiddleware(cdp_url="ws://127.0.0.1:9222/devtools/browser/abc"),
])
```

One class, two optional kwargs, three tools. No connection protocol, no
provider registry, no toggles -- there is one real choice here (whose browser)
and it is one argument. If a third kind of connection ever shows up, that is
when to grow an abstraction, not before.

## The tools

Stagehand's facade contract, unchanged:

- **`snapshot()`** -- accessibility tree with a bracketed ID on every
  interactive element.
- **`run(code=...)` or `run(actions=[...])`** -- exactly one. `code` is
  JavaScript evaluated in the page; `actions` are
  `{"op": ..., "id": ...}` steps against the last snapshot's IDs.
  Ops: goto, click, hover, fill, type, press, select.
- **`screenshot()`** -- returns an image content block.

No `act`/`extract`/`observe`. Those spend a Stagehand-side model call that never
enters your trace, which is most of why the hosted-MCP arm cost 7.3x what this
one did on the same 100 tasks.

## Sketch

```python
class BrowserMiddleware(AgentMiddleware):
    """Give the agent a browser: snapshot, run, screenshot."""

    def __init__(self, *, cdp_url: str | None = None, api_key: str | None = None) -> None:
        self._cdp_url = cdp_url
        self._api_key = api_key
        self._browsers: dict[str, StagehandBrowser] = {}
        self.tools = [snapshot, run, screenshot]

    async def _open(self):
        if self._cdp_url:
            return await local_browser.connect(cdp_url=self._cdp_url)
        key = self._api_key or os.environ["BROWSERBASE_API_KEY"]
        # TTL is a backstop; the session is closed on after_agent. A long TTL is
        # how one finished benchmark suite held all 75 of a project's
        # concurrency slots and starved every other run.
        return await browserbase.launch(api_key=key, timeout=300)

    async def aafter_agent(self, state, runtime):
        """Release the browser. Nothing else does."""
        browser = self._browsers.pop(_thread_id(runtime), None)
        # Only close what we opened: a CDP endpoint belongs to whoever started it.
        if browser is not None and not self._cdp_url:
            await browser.close()
```

Opened-vs-attached is the one distinction worth encoding: a Browserbase session
we created must be closed, a CDP endpoint we attached to must not be.

## Three behaviours to build in, each already paid for

**One browser per thread, with no fallback key.** Keyed on `thread_id`. A
literal `"default"` in a long-lived server process hands one thread's dead
session to every later thread, and every run after the first failure dies on
`RPC client is closed`.

**Relaunch on a dead transport, never on a page error.** A dropped websocket
should evict and relaunch once. `RPCError: Uncaught` from a selector that missed
must not, or a bad snippet destroys a healthy browser and loses the page.

**Tool failures are messages, not exceptions.** LangGraph's
`_default_handle_tool_errors` re-raises, so one exception ends the run -- six of
the first seventeen runs of one arm died that way. Returning
`error: RPCError: Uncaught` lets the next turn fix its own selector.

## Two divergences from the TypeScript facade

The reference `run` executes code with Playwright-shaped `page`, `context` and
`browser` in scope. Python's `page.evaluate` runs in the **page**, so:

1. `await page.goto(url)` -- the form the contract documents and every model
   writes -- fails. Navigation is an action: `{"op": "goto", "url": ...}`.
2. `evaluate` takes an *expression*, so a top-level `return` is a syntax error.
   Wrap snippets containing `return` in an async IIFE.

Both were found by a smoke test failing every navigation. Worth documenting in
the tool docstring rather than papering over.

## Don't wrap this in a REPL

Measured, not assumed. A sixth arm put these same three tools behind QuickJS:
**1.67 `tools.*` calls per `eval`**, 11 of 42 snippets calling exactly one tool,
model turns 50 against 54. Score was a tie (86 vs 85) at 21% higher cost.

`run({code})` is already code mode. When tools are atomic -- `navigate`,
`extract`, `act` -- a REPL collapses a sequence into one turn and wins. When one
tool is "execute arbitrary JavaScript", the batching is already inside it.

## What this scored

webvoyager-100, against the four other arms:

| arm | correct | blocked | tokens | **total $** | secs |
|---|---|---|---|---|---|
| CLI (sandbox Chrome) | 80 | 15 | 49,270 | $11.65 | 63 |
| REPL (Stagehand in QuickJS) | 74 | 21 | 34,492 | $4.45 | 51 |
| MCP (hosted Browserbase) | 83 | 13 | 101,206 | $17.87 | 30 |
| **facade** | **85** | **10** | **31,157** | **$2.46** | **22** |
| facade in QuickJS | 86 | 7 | 40,825 | $2.98 | 30 |
