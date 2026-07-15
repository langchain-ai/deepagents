# deepagents-browser

Experimental Playwright browser middleware for Deep Agents.

- Distribution: `deepagents-browser`
- Import: `deepagents_browser`
- Python: 3.11+

## Installation

For a normal Deep Agents Code installation, install the browser extra through dcode. This installs
the Python package and provisions the matching Playwright Chromium runtime in the same tool
environment:

```bash
dcode --install browser
```

Then relaunch with browser capability enabled:

```bash
dcode --browser
```

For an editable development checkout, install the extra and Chromium from `libs/code`:

```bash
uv sync --extra browser
uv run playwright install chromium
```

Neither importing this package nor constructing `BrowserMiddleware` downloads a browser,
initializes Playwright, or launches Chromium. Browser provisioning remains an explicit install
step rather than occurring during import or the first tool call.

## Usage

```python
from deepagents_browser import BrowserLimits, BrowserMiddleware

browser = BrowserMiddleware(
    limits=BrowserLimits(
        max_contexts=8,
        max_tabs_per_context=4,
        max_snapshot_nodes=200,
        max_snapshot_chars=32_000,
        max_screenshot_bytes=2_000_000,
        max_requests_per_context=1_000,
    )
)
```

The middleware eagerly exposes exactly five tools: `browser_navigate`, `browser_snapshot`,
`browser_act`, `browser_screenshot`, and `browser_tabs`. Tools are removed from model requests
unless private agent state contains the strict boolean `{"_browser_enabled": True}`. A truthy string or integer does not activate access. Tool
execution checks the same flag again before touching the runtime. Tools are async-only. Call
`await browser.aclose()` at application shutdown.

## Security model

- only HTTP(S) navigation is accepted;
- URL credentials, local/private/link-local/multicast/reserved/unspecified/CGNAT destinations,
  and mixed public/private DNS answers are rejected;
- asynchronous DNS validation happens before navigation with a hard timeout;
- Playwright routing validates every intercepted request and redirect target, while service
  workers are disabled to prevent route bypass;
- each LangGraph thread gets an isolated incognito context;
- contexts, tabs, requests, snapshot nodes/chars, fixed-viewport screenshot bytes, DNS time, and
  action time are bounded with non-overridable hard caps;
- actions accept only latest-generation opaque references and allowlisted `click`, `type`, `press`,
  or `select` discriminators;
- references pin the exact snapshotted element identity, cannot retarget after DOM reorder or
  replacement, and are invalidated by top-level navigation;
- password and payment controls are omitted from snapshots and blocked again immediately before
  actions;
- stale, changed, detached, sensitive, or navigation-invalidated references fail closed with a
  stable machine-readable `BrowserRuntimeError.code`;
- arbitrary JavaScript, CDP, raw user selectors, file uploads, downloads, clipboard access, and
  persistent browser profiles are not exposed.

### Network isolation warning

**DNS validation and Playwright request interception are not complete DNS-rebinding protection.**
DNS can change between validation and connection, and application checks are not transport-level
egress controls. Production deployments handling untrusted URLs must additionally force traffic
through a policy-enforcing egress proxy or run the browser in a network namespace/container whose
firewall permits only approved public destinations. Do not treat this package alone as an SSRF
sandbox.

## Resource lifecycle

The first activated operation lazily starts one Playwright driver and Chromium browser. Concurrent
first calls are single-flight. A context is created per stable LangGraph `thread_id` and retained
until idempotent asynchronous cleanup.

## Testing

Unit tests use deterministic fake Playwright objects and make no network calls:

```bash
make test
```
