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

`browser_snapshot` preserves its original node fields and adds bounded semantic metadata gathered
from each exact Playwright element handle: tag, accessible-ish role/name, and
`disabled`/`checked`/`selected`/`expanded`/`readonly`/`required`/`focused`/`editable` states.
`browser_screenshot` returns standard text and image content blocks. Its concise text block contains
only page reference, MIME type, and byte count. The tool artifact repeats that metadata and the
standard JSON-compatible image block, so checkpoints never need to serialize raw bytes; base64 is
confined to structured image data and is never included in the plain-text block.

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
- actions accept only allowlisted strict discriminators: exact-reference `click`, `type`, `press`,
  `select`, and `scroll_into_view`, plus reference-free `scroll` with fixed direction and
  viewport-relative distance literals;
- page scrolling runs only a package-owned script with validated literals—never user JavaScript,
  coordinates, amounts, selectors, or deltas—and reports bounded before/after positions plus whether
  the page moved;
- every action uses the configured operation timeout, invalidates all element references after the
  attempt, and returns bounded diagnostics on success;
- operational action failures return stable error codes plus resnapshot guidance, while attempts on
  sensitive controls still raise and fail closed;
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
first calls are single-flight. A context is created per stable LangGraph `thread_id`. Middleware
holds a manager lease for every complete tool operation, so an in-flight context cannot be evicted.
When `max_contexts` is full, the least-recently-used zero-lease session is closed before a new
isolated context is created; if all sessions are leased, the operation fails with
`context_limit_reached`. `BrowserRuntimeManager.get_session()` remains available for advanced direct
usage, but its return value is intentionally unleased and immediately eligible for eviction; hosts
that span retrieval and operation must use `lease_session()`. Advanced hosts can call
`aclose_session(thread_id)` for targeted idle cleanup, and middleware shutdown waits for bounded
in-flight leases before closing all remaining resources.

## Testing

Unit tests use deterministic fake Playwright objects and make no network calls:

```bash
make test
```
