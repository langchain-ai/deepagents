# Five browsing harnesses, one benchmark

Five Managed Deep Agents solve the same WebVoyager tasks. They differ only in
**how the agent reaches a browser** -- same dataset, same judge, same grading
thresholds. This is the write-up of what that difference bought.

| | CLI | REPL | MCP | facade | facade-repl |
|---|---|---|---|---|---|
| Browser | **sandbox Chrome** | Browserbase | Browserbase | Browserbase | Browserbase |
| Tool surface | `execute` (shell) | `eval` (JS) | 6 MCP tools | 3 facade tools | `eval` (JS) |
| Browser API | `browse` CLI | Stagehand v4 SDK | hosted MCP | Stagehand v4 SDK | Stagehand v4 SDK |
| Second model for act/extract | no | yes | yes | **no** | **no** |
| Sandbox | yes | no | no | no | no |

Projects: `browser-mda`, `browser-mda-repl`, `browser-mda-mcp`,
`browser-mda-facade`, `browser-mda-facade-repl`.

## The three arms

### CLI -- `browser-mda`

Google Chrome installed into the agent's own sandbox at snapshot-bake time and
driven headless through Browserbase's `browse` CLI. The agent's only browser
affordance is the built-in `execute` shell tool, so it composes work with pipes:

```sh
web-open 'https://www.allrecipes.com/search?q=cauliflower+pizza+crust' \
  && browse get markdown | head -c 6000
```

`web-open` is a wrapper pinning `--local --headless --no-sandbox
--disable-dev-shm-usage` and a `--host-resolver-rules` deny of the metadata IP.
This is the only arm where the browser runs inside the sandbox, and the only one
that pays a browser cold start -- see `benchmarks/webvoyager/LATENCY_BUDGET.md`.

### REPL -- `browser-mda-repl`

Stagehand v4 against a Browserbase session, but the tools are **never given to
the model**. `CodeInterpreterMiddleware(ptc=BROWSER_TOOLS)` exposes them inside
a QuickJS interpreter as `tools.navigate()`, `tools.extract()` and so on, so the
single tool the model can call is `eval`. One snippet navigates, extracts,
filters and loops, and the snippet decides what comes back.

Tool surface is held equal to MCP's by default:

```python
_PARITY = [navigate, extract, act, observe]
_EXTRAS = [evaluateJs, currentUrl]   # REPL_BROWSER_EXTRAS=1
```

`navigate` deliberately returns `f"{url} | {title}"` and nothing else. That one
decision drives most of the cost result below -- and, it turns out, most of the
accuracy result too.

### facade -- `browser-mda-facade`

Stagehand's facade contract -- `snapshot`, `run`, `screenshot` -- handed to the
model directly. `snapshot` returns an accessibility tree with bracketed IDs;
`run` takes either JavaScript or a batch of actions against those IDs.

No `act`/`extract`/`observe`, which is the point: those spend a Stagehand-side
model call that never enters this agent's trace. Here the agent reads the tree
and decides for itself, so every token is ours and visible.

### facade-repl -- `browser-mda-facade-repl`

The same three tools, reached through `CodeInterpreterMiddleware` instead of
being given to the model. The only clean A/B in the set: identical tools,
identical browser, identical model, invocation style the only variable.

### MCP -- `browser-mda-mcp`

Browserbase's hosted MCP (`stagehand-api`) as the whole browser layer. No
sandbox, no `execute`, no provisioning. The server exposes exactly six tools:

```
start  end  navigate  act  observe  extract
```

Verified live against `https://mcp.browserbase.com/mcp`; the self-hosted
`@browserbasehq/mcp@3.0.0` is byte-for-byte the same surface. The deprecated
`@browserbasehq/mcp-server-browserbase@2.4.3` had nine, including
`browserbase_screenshot`, `browserbase_stagehand_get_url` and
`browserbase_stagehand_agent`, all dropped in v3.

> The `snapshot` / `run` / `screenshot` trio in the Stagehand docs is a
> *different thing*: the agent-framework adapter contract shipped in the
> Stagehand repo (Deep Agents, CrewAI, Mastra, Vercel AI SDK, Eve). It is not
> the MCP server's surface, and MCP is only one possible transport for it.

## Results -- `webvoyager-100`, 100 tasks

`benchmarks/webvoyager/RESULTS_WV100.md` reports the same run with a different
cut of the same feedback, so the two tables disagree on one column and it is
worth saying why. That doc averages the binary `correct` score across all graded
rows; a row can be both blocked and correct (the agent was walled and said so
accurately). The table below is a strict three-way split where `BLOCKED` wins
the tie, so its `CORRECT` column is the smaller number. Blocked rates are
identical in both. Neither is wrong; use RESULTS_WV100 for the pass rate the UI
shows, this one for the decomposition.


| arm | graded | CORRECT | BLOCKED | INCORRECT | **rate** | med steps | med tokens | med $ | **total $** | med secs |
|---|---|---|---|---|---|---|---|---|---|---|
| CLI | 99 | 80 | 15 | 4 | 80.8% | 8 | 49,270 | $0.0511 | $11.65 | 63 |
| REPL | 99 | 74 | 21 | 4 | 74.7% | 10 | 34,492 | $0.0146 | $4.45 | 51 |
| MCP | 100 | 83 | 13 | 4 | 83.0% | 13 | 101,206 | $0.0770 | $17.87 | 30 |
| **facade** | 100 | 85 | 10 | 5 | 85.0% | 10 | **31,157** | $0.0171 | **$2.46** | **22** |
| facade-repl | 100 | **86** | **7** | 7 | **86.0%** | 11 | 40,825 | $0.0185 | $2.98 | 30 |

**The facade arm is 7.3x cheaper than hosted MCP at higher accuracy** -- $2.46
against $17.87, a third of the tokens, a quarter of the wall clock.

## Does code mode help? Only when the tools are atomic

`facade` and `facade-repl` run the same three tools over the same browser; the
only difference is whether the model calls them or writes code that calls them.
The scores tie (86 vs 85, with 4-7 INCORRECT either way) and the REPL costs 21%
more.

The traces say why. Across 10 traces the REPL issued **1.67 `tools.*` calls per
`eval`**, and 11 of 42 snippets called exactly one tool -- a pass-through
paying an interpreter hop for nothing. Model turns barely moved: 50 against 54.
One snippet makes it plain:

```js
for (let i = 0; i < 6; i++) {
  await tools.run({code: `const f = document.querySelector('[role=feed]');
                          f.scrollTop = f.scrollHeight; return 1`});
}
```

Six round trips through QuickJS to run a loop that belonged inside the page JS.

**`run({code})` is already code mode.** When the tools are atomic and
unprogrammable -- `navigate`, `extract`, `act`, `observe` -- a REPL collapses a
sequence into one model turn and wins, which is what the original REPL arm
showed against MCP. When one of the tools is "execute arbitrary JavaScript",
the batching is already inside it and the outer interpreter only adds a layer.

## A session leak that invalidated a run, and how to spot one

The first facade run scored 66/100 with 30 BLOCKED. It was void. The split
gives it away:

| | CORRECT | BLOCKED |
|---|---|---|
| first 50 rows | 45 | 4 |
| last 50 rows | 21 | 26 |

A harness does not get six times worse at browsing halfway through a run.
Sampling the blocked answers, 9 of 14 said some version of *"every browser call
failed with BrowserbaseSessionError: Failed to create a Browserbase session"*.
Live sessions were pinned at the project cap of 75.

`browserbase.launch(timeout=900)` sets a session TTL and **nothing closed the
session**: the thread ended, the session sat there holding a concurrency slot
for 15 minutes. At concurrency 8 with 35s tasks that is demand for ~205
sessions against a cap of 75. The hosted-MCP arm leaks the same way for a
different reason -- it calls `browserbase_start` and never `browserbase_end`,
with `keepAlive=true` on the URL.

Two fixes, both in the arms now: close the session on `after_agent`, and drop
the TTL to 300s as a backstop for runs that die before teardown. Afterwards,
two suites at concurrency 6 held 36 live sessions instead of saturating 75.

**The general lesson: a blocked rate that climbs through a run is an
infrastructure signal, not a site signal.** Check the first half against the
second before reporting any number.

## Why CLI and REPL are so much cheaper

It is one number. Tool-output bytes over 8 traces per arm:

| arm | tool output | the dominant call |
|---|---|---|
| REPL | 131,905 ch | `eval` 46.6%, `extract` 34.0%, **`navigate` 1.1%** |
| MCP | 216,293 ch | **`navigate` 80.3%** |
| CLI | 962,644 ch | `read_file` 78.5% (the skill, read once and cached), `execute` 21.5% |

**`browserbase_navigate` returns a median 16,968 characters. The REPL's
`navigate` returns 117.** A 145x difference on the single most frequent verb in
a browsing task. MCP's navigate serializes a CDP object graph whether or not the
agent wanted one, and the protocol gives no way to ask for less -- every tool
takes `sessionId` plus at most one natural-language string. The constraint is in
the tool surface, not in how we call it.

CLI's raw character total is the largest but misleads: 78.5% is `read_file`
pulling in the web-browsing skill, which is read once and prompt-cached. Its
actual per-action payload (`execute`, median 626 ch) is the smallest of the
three, because the agent chooses what to pipe through `head -c`.

Steps move the opposite way to tokens. MCP needs 13 model turns to CLI's 8
because every browser action is one round trip, where a single `execute` chains
`web-open … && browse get markdown | head -c 6000`. The REPL sits at 10 steps --
*more* than CLI -- yet is still the cheapest arm, because each of its steps
carries almost nothing.

## Why MCP scores higher -- and why that is not what it looks like

Split the grade into "did it get walled" versus "did it answer correctly once
it got through":

| arm | CORRECT | INCORRECT | BLOCKED | rate overall | **rate excluding blocked** |
|---|---|---|---|---|---|
| CLI | 80 | 4 | 15 | 80.8% | **95.2%** |
| REPL | 74 | 4 | 21 | 74.7% | **94.9%** |
| MCP | 83 | 4 | 13 | 83.0% | **95.4%** |

**All three answer correctly ~95% of the time when the site lets them in, and
all three get exactly 4 wrong.** The entire 8-point spread in the headline
number is bot-walling. MCP is not better at browsing; it is blocked less often.

Two further facts make the ranking weaker still:

1. **No task is blocked in all three arms.** Of 49 blocked rows, zero overlap
   across all three; 8 are CLI-only, 12 REPL-only, 5 MCP-only. Blocking is not a
   property of the site, it is a property of the attempt.
2. **REPL and MCP share a backend.** Both run on Browserbase, so infrastructure
   cannot explain REPL being blocked 21 times to MCP's 13.

What *can* explain it is the payload decision above. MCP's fat `navigate` hands
the agent the whole page on arrival, so an interstitial, a cookie wall or a
soft-block is visible immediately and often navigable -- sometimes the answer is
even sitting in the returned graph. The REPL's `navigate` returns a URL and a
title, so the agent is blind until it spends an `extract`, and it gives up
earlier. The behavioural signature is right there in the step counts:

| arm | CORRECT | BLOCKED |
|---|---|---|
| CLI | 8 steps / 61 s | 13 steps / 81 s |
| REPL | **6 steps / 24 s** | 12 steps / **159 s** |
| MCP | 12 steps / 24 s | 39 steps / 186 s |

REPL's successful runs are the fastest and leanest of any arm by a wide margin
-- 6 steps, 24 seconds. Its blocked runs are its worst: it thrashes for 159
seconds. The narrow payload is both why it is 4x cheaper than MCP and why it
walls more.

**So the honest reading is: token cost and accuracy are trading against each
other through exactly one design choice -- how much page state `navigate`
returns.** That is a dial, not a verdict, and nobody has tried the middle of it.

## What to try next

- **Adaptive navigate.** Return `url | title` by default; fall back to a
  snapshot when the title or status looks like an interstitial. Should recover
  most of REPL's 21 blocked rows at a fraction of MCP's per-call cost. This is
  the single highest-value experiment outstanding.
- **`REPL_BROWSER_EXTRAS=1`.** Measures code mode's ceiling rather than its
  MCP-matched floor. `page.snapshot()` is the interesting one -- it is the
  accessibility tree that `extract` currently burns a model call to interpret.
- **Kill the sandbox cold start.** 29.5% of CLI wall clock, and the mitigation
  needs no model work. See `benchmarks/webvoyager/LATENCY_BUDGET.md`.

## Caveats, stated plainly

- MCP's and REPL's `act`/`extract`/`observe` run model calls on Browserbase's
  side that never enter our traces. **Both arms' step counts and costs
  understate real work**, and are not like-for-like with a variant whose
  reasoning is all local. CLI is the only arm whose trace is complete.
- The Stagehand reasoning model is pinned to `google/gemini-2.5-flash-lite` in
  both Browserbase arms so they differ in interface rather than model. It is
  still a different model from the one driving the agent.
- Only CLI runs the browser in a sandbox. Comparing its latency to the other two
  conflates sandbox cold start with tool surface.
- 4 INCORRECT out of ~100 in every arm is small enough that differences in that
  column are noise.
