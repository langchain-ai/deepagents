# Three browsing harnesses, one benchmark

Three Managed Deep Agents solve the same WebVoyager tasks. They differ only in
**how the agent reaches a browser** -- same dataset, same judge, same grading
thresholds. This is the write-up of what that difference bought.

| | `browser-mda` (CLI) | `browser-mda-repl` (REPL) | `browser-mda-mcp` (MCP) |
|---|---|---|---|
| Browser | **Local Chrome in a LangSmith sandbox** | Browserbase hosted | Browserbase hosted |
| Agent's tool surface | `execute` (shell) | `eval` (JavaScript) | 6 MCP tools |
| Browser API | `browse` CLI | Stagehand v4.2.0 Python SDK | Browserbase hosted MCP |
| Reasoning for act/extract | local model | Browserbase-side model | Browserbase-side model |
| Sandbox | yes | no | no |

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
| REPL | 99 | 74 | 21 | 4 | 74.7% | 10 | **34,492** | **$0.0146** | **$4.45** | 51 |
| MCP | 100 | **83** | 13 | 4 | **83.0%** | 13 | 101,206 | $0.0770 | $17.87 | **30** |

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
