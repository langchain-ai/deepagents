# Nine browsing harnesses, one benchmark

Nine browsing agents solve the same WebVoyager tasks. They differ only in
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
sandbox, no `execute`, no provisioning.

**The server exposes exactly six tools -- four verbs and two lifecycle calls.**
It is easy to count four, because the lifecycle pair is invisible in a task
description and `end` is never called at all (see below). From a live
`tools/list` against `https://mcp.browserbase.com/mcp`:

| tool | parameters | description | AI call? |
|---|---|---|---|
| `start` | `sessionId` | Create or reuse a Browserbase session | no |
| `end` | `sessionId` | Close the current Browserbase session | no |
| `navigate` | `url`, `sessionId` | Navigate to a URL | no |
| `act` | `action`, `sessionId` | Perform an action on the page | **yes** |
| `observe` | `instruction`, `sessionId` | Observe actionable elements on the page | **yes** |
| `extract` | `instruction`, `sessionId` | Extract data from the page | **yes** |

That is the entire surface -- every tool, every parameter. No snapshot, no
screenshot, no JavaScript, no tab control. Note how thin the signatures are:
`sessionId` plus at most one natural-language string. There is no way to ask
`navigate` for a smaller payload, which is why it returns a median 16,968
characters whether or not the agent wanted them.

Three of the six route to a model on Browserbase's side, which is both why this
arm is the most expensive and why its trace understates the work done.

Observed in our runs: `browserbase_act` 111 calls, `browserbase_extract` 61,
`browserbase_navigate` 50, `browserbase_observe` 34, `browserbase_start` 12 --
and **`browserbase_end` zero times**. Nothing ever releases a session; combined
with `keepAlive=true` on the URL, that is the leak that pinned the project at
its concurrency cap.

The self-hosted `@browserbasehq/mcp@3.0.0` is byte-for-byte the same six. The
deprecated `@browserbasehq/mcp-server-browserbase@2.4.3` had nine, including
`browserbase_screenshot`, `browserbase_stagehand_get_url` and
`browserbase_stagehand_agent`, all dropped in v3.

For scale, the three surfaces we looked at: this server's **six**, Stagehand's
facade **three** (`snapshot`, `run`, `screenshot`), and Anthropic's
`browser_toolset_20260801` **thirty** members. Stagehand's adapter executes all
thirty Anthropic members against four facade primitives, so member count and
capability are not the same axis.

> The `snapshot` / `run` / `screenshot` trio in the Stagehand docs is a
> *different thing*: the agent-framework adapter contract shipped in the
> Stagehand repo (Deep Agents, CrewAI, Mastra, Vercel AI SDK, Eve). It is not
> the MCP server's surface, and MCP is only one possible transport for it.

## The nine arms

Five are Managed Deep Agents driving a remote or sandboxed browser; four run
the agent, its tool server and its browser in one process tree. Every arm uses
the same dataset, the same `claude-sonnet-5-5`, the same Jev judge and the same
thresholds.

| arm | tool surface | browser | where it runs |
|---|---|---|---|
| CLI | `execute` (shell) → `browse` CLI | Chrome in a LangSmith sandbox | deployment |
| REPL | `eval` (JS) → Stagehand v4 | Browserbase | deployment |
| MCP | 6 hosted MCP tools | Browserbase | deployment |
| facade | `snapshot`/`run`/`screenshot` | Browserbase | deployment |
| facade-repl | the same three, inside QuickJS | Browserbase | deployment |
| playwright MCP | 25 tools over stdio MCP | its own Chromium | CI runner |
| browser-use | 16 tools over stdio MCP | **Browserbase**, one session per task | CI runner |
| agent-browser | `execute` (shell) → `agent-browser` CLI | Chrome on the runner | CI runner |
| ~~wdio~~ | 17 tools over stdio MCP | — | **blocked** |

`wdio` is recorded rather than deleted: `provider: external` wants a W3C
WebDriver endpoint on localhost, Browserbase speaks CDP, and no URL satisfies
both. Its remote options are the named providers (browserstack, saucelabs,
testingbot), which need their own accounts.

Two things had to be true before the local arms produced anything meaningful,
and neither was obvious:

**A broken browser scores INCORRECT, not ERROR.** An agent whose Chrome never
starts reports politely that it could not browse, and the judge marks that
wrong. Two arms returned a clean-looking 0/100 with green CI before this was
caught, so the provisioning step now asserts a headless launch before any task
runs.

**browser-use needed a hosted browser, not a better local one.** Six attempts
at installing Chrome on a runner failed; pointing it at a Browserbase session
per task worked first time. `keepAlive` is mandatory -- without it the session
ends on the first disconnect, which happens during MCP tool discovery, and the
real connection gets `HTTP 410` -- and sessions must be released explicitly,
because `keepAlive` means nothing else will.


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
| facade-repl | 100 | **86** | 7 | 7 | **86.0%** | 11 | 40,825 | $0.0185 | $2.98 | 30 |
| **facade** | 100 | 85 | 10 | 5 | 85.0% | 10 | 31,157 | $0.0171 | $2.46 | 22 |
| MCP | 100 | 83 | 13 | 4 | 83.0% | 13 | 99,282 | $0.0768 | $15.98 | 29 |
| browser-use | 100 | 82 | 13 | 5 | 82.0% | 14 | 57,579 | $0.1202 | **$58.81** | 46 |
| CLI | 99 | 80 | 15 | 4 | 80.8% | 8 | 47,552 | $0.0492 | $8.23 | 63 |
| agent-browser | 100 | 77 | 18 | 5 | 77.0% | 10 | 34,495 | $0.0206 | $3.09 | **17** |
| playwright MCP | 100 | 75 | 21 | 4 | 75.0% | 26 | 103,728 | $0.2188 | $23.80 | 53 |
| REPL | 99 | 74 | 21 | 4 | 74.7% | 10 | 34,417 | $0.0142 | **$2.42** | 35 |

**The facade arm is 7.3x cheaper than hosted MCP at higher accuracy** -- $2.46
against $17.87, a third of the tokens, a quarter of the wall clock.

## WebTailBench-150: three arms, and three surprises

A second benchmark, pulled from the Stagehand evals package: 609 real-world
"tail" tasks across 11 categories, of which a proportionally stratified 150
were run. The judge differs from WebVoyager's because the dataset does --
every row ships a 3-6 item rubric instead of a reference answer, so the score
is the fraction of criteria met. Partial credit is the normal case here: an
agent that finds the flights but not the fare difference is not simply wrong.

| arm | graded | **score** | med tokens | **total $** | med steps | med secs |
|---|---|---|---|---|---|---|
| **facade-noshot** | 150 | **0.797** | **76,328** | **$10.43** | 17.5 | 51 |
| agent-browser | 150 | 0.777 | 118,116 | $10.44 | 22.0 | 119 |
| facade | 149 | 0.769 | 77,061 | $11.85 | 17.0 | 50 |

### 1. Removing screenshots made the facade arm better, not just cheaper

Screenshots were **93.8% of the facade arm's tool output** -- 42 calls across
11 smoke tasks at a median 89KB of base64 each. Dropping them was expected to
trade accuracy for cost. It did not: `facade-noshot` scores **0.797 against
0.769**, at $10.43 against $11.85, on the same tokens and the same wall clock.

The screenshots were not buying sight so much as spending budget. The
accessibility tree `snapshot` returns is apparently enough for these tasks,
and the images mostly displaced useful context. This is the one result here
that is directly actionable: ship the facade without `screenshot` by default.

### 2. The 11-task smoke was wrong about everything that mattered

The smoke said facade **0.80** and agent-browser **0.67** -- a 17-point gap --
and that agent-browser used 16% *fewer* tokens. At n=150 the gap is 0.769 vs
0.777, with agent-browser slightly **ahead**, and it uses **53% more** tokens
(118,116 against 77,061).

Both smoke conclusions inverted. Eleven tasks across eleven categories felt
like reasonable coverage and was not: one task per category means one draw
from each of eleven different distributions. Worth remembering before any
future "the smoke looks good, ship it" -- including the one in this PR's own
history.

### 3. agent-browser is 2.4x slower, and fixed sleeps are why

119s median against 51s. It is **not** cold start -- the first tool call takes
a median 1.4s, versus 3.9s for the facade -- and it is not call volume, at
14.1 calls per task against 15.2. It is the tail:

| | median call | p75 | p90 | calls over 10s |
|---|---|---|---|---|
| facade | 2.5s | 4.2s | **7.2s** | 7% |
| agent-browser | 3.6s | 9.1s | **31.0s** | 23% |

The slowest calls say why outright:

```
180.3s  agent-browser click @e1532; sleep 5; agent-browser ...
152.6s  sleep 3; agent-browser snapshot -i 2>&1 | head -6
129.1s  agent-browser open https://www.americanstandard-us.com/...
```

**The model writes `sleep` in shell.** 56 of the 81 calls over 10 seconds are
`open`, whose page-load wait has no cap, and the rest are dominated by
hand-written sleeps.

The facade arm does not have this problem because `page.waitForTimeout(n)` is
reinterpreted there as "wait until the page goes quiet, capped at n" -- a
measured `waitForTimeout(8000)` returns in 0.6s. That override reaches the
Playwright API and nothing else. **A shell arm can always `sleep`, and no
amount of tool-level cleverness intercepts it.**

Which sharpens the original finding. Capping fixed sleeps is worth ~30% of
agent time, but only on a surface where waits are expressed through an API you
control. Give the model a shell and the remedy is unavailable; the only
remaining lever is the prompt, which measured *worse* upstream.

## Playwright MCP: excellent tools, expensive surface

`@playwright/mcp` 1.64.0-alpha exposes **25 tools** -- `browser_snapshot`,
`browser_click`, `browser_type`, `browser_fill_form`, `browser_evaluate`,
`browser_run_code_unsafe`, `browser_network_requests`, `browser_console_messages`,
`browser_tabs`, `browser_wait_for` and more. It is the richest surface in this
comparison by some distance, and it is also the only arm where that is a
liability.

It costs **$23.80 against the facade's $2.46 -- 10x -- for ten fewer correct
answers.** 103,728 median tokens to the facade's 31,157; 26 median steps to 10.

An earlier revision of this section reported $66.16 and 82.7%. That run was
measured on `claude-sonnet-5` while every other arm ran `claude-sonnet-5-5`,
which is a mistake worth recording: the same arm, same dataset and same judge
moved from $66.16 to $25.54 and from 82.7% to 77.7% on the model change alone.
Any table mixing the two is meaningless.

Nothing is wrong with the tools. Only **4 INCORRECT** in 100 graded rows: this
arm almost never answers wrongly, it gets walled -- 21 times, the joint worst
in the set. The cost is structural. Twenty-five fine-grained verbs mean a model
turn per action, and `browser_snapshot` returns a full page representation
where the facade's `run` returns whatever the snippet chose.

That is the same mechanism that made hosted MCP expensive, further along the
same axis, and it is the clearest statement of the pattern this benchmark keeps
producing: **what a tool returns matters more than how good the tools are.**

This arm ran on a laptop rather than a deployment, so its wall clock is not
comparable to the rows above it. Token counts and cost are.

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

## What the whole set says

**Cost spans 24x and accuracy spans 11 points, and they are barely related.**
The two cheapest arms (REPL at $2.42, facade at $2.46) bracket the accuracy
range -- last and second -- while the most expensive (browser-use, $58.81)
lands fourth and the richest tool surface (playwright MCP, 25 tools) is seventh
of eight.

What separates them is how much a tool returns and how many turns it costs to
get it:

* `browserbase_navigate` returns a median 16,968 characters whether or not the
  agent wanted them; the facade's `run` returns whatever the snippet chose.
* Playwright's 25 fine-grained verbs mean a model turn per action -- 26 median
  steps against the facade's 10.
* facade ($2.46) and agent-browser ($3.09) have nothing in common except that
  the agent decides what comes back: one writes JavaScript, the other writes
  shell. Both run ~10 steps a task.

**Most accuracy differences are bot-walling, not browsing skill.** Across the
set, INCORRECT counts cluster between 3 and 7 regardless of arm, while BLOCKED
ranges from 7 to 21. Arms do not mostly answer wrongly; they mostly get walled,
and which site walls which run is close to arbitrary -- of 49 blocked rows in
the first three arms, none overlapped across all three.

**Caveats worth carrying.** Every arm is now n=100 bar CLI and REPL at 99,
which lost one row each to an ungradeable empty answer. Three CI jobs were
SIGTERMed at the old 240-minute cap and finished by resuming only their missing
examples, so no arm rests on a truncated base. Wall clock is not comparable
between deployment arms and CI arms. And three arms spend a second model on Browserbase's side that
never enters our traces, so their token counts understate the real work.

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
