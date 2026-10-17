# Why the MCP variant costs 4x and runs longer

Same 30 WebVoyager tasks, 3 repetitions, same model, same judge. The only
variable is the browser surface: `browser-agent` drives Chrome through the
`browse` CLI over one `execute` tool; `browser-agent-mcp` drives Browserbase
through `browserbase_{start,navigate,act,observe,extract}`.

| | CLI (`350a640d`) | MCP (`00f12e88`) | ratio |
|---|---|---|---|
| total cost, 90 runs | **$8.97** | **$36.38** | 4.1x |
| cost, median run | $0.060 | $0.140 | 2.3x |
| prompt tokens, mean | 258k | 1.91M | 7.4x |
| steps, median | 10 | 22 | 2.2x |
| seconds, median | 82 | **67** | **0.8x** |
| seconds, mean | 147 | 180 | 1.2x |

## MCP is not slower per step -- it has a fat tail

The median MCP task finishes *faster* than the median CLI task. Its individual
operations are quicker: median tool-call latency is 3.6s against the CLI's
10.4s, because one `browse` invocation does more work per call than one
`browserbase_act`.

What moves the mean is the tail:

| | runs >= 60 steps | share of variant cost | share of wall-clock | worst run |
|---|---|---|---|---|
| CLI | 3 / 90 | 18% | 16% | 109 steps, $0.59, 812s |
| MCP | **17 / 90** | **72%** | **70%** | **386 steps, $4.48, 1862s** |

Seventeen MCP runs -- under a fifth of the suite -- account for nearly three
quarters of everything it spent. ArXiv--41 took 221 steps and $2.10 on MCP
against 41 steps and $0.22 on the CLI; Booking--4 took 257 steps and $2.71
against 84 and $0.54. Capping steps would cut MCP's bill far more than any
per-call optimisation.

## Two mechanisms multiply

**1. Step inflation (2.2x).** The MCP surface splits one intention across
several model turns. Over 6 matched traces: 53 `navigate`, 111 `act`, 62
`extract`, 35 `observe` -- 261 tool calls where the CLI used 110 `execute`.
Every one of those is a full round trip through the model, which is also why
MCP burns 323 LLM calls per 14 traces against the CLI's 180.

**2. Payload inflation (1.8x prompt tokens per step).** Median tool output:

| tool | median output | max |
|---|---|---|
| `browserbase_navigate` | **20,014 chars** | 27,514 |
| `browserbase_extract` | 580 | 354,988 |
| `browserbase_act` | 1,041 | 3,635 |
| `execute` (CLI) | **585** | 77,359 |

`navigate`'s 20KB is not a page dump -- it is worse than that. Sampled across
56 navigate results, the payload is a serialised Playwright/CDP connection
object that opens
`{"success":true,"data":{"result":{"page":{"conn":{"ws":{"_events":...,"_closeCode":1006,...`
and runs on into raw HTTP response headers. Zero occurrences of `"url"`,
`"title"`, `"markdown"` or any accessibility tree. **The agent pays ~20KB per
page load and learns nothing**, then has to spend an extra `extract` or
`observe` round trip to find out what it just loaded. That is both halves of the
problem in one call: the context cost and the step cost.

The CLI returns 585 characters at the median because the agent writes a command
that asks for exactly what it wants, and pulls the large thing with `read_file`
only when it decides it needs it. Median prompt tokens per step: 11,707 for MCP
against 6,601 for the CLI.

Those compound -- 2.2x the steps, each carrying 1.8x the context, against a
conversation history that is re-sent in full on every turn.

## What would actually move the number

1. **Cap the step budget per task.** 17 runs drive 72% of cost; a cap near the
   p75 (40 steps) bounds the blow-ups without touching the median task.
2. **Fix `navigate`'s return value.** It currently serialises the CDP
   connection object -- ~20KB of websocket internals with no page content. A
   status line plus the final URL and title would be strictly more informative
   at ~0.1% of the size, and would remove the follow-up `extract` whose only job
   is to discover what the page is. This is the single clearest defect found.
3. **Collapse observe+act.** Most `observe` calls exist to find a target for the
   `act` that follows; merging them removes roughly a third of the round trips.

Correctness is analysed separately in `MCP_CORRECTNESS_ANALYSIS.md`; the step
tails and the correctness gap are likely the same phenomenon seen twice.
