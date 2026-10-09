# Harness findings: CLI vs MCP

From 180 traced runs (30 WebVoyager tasks x 3 reps x 2 variants), experiments
`webvoyager-cli-native-1791582955` and `webvoyager-mcp-native-1791582963`.
Token figures are measured from per-call `usage_metadata`, not estimated.

| | CLI (sandbox Chrome + `browse`) | MCP (Browserbase hosted) |
| --- | --- | --- |
| correct >= 0.5 | 46/90 | 36/90 |
| blocked >= 0.5 | 22/90 | 10/90 |
| hard errors, no answer | 0 | **19** |
| median tokens / task | 67,361 | **176,863** |
| median cost / task | $0.057 | $0.114 |
| total spend | $8.55 | $17.51 |
| median wall clock | 68.7 s | 55.2 s |

## 1. `browserbase_navigate` returns the CDP object graph

Every navigate injects a serialized Playwright/WebSocket object: `_events`,
`_eventsCount`, `_binaryType`, `_closeCode`, `_receiver._writableState`, and --
the bulk of it -- raw CDP frames encoded as JSON arrays of integer byte values.
In the largest payload seen (79,389 chars), **57,680 chars (72.7%) were
`{"type":"Buffer","data":[123,34,...]}`**.

Payload size across 238 navigate calls: p25 18,416 / median 21,372 / p75 23,566
/ p90 32,503 / max 79,389 chars.

Cost, measured by differencing `input_tokens` between consecutive model calls:

| tool | calls | median tokens injected | total carried | share of MCP input |
| --- | --- | --- | --- | --- |
| **`browserbase_navigate`** | 238 | **11,160** | **20,712,898** | **66.7%** |
| `browserbase_act` | 196 | 418 | 1,662,882 | 5.4% |
| `browserbase_extract` | 287 | 233 | 1,174,816 | 3.8% |
| CLI `execute` | 627 | **280** | 15,205,706 | 65.0% |

**One navigate costs 40x one `browse` shell command, and roughly 99% of it is
machine noise with no information value.** Because a tool result stays in context
for every later turn, 238 navigate calls carry 20.7M tokens -- two thirds of the
MCP variant's entire input budget.

The CLI is cheap for a specific reason: the agent can trim at the source.
84/90 CLI runs open with `browse get markdown`, routinely piped --
`| head -150`, `| grep -oE '[0-9]+ kg CO2e'`, `browse eval
'document.querySelector("#score").innerText'`. The MCP agent has no equivalent;
the payload arrives whole or not at all.

**Fix:** wrap the MCP tools and replace `data.result` with `{url, status, title}`.
Removes ~20.5M of 31.0M input tokens on completed MCP runs (66%), ~$6-7 of the
$11.70 spent on them, and probably most of the crashes -- one died on
`'Server-sent event exceeded the 1048576 byte limit'`, and a 79KB payload is
within a factor of 13 of that ceiling.

## 2. CLI wall clock is mostly Chrome cold start

Span decomposition (rep 1 of all 30 tasks, overlapping spans unioned):

| | CLI | MCP |
| --- | --- | --- |
| median wall clock | 68.7 s | 55.2 s |
| LLM share | **11%** | 35% |
| tool share | 88% | 64% |

`execute` spans by kind:

| | n | median | total |
| --- | --- | --- | --- |
| **first `web-open` of a task** | 29 | **47.1 s** | 1,344 s |
| later `web-open` | 89 | 12.7 s | 1,933 s |
| other `execute` | 105 | 9.8 s | 1,349 s |

The first `web-open` is **70% of the median task's wall clock**, and it is
44-55 s on every task regardless of page weight -- the signature of a fixed
startup cost (Chrome launch + daemon handshake), not page load. The agent is not
thinking slowly: model time is 11%.

MCP has no equivalent (`browserbase_start` median 1.8 s), which is why it is
faster per task despite twice the local token cost. Note the asymmetry: the CLI's
dominant cost is fixed and removable; MCP's is variable, invisible to our traces,
and outside our control.

**Fix:** pre-warm the daemon and a blank page in the sandbox snapshot. ~34 s per
task, ~51 minutes across a 90-run benchmark.

## 3. The two variants are not yet measuring the same thing

| task | CLI blocked | MCP blocked |
| --- | --- | --- |
| Allrecipes--20/21 | **6/6** | 0/6 |
| Booking--4 | **3/3** | 0/3 |
| Google Search--13/14 | **6/6** | 2/6 |
| Google Map--33 | **3/3** | 1/3 |
| BBC News--4 | 0/3 | **3/3** |

The sandbox's egress IP reads as a datacenter IP to Allrecipes, Booking and
Google; Browserbase's proxying gets through nearly all of it. **24% of CLI runs
(22/90) are lost to this.** Until the sandbox has comparable egress, CLI 46/90
and MCP 36/90 are not comparable numbers.

## 4. A third of MCP spend bought nothing

| | runs | tokens | cost |
| --- | --- | --- | --- |
| errored | 19 | 23.3M | **$5.81** |
| completed | 71 | 31.2M | $11.70 |

17 of 19 are `MCPError(-32000, 'SSE stream ended without a response')`. Booking--4
alone burned 12.6M tokens across three reps and returned nothing each time.
These are transport failures and must never be scored as agent failures -- the
runner now retries them.

## 5. Failure causes, excluding blocks

**CLI: 22 non-blocked failures, none below 0.25.** There are no catastrophic
failures; every one is a partial-credit near miss.

| cause | n | tasks |
| --- | --- | --- |
| stale reference; agent's answer defensible | 8 | GitHub--37 x3, GitHub--38 x3, Coursera--7 x2 |
| unverified inference, or hedging that undercuts a right answer | 4 | ESPN--23 x2, ESPN--24 x2 |
| answered off an in-progress page | 3 | BBC News--3 x3 |
| partial completion of a multi-part task | 2 | ArXiv--41 x2 |
| browser crash blocked enumeration | 2 | Google Flights--3 x2 |
| shallow search for "latest" | 2 | BBC News--4 x2 |
| incomplete enumeration -> wrong negative | 1 | Google Map--32 |

**MCP: 44 non-blocked failures, dominated by infrastructure** -- 19 transport
crashes, then 6 widget-driving loops, 7 stale-reference, 3 `extract` returning
nothing usable.

Measured MCP tool health: `act` returned "Failed to perform act: No action found"
on **15/196 calls (7.7%)**; **108/287 `extract` results were under 400 characters**
(effectively empty); 33 identical calls were repeated within a single run.

## 6. Instruction compliance

Mostly good. This is not a compliance problem except in two places.

| instruction | CLI |
| --- | --- |
| read SKILL.md first | 87/90 |
| prefer reading over clicking | 84/90 open with `get markdown` |
| budget screenshots | 21 across 90 runs |
| keep notes in `/workspace` | 29/90 (short runs do not need it) |
| **final answer is the answer, no narration** | **violated 67/90** |

The narration rule is costing real score. ESPN--24 reached the right answer in
**two tool calls**, then spent four lines explaining it could not confirm the
game aired on an ESPN channel; it scored 0.32. The instruction currently says
both "do not narrate your process" and "state what you checked", and the model
resolves that by narrating.

MCP: `sessionId` is omitted **57 times**, forcing a duplicate navigate in
**55/90 runs**, against an instruction that says in bold to pass it on every
call. And the "change approach after ~10 calls" rule is phrased as advice and
treated as such: Google Flights--3 made **14 `act` calls at one dropdown across
33 consecutive turns**; Google Map--32 made **15 consecutive scroll attempts** on
one panel. Those two runs cost 3.36M tokens and $0.61.

## Ranked fixes

1. **Truncate `browserbase_navigate` to `{url, status, title}`** -- 66% of MCP
   tokens, ~$6-7, and probably most of the 19 crashes.
2. **Pre-warm Chrome in the sandbox snapshot** -- ~34 s/task, 70% of median CLI
   wall clock.
3. **Give the sandbox comparable egress** (residential proxy) or accept that
   24% of CLI tasks are untestable and exclude them from headline numbers.
4. **Bind `sessionId` server-side** in an MCP wrapper -- removes a wasted round
   trip from 55/90 runs.
5. **Make the retry budget a counted rule**, not advice: "if the same element or
   objective has failed twice, change route or report the blocker." Would have
   cut Google Flights--3 from 38 calls to ~8.
6. **Rewrite the final-answer rule** so it cannot be read as inviting narration:
   answer first and in full, one `Source:` line, a caveat only if it changes the
   answer, never explain what you did not do.
7. **Give the MCP agent a cheap read path** (`get_markdown`/`get_html`).
   `extract` returned under 400 chars on 38% of calls, and all three ESPN--24
   reps worked around it by hand-navigating to a JSON API.
8. **Disambiguate "most recent"** in both instruction files -- BBC News--3 failed
   6/6 across both variants by answering off a mid-tournament leaderboard.
9. **Require opening the item page before quoting a count** -- "2.9K reviews" off
   a listing card is not an answer.
10. **Refresh the dataset.** 15 of 66 non-blocked failures are the agent being
    right against a reference the live web has moved past.

## Known unknown

The Browserbase-side Stagehand model calls (`act`/`extract`/`observe`, default
`gemini-2.5-flash-lite`) are invisible to our traces by construction. MCP's true
all-in cost is strictly higher than the $17.51 measured here by an unknown
amount, and its 4.1 s median `extract` latency and 38% thin-result rate are the
only observable proxies for quality on that side.
