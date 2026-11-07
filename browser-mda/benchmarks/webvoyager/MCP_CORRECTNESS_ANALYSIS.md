# Why the MCP variant scores lower on correctness (WebVoyager-30)

Dataset `e06b6845-b213-4c5f-9369-3a6cfdee9331`, 30 examples x 3 repetitions.

| variant | project_id | mean `correct` | mean `blocked` | mean steps | median steps | mean seconds | mean `correct_confidence` |
|---|---|---|---|---|---|---|---|
| CLI | `350a640d-8258-4147-8ffd-d6a44666caad` | 0.733 | 0.256 | 17.9 | 10 | 147 | 0.569 |
| MCP | `00f12e88-8cc5-47b1-b0d3-effd95f3aa0f` | 0.622 | 0.189 | 45.8 | 25 | 188 | 0.514 |

Grading is the Jev typed judge in `experiment_native.py`. `correct` is thresholded at
`correct_confidence >= 0.5`, and the `false` criterion reads: *"Something asked for is
missing or **hedged**, or a value was guessed, recalled, or worked out without visiting
the site."* That wording matters for several rows below.

## 1. Per-task / per-run failure modes

Modes: **A** act() could not perform the interaction, **B** extract()/observe() returned
the wrong or lossy slice, **C** judge penalised a hedge on a factually correct answer,
**D** wrong page or wrong entity reached, **E** run died on MCP transport, **F** bot wall
and no primary source reached.

| task | run (prefix) | mcp `correct` | cli | mode | evidence quoted from the trace |
|---|---|---|---|---|---|
| ArXiv--41 | `01a122f8` | ERROR (ungraded) | 0.667 | E | run status `error`; `MCPError(-32001, "Request 'tools/call' timed out")` after 215 steps / 798.9 s |
| ArXiv--41 | `01a12303` | 0.0 | | A | 41 `act` calls, 18 hard errors; final: *"I didn't open the cart page itself."* |
| ArXiv--41 | `01a12314` | 0.0 | | A | 8 consecutive rephrasings of the same click (`"select size XL"` → `"click the radio button for XL size"` → `"choose XL"`), each `AI_NoObjectGeneratedError: No object generated: response did not match schema.`; final: *"The browser tool's click action kept failing on the size options."* |
| ArXiv--41 | `01a1233b` | 0.0 | | A/B | *"The cart page showed its size as \"L\", so the size may be wrong."* |
| BBC News--3 | `01a122fc` | 1.0 | 1.0 | — | correct |
| BBC News--3 | `01a1230d` | 0.0 (conf 0.47) | | **C** | *"Zero players have a Total of -10."* — the same fact the three CLI runs stated and were scored 1.0 for |
| BBC News--3 | `01a1232b` | 0.0 (conf 0.40) | | **C** | *"**No players have a Total of -10.** The leader is Grant Forrest at -9"* |
| BBC News--4 | `01a122f8` | 0.0, blocked 1.0 | 0.333 | A/B | *"Clicking a result never left the search page, and the article URLs I tried came back as a 404."* |
| BBC News--4 | `01a12303` | 0.0, blocked 1.0 | | A/B | three `act` clicks on the headline error out; a fourth reports `"performed successfully on selector: xpath=…/a[1]"` yet the next extract returns *"Search results page"* |
| BBC News--4 | `01a12318` | 0.0, blocked 1.0 | | A/B | *"The two article URLs I was given turned out to be guesses and returned a 404"*; extract on the guessed URL: *"Error 404 - Oops, the page you're looking for is no longer here"* |
| Cambridge Dictionary--35 | `01a122f9` | 0.0 | 1.0 | D | landed on the *Prepositional phrases* page, not *Prepositions: uses*; *"The page lists them without ranking, so it doesn't say which is most common."* |
| Coursera--6 | `01a122f9` | 0.0 | 1.0 | B/D | *"The syllabus extraction gave \"4 assignments\" for each of the 4 modules, which looks unreliable."* |
| Coursera--6 | `01a12304` | 0.0 | | B/D | `act` reports a successful click on the search result, but the next extract returns *"URL: https://www.coursera.org/"*; five different extract instructions all return the same string, *"3 assignments"*, and the per-module extract returns *"Module 1 … 3 assignments / Module 2 … 3 assignments / Module 3 … 3 assignments"* |
| Coursera--6 | `01a1231a` | 0.0 | | B/D | *"The page does not list quizzes… I can't give a quiz count."* CLI read the same style of page and answered *"Quiz 1 through Quiz 8, each 30 minutes"* |
| Coursera--7 | `01a122f8` | 0.0 (conf 0.43) | 1.0 | **C** | *"Fundamentals of Reinforcement Learning … 4.8 stars … about 2.9K reviews"* + *"I didn't open the course page"*. CLI run `01a1230e` says *"I … did not open the course page itself"* and scored 1.0 |
| Coursera--7 | `01a12303` | 0.0 (conf 0.45) | | **C** | same answer, same disclosure |
| Coursera--7 | `01a12314` | 1.0 | | — | same answer, no second disclosure sentence |
| ESPN--24 | `01a122ff` | 0.0, blocked 1.0 | 0.667 | C/B | *"The scoreboard page at espn.com/nfl/scoreboard loaded but wouldn't give me its text."* Score reported (*"Tampa Bay Buccaneers 24, Dallas Cowboys 16"*, Oct 8 2026) matches the CLI answers graded 1.0 |
| ESPN--24 | `01a12310` | 0.0 | | B/D | answered a different game — *"Atlanta Falcons 45, New Orleans Saints 24"* — after reading the network field out of the JSON feed |
| ESPN--24 | `01a12331` | 0.0 | | C/B | extract on the scoreboard returned *"null"*, then *"Something went wrong!"*, then stale *"SUNDAY NIGHT FOOTBALL / DEC 31, 2023 / LIONS VS. VIKINGS / 42-32"*; the agent fell back to `site.api.espn.com` and reported the same Bucs 24 – Cowboys 16 result the CLI was marked correct for |
| GitHub--37 | `01a12302` | 0.0 | 1.0 | B | extract flattened the page to a heading list; the agent then reported the trio *"Manage agent-driven work from one place / Make your editor your most powerful accelerator / Ship faster with AI that work alongside you"*, while all three CLI runs returned the real card trio *"Go beyond one-size-fits-all / Use your agents, your way / Stay in your flow"* |
| Google Search--13 | `01a122fb` | 0.0 | 1.0 | F | *"I got this from Google's search results page and its snippets of IMDb and Reddit. I didn't open IMDb or any other page directly."* |
| Google Search--13 | `01a1230c` | 0.0, blocked 1.0 | | F | *"This comes from Google's AI Overview, which cited IMDb"*; *"IMDb's page for him returned a bot-challenge page"* |
| Google Search--13 | `01a12329` | 0.0, blocked 1.0 | | F | extract returned *"Our systems have detected unusual traffic from your…"*; the answer rests on *"Bing's AI-generated search summary"*. CLI instead reached TMDB directly and cited `themoviedb.org/person/111512-jerry-trainor` |
| Google Search--14 | `01a122fb` | 0.0, blocked 1.0 | 0.333 | A/F | *"I could not filter Transfermarkt by retirement year or open every profile"* |
| Google Search--14 | `01a1230b` | 0.0, blocked 1.0 | | A/F | *"I couldn't open their profiles because my browser session timed out."* |
| Google Search--14 | `01a12322` | 0.0, blocked 1.0 | | A/F | *"I couldn't open individual profiles from the results page because clicks wouldn't navigate. One of my profile reads returned obviously invented data, so I discarded it."* |
| Huggingface--2 | `01a122f8` | 1.0 | 0.667 | — | correct |
| Huggingface--2 | `01a12300` | 0.0 | | B | *"The listing shows \"last updated\" dates, not creation dates."* CLI bypassed the page entirely: `curl -s "https://huggingface.co/api/models?pipeline_tag=translation&sort=trendingScore…"` filtered on `createdAt >= 2026-09-09`, and answered *"was created Sep 28, 2026"* |
| Huggingface--2 | `01a12312` | 0.0 | | B | *"Hugging Face lists \"last updated\" rather than creation dates, so the \"released\" timing below comes from those update dates"* |
| Huggingface--3 | `01a122ff` | 1.0 | 1.0 | — | correct |
| Huggingface--3 | `01a12311` | 0.0 (conf 0.46) | | **C** | *"**replit/replit-code-v1-3b**, a text-generation model with 745 likes"* — exactly the reference answer — followed by *"I didn't open the replit model's own page to confirm its license tag."* |
| Huggingface--3 | `01a12333` | 1.0 | | — | same answer, model page opened |

## 2. Failure modes ranked by frequency

Across the 28 MCP rows in these 12 tasks that did not score `correct = 1.0`:

| rank | mode | rows | share |
|---|---|---|---|
| 1 | **A — `act()` could not perform the interaction** (hard schema error, or a click reported successful that never navigated) | 9 | 32% |
| 2 | **B — `extract()`/`observe()` returned the wrong or lossy slice** (no structure, no hrefs, stale content, `"null"`) | 7 | 25% |
| 2 | **C — judge penalised a hedge on a factually correct answer** | 7 | 25% |
| 4 | **F — bot wall, answer built from an AI-generated search summary rather than a primary page** | 3 | 11% |
| 5 | **D — wrong page / wrong entity reached** | 1 | 4% |
| 6 | **E — run died on MCP transport** (`tools/call` timed out) | 1 | 4% |

Tool-level base rates over the 34 MCP runs analysed:

| tool | calls | hard errors | silent `"success": false` | error rate |
|---|---|---|---|---|
| `browserbase_act` | 233 | 88 | 29 | **37.8% hard, 50% including silent** |
| `browserbase_extract` | 283 | 13 | — | 4.6% |
| `browserbase_navigate` | 205 | 29 | — | 14.1% |
| `browserbase_observe` | 102 | 4 | — | 3.9% |

82 of the 88 `act` errors are the identical string
`AI_NoObjectGeneratedError: No object generated: response did not match schema.`
Session loss accounts for 38 more results (`No active session` x28,
`Cannot connect to session: session has completed or timed out` x10,
`Error: -32001 Session with given id not found.` x7).

## 3. Mechanism

Three properties of the MCP tool surface produce these modes.

**`navigate` returns no page content.** 175 of the 176 successful `browserbase_navigate`
results in these runs are an ~18–20 KB serialised Playwright/CDP object that begins
`{"success":true,"data":{"result":{"page":{"conn":{"ws":{"_events":{"error":[null,null]},"_eventsCount":3,"_binaryType":"nodebuffer","_closeCode":1006,…`
and ends in raw HTTP response headers. One occurrence of the key `"url"`; zero occurrences
of `"title"`, `"text"`, `"content"`, `"markdown"` or `"accessibility"`. So every page load
costs ~20 KB of context and delivers no information, and the agent must spend an additional
`extract` or `observe` round-trip to learn anything. This is the "huge navigate dump" — it
is worse than noisy, it is pure noise. It is also why MCP's median step count is 2.5x the
CLI's (25 vs 10) at a lower score.

**Reading goes through a natural-language bottleneck.** The only way to see the page is
`extract(instruction=...)`, which returns a short free-text summary produced by another
model. The structure is gone, so three things follow: it collapses repeated rows
(Coursera--6 got `"3 assignments"` back from five differently-worded instructions and from
the per-module request), it loses document order and nesting (GitHub--37 `01a12302` got a
flat `<h2>` list and picked the wrong trio), and it can fail outright (`"null"`,
`"Something went wrong!"`, stale December-2023 ESPN cards). `observe` returns an xpath but
almost never an href — 1 of 98 results contained the string `href` — so the agent cannot
read a link and navigate to it directly. That is the proximate cause of the BBC News--4
404s: unable to extract the article URL, it guessed one.

**Writing goes through the same bottleneck, and fails half the time.** `act(action=...)`
must re-ground a sentence into a selector on every call; it errors hard 37.8% of the time
and, when it does not, it can report
`"Action [click] performed successfully on selector: xpath=…/a[1]"` while the page never
changes. The agent has no way to distinguish the two, so it loops: ArXiv--41 `01a12314`
issued eight successive rewordings of "click XL" and still finished with *"I couldn't get
the shirt into the cart as size XL."*

The CLI variant has none of these constraints, because its `execute` tool gives it a shell.
It pipes a whole page through grep in one call
(`browse get markdown | grep -oE '\* (Quiz|…)…'` → the 8 quizzes MCP never found), follows
links by URL because markdown carries them, and sidesteps the browser entirely when an API
is better (`curl -s "https://huggingface.co/api/models?…"` → real `createdAt` dates, which
is exactly the field the HF page does not display and the whole of the Huggingface--2 gap).

**The judge's hedge rule amplifies all of this.** Because MCP agents are routinely unsure
whether their click landed or their extract was complete, they say so. The Jev `false`
criterion fires on "missing **or hedged**", so an honest caveat costs the same as a wrong
answer. Seven of the 28 failing rows carry an answer that is factually right. MCP's mean
`correct_confidence` is 0.514 against CLI's 0.569 — the whole MCP distribution sits closer
to the 0.5 threshold, so hedging converts a confidence difference into a score difference.

## 4. Rows where the MCP answer looks correct and the judge is wrong

Flagged, in descending confidence that the judge erred:

1. **Huggingface--3 `01a12311`** (conf 0.46). Answer: *"replit/replit-code-v1-3b, a
   text-generation model with 745 likes"*. The reference answer is
   `replit/replit-code-v1-3b`. The only difference from the two sibling runs that scored
   1.0 is the closing sentence *"I didn't open the replit model's own page to confirm its
   license tag."* **The judge is wrong here.**
2. **Coursera--7 `01a122f8` and `01a12303`** (conf 0.43, 0.45). Both name
   *Fundamentals of Reinforcement Learning*, University of Alberta, 4.8, ~2.9K reviews —
   the same content as MCP `01a12314`, which scored 1.0, and the same content as CLI
   `01a1230e`, which scored 1.0 while likewise disclosing *"I … did not open the course
   page itself."* **The judge is wrong on both.**
3. **BBC News--3 `01a1230d` and `01a1232b`** (conf 0.47, 0.40). Both give the tournament
   (*Open de España*) and the count (*zero* players at -10). All three CLI runs gave the
   identical pair of facts and scored 1.0. **The judge is wrong on both**; these sit just
   under the threshold rather than reflecting any difference in the work.
4. **ESPN--24 `01a12331`** (conf 0.31) and **`01a122ff`** (conf 0.24). Both report
   *"Tampa Bay Buccaneers 24, Dallas Cowboys 16 (Final) … Thursday, October 8, 2026"*,
   which is exactly what CLI `01a122ff` and `01a12316` reported and were graded 1.0 for.
   The MCP runs additionally disclose that the HTML scoreboard would not yield text and
   that the figure came from `site.api.espn.com`, and that they could not confirm the
   broadcasting network. **The judge is probably wrong**, though the task does ask for the
   game "broadcast on ESPN" and the MCP answers explicitly say this game aired on Prime
   Video — a grader could legitimately read that as the answer disqualifying itself.
   ESPN--24 `01a12310` is a genuine miss: it answered a different game.

Net effect: if the seven C-mode rows had been graded the way the equivalent CLI answers
were, MCP's mean `correct` over the full 90 graded rows would rise from 0.622 to
roughly 0.70 — close to the CLI's 0.733. **About two thirds of the headline gap is judge
sensitivity to hedging; the remaining third — ArXiv--41, BBC News--4, Coursera--6,
Google Search--13/14 — is real, and is caused by `act` and `extract` failing where the
CLI's shell does not.**
