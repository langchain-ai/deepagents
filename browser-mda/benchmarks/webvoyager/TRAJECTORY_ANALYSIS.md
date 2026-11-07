# WebVoyager-15 trajectory analysis

All 15 traces from experiment `28cb9faf-6d93-43f4-8827-096026556c4e`, agent
`browser-agent` (sandbox Chrome + `browse` CLI). 132 tool calls, 133 model
calls, 3,144,293 input tokens, 24,735 output tokens.

A "step" in the experiment output is a message, not a browser action. The agent
batches several `browse` commands per `execute`, so the real unit is the tool
call: **median 5 per task**.

| task | steps | tool calls | secs | input tokens | result |
| --- | --- | --- | --- | --- | --- |
| Booking--4 | 57 | 28 | 322 | 669,910 | BLOCKED |
| ArXiv--41 | 46 | 23 | 217 | 888,187 | SUCCESS |
| Cambridge--34 | 45 | 22 | 309 | 274,577 | SUCCESS |
| Google Search--13 | 33 | 16 | 145 | 527,851 | SUCCESS |
| ESPN--23 | 16 | 8 | 135 | 120,465 | SUCCESS |
| Amazon--9 | 12 | 6 | 103 | 96,422 | SUCCESS |
| BBC News--3 | 12 | 6 | 73 | 101,547 | FAILURE* |
| Coursera--6 | 12 | 6 | 125 | 109,851 | SUCCESS |
| Allrecipes--20 | 11 | 5 | 62 | 45,155 | BLOCKED |
| GitHub--37 | 8 | 4 | 73 | 37,688 | SUCCESS |
| Wolfram--5 | 8 | 4 | 72 | 36,429 | SUCCESS |
| Huggingface--2 | 8 | 4 | 62 | 48,548 | SUCCESS |
| Google Map--32 | 8 | 4 | 73 | 106,204 | SUCCESS |
| Apple--25 | 6 | 3 | 63 | 37,900 | SUCCESS |
| Google Flights--3 | 6 | 3 | 73 | 43,559 | SUCCESS |

\* see "Grading" below -- the agent was right and the gold label is stale.

## 1. `| head` and `| grep` on `browse` output do nothing

`browse` prints pretty-printed JSON in which the whole page is a **single**
string. Measured on the trace payloads:

| task | `get markdown` bytes | real newlines | escaped `\n` in the string |
| --- | --- | --- | --- |
| Apple--25 | 25,919 | 19 | 663 |
| GitHub--37 | 82,518 | 19 | 966 |
| Google Flights--3 | 34,544 | 23 | 2,380 |
| Google Map--32 | 55,765 | 22 | 170 |
| Coursera--6 | 23,338 | 19 | 284 |

So `| head -150` returns the entire page and `| grep` matches the one line and
prints all of it. **The agent wrote 36 such commands**, returning 1,998,528
bytes, believing each was a bounded read.

- **108,659 of 461,029 new-context tokens (23.6%)** entered through pipes that
  did nothing. New context is what is billed at cache-*write* rate, so this is
  ~24% of the dominant cost component.
- Counting re-sends: **745,601 of 3,144,293 input tokens (23.7%)**.
- Per task: ArXiv 37%, Google Search 39%, Google Map 30%, Flights 31%, Apple 29%.
- Worst single calls: Google Map `head -150` **+35,409 tokens**; Google Search's
  Wikipedia `grep` **+29,498**; Apple's only substantive call **+13,599** for a
  three-word answer.

The agent found the workaround itself in 5 tasks (piping into `python3 -c
"json.load..."`) but only after paying for the big read, and never generalised.
`head -c N` works and was used correctly 4 times.

## 2. Where steps were wasted

**Ref discipline is working:** zero "Unknown ref" errors in 132 calls; 2 xpath
misses.

| outcome | n | tasks |
| --- | --- | --- |
| Cloudflare interstitial | 10 | Cambridge (9), Booking (1) |
| `read_file` on a screenshot not yet written | 3 | ArXiv (2), Allrecipes (1) |
| daemon "running but not responding" | 2 | Cambridge |
| `browse stop --force` rejected | 2 | Cambridge |
| HTTP 403 | 2 | Google Search (IMDb twice) |
| site access-denied wall | 3 | Allrecipes |
| CAPTCHA | 2 | Google Search |

Patterns worth fixing:

- **Screenshot/`read_file` race.** `browse screenshot --path X` and
  `read_file(X)` issued as *parallel* calls in one turn, so the read ran before
  the file existed. 3 calls, all wasted.
- **Retrying a dead source.** Google Search hit IMDb → 403, then tried a
  different guessed IMDb id → 403 again.
- **Repeating a no-op shape.** Booking calls 6 and 7 were both
  `browse snapshot | head/grep` → both offloaded as too large; nothing learned
  from the first. Cambridge called `get markdown` **16 times** against a
  Cloudflare interstitial.
- **Wrong recovery command in SKILL.md.** It documents `browse stop --force`;
  the installed CLI rejects it and wants `browse stop --session default
  --force`. Two calls lost, and even the correct form returned
  `"stopped": false`.

## 3. The expensive tail

Mean 19.2 steps vs median 12 is four tasks: Booking (57), ArXiv (46),
Cambridge (45), Google Search (33) -- 89 of 132 tool calls.

**Booking--4, 28 calls, 322 s, scored 0.** The agent had decisive evidence at
**tool call 3** that the site strips query parameters: it opened
`…?ss=Kashi+Vishwanath+Temple&checkin=2026-12-25&checkout=2026-12-26&…` and
`browse get url` returned `…?nflt=mealplan%3D1` -- destination and both dates
gone. It then spent **15 of the remaining 25 calls driving the date picker**:
9 screenshots, 6 PNG `read_file`s, `eval` to discover the visible month range,
mouse-clicks to page the calendar, then Search -- landing back on the landmark
page with no dates -- and repeated the cycle. Net result: zero date-specific
data, correctly reported as blocked.

**ArXiv--41, 23 calls, 888k tokens** (most expensive task). Succeeded. 14
snapshots; call 3 was `snapshot --full | head -80` → **+17,672 tokens**. Calls
15-20 were six consecutive attempts to find a "short sleeve" variant; call 17's
`innerText` already contained the full answer, so ~4 were redundant.

**Cambridge--34, 309 s.** 15 of 21 calls hit Cloudflare or a daemon error. The
agent accumulated **82 s of self-imposed `wait`/`sleep`** inside a 266 s budget,
cycling wait → stop → reopen about five times.

### Latency is mostly cold start, not the agent

```
first execute of each task : 769 s over 15 calls (mean 51.3 s)
every other execute        : 762 s over 93 calls (mean  8.2 s)
tool time 1,535 s   |   model time 299 s
```

Every first `web-open` takes 40.8-64.3 s; every later one averages 8.2 s
including page load. **50% of all browser time is Chrome/daemon cold start.**
Model time is 299 s of 1,834 s -- the agent is not thinking slowly, it is
waiting for a browser to boot.

## 4. Where the tokens go

| bucket | tokens | share |
| --- | --- | --- |
| system prompt + task, re-sent across 133 calls | ~886,000 | ~28% |
| page markdown / `get text` | 518,470 | 16.5% |
| accessibility snapshots | 445,789 | 14.2% |
| **SKILL.md, read fresh every task** | **184,086** | **5.9%** |
| screenshots read back | 80,240 | 2.6% |
| nav/eval/click output | 25,059 | 0.8% |

- **Images are not the problem.** A 512 KB base64 PNG is ~1,180 tokens.
- **The SKILL.md read is pure overhead** -- a static 159-line file, 1,460 tokens,
  re-sent every turn, plus 15 extra round-trips and 15 tool calls.
- **Cost is cache writes, not reads**: 2,741,518 cache-read vs 402,775
  fresh/write vs 24,735 output. Every byte that ever enters context is paid at
  write rate then re-read forever -- which is why finding 1 is the top lever.

## 5. Agent error vs environment

- **Allrecipes--20 (0)** -- genuine site block. All four URLs returned
  access-denied. The agent refused to answer from memory, in 5 calls. Correct
  behaviour, scored as failure.
- **Booking--4 (0)** -- environment *and* agent. The wall is real; the 25 calls
  after it was provable are not.
- **Cambridge--34**, **Google Search--13** -- environment (Cloudflare, CAPTCHA,
  403); both routed around it and succeeded.

**No task failed for a reasoning or navigation mistake.**

### Grading: BBC News--3 is a stale label, not an agent error

The agent read the live leaderboard -- Open de España, 8-11 Oct 2026, round 3
pending, best score −9 -- and correctly reported no one at −10. The WebVoyager
gold answer was authored for a different tournament. **True quality is 13/15,
not 12/15.** Two minor criticisms stand: it guessed a leaderboard URL and 404'd
first, and it treated an *empty* `grep` as proof of absence, which is
indistinguishable from a page that failed to load.

## 6. Is existing guidance followed?

| instruction | verdict |
| --- | --- |
| read the skill before browsing | **15/15** -- and the compliance is the problem (5.9% of tokens) |
| prefer reading over clicking | **13/15**; the 2 exceptions are the expensive tail |
| prefer a direct URL over click sequences | **14/15** -- clearly working |
| stop after ~10 unproductive calls | **violated 2/15** (Booking, Cambridge) |
| stop after 5 screenshots on one subtask | **violated 1/15** (Booking took 9) |
| keep notes in `/workspace` | **0/15** -- never used, including on 28- and 23-call tasks |
| cite URLs visited | **15/15** |
| do not narrate process in the final answer | **~6/15 violated** -- the model reads "I did not check X" as a caveat rather than narration |

## 7. Recommended changes, ranked

1. **Ship a `page` helper with a hard byte cap** (`page`, `page --grep`,
   `page --sel`, `tree`, `tree --grep`) that JSON-decodes and truncates. Fixes
   all 36 broken pipes: **~109k new-context tokens, ~746k input tokens (≈24%)**,
   plus ~6 calls that existed only to re-slice an oversized blob. If binaries
   cannot be added, document in SKILL.md that `head -N`/`grep` do not work on
   `browse` output and that `head -c 4000` is the bounded read.
2. **Move SKILL.md into the cached system prompt** and drop "read it first":
   184k tokens, 15 round-trips, 15 tool calls.
3. **Pre-warm the browser before the first model call**: cuts median latency
   73 s → ~25-30 s, mean 127 s → ~90 s, with no behavioural change.
4. **Fix the recovery command** in SKILL.md to
   `browse stop --session default --force`.
5. **Add a bot-wall stopping rule**: if `browse get url` differs from the URL
   requested, or a CAPTCHA/access-denied appears, confirm once and report.
   Applied at Booking call 3 this ends the task at ~5 calls / 90 s instead of
   28 / 322 s.
6. **Never drive a date picker by screenshot** -- set dates in the URL or via
   `eval`, verify with `get url`.
7. **Screenshot and read it back in separate turns** -- parallel calls race.
8. **Tighten the negative-evidence rule**: an empty command result is not
   evidence of absence; confirm the page loaded and print surrounding text.
9. **Re-tighten the final-answer rule** -- the only process worth mentioning is a
   gap the reader needs, not what the agent chose not to open.
10. **Lower the large-result offload threshold** from ~80 KB to ~20 KB: three
    payloads landed just under it (79,128 B / 55,765 B / 42,505 B) and went into
    context whole, ~75k tokens across three tasks.

## Caveats

- The 51 s cold start is inferred from timing; the traces do not separate
  daemon launch from navigation. The gap is uniform across 15 different sites,
  which makes a page-dependent explanation unlikely.
- `total_cost` was empty in the trace payloads, so cost attribution is derived
  from the token-type split. It lands in the right range for the $0.083 median.
- "Stale gold label" for BBC is inferred from the live page; the dataset's gold
  answer was not visible.
