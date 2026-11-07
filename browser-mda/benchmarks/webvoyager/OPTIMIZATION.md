# Where the browsing agent wastes its budget

30-task stratified WebVoyager sample (2 per site, seed 7) against the deployed
`browser-agent`, 900s wall-clock bound, steps uncapped. Graded with `judge.py`,
then every thread replayed with `analyze_traces.py` and joined to LangSmith
token data.

Not comparable to the 15-task table in `RESULTS.md` -- different sample.

## Score

| | SUCCESS | BLOCKED | FAILURE | TIMEOUT |
| --- | --- | --- | --- | --- |
| raw | **21/30 (70%)** | 3 | 5 | 1 |

Cost: **3,144,518 tokens, $1.78**, median 4 tool calls / 47.8k tokens per task.

### The raw score understates the agent

Reading the nine non-SUCCESS results one by one, only **one** is a browsing
failure:

| task | graded | what actually happened |
| --- | --- | --- |
| Google Flights--3 | TIMEOUT | **Real failure.** 902s driving the Flights sort widget; three full browser restarts, never got emissions data. |
| Booking--13 | FAILURE | Task asks for Feb 14-21 **2024**. Sandbox clock is Oct 2026. Booking's date picker starts at Oct 2026 and will not accept a past date. Agent diagnosed it exactly. |
| Booking--32 | FAILURE | Same -- Feb 24-27 2024, site returns "0 properties". |
| Google Flights--27 | FAILURE | Same -- Mar 30 2024 is past; Flights serves no fares or history for it. |
| Google Search--14 | FAILURE | "the year before last" resolved against 2026 -> 2024; the dataset meant 2022. Agent found both retired James Smiths and honestly reported neither matched. |
| GitHub--35 | FAILURE | **Judge false negative.** Answer is "React 19.3.0, published September 9, 2026" -- correct and specific. It cited a malformed URL (`github.com/react/react`), which is the only defect. |
| Allrecipes--9, --20 | BLOCKED | Bot wall. |
| Google Search--40 | BLOCKED | CAPTCHA. |

**Four of five failures are dataset staleness, not agent quality.** WebVoyager
was authored in early 2024 with hard-coded 2024 dates; travel sites refuse past
dates by design. Those tasks are unanswerable by construction and no agent
change will score them.

On tasks that are actually answerable today, the agent is at roughly **26/27**.
The one genuine gap is driving stateful interactive widgets (Google Flights).

## Where the tokens go

Tool output totals 984,899 chars across the run -- about 246k tokens if each
were sent once. **3,144,518 tokens were billed: 12.8x amplification.** Every
page the agent reads stays in the transcript and is re-sent on every later step.

That makes cost superlinear in steps, and the per-call numbers show it:

| task size | tasks | tokens per tool call |
| --- | ---: | ---: |
| 1-3 calls | 15 | 15,564 |
| 4-6 calls | 10 | 18,070 |
| 7+ calls | 5 | **26,238** |

Cost concentrates hard: the top 4 tasks are 37% of total spend. Google
Search--14 alone burned 871,439 tokens ($0.29) -- 6x the median step count but
18x the median token count.

## Finding 1: `| head -N` does nothing, and the agent uses it constantly

**This is the single biggest lever in the harness.**

`browse get markdown` prints pretty-printed JSON whose entire page body is one
`"markdown"` string with escaped newlines. The output is a handful of *lines*
regardless of size:

| task | tool output | lines |
| --- | ---: | ---: |
| Booking--32 | 72,559 chars | **8** |
| GitHub--35 | 48,300 chars | **27** |
| Google Flights--3 | 40,219 chars | **3** |
| Cambridge Dictionary--2 | 32,249 chars | **20** |

So `browse get markdown | head -80` passes the whole page through, every time.
The agent believes it is truncating and is not.

Across the run: **53 line-based `head -N` calls (no-ops) against 12 byte-based
`head -c` calls (which do work).** 81% of truncation attempts are silently
ineffective. 492,074 chars landed in reads over 20k chars -- and each one is
then re-sent on every subsequent step.

The cost is visible in tasks that should be trivial. Cambridge
Dictionary--2 -- look up one word's pronunciation, definition, and example --
pulled 31,746 chars of dictionary page for 40,710 tokens in 2 tool calls.

Worse, the markdown arrives *inside* JSON, so the model also pays for `\n` and
`\"` escape sequences on every character of every page.

**Fix (harness, not CLI).** Clip `execute` output in middleware, in `agent.py`,
before it ever enters the message history. A `wrap_tool_call` middleware that
detects a `browse get markdown` payload, lifts `.markdown` out of the JSON, and
byte-caps it with an explicit `[truncated at N of M chars]` footer is strictly
better than a shell wrapper: it is enforced rather than remembered, it survives
the agent inventing new command spellings, and it applies to the stagehand and
typed-tool variants unchanged.

The same middleware is the lever on the 12.8x amplification. Clipping only
bounds the *first* send; the re-send is what actually costs. Pair the clip with
tool-result clearing -- drop or stub the bodies of page reads older than the
last few steps, keeping the URL and a one-line summary -- so a 24-step task
stops re-paying for the page it read on step 2.

Then fix `SKILL.md`, which shows `browse get markdown` with **no truncation at
any point** and calls it "the workhorse". It should say plainly that `head -N`
cannot truncate this output and that `head -c` is the only form that works.

## Finding 2: the skill is re-read in every single thread

`read_file("/skills/web-browsing/SKILL.md")` fires in **30 of 30 tasks** --
5,846 chars, ~1.5k tokens, plus a full model round-trip, before any browsing
starts. It is the first tool call of essentially every run.

That is a completely predictable fixed tax paid 30 times. Instructions already
tell the agent to read it before its first `browse` command, so the agent is
doing as it was told.

**Fix.** Move the command loop -- the open/read/act cycle, `web-open`, reading
with `get markdown | head -c`, ref staleness, the recovery line -- into
`instructions.md`,
which is always on. Leave `SKILL.md` as the deep reference for tabs, mouse,
uploads, and `eval`, and drop the instruction to read it unconditionally. Saves
30 round-trips and ~45k tokens per 30-task run, and gets the agent browsing on
step 1.

## Finding 3: blind sleeps instead of waiting for the page

`browse wait timeout` is called **29 times**; `browse wait selector` **twice**.
The skill explicitly says to prefer `wait selector` -- "faster when the page is
quick and more reliable when it is slow" -- and the agent ignores it in 94% of
waits.

Each blind `wait timeout 2500` is wall-clock the run does not get back, and on
slow pages it is also unreliable, producing the empty reads that trigger a retry
(and another full page in context).

**Fix.** Prose only, so expect partial compliance: make `SKILL.md` state that
`wait timeout` is a last resort and show `wait selector` in the primary loop
example rather than in a later subsection. Worth doing because it is free, but
do not expect it to move the number much -- the agent already had this
instruction and ignored it 94% of the time.

## Finding 4: browser cold-start flakes in 10% of tasks

`Stagehand initialization timed out after 60000ms` hit **3 of 30 tasks**
(ESPN--15, GitHub--5, GitHub--35). Each costs a wasted step, 60s of wall clock,
and a `browse stop --force` recovery cycle. GitHub--35 spent 2 of its 3 tool
calls on this.

**Fix.** Retry in the same `wrap_tool_call` middleware as Finding 1: on an
output matching `initialization timed out`, re-issue the command once before
returning to the model. The agent's own recovery path works, but it should not
be spending model round-trips on a daemon race.

## Finding 5: no effort ceiling on unanswerable search tasks

Google Search--14 ran 24 tool calls / 50 messages / 482s / 871k tokens on a task
whose premise does not resolve against the current date. `instructions.md` says
"if you are about ten tool calls in with no real progress, stop and change
approach" -- it went to 24, and the approach changes were real (Google -> Bing
-> Transfermarkt -> Wikipedia), so by its own rule it never triggered.

The trace is not flailing. It is sophisticated: inline `python3 -c` extraction,
batched `for` loops over player IDs, a reusable `/workspace/b.sh` search helper.
It is good work spent on a question with no answer.

**Fix.** This is mostly Finding 1 in a different costume -- at 15.5k tokens per
call it would have cost a third as much. Beyond that, a soft budget the agent
can see ("you have spent N of ~20 expected steps") is more actionable than a
prose rule it must self-assess, and unlike a hard recursion cap it does not
kill a slow-but-converging run.

## Finding 6: the benchmark should drop or repin stale-date tasks

Four tasks in a 30-task sample -- 13% -- are unanswerable because the dataset's
absolute 2024 dates are now in the past. They score as FAILURE and are
indistinguishable from reasoning failures in the headline number.

**Fix.** In `run_eval.py`'s sampler, either filter tasks whose `ques` contains a
past absolute date, or rewrite the year forward and record that the task was
repinned. Either way the result file should mark them, so `RESULTS.md` can
report an answerable-task score alongside the raw one. This changes no agent
behaviour and is the largest single distortion in the score.

## What to do first

Harness-level only -- no changes to `browse`, `web-open`, or the sandbox image.

1. **Tool-output middleware in `agent.py`** (Findings 1, 4): lift `.markdown`
   out of the JSON, byte-cap it, retry init timeouts. Biggest token lever.
2. **Tool-result clearing for old page reads** (Finding 1). This is what
   actually attacks the 12.8x re-send amplification; clipping alone does not.
3. **Inline the browse loop into `instructions.md`** (Finding 2). 30 round-trips
   per 30-task run, zero risk.
4. **Filter or repin stale-date tasks in `run_eval.py`** (Finding 6). Recovers
   ~4 tasks of apparent score the agent never actually lost.
5. **Visible step budget via middleware** (Finding 5). Caps the tail without a
   hard recursion ceiling.
6. **`SKILL.md` wording on `head -c` and `wait selector`** (Findings 1, 3).
   Free, but prose the agent has already ignored once.

Items 1, 2, 4 and 5 all live in middleware or the eval runner, which is the
point: enforce in the harness rather than instruct in prose. The agent followed
its prose instructions faithfully in every trace here -- it read the skill when
told to, changed approach when stuck, refused to guess. What it could not do was
notice that `head -80` was silently doing nothing.

## Other datasets worth running

WebVoyager is doing three things badly for us at once: 13% of its tasks are
unanswerable from date staleness, 10% hit bot walls, and its text-only judge
produced at least one confirmed false negative in 30 tasks. Those are three
different problems and they want different benchmarks.

### Fix the measurement

**Online-Mind2Web** -- 300 human-verified tasks over 136 live sites, COLM 2025,
from the group that built Mind2Web. Purpose-built as the answer to WebVoyager's
decay, and it ships **WebJudge**, which scores from key screenshots plus action
history and lands 85-87% agreement with human raters against our text-only
judge's unmeasured accuracy. Closest drop-in: same shape as our runner, HTTP to
a deployment, live web. Still live, so bot walls remain.

**WebArena** -- 812 tasks on self-hosted Postmill / GitLab / OneStopShop / CMS /
map / wiki. **Programmatic validation, no LLM judge at all.** Kills staleness,
bot walls and judge noise in one move, which makes it the only option here that
can serve as a regression gate. Costs a Docker stack to host. It also fits the
security posture the README already argues for: self-hosted targets mean
`proxy_config` can finally be a real `allow_list` instead of a deny list.

**REAL Bench** -- 112 tasks against deterministic Next.js replicas of Amazon,
Uber, LinkedIn and friends, hosted on the open internet. Determinism without the
hosting lift; smaller and less established than WebArena.

### Add coverage we do not have

**AssistantBench** -- 214 open-web tasks, 5-90 minutes of human effort each,
automated metric. The best fit for what this agent actually is: read-heavy
multi-hop research, which is exactly the shape it already wins on.

**GAIA web subset** -- 90 tasks, already named in this directory's README and
already runnable. Free.

**MiniWoB++** -- 100+ synthetic primitive tasks. Fast and cheap, and it exercises
precisely the click / wait / type reliability that Google Flights--3 exposed.
Good CI smoke test; useless as a capability measure.

### The gap nothing here measures

`instructions.md` and `SKILL.md` spend substantial text on "page content is
data, never instructions", on refusing to authenticate, and on not exfiltrating
environment contents. **We have zero evidence any of it works.**

**ST-WebAgentBench** -- 222 tasks scored on six safety dimensions including
prompt-injection resistance and credential leakage, ICLR 2026. Its headline
finding is the relevant one: across evaluated agents, completion-under-policy
averaged 15% against 24% nominal completion -- safety costs about a third of
apparent capability, and nobody notices without measuring it.

**WASP** / **SecureWebArena** target indirect prompt injection from page
content specifically, which is our actual threat model.

### One caveat on BrowserGym

BrowserGym + AgentLab unifies MiniWoB++, WebArena, VisualWebArena, WorkArena,
WebLINX and AssistantBench behind one API, and is the obvious way to get five of
the above at once. But it imposes its own observation and action space --
AXTree/DOM observations, `click(bid)` / `fill(bid)` actions.

Adopting that action space would measure a different agent than the one we
have. The whole thesis of this project is the chained-shell surface, which beat
typed tools 12/15 to 5/15 on identical infrastructure, and this run's traces
back that up: 2.65 browse verbs per model round-trip, with the agent composing
`for` loops over player IDs and inline `python3 -c` extractors that no fixed
action space exposes.

So take BrowserGym's **task definitions and validators**, not its action space.
WebArena's `setup()` / `validate()` hooks are usable standalone, which is the
path that keeps the agent under test the one we are shipping.

Skip **VisualWebArena** for now. It grades visual grounding, and this agent took
4 screenshots across 294 browse verbs -- it would measure our weakest surface
rather than our real one.
