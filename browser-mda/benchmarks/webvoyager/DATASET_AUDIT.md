# WebVoyager-30 dataset audit

Are the rows that score `correct=0` actually agent failures, or are they flaws in
the benchmark?

**Scope.** Dataset `e06b6845-b213-4c5f-9369-3a6cfdee9331` ("webvoyager-30"), 30
examples. Experiments `350a640d` (CLI, 92 root runs) and `00f12e88` (MCP, 93 root
runs), ~3 reps per example per variant, graded by the typed Jev judge in
`experiment_native.py` on feedback keys `correct` / `blocked` /
`correct_confidence`.

**19 of 30 examples had at least one run score `correct=0`.** Every one of those
19 is audited below; reference answers and agent answers were checked against the
live web on 2026-10-10 wherever the verdict turned on a fact.

This extends the prior passes in `OPTIMIZATION.md` (~4/30 stale dates, ~3/30 bot
walls) and `HARNESS_FINDINGS.md`. Both undercounted. The real figure is **11/30
rows with a defect in the question or the reference**, plus 5 more that are
unscoreable from the CLI sandbox's egress and 3 where the row is fine and the
*judge* is wrong.

---

## 1. Per-example findings

`fail` is runs scoring `correct=0` / total graded runs. `conf` is the range of
`correct_confidence`; the threshold is 0.5.

| task_id | question (truncated) | reference answer | agent answer (truncated) | classification | evidence |
| --- | --- | --- | --- | --- | --- |
| **Booking--5** | "Search a hotel with free WiFi and air conditioning in Bali **from Jan 1 to Jan 4, 2024**." | `Palasari Villa, free WiFi and air conditioning` | "I couldn't run this search… Jan 1–4, 2024 is in the past. The date picker starts at October 1, 2026… My direct search link with those dates redirected to the homepage." | **STALE_REFERENCE** (unanswerable) | fail 6/6 (CLI 0/3, MCP 0/3), conf 0.06–0.38. The stale part is literally in the question: `Jan 1 to Jan 4, 2024`. Booking.com greys out past dates by design; one run verified the control by re-running Jan 1–4 **2027** and getting 782 Bali properties. **The only row in the set no agent can pass.** |
| **Google Search--14** | "Find the retired players **the year before last** named James Smith and tell me which club he has been a member of from 2020–2021." | `Yeovil Town` | "I read 'the year before last' as retiring in 2024, since the site's current date is October 2026… Transfermarkt lists only two retired James Smiths, retired 2014 and 2013. Neither fits." | **STALE_REFERENCE** (relative-date drift) | fail 5/6, conf 0.28–0.45. Verified: the intended subject is **Jimmy Smith (James Dean Smith, b. 1987)**, Yeovil Town 2020–21, who retired professionally in 2021 and stopped non-league play in 2022. The phrase resolved to 2022 when the dataset was authored; in Oct 2026 it resolves to 2024 and matches nobody. Compounded by a Google CAPTCHA on 5/5 failing runs. |
| **Google Search--13** | "What are Jerry Trainor's **upcoming** projects?" | `Strange Planet, 2023` | "The only upcoming project I could find is an untitled **iCarly movie**… IMDb lists it as 'Upcoming' and in pre-production." | **STALE_REFERENCE** | fail 3/6 (MCP 0/3, CLI 3/3), conf 0.28–0.34. *Strange Planet* premiered on Apple TV+ in **August 2023** — it has not been "upcoming" for three years. The agents' iCarly-movie answer is the live-correct one and was graded 0 for matching the present rather than 2023. |
| **BBC News--3** | "Check the leaderboard for Golf's DP World Tour… what was **the most recent tournament**, and how many **teams** have a Total of -10 strokes." | `Mauritius Open; 5` | "The most recent tournament is the **Open de España** (8–11 Oct 2026)… **No players have a Total of -10.** Leader Grant Forrest at -9. The leaderboard lists individual players rather than teams." | **STALE_REFERENCE** (+ ill-posed) | fail 2/6, conf 0.40–0.47. `Mauritius Open` is a December event last relevant in 2023; the DP World Tour leaderboard rotates weekly, so this reference is stale within 7 days of authoring. Secondary defect: the DP World Tour is **individual stroke play** — "how many teams" has no referent, and both agents said so explicitly. |
| **GitHub--37** | "List the 3 features mentioned in GitHub's Copilot product page." | `The AI coding assistant elevating developer workflows. Get AI-based suggestions in real time. Docs that feel tailored for you.` | "The page has a 'Code, command, and collaborate' section listing three features: Manage agent-driven work from one place / Make your editor your most powerful accelerator / Ship faster with AI that work alongside you." | **STALE_REFERENCE** | fail 1/6, conf 0.39. **Verified live 2026-10-10:** none of the three reference phrases appear on `github.com/features/copilot`. Current hero is "Command your craft". The agent quoted the page verbatim and was graded against 2024 marketing copy. Secondary: "the 3 features" is ambiguous — the live page has ~8 candidate section headings, and the agent flagged exactly that. |
| **Allrecipes--20** | "Find a recipe for a cauliflower pizza crust… under 30 minutes, ≥4 stars. Include calories per serving." | `'Cauliflower Pizza Crust', 4.2 stars, Prep Time: 15 mins, 59 Calories per serving` | "I couldn't find a recipe, because Allrecipes blocks this sandbox. Homepage, `/search?q=cauliflower+pizza+crust` and a direct recipe URL all returned an 'access issue' notice from support@people.inc." | **BOT_WALL** | fail 3/6 — **CLI 0/3, MCP 3/3**. `blocked=1.0` on all 3 failing runs, conf 0.07–0.26. The question and reference are sound; the CLI sandbox's datacenter egress is refused by People Inc. Browserbase's proxy passes 3/3. Not a dataset defect. |
| **Allrecipes--21** | "Locate a high-rated gluten-free brownie recipe with ≥50 reviews. List main ingredients and total time." | `'Gluten-Free Fudge Brownies', 4.1 stars, 69 reviews, <Ingredients>, Prep Time: 15 mins, Total Time: 1 hr` | "I couldn't get a recipe from Allrecipes… every page returned an access-issue notice. I won't guess from memory." | **BOT_WALL** | fail 3/6 — **CLI 0/3, MCP 3/3**, `blocked=1.0` on all 3, conf 0.07–0.11. Same wall. Note the reference also carries an unexpanded `<Ingredients>` placeholder, so even an unblocked run has nothing to match on the ingredient list. |
| **Cambridge Dictionary--34** | "Use the Cambridge Dictionary to understand the rules for comparative and superlative adjectives… including example sentences." | `This car is more expensive than my last one; Joe used to be the slowest runner in the class.` | "Every page stopped at a Cloudflare 'Just a moment… Performing security verification' check. I never saw any grammar content, so I'm not giving the rules from memory." | **BOT_WALL** | fail 1/6 (CLI 2/3, MCP 3/3), `blocked=1.0`, conf **0.03**. Transient Cloudflare interstitial, not a dataset problem — the other 5 reps passed. |
| **Booking--4** | "Search for the cheapest Hotel near Kashi Vishwanath Temple that offer breakfast **from Dec 25th - Dec 26th**." | `Moonlight Residency, Breakfast included, US$14` | CLI: "Every search said '0 properties are available' and reset the dates to today. It also showed a 'Verifying…' check." MCP: "Nakshatra Paying Guest House, about $10" / "Bhadra Kali Guest House, about $19". | **BOT_WALL** (+ minor ambiguity) | fail 5/10, `blocked=1.0` on 4/5, conf 0.02–0.46 — **CLI 0/3, MCP 1/3**. CLI sees Booking's anti-bot "Verifying…" path and a 0-result page; MCP reaches results but the sort/filter controls fail. Minor secondary defect: the question states **no year**, so "Dec 25–26" is agent-resolved (all picked 2026). The `US$14` reference price is naturally stale but the judge's rubric already forgives changed values. |
| **Google Map--33** | "Check Denver International Airport's info: 1) which **level** has the least proportion in reviews; 2) its Accessibility and Amenities." | `star 2 has the least proportion; Accessibility: Assistive hearing loop; Wheelchair accessible entrance/parking lot/restroom/seating; Amenities: Baggage storage; Wi-Fi; Free Wi-Fi` | CLI: "Maps served a limited, signed-out view with no Reviews tab and no star histogram. I'm not guessing." (then gave all five Accessibility items verbatim). MCP: "2 stars — 2,311 of 51,141, about 4.5%" but could not open the About tab. | **BOT_WALL** | fail 4/6 — **CLI 0/3, MCP 2/3**, `blocked=1.0` on all 4, conf 0.19–0.25. Google serves the sandbox a signed-out "limited view" that omits the rating histogram entirely. Note the two variants each got a *different half* right: CLI nailed Accessibility, MCP nailed the 2-star answer. The question is sound; the access is not. Secondary: "level" for "star rating band" is unidiomatic and one agent asked whether terminal floors were meant. |
| **ESPN--24** | "Find the final score from **the most recent NFL game broadcast on ESPN**, including team names and date." | `NFL <score>, latest, eg, January 29, 2024, Chiefs - Ravens, 17 - 10` | "Tampa Bay 24, Dallas 16, Thu Oct 8 2026 — ESPN lists the broadcaster as **Prime Video, not ESPN**. If you meant a game aired on the ESPN network itself, this isn't one; that would be Falcons 45, Saints 24 on Mon Oct 5." | **AMBIGUOUS_QUESTION** | fail 4/6 (CLI 2/3, MCP 0/3), conf 0.24–0.49. "Broadcast on ESPN" has two defensible readings — *aired on the ESPN network* vs *listed on espn.com's scoreboard* — and three of four failing runs explicitly answered both and were still graded 0. Reference is also a template (`<score>`) with a stale Jan-2024 example baked into it. |
| **Coursera--6** | "Identify **a course** on Coursera related to 'AI for Healthcare' and note the duration along with the number of **quizzes** in Assessments." | `Fundamentals of Machine Learning for Healthcare; 14 hours (approximately); 19 quizzes` | "AI for Medical Diagnosis (DeepLearning.AI)… Its Assessments field says '**3 assignments**', so I can't give a quiz count." | **AMBIGUOUS_QUESTION** (+ site changed) | fail 3/6 — **CLI 3/3, MCP 0/3**, conf 0.24–0.32. "A course related to AI for Healthcare" admits dozens; the reference pins one. **Verified live:** the referenced course now reads "**19 assignments**" and "1 week at 10 hours a week" / "Estimated Time to Complete: 11 hours" — Coursera has removed the word *quizzes* from its UI and the "14 hours" figure is gone. The quiz count the question asks for no longer exists as a labelled field anywhere on Coursera. |
| **Coursera--7** | "Find **a course** that teaches Reinforcement Learning for Intermediate with rating ≥4.5. Provide name, institution, number of reviews." | `Reinforcement Learning, Specialization; University of Alberta; 3.3K reviews` | "**Fundamentals of Reinforcement Learning**, University of Alberta, 4.8 stars, about **2.9K reviews**, labeled Intermediate." | **AMBIGUOUS_QUESTION** (+ stale count) | fail 2/6 — **CLI 3/3, MCP 0/3**, conf 0.43–0.45. The question says "a course"; the reference is a **Specialization**. **Verified live:** the Specialization now shows **3,601 reviews (3.6K)**, not `3.3K`; the agent's course shows **2,910 reviews**, matching its "2.9K" exactly. The agent gave a verified-correct answer to the question as literally worded and lost on the reference's choice of artifact. Borderline JUDGE_ERROR. |
| **Huggingface--2** | "Discover three new and popular open-source NLP models for language translation **released in the past month**." | `<model 1>; <model 2>; <model 3>; (last month, recently created)` | "IndexTeam/Index-Translate-2B (created 28 Sep 2026, 49 likes), IndexTeam/Index-Translate-9B, netease-youdao/Confucius4-T3PO… Hugging Face lists 'last updated' rather than creation dates." | **AMBIGUOUS_QUESTION** (ungradeable reference) | fail 3/6 (CLI 2/3, MCP 1/3), conf 0.31–0.49. The reference is an **unexpanded placeholder** — the judge is handed `<model 1>; <model 2>; <model 3>` and has literally nothing to compare against, so the grade is pure judge prior. Compounded: HF's model index exposes *last-updated*, not *released*, so "released in the past month" is not directly filterable and every run had to approximate it differently. |
| **BBC News--4** | "Find the **latest** article regarding the economic implications of climate change in Europe and summarize the central points." | `<summary> (economic implications of climate change in Europe)` | CLI: "'Spain battles climate change on several fronts', 22 Sept 2026 — wildfires ~80,000 ha, ~300,000 ha burned this year, roughly half the EU total…" MCP: "I couldn't open the full text of any BBC article; this comes only from the two-line teasers." | **AMBIGUOUS_QUESTION** (ungradeable reference) — *with a real agent-fail component* | fail 5/6 (CLI 1/3, MCP 0/3), conf 0.08–0.45. Reference is the placeholder `<summary>`; there is no gradeable target, and "the latest article" on a topic with no BBC tag is unverifiable by construction (both CLI runs noted BBC site search is not exhaustive). **However**: the 3 MCP failures (conf 0.08–0.19, `blocked=1.0`) are genuine — clicks never navigated and guessed URLs 404'd, so they summarised from search teasers. Split verdict: dataset-defective row, but 3/5 failures are real. |
| **ArXiv--41** | "Find the button to share arxiv non-profit store and **follow the QR code** to share the shop. Then add arXiv Forever short sleeve (XL) to your cart." | `QR code image, Action: add to chart` | "I clicked 'Share this store' and then 'QR Code' and a dialog appeared. **I couldn't scan or decode the QR code in this sandbox**, so I didn't follow it to a destination. The XL tee is in the cart." | **UNANSWERABLE / SITE_CHANGED** — *with a real agent-fail component* | fail 4/7 (CLI 2/3, MCP 0/3), conf 0.11–0.48, `blocked=0` throughout. "Follow the QR code" requires decoding a rendered image; no run could, and all four flagged it. Secondary: arXiv's homepage has no store link at all now — every run had to route via `info.arxiv.org/brand/swag.html`. **Genuine failure in 1 of 4**: one MCP run clicked the 4th size radio and the cart showed "L", which it honestly reported. |
| **ArXiv--42** | "Find an article published between 1 Jan 2000 and 1 Jan 2005 requiring Support Vector Machines in the title whose journal ref is ACL Workshop." | `'Using a Support-Vector Machine for Japanese-to-English Translation of Tense, Aspect, and Modality'` | "**'Using a Support-Vector Machine for Japanese-to-English Translation of Tense, Aspect, and Modality'** — arXiv cs/0112003, submitted 5 Dec 2001, journal ref 'ACL Workshop, the Data-Driven Machine Translation, 2001'. The title says 'Support-Vector Machine' (singular, hyphenated) rather than 'Support Vector Machines'." | **JUDGE_ERROR** | fail 2/6, conf **0.45 and 0.49** — both within 0.05 of passing. The answer is an **exact string match** to the reference plus the correct arXiv id and journal ref. The only distinguishing feature of the two failing runs is the trailing honesty caveat about hyphenation, which the `false` criterion ("missing or **hedged**") appears to have punished. The agent is right and the row is fine. |
| **Huggingface--3** | "Look up a model with a license of cc-by-sa-4.0 with the most likes on Hugging Face." | `replit/replit-code-v1-3b` | "**replit/replit-code-v1-3b**, 745 likes. Next three: defog/sqlcoder-7b-2 (446), nlpaueb/legal-bert-base-uncased (324), stabilityai/stablelm-3b-4e1t (316). I didn't open the model's own page to confirm its license tag." | **JUDGE_ERROR** | fail 1/6, conf **0.46**. **Verified live 2026-10-10** on `huggingface.co/models?license=license:cc-by-sa-4.0&sort=likes`: 745 / 446 / 324 / 316 — the agent's four counts are exact, and the top entry matches the reference string character for character. Graded 0 for the final sentence declining to double-check the license tag. |
| **Cambridge Dictionary--35** | "Find **the most common** prepositions that consist of groups of words." | `ahead of; except for; instead of; owing to; apart from; in addition to ...` | "The grammar page 'Prepositional phrases' has a section 'Common multi-word prepositions': according to, ahead of, along with, apart from, as far as, … in addition to, … instead of, … owing to … **The page lists them without ranking, so it doesn't say which is most common.**" | **JUDGE_ERROR** (+ ill-posed question) | fail 1/6, conf **0.47**. The agent returned the site's own list, which is a **superset** of the reference (`ahead of`, `apart from`, `instead of`, `in addition to`, `owing to` all present). Graded 0 for the closing caveat. The caveat is itself correct and exposes a question defect: "the **most** common" implies a ranking Cambridge does not publish, so no run can supply the superlative the question asks for. |

---

## 2. Counts and the benchmark ceiling

### Classification counts (one primary label per affected row, 19 rows)

| classification | rows | task_ids |
| --- | ---: | --- |
| **STALE_REFERENCE** | 5 | Booking--5, Google Search--14, Google Search--13, BBC News--3, GitHub--37 |
| **BOT_WALL** | 5 | Allrecipes--20, Allrecipes--21, Cambridge Dictionary--34, Booking--4, Google Map--33 |
| **AMBIGUOUS_QUESTION** | 5 | ESPN--24, Coursera--6, Coursera--7, Huggingface--2, BBC News--4 |
| **JUDGE_ERROR** | 3 | ArXiv--42, Huggingface--3, Cambridge Dictionary--35 |
| **UNANSWERABLE / SITE_CHANGED** | 1 | ArXiv--41 |
| **AGENT_FAIL** (as the row's primary cause) | **0** | — |
| clean (no failing run) | 11 | Amazon--9/--10, Apple--25/--26, ESPN--23, GitHub--38, Google Flights--3/--4, Google Map--32, Wolfram Alpha--5/--6 |

**Not one of the 19 rows is primarily an agent failure.** Two rows carry a real
agent-fail *component* inside a defective row — ArXiv--41 (one MCP run put size L
in the cart) and BBC News--4 (three MCP runs summarised from search teasers
because article navigation failed). At run granularity that is roughly **4 of 55
failing runs (7%)** attributable to the agent.

### Implied ceiling

| | rows | running total gradeable |
| --- | ---: | ---: |
| total | 30 | 30 |
| − defects in the **question or reference** (stale 5 + ambiguous 5 + unanswerable 1) | 11 | **19** |
| − unscoreable from **CLI sandbox egress** (bot walls) | 5 | **14** |

- **19 / 30 rows are cleanly gradeable today** as benchmark items. The other 11
  (37%) carry a defect that no agent change can fix.
- **14 / 30 rows measure CLI agent quality today**, because 5 of the 19 clean
  rows are refused by the sandbox's datacenter IP.
- The bot walls are demonstrably **environment, not dataset**: Allrecipes--20 and
  --21 are CLI **0/3** and MCP **3/3**; Google Map--33 is CLI 0/3, MCP 2/3. The
  Browserbase proxy passes what the sandbox cannot reach. Fix egress and those 5
  rows return.
- **Booking--5 is the only row that is impossible for every variant** (0/6). The
  rest of the defects raise variance rather than imposing a flat penalty — the
  same stale GitHub--37 reference scored CLI 3/3 and MCP 2/3; stale Google
  Search--13 scored CLI 3/3 and MCP 0/3. The Jev rubric's "current values that
  differ from the reference are fine" clause rescues staleness **sometimes**,
  which is worse for measurement than failing consistently: a headline score
  moves on whether the judge happened to invoke that clause.

### Two systematic judge defects, separate from the dataset

1. **Placeholder references are handed to the judge raw.** Four examples ship an
   unexpanded template — `<summary>` (BBC News--4), `<model 1>; <model 2>;
   <model 3>` (Huggingface--2), `NFL <score>` (ESPN--24), `<Ingredients>`
   (Allrecipes--21) — and ESPN--23 has one too. The judge receives
   `REFERENCE ANSWER (type: possible)\n<summary>` and grades on prior alone.
2. **The `false` criterion punishes honesty.** All three JUDGE_ERROR rows
   (ArXiv--42, Huggingface--3, Cambridge Dictionary--35) scored **0.45–0.49** —
   inside 0.05 of the threshold — on answers that match the reference or the live
   site exactly, and in each case the only distinguishing feature is a trailing
   caveat. `"Something asked for is missing or **hedged**"` is reading a correct
   answer plus a scope note as a hedge. This is the same false-negative mode
   `OPTIMIZATION.md` found on GitHub--35 and it has not been fixed; it is now
   measured at **3 of 30 rows**.

---

## 3. Recommended fixes, per bad row

### Drop (2)

| task_id | reason |
| --- | --- |
| **Booking--5** | `Jan 1 to Jan 4, 2024` is a hard past date on a site that refuses past dates. 0/6 across both variants. Repinning the year changes the reference answer (`Palasari Villa` was a Jan-2024 result), so there is nothing to preserve. Drop, or regenerate from scratch against a future date. |
| **Google Search--14** | "The year before last" has drifted two years off its intended subject and the reference (`Yeovil Town`) is tied to a 2022 retirement. Rewriting the relative phrase to "in 2022" makes it answerable again — but then it is a trivia lookup, not a web task, and Google CAPTCHAs the sandbox anyway. Drop. |

### Update the reference (4)

| task_id | fix |
| --- | --- |
| **GitHub--37** | Replace the 2024 marketing copy with the live trio under "Code, command, and collaborate" — *Manage agent-driven work from one place* / *Make your editor your most powerful accelerator* / *Ship faster with AI that work alongside you*. Better: also rewrite the question to name the section (see below), since the page will churn again. |
| **Coursera--7** | `3.3K reviews` → `3.6K` (live: 3,601). And see the rewrite below. |
| **Coursera--6** | `14 hours (approximately); 19 quizzes` → `~11 hours; 19 assignments`. Coursera's UI no longer uses the word *quizzes* anywhere. |
| **Google Search--13** | `Strange Planet, 2023` → the untitled iCarly film (currently listed Upcoming / pre-production on IMDb). Low-value fix: an "upcoming projects" question will go stale again within months. Consider dropping instead. |

### Rewrite the question (5)

| task_id | fix |
| --- | --- |
| **ESPN--24** | Disambiguate the two readings. Either *"…the most recent NFL game listed as Final on ESPN's scoreboard"* or *"…the most recent NFL game whose ESPN-listed broadcaster is ESPN or ABC"*. Pick one; today the agents answer both and lose. Also expand the `<score>` placeholder or mark the row as reference-free. |
| **BBC News--3** | `teams` → `players` (DP World Tour is individual stroke play). Pin the tournament by name or accept that the correct answer changes weekly — as written, both the tournament name and the −10 count are unstable, and the leaderboard is often mid-round. Strong candidate for dropping. |
| **Coursera--7** | "Find a course" but the reference names a **Specialization**. Either say "Find a Specialization" (reference stays `Reinforcement Learning Specialization`, 3.6K) or keep "course" and change the reference to `Fundamentals of Reinforcement Learning; University of Alberta; 2.9K reviews`, which is what the agents verifiably returned. |
| **Cambridge Dictionary--35** | Drop the superlative. "**The most common** prepositions" asks for a ranking Cambridge does not publish. Rewrite as *"List the multi-word prepositions Cambridge's 'Prepositional phrases' page gives as common examples."* |
| **Coursera--6** | "Identify a course related to 'AI for Healthcare'" is open-ended while the reference pins one course. Name it: *"On the Coursera course 'Fundamentals of Machine Learning for Healthcare', report the duration and the number of assignments in Assessments."* |

### Rewrite or drop — reference is a placeholder (2)

| task_id | fix |
| --- | --- |
| **Huggingface--2** | `<model 1>; <model 2>; <model 3>` is ungradeable, and HF exposes *last updated*, not *released*. Either convert to a **rubric** row (criteria: three distinct translation models, each with an HF-verifiable update date inside 30 days, each with a stated like/download count) or replace with a deterministic variant like Huggingface--3. |
| **BBC News--4** | `<summary>` is ungradeable and "the latest article on topic X" is unverifiable without an exhaustive index. Convert to a rubric row (criteria: names a specific BBC article URL published within N days, summary contains ≥3 claims traceable to that article) or drop. |

### No dataset change — fix the environment (5)

`Allrecipes--20`, `Allrecipes--21`, `Booking--4`, `Google Map--33`,
`Cambridge Dictionary--34`. All five are good questions blocked by the CLI
sandbox's egress; MCP/Browserbase passes most of them. Give the sandbox a
residential proxy, or report a CLI score over the 25 non-walled rows and say so.
Minor, independent of the wall: **Booking--4 states no year** ("Dec 25th - Dec
26th") — pin it to a future year so the row is reproducible.

### No dataset change — fix the judge (3)

`ArXiv--42`, `Huggingface--3`, `Cambridge Dictionary--35`. These rows are correct
and the agent answers are correct. Two changes to `JEV_QUESTIONS` in
`experiment_native.py`:

1. Remove `"or hedged"` from the `false` criterion, or qualify it to
   *"the answer is hedged such that no definite value is given"*. A caveat that
   sits **after** a complete answer is not a hedge, and it is currently costing
   three rows at confidence 0.45–0.49.
2. Detect `<…>` placeholder references before grading and either expand them or
   switch that row to a reference-free rubric. Five of 30 references are
   templates; the judge currently treats them as ground truth.

Expected effect: the three JUDGE_ERROR rows pass, lifting the measured score by
~3 rows with no agent change — on top of the ~11 rows the dataset fixes recover.

---

## 4. What this means for the headline number

The CLI variant's 46/90 and the MCP variant's 36/90 in `HARNESS_FINDINGS.md` are
each measuring, at most, 19 of 30 real questions, and the CLI number is further
cut to 14 by egress. **Roughly a third of the benchmark is not grading the agent.**

Priority order, by rows recovered per unit of work:

1. **Judge rubric: drop "or hedged", handle `<…>` placeholders** — 3 rows plus 5
   placeholder rows de-noised. One edit to `JEV_QUESTIONS`.
2. **Sandbox egress (residential proxy)** — 5 rows, already proven recoverable by
   the MCP variant's results on the same questions.
3. **Rewrite/repin the 9 question-and-reference defects** listed above — the work
   is per-row but each fix is permanent.
4. **Drop Booking--5 and Google Search--14** — unrecoverable.

And the broader conclusion from `OPTIMIZATION.md` holds with a bigger number
behind it: a live-web benchmark with hand-written 2024 references decays faster
than an agent improves. The 11 defective rows found here are ~37% of a 30-row
sample two years after authoring. Programmatically-validated suites (WebArena) or
re-verified live suites (Online-Mind2Web) are the way out; this dataset can be
patched but not stabilised.
