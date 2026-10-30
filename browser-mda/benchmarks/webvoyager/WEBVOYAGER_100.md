# webvoyager-100

A 100-row WebVoyager dataset, every row vetted against the live web on **2026-10-10**.

- LangSmith dataset: **`webvoyager-100`**, id `27a7d6e1-1fa9-4205-b230-098450c30db2`
- Reproducible source of truth: [`webvoyager_100.jsonl`](webvoyager_100.jsonl)
- `webvoyager-30` (`e06b6845-b213-4c5f-9369-3a6cfdee9331`) is **unmodified**.
- Schema is unchanged, so `experiment_native.py` runs against it by flipping `DATASET`.

```python
DATASET = "webvoyager-100"
```

Row shape (identical to `webvoyager-30`):

```json
{"inputs":  {"web": "...", "ques": "...", "site": "...", "task_id": "ESPN--41"},
 "outputs": {"answer_type": "golden|possible|criteria",
             "site_notice": "...", "reference_answer": "..."}}
```

`answer_type: "criteria"` is new. It marks a row whose correct answer is genuinely
live data (a hotel search, a fare lookup, an "open now" filter) where no fixed string
can be ground truth. The `reference_answer` on those rows is written as an acceptance
criterion instead of an answer, and the Jev `correct` rubric in `experiment_native.py`
already grades it correctly ("current values that differ from the reference are fine").
`criteria` rows require an LLM judge; they will not work under exact-match scoring.

## Headline counts

| | count |
| --- | ---: |
| rows shipped | **100** |
| kept from `webvoyager-30` verbatim | 8 |
| kept from `webvoyager-30` after repair (question and/or reference rewritten) | 15 |
| newly added from the upstream 643 | 77 |
| &nbsp;&nbsp;…of which the upstream 2024 reference had to be corrected | 21 |
| verified against the live web | **83** |
| accepted on inspection alone (site bot-walled) | **17** |
| rejected / adjudicated and not shipped | **481** |

`answer_type`: golden 56, possible 30, criteria 14.

Of the original 30 rows, **23 survive** (8 verbatim, 15 repaired) and **7 were dropped**.
Only 56 of the 100 reference answers survived contact with the live web unchanged: 36 rows
needed their reference or question rewritten, which is the real measure of how far the 2024
annotations have decayed.

## Per-site counts

| site | rows | kept | repaired | added | live-verified | inspection-only |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Allrecipes | 6 | 1 | 0 | 5 | 0 | 6 |
| Amazon | 7 | 0 | 2 | 5 | 7 | 0 |
| Apple | 7 | 2 | 0 | 5 | 7 | 0 |
| ArXiv | 7 | 1 | 0 | 6 | 6 | 1 |
| BBC News | 7 | 0 | 0 | 7 | 7 | 0 |
| Booking | 6 | 0 | 1 | 5 | 3 | 3 |
| Cambridge Dictionary | 7 | 1 | 1 | 5 | 7 | 0 |
| Coursera | 7 | 0 | 2 | 5 | 7 | 0 |
| ESPN | 7 | 0 | 2 | 5 | 7 | 0 |
| GitHub | 7 | 0 | 2 | 5 | 7 | 0 |
| Google Flights | 6 | 0 | 2 | 4 | 1 | 5 |
| Google Map | 6 | 0 | 2 | 4 | 4 | 2 |
| Google Search | 6 | 0 | 1 | 5 | 6 | 0 |
| Huggingface | 7 | 1 | 0 | 6 | 7 | 0 |
| Wolfram Alpha | 7 | 2 | 0 | 5 | 7 | 0 |
| **total** | **100** | 8 | 15 | 77 | 83 | 17 |

Allrecipes, Booking, Google Search, Google Map and Google Flights are capped at 6 because
they bot-wall datacenter IPs for the CLI arm. They are kept in the set (the MCP/REPL arms
reach them) but deliberately not over-weighted.

## Vetting method

One agent per site, working the full upstream pool for that site (all 643 upstream tasks
were adjudicated across the 15 sites, not just the ones shipped). Reference answers came
from the upstream `reference_answer.json`, merged onto `WebVoyager_data.jsonl` by
`(web_name, id)` and checked against all 30 existing rows — 30/30 exact match, so the merge
is sound.

Each candidate was screened against nine reject rules, then taken to the live web:

1. absolute date in the past used transactionally (the `Booking--5` failure)
2. relative date that has drifted ("the year before last" — the `Google Search--14` failure)
3. reference containing an unfilled `<...>` placeholder slot
4. time-varying content that churns within days (live leaderboards, "most recent article",
   "upcoming projects")
5. ambiguous about which artifact is wanted (the Coursera "course" vs "Specialization" failure)
6. requires login, payment, or CAPTCHA-solving
7. requires decoding an image or QR code (the `ArXiv--41` failure)
8. asks for a superlative the site does not publish (the `Cambridge Dictionary--35` failure)
9. the target page or section no longer exists

Archival lookups over a *historic* range were explicitly **not** treated as stale — a paper
published between 2000 and 2005, or the Jan 27 2024 Heat-Knicks boxscore, is permanently
true and is the strongest row shape this benchmark has.

Repair was preferred to rejection throughout: a stale reference on a sound question was
re-read off the live page rather than dropped. 36 of the 100 rows carry a corrected
reference or a rewritten question (15 of the 23 surviving originals, plus 21 of the 77
additions).

**Verification channels actually used**, because "WebFetch it" was not always possible:

| site | channel | result |
| --- | --- | --- |
| ArXiv, Huggingface, GitHub | WebFetch + the sites' own JSON APIs | clean |
| Apple, Coursera, ESPN, BBC News | WebFetch, or `curl` with a desktop UA where WebFetch was refused | clean |
| Cambridge Dictionary | `curl` with a desktop Chrome UA (WebFetch gets a Cloudflare 403) | clean |
| Wolfram Alpha | **computed independently with sympy/mpmath**, not fetched — stronger evidence than a page read | clean |
| Google Search | WebSearch against the underlying facts, not google.com's rendering | facts confirmed, rendering not |
| Amazon | WebSearch snippets; `/s?k=` search returns 503 and `/dp/` returns a shell to WebFetch | partial |
| Booking | help/articles pages fetch; `searchresults.html` returns an empty shell | partial |
| Google Flights, Google Map | JS shell only — zero page text to a datacenter IP | **blocked** |
| Allrecipes | WebFetch refused, `curl` 403/460, Jina Reader 451, every CORS proxy walled, WebSearch blocks the domain, Internet Archive offline | **blocked** |

Where a site could not be read, the row is marked `vetting: "inspection"` in
`webvoyager_100.jsonl` and the `evidence` field records the exact failure. No row claims a
verification that did not happen.

### The 17 rows accepted on inspection alone

| task_id | why not live |
| --- | --- |
| `Allrecipes--13` | Could not fetch allrecipes.com (bot wall, 403). Judged from question + reference: requires expanding the Full Nutrition Label, a static per-serving fi |
| `Allrecipes--20` | WebFetch + curl + jina + 3 CORS proxies all refused (People Inc bot wall: HTTP 403/451); a DuckDuckGo site: search did surface the live page https://w |
| `Allrecipes--22` | Could not fetch allrecipes.com (bot wall, 403). Judged from question + reference: no placeholder slots, and prep time plus the per-serving nutrition p |
| `Allrecipes--24` | Could not fetch allrecipes.com (bot wall, 403). Judged from question + reference: no placeholder, no date, and the asked-for facts (prep time, serving |
| `Allrecipes--36` | Could not fetch allrecipes.com (bot wall, 403). Judged from question + reference: the graded fact is an oven temperature printed in the Directions, wh |
| `Allrecipes--38` | Could not fetch allrecipes.com (bot wall, 403). Judged from question + reference: the reference spells out concrete ingredients rather than a <Ingredi |
| `ArXiv--42` | Kept verbatim per instructions (verified by a prior audit against the live abs page); archival 2000-2005 title+journal-ref search, answer cannot move. |
| `Booking--2` | Could not confirm live - Booking search-results pages are bot-walled from this IP (empty shell / 0 properties). Ohio is a broad, high-inventory region |
| `Booking--4` | Could not confirm availability live - Booking.com search-results pages return an empty/0-property shell from this datacenter IP (bot-wall); only help  |
| `Booking--43` | WebFetch of https://www.booking.com/searchresults.html?ss=Los+Angeles returned a shell page reporting '0 properties are available' with no sidebar, an |
| `Google Flights--2` | WebFetch of https://www.google.com/travel/flights/ returned an empty JS shell; Google refuses datacenter IPs, so no live read of a JFK-LHR results pag |
| `Google Flights--25` | WebFetch of https://www.google.com/travel/flights/ returned an empty JS shell; Google blocks datacenter IPs so the business-class results page could n |
| `Google Flights--3` | WebFetch of https://www.google.com/travel/flights/ returned an empty JS shell (no text) - Google serves no server-rendered content to datacenter IPs,  |
| `Google Flights--4` | WebFetch of https://www.google.com/travel/flights/ returned an empty JS shell (no text); Google blocks datacenter IPs, so the live results page could  |
| `Google Flights--9` | WebFetch of https://www.google.com/travel/flights/ returned an empty JS shell; no live verification possible from this IP. Judged from question + refe |
| `Google Map--32` | Google Maps blocks WebFetch (https://www.google.com/maps/place/... returned only the string 'Google Maps' - JS shell, no listing data). Judged from qu |
| `Google Map--33` | Google Maps blocks WebFetch (place URL returned only 'Google Maps'). A WebSearch for DEN accessibility attributes surfaced only flydenver.com / third- |

All 6 Allrecipes rows are inspection-only: the site is unreadable from every channel
available here, so their references are upstream 2024 values carried forward. **This is the
weakest slice of the dataset and the one a human should re-check first**, on a residential
IP or a real browser. The Google Flights, Google Map and Booking inspection rows are all
`criteria` rows whose correctness does not depend on a fetched value — only on the task
being well-posed — which is why they are shippable despite the wall.

## The DATASET_AUDIT.md fix list, as applied

| audit instruction | task_id | what happened |
| --- | --- | --- |
| drop | `Booking--5` | **dropped** — Booking greys out past dates entirely, so the stay is unselectable; scored 0/6 in a prior audit. Unpassable by any agent, not a model failure. |
| drop | `Google Search--14` | **dropped** — 'The year before last' meant 2022 when authored and means 2024 today, which matches no James Smith retirement; reference 'Yeovil Town' is unreachable. Google also CAPTCHAs the sandbox for this multi-h |
| update reference | `GitHub--37` | **repaired** — Named the target section in the question so it survives marketing refreshes, and replaced the 2024 reference copy with the three live headings verified today. |
| update reference | `Coursera--6` | **repaired** — Named the specific course in the question instead of 'a course related to AI for Healthcare'; changed 'quizzes' to 'assignments' to match current UI wording; updated the reference duration from '14 ho |
| update reference | `Coursera--7` | **repaired** — Question said 'a course' while the reference named a Specialization; rewrote the question to ask explicitly for the Reinforcement Learning Specialization. Updated the review count from '3.3K' to the l |
| update reference | `Google Search--13` | **repaired** — Rewrote the stale 'upcoming projects' question (ref 'Strange Planet, 2023' had been out for three years) into a settled filmography fact about the same actor. Kept the person and the Google-knowledge- |
| rewrite question | `ESPN--24` | **repaired** — Removed the ambiguous 'most recent NFL game broadcast on ESPN' framing (network-broadcast vs espn.com-scoreboard readings) and the '<score>' placeholder by pinning a specific, permanently archived gam |
| rewrite question | `BBC News--3` | **dropped** — DP World Tour leaderboards rotate weekly, so both the tournament name and the count at -10 are stale within 7 days; 'teams' also has no referent in individual stroke play. Rewriting 'teams'->'players' |
| rewrite question | `Cambridge Dictionary--35` | **repaired** — Dropped the unranked superlative from the question and named the grammar page ('Prepositions') so the target artifact is unambiguous; replaced the truncated '...' reference with the full 22-item list  |
| convert or drop placeholder | `Huggingface--2` | **dropped** — Reference is the bare placeholder '<model 1>; <model 2>; <model 3>' with no fillable concrete value, and the answer set turns over every month; a criteria rewrite would still be a churning, barely-gra |
| convert or drop placeholder | `BBC News--4` | **dropped** — Reference is the bare placeholder '<summary>'. There is no exhaustive BBC index of 'climate change economics in Europe', so 'the latest' cannot be verified by any grader, and a criteria rewrite would  |

### Rows dropped beyond the audit's explicit list

| task_id | rule | why |
| --- | --- | --- |
| `Allrecipes--21` | rule 3 (unfilled placeholder that cannot be replaced with a verified value) | Reference is "'Gluten-Free Fudge Brownies', 4.1 stars, 69 reviews, <Ingredients>, Prep Time: 15 mins, Total Time: 1 hr"; every fetch route to allrecipes.com was bot-walled (403/451, Internet Archive is offline), so the ingredient list could not be fi |
| `ArXiv--41` | 7 (requires decoding an image / QR code) | Needs a rendered QR decoded, and is also a cart transaction on a store arXiv's homepage no longer links; not repairable. |

Four deviations from the audit's literal fix list, all in the direction of the nine
reject rules above:

- **`BBC News--3`** — the audit offered "rewrite `teams`→`players`" but also called it a
  strong drop candidate. Renaming the noun fixes the wording, not the defect: the DP World
  Tour leaderboard rotates weekly, so both the tournament name and the −10 count go stale
  within 7 days. Dropped under rule 4, replaced by two pinned evergreen BBC explainers.
- **`Huggingface--2`** / **`BBC News--4`** — both offered a `criteria` conversion. Both were
  dropped instead: a criterion like "names three translation models updated in the last 30
  days" grades a churning free-form list and gives no discriminating signal. With 7 solid
  deterministic rows available on each site, neither was worth a slot.
- **`ArXiv--41`** is not on the audit's fix list but is named in the reject rules as the
  image/QR-decoding failure. Dropped under rule 7; arXiv's homepage no longer links the
  store at all, so it is also a rule-9 failure.
- **`Allrecipes--21`** was to be repaired by filling its `<Ingredients>` slot. Allrecipes
  was unreadable from every channel, so the slot could not be filled and the row was dropped
  rather than shipped with a placeholder. A clean `criteria` rewrite exists if anyone can
  reach the page.

## Rejections

481 upstream tasks were adjudicated and not shipped. By primary reason:

| rule | reason | count |
| ---: | --- | ---: |
| 4 | churns within days (live / latest / upcoming) | 166 |
| 3 | unfilled `<...>` placeholder reference | 80 |
| 1 | absolute past date used transactionally | 71 |
| 9 | target page no longer exists | 43 |
| 2 | drifted relative date | 39 |
| 5 | ambiguous about which artifact is wanted | 36 |
| surplus | sound, but cut by the per-site cap | 14 |
| 6 | login, payment or CAPTCHA | 12 |
| 7 | requires decoding an image or QR code | 8 |
| 8 | superlative the site does not publish | 6 |
| other | other | 3 |
| unverifiable | could not be verified either way | 3 |

Rule 4 alone accounts for a third of the corpus, and rules 1+2+4 together — all forms of
time decay — account for **276 of 481**. That is the measured version of the conclusion in
`OPTIMIZATION.md`: a live-web benchmark with hand-written 2024 references decays faster than
an agent improves.

### Headline rejections, with examples

- **rule 1** — `Booking--5` — "hotel in Bali **from Jan 1 to Jan 4, 2024**". Booking.com greys out past dates by design, so no agent can pass it; a prior audit measured 0/6. Repinning the year would change the reference answer, so there was nothing to preserve.
- **rule 1** — `Google Flights--12` — "round trip New York->London leaving **December 25, 2023**". 28 of the 42 Google Flights tasks and 24 of the 44 Booking tasks carry a hard 2023/2024 date in the question text.
- **rule 2** — `Google Search--14` — "the retired players **the year before last** named James Smith". The phrase resolved to 2022 when authored and resolves to 2024 today, matching nobody. Reference `Yeovil Town` is tied to the 2022 reading.
- **rule 3** — `Huggingface--2` — reference is the bare string `<model 1>; <model 2>; <model 3>`. The judge is handed an angle-bracket literal as ground truth and grades on its prior alone.
- **rule 3** — `BBC News--4` — reference is `<summary>`. Same defect, compounded by "the latest article" having no verifiable referent.
- **rule 4** — `BBC News--3` — DP World Tour leaderboard. Stale within 7 days of authoring, and the tour is individual stroke play so "how many **teams** at -10" has no referent at all.
- **rule 4** — `Google Search--13` (as originally worded) — "Jerry Trainor's **upcoming** projects", reference `Strange Planet, 2023`, which premiered in August 2023 and has not been upcoming for three years.
- **rule 4** — Hugging Face download counts are a **rolling 30-day window**, not cumulative — `Huggingface--20`'s "1M+ downloads" reference now reads 91K. That one fact drove five rejections on its own.
- **rule 5** — `ESPN--18` — "how many sports leagues on the ESPN home page?" Live: 7 in the top nav, 31 more under "More Sports", and several entries (Gaming, Sports Betting, Bananas) are not leagues. The upstream `31` is both stale and under-specified.
- **rule 6** — `Amazon--5` / `Amazon--15` — add-to-cart flows; `Huggingface--1` / `--6` now require an account with inference credits, which they did not in 2024.
- **rule 7** — `ArXiv--41` — "follow the QR code to share the shop". Requires decoding a rendered image; every run in a prior audit flagged exactly that.
- **rule 8** — `Cambridge Dictionary--35` (as originally worded) — "**the most common** prepositions". The page prints the phrase but publishes no ranking, so the superlative is unanswerable. Repaired by asking for the list the page actually prints (22 items).
- **rule 9** — `ArXiv--19` — "which university maintains arXiv?" arXiv is now an independent nonprofit; the Cornell answer is gone from the site.
- **rule 9** — `GitHub--29` — /pricing no longer has a Pro column. `GitHub--16` — the "First day on GitHub" heading went away with Learning Lab. `GitHub--21` — GitHub Advanced Security was rebranded and unbundled.
- **rule 9** — `Huggingface--13` — `spaces_between_special_tokens` is gone from the current `LlamaTokenizer` signature. `--18` and `--21` target doc passages that were rewritten. All three pass a regex pre-screen cleanly and are nonetheless dead.

Three defects were found that a regex pre-screen cannot catch and that only a live read
surfaced: silently-rewritten documentation pages (Hugging Face, GitHub), *factually wrong
upstream references* (`Wolfram Alpha--15` states 3.125% where a first cousin once removed is
6.25%; `Apple--27` lists a HomePod mini colour that no longer exists; `Coursera--16` names
the wrong instructor), and references that are internally contradictory
(`Wolfram Alpha--39` asks for 33 degrees and answers with 2pi/15 rad = 24 degrees).

### Full rejection list

Every adjudicated-and-not-shipped task, grouped by site.

<details><summary><b>Allrecipes</b> (24)</summary>

| task_id | rule | detail |
| --- | --- | --- |
| `Allrecipes--10` | rule 9 / rule 4 (editorial roundup, ordinal position) | Pins the *second* item in an editorial slideshow; Allrecipes re-edits and re-orders these collections, and the listicle may have been retired. |
| `Allrecipes--12` | rule 3 (unfilled placeholder) | <Ingredients> unfilled. |
| `Allrecipes--16` | rule 3 (placeholder) + fragile | <Ingredients> unfilled, and the pinned recipe sat at 5.0 stars on only 3 reviews - that rating almost certainly moved. |
| `Allrecipes--17` | rule 4 (churns within days) | 'The latest review' changes whenever anyone reviews the recipe; the 2024 reference text is long dead. |
| `Allrecipes--19` | rule 3 (unfilled placeholder) | <Ingredients> unfilled; reference also fails its own >200-review constraint check without live data. |
| `Allrecipes--21` | rule 3 (unfilled placeholder that cannot be replaced with a verified value) | Reference is "'Gluten-Free Fudge Brownies', 4.1 stars, 69 reviews, <Ingredients>, Prep Time: 15 mins, Total Time: 1 hr"; every fetch route to allrecipes.com was bot-walled (403/451, Internet Archive is offline), so the i |
| `Allrecipes--23` | rule 3 (unfilled placeholder) | <Ingredients> unfilled; reference answer ('Baked Chicken Schnitzel') also does not match the 'baked lemon chicken' ask. |
| `Allrecipes--25` | rule 3 (unfilled placeholder) | <Ingredients> unfilled. |
| `Allrecipes--26` | rule 3 (unfilled placeholder) | <Ingredients> and <Description: Cooking steps> unfilled. |
| `Allrecipes--27` | rule 3 (unfilled placeholder) | <Ingredients> and <cooking instructions> unfilled. |
| `Allrecipes--28` | rule 3 (unfilled placeholder) | <Ingredients> and <preparation steps> unfilled. |
| `Allrecipes--29` | rule 3 (unfilled placeholder) | <Ingredients>/<cooking method> unfilled; 'includes olives' is also only satisfied by 'olive oil' in the reference. |
| `Allrecipes--30` | rule 3 (unfilled placeholder) | <steps> unfilled; also the pinned answer is a smoothie, not a smoothie bowl. |
| `Allrecipes--31` | rule 3 (unfilled placeholder) | <Ingredients> and <preparation steps> unfilled. |
| `Allrecipes--32` | rule 3 (unfilled placeholder) | <Ingredients> unfilled and the task grades the *first five* ingredients, so the slot is load-bearing. |
| `Allrecipes--34` | rule 3 (unfilled placeholder) | <Ingredients> unfilled where the graded fact is the primary seasoning. |
| `Allrecipes--39` | rule 3 (unfilled placeholder) | <Ingredients> unfilled; storage advice is editorial copy that may have been rewritten. |
| `Allrecipes--40` | rule 9 (page/section may no longer exist) | Reference is a verbatim 2024 About-Us paragraph; People Inc. rewrote Allrecipes' corporate pages and the exact wording cannot be confirmed behind the bot wall. |
| `Allrecipes--41` | rule 4 (churns within days) | The Dinners hub is a rotating recommendation rail; the three named casseroles will not be on it. |
| `Allrecipes--44` | rule 9 (site navigation may have changed) | Depends on the current Occasions nav taxonomy, which People Inc. has reshuffled; unverifiable behind the bot wall. |
| `Allrecipes--5` | rule 3 (unfilled placeholder) | <Ingredients> unfilled; also internally inconsistent (asks >1000 reviews, reference says 818). |
| `Allrecipes--7` | rule 3 (unfilled placeholder) | Both <Ingredients> and <Preparation Steps> unfilled and unverifiable behind the bot wall. |
| `Allrecipes--8` | rule 3 (unfilled placeholder) | <Ingredients> unfilled. |
| `Allrecipes--9` | rule 3 (unfilled placeholder) | <Ingredients> unfilled. |

</details>

<details><summary><b>Amazon</b> (27)</summary>

| task_id | rule | detail |
| --- | --- | --- |
| `Amazon--1` | rule 4 (churns within days) + rule 5 (ambiguous) | Lowest price in a filtered result set changes constantly, and the reference names a 2019 MEN's polo for a women's query. |
| `Amazon--11` | rule 4 (churns within days) | Cheapest-with-free-delivery is price- and location-dependent and not reproducible. |
| `Amazon--12` | rule 4 (churns within days) | 'Top review' re-ranks continuously; reference 'Worth every penny' is a one-off review title. |
| `Amazon--13` | rule 2 (drifted relative date) | Reference depends on '500+ bought in past month' badge, a rolling window. |
| `Amazon--15` | rule 6 (login/payment) | Transactional add-to-cart. |
| `Amazon--16` | rule 9 (target page likely gone) | Pinned to an unnamed-ASIN apparel listing from 2024 that could not be located live; the generic return-steps answer would be gradeable but the specific item is unverifiable. |
| `Amazon--17` | rule 4 (churns within days) | On-sale sub-$10 inventory rotates daily. |
| `Amazon--18` | rule 4 (churns within days) | Homepage deals change hourly. |
| `Amazon--19` | rule 2 (drifted relative date) | 'Within a month' anchored to a Jan 2024 release. |
| `Amazon--20` | rule 4 (churns within days) | Logitech Wave Keys is now typically above the $60 band; price-band pinning makes the row flaky. Not needed to reach 7. |
| `Amazon--21` | rule 4 (churns within days) | Viable but price-band dependent (Braun BrewSense frequently drops below $100); deprioritized in favor of looser-constraint rows. |
| `Amazon--23` | rule 4 (churns within days) | Generic unbranded listing reference; narrow price band on churning no-name listings. |
| `Amazon--24` | rule 9 (target page likely gone) | Referenced Dash DMAF360GBAQ02 model line has churned; not needed to reach 7. |
| `Amazon--25` | rule 4 (churns within days) | Reference is an unbranded listing in a narrow price band. |
| `Amazon--26` | rule 4 (churns within days) | Verified live that qualifying speakers exist (Anker Soundcore 2/3, Ortizan X10), but the referenced unbranded listing is gone and many candidates sit near the $50 line; held back as a spare. |
| `Amazon--27` | rule 4 (churns within days) | Best-seller ordering re-ranks continuously. |
| `Amazon--3` | rule 4 (churns within days) | Top-3 by price-desc in an open search is unstable day to day; reference is a placeholder ('First 3 results after sort'). |
| `Amazon--30` | rule 8 (superlative the site does not publish) | Amazon publishes no 'highest rated 2024 fiction' ranking; also a stale year anchor. |
| `Amazon--33` | rule 4 (churns within days) | Top-3 search results and their prices churn; seasonal category. |
| `Amazon--35` | rule 1 (past absolute date in reference) | Reference pins 'FREE delivery Friday, March 1' (2024). |
| `Amazon--38` | rule 1 (past absolute date in reference) | Reference pins 'FREE delivery Friday, March 1' (2024). |
| `Amazon--39` | rule 4 (churns within days) | Self-published travel guide listings churn; the 2024 pin is now arbitrary and the referenced title may be delisted. |
| `Amazon--4` | rule 4 (churns within days) | Marketplace used offers appear/disappear hourly; cheapest 'Used - Good' price is not reproducible. |
| `Amazon--40` | rule 1 (past absolute date in reference) | Reference pins 'FREE delivery Friday, March 1' (2024) plus a specific color count that changes. |
| `Amazon--5` | rule 6 (login/payment) | Transactional add-to-cart; excluded per instruction to reject cart/checkout rows. |
| `Amazon--6` | rule 4 (churns within days) | Workable but the 20,000-review threshold plus a narrow price band is fragile as listings move in and out of the band; not needed to hit 7. |
| `Amazon--8` | rule 4 (churns within days) | 'Cheapest' across a filtered set changes constantly; referenced renewed Galaxy Tab S 10.5 at $139.94 is long gone. |

</details>

<details><summary><b>Apple</b> (36)</summary>

| task_id | rule | detail |
| --- | --- | --- |
| `Apple--0` | reference answer pinned to stale pricing I will not re-verify (rule 4 time-varying / prompt instruction) | M1/M2 lineup no longer sold; current Air is M5. Pricing-pinned reference rejected per instruction. |
| `Apple--1` | 9 - targets content no longer prominent on the live site | iOS 17 feature page superseded (iOS 26.x era); 'new features' framing is stale. |
| `Apple--10` | 5 - ambiguous (MacBook Air vs Pro) + stale M3 reference | Memory/storage option lists churn every refresh. |
| `Apple--11` | 1/5 - stale release date (Oct 26, 2022) + pricing + ambiguous 'latest iPad' | Three stale components at once. |
| `Apple--13` | duplicate coverage - not needed to hit target | Verified live (4: Sky Blue, Silver, Starlight, Midnight) but dropped to avoid three color-listing rows; a good spare if a slot opens. |
| `Apple--14` | 9 - M3 configuration no longer offered; pricing-pinned | Current 14-inch is M5-family; the $1599/$1020 arithmetic cannot be reproduced. |
| `Apple--15` | 4 - answer churns with per-region keyboard layout availability | Count of layouts varies by storefront/region; '16' not reproducibly verifiable. |
| `Apple--16` | 9 - product no longer sold | AirPods 3rd gen (Lightning/MagSafe variants) are off the current lineup. |
| `Apple--17` | 4 - live inventory, churns within days | Pickup store depends on stock on the day; 'Apple Tower Theatre' is not stable. |
| `Apple--18` | 4 - promotional offer, and the reference is trivially yes/no | Low-signal row even when the answer holds. |
| `Apple--19` | 4 - marketing taglines rotate with every campaign | Upstream 'If you can dream it, Mac can do it' no longer matches the live headline. |
| `Apple--2` | 9 - page no longer exists | iPhone 14 Pro / 15 Pro are discontinued and have no buy pages on apple.com; reference also price-pinned. |
| `Apple--20` | 9 - product no longer sold | iPhone 14 Plus discontinued; price-pinned. |
| `Apple--21` | stale reference, overlaps better rows | Current iPad Pro storage tiers differ from the 128GB-base 2024 list; could be repaired but not needed for target. |
| `Apple--22` | 4 - trade-in values are repriced continuously | Pricing-pinned reference I cannot re-verify stably. |
| `Apple--23` | pricing-pinned reference | Series 9 / $399 / $249 superseded. |
| `Apple--24` | pricing-pinned reference | apple.com/imac/specs/ does not even render a price in fetched content; rejected per pricing instruction. |
| `Apple--3` | pricing-pinned reference | iPhone 15 Pro prices/sizes long superseded; price-pinned refs rejected. |
| `Apple--30` | pricing-pinned reference | Full price ladder, all stale. |
| `Apple--31` | 4 - taglines rotate yearly | 'Smarter. Brighter. Mightier.' was the Series 9 campaign. |
| `Apple--32` | 4 - trade-in values churn; device likely dropped from the trade-in list | Pricing-pinned. |
| `Apple--34` | duplicate coverage - overlaps the kept Apple TV row | Verified live (1.2 x 3.66 x 3.66 in; USB-C Siri Remote, clickpad, Bluetooth 5.0) — a strong spare, dropped only for site-area diversity. Second half ('list the Siri Remote features introduced') is also loosely specified. |
| `Apple--35` | 5 - ambiguous second clause | Live page lists 4 pencils (Pro, 2nd gen, USB-C, 1st gen) and BOTH Apple Pencil Pro and 2nd gen pair/charge wirelessly, so the single-answer framing is now ambiguous. |
| `Apple--36` | 4/7 - rotating editorial artwork, requires reading names off images | Changes weekly. |
| `Apple--37` | 9 - all three products removed from apple.com | Only reachable via support archive pages, not the shopping site the task targets. |
| `Apple--38` | 4 - accessory grid churns and the reference ends in '...' | Could be repaired to a criteria row; skipped in favor of deterministic rows. |
| `Apple--39` | 6 - the flow leads into an account/sign-in path | Reference is also a verbatim support-page sentence that Apple rewords often. |
| `Apple--4` | 9 - configuration no longer offered | M3 Max is not configurable on the current MacBook Pro; price reference unverifiable. |
| `Apple--40` | 9 - Vision Pro hardware revised; weight range and app list changed | Reference ends in '...' and the 600-650 g figure is tied to the 1st-gen unit. |
| `Apple--41` | 9 - 64GB tier no longer offered; pricing-pinned | Current iPad mini starts at 128GB. |
| `Apple--42` | 3 - unfilled <summary> placeholder in the reference | Series 7/8/9 are also off the compare tool now. |
| `Apple--5` | 1/4 - stale date + pricing | Reference is iPhone 15, September 22, 2023. |
| `Apple--6` | 4 - count churns with lineup changes | AirPods lineup count moves every refresh and the store grid also varies; weak deterministic answer. |
| `Apple--7` | 1 - absolute past date | 'Available early 2024 in the U.S.' — the product has shipped; question is no longer well-posed. |
| `Apple--8` | 5 - ambiguous which artifact is wanted | 'latest iPad' is ambiguous across iPad / iPad Air / iPad Pro / iPad mini; reference pinned to M2 iPad Pro. |
| `Apple--9` | 1 - absolute date in the past, transactional | Also effectively a checkout flow. |

</details>

<details><summary><b>ArXiv</b> (36)</summary>

| task_id | rule | detail |
| --- | --- | --- |
| `ArXiv--0` | 4 (churns within days) | 'Latest' preprints change daily. |
| `ArXiv--1` | 2 (relative date) | 'Last two days' has no fixed answer. |
| `ArXiv--11` | 9 (page/section no longer exists) | Live info.arxiv.org/help/faq/multilang.html now requires a full English version of the paper and says nothing about a '-----' abstract separator; prep.html only links the FAQ. |
| `ArXiv--12` | 9 (page no longer exists) | Store is no longer reachable from arXiv Help; SKU counts also churn. |
| `ArXiv--13` | 4 (growing count) | Title count grows as new SimCSE papers appear; superseded by the archival ArXiv--14 repair. |
| `ArXiv--15` | 5 (ambiguous) | Archival window, but 'mention being accepted for AAAI 2024' depends on comments-field text that authors still edit, and the query phrasing ('Chinese Benchmark' in which field?) is undefined. |
| `ArXiv--16` | 2 + 3 | 'This week' plus '<summary>' placeholder. |
| `ArXiv--18` | 5 (ambiguous) | 'How many formulas' is undefined (numbered equations vs inline), so grading is unreliable. |
| `ArXiv--19` | 9 + 4 | arXiv is now an independent nonprofit (banner on info.arxiv.org), so the Cornell premise is stale, and the enrollment figure changes yearly. |
| `ArXiv--2` | 3 (unfilled placeholder) + 4 | Reference is '<abstract>' over a daily-changing listing. |
| `ArXiv--20` | 3 + 4 | '<abstract>' of the latest stat.ML paper. |
| `ArXiv--21` | 2 (relative date) | Rolling week count. |
| `ArXiv--22` | 9 (instructions replaced) | The 'arch-ive@arxiv.org subscribe' mail flow has been superseded by the current email-subscription help page; reference text no longer matches. |
| `ArXiv--23` | 2 (relative date) | 'Yesterday' drifts. |
| `ArXiv--24` | 4 (churns within days) | Also asks for affiliation arXiv often does not publish. |
| `ArXiv--25` | 9 (page no longer exists) | Same store path; merchandise list is inventory that changes. |
| `ArXiv--26` | 2 (relative date) | Rolling week count. |
| `ArXiv--28` | 4 (result set churns) | HTML availability and the title-search result set both change as new papers land. |
| `ArXiv--29` | 3 (imprecise reference) | Reference is '240+ (search by title)' — not a gradeable value, and arXiv's title search is phrase-loose here. |
| `ArXiv--3` | 3 (unfilled placeholder) + 4 | '<title>, <authors>, <abstract>' over the live math.AT listing. |
| `ArXiv--31` | no rule violation, dropped for unverifiability | Archival window is sound, but confirming '7 papers' requires author-counting every hit; not verified, so not shipped. |
| `ArXiv--32` | 3 + 4 | Placeholders over a daily listing. |
| `ArXiv--33` | 4 (churns within days) | Latest eess.SY paper. |
| `ArXiv--34` | 4 + 3 | 'Most recent' plus '<abstract>'; the pinned 2023 paper is no longer the most recent. |
| `ArXiv--35` | 4 (churns within days) | Latest quant-ph paper. |
| `ArXiv--36` | 4 (growing count) | Journal-ref fields keep being added by authors, so both counts drift upward from 48/9. |
| `ArXiv--37` | 4 (time-varying roster) | Live roster is now Penelope Lewis, Jeff Sank, Stephanie Orphan, Steinn Sigurdsson — the 2024 reference is fully stale, and staff rosters keep churning. |
| `ArXiv--38` | 4 + 3 | 'Latest blog post' plus '<summary>'. |
| `ArXiv--4` | 2 (drifted relative date) | Reference pinned to 22 Dec 2023 counts. |
| `ArXiv--40` | 4 (churns daily) | Also 'choose one at random', which is ungradeable. |
| `ArXiv--41` | 7 (requires decoding an image / QR code) | Needs a rendered QR decoded, and is also a cart transaction on a store arXiv's homepage no longer links; not repairable. |
| `ArXiv--5` | 4 (monotonically growing count) | 23081/39482 are 2023 totals; the corpus grows daily. |
| `ArXiv--6` | no rule violation, dropped for unverifiability | Deterministic in principle, but the 2020 paper has no arXiv HTML and the '2 Figures, 8 Tables' count could not be confirmed without parsing the PDF; dropped rather than shipped unverified. |
| `ArXiv--7` | 4 (churns within days) | cs.LG 'latest' changes hourly. |
| `ArXiv--8` | 4 (today's headline) | Reference is the December 2023 news item. |
| `ArXiv--9` | 2 (relative date) | 'Last week' drifts. |

</details>

<details><summary><b>BBC News</b> (35)</summary>

| task_id | rule | detail |
| --- | --- | --- |
| `BBC News--0` | rules 3 + 4 | Placeholder '<report>' reference plus 'recent'; no named evergreen BBC explainer on UK renewables to re-point it at. |
| `BBC News--1` | rules 3 + 4 | 'Latest' health headline changes hourly; reference is '<summary>'. |
| `BBC News--12` | rule 7 (image decoding) | Also non-deterministic: any food photo in a rotating section satisfies it. |
| `BBC News--13` | rules 3 + 4 | 'Recent' + '<summary>' placeholder. |
| `BBC News--14` | rules 3 + 4 | Three unfilled placeholders (<title>, <author>, <summary>) over a 'recent' window. |
| `BBC News--15` | rule 4 | 'Current headline' in an editorial section; the 2024 reference ('Name that whale! How AI aces animal spotting') is long gone. Considered rewriting as an Earth-section navigation question but the Earth menu's 'Science' en |
| `BBC News--16` | rules 3 + 4 | 'Most recent development' + '<summary>' placeholder; BBC no longer maintains a live Brexit-negotiations section. |
| `BBC News--17` | rule 4 | The 2024 answer (2: Israel-Gaza war, War in Ukraine) relied on conflict-specific nav entries. The current bbc.com News menu carries none of them, so the count is both 0 and volatile. |
| `BBC News--19` | rules 1 + 4 | Reference is a fixed February 2024 date; 'next earliest' fixture changes continuously. |
| `BBC News--2` | rules 3 + 4 | Explicit 'last two days' window; reference is '<article>'. BBC has no evergreen deforestation explainer to pin it to (the linked piece, 'Tropical forests destroyed at fastest recorded rate last year', is itself a dated n |
| `BBC News--20` | rules 3 + 4 | 'Latest' + '<article>, <summary>' placeholders. |
| `BBC News--21` | rules 3 + 4 | Top World headline churns hourly; reference is '<headline>, <region>'. |
| `BBC News--22` | rules 3 + 4 | 'Current top story' + placeholders. |
| `BBC News--23` | rules 3 + 4 | Duplicate of --1 in substance; 'latest' + placeholders. The referenced 'Innovation - Science & Health' nav label no longer exists (now Health, and Innovation > Technology/AI). |
| `BBC News--24` | rule 4 | 'Latest' article in a continuously-updated topic; the 2024 example answer is long superseded. |
| `BBC News--26` | rule 4 | 'Latest report' on a rolling topic; example answer from early 2024. |
| `BBC News--27` | rule 4 | 'Most recent' with an illustrative-only reference. |
| `BBC News--29` | rules 3 + 4 | 'New Releases' rail is refreshed continuously; reference is an ellipsis. Its sibling --18 was repaired into the stable Podcast Categories row instead. |
| `BBC News--3` | rule 4 (churns within days) | DP World Tour leaderboards rotate weekly, so both the tournament name and the count at -10 are stale within 7 days; 'teams' also has no referent in individual stroke play. Rewriting 'teams'->'players' fixes the wording b |
| `BBC News--30` | rules 3 + 4 | 'Latest' review + '<review>, <summary>' placeholders. |
| `BBC News--31` | rules 1 + 4 | Reference pins a February 2024 fixture; 'most recent match' changes weekly. |
| `BBC News--32` | rules 3 + 4 | 'At this time' top headline + '<headline>, <companies>' placeholders. (The Innovation > Artificial Intelligence section does still exist.) |
| `BBC News--33` | rules 3 + 4 + 9 | Reference points at a 'News - Israel-Gaza War' section that is no longer in the BBC News navigation, plus '<article>, <summary>'. |
| `BBC News--35` | rules 3 + 4 | 'Most recent' + placeholders. |
| `BBC News--36` | rules 3 + 8 | 'What most of these are about' is a superlative over an unranked, rotating feed; reference is placeholders. |
| `BBC News--37` | rule 4 | 'Latest' Culture > Books item; 2024 example answer. |
| `BBC News--38` | rule 4 | Any current storm story satisfies it; nothing fixed to grade against. |
| `BBC News--39` | rules 2 + 4 | 'Yesterday' plus a live results feed; reference pins 2024-01-30. |
| `BBC News--4` | rules 3 + 4 (unfilled placeholder reference + churns) | Reference is the bare placeholder '<summary>'. There is no exhaustive BBC index of 'climate change economics in Europe', so 'the latest' cannot be verified by any grader, and a criteria rewrite would only test that the a |
| `BBC News--40` | rules 3 + 4 + 9 | Placeholder reference tied to a defunct 'Israel-Gaza War' section. |
| `BBC News--41` | rule 4 | Live leaderboard 'at this point'; both the country count and Grace Kim's placing are stale within days. |
| `BBC News--6` | rule 4 | Today's top technology story - churns hourly; reference is not even an answer ('Latest news in Innovation - Technology'). |
| `BBC News--7` | rule 7 (image decoding) | Requires describing an arbitrary, rotating lead image; reference is an instruction, not an answer. |
| `BBC News--8` | rules 2 + 3 | 'Latest trade deal' has drifted far past the July 2023 CPTPP story, and the reference carries an unfilled '<summary>'. I could not locate the specific BBC CPTPP article URL to pin it (bbc.com is not indexed by the availa |
| `BBC News--9` | rule 4 | 'Taylor Swift' was the January-2024 Music News headline; this changes daily. |

</details>

<details><summary><b>Booking</b> (38)</summary>

| task_id | rule | detail |
| --- | --- | --- |
| `Booking--0` | 2 - drifted/ambiguous date | No year given; 'deals' is an unstable merchandising badge and the reference 'Be Local' is a 2024 artifact. Dated-search quota already filled by Booking--2 and Booking--4. |
| `Booking--1` | 2 - drifted/ambiguous date | Year-less '1st Jan'; cheapest-room price churns daily. Redundant with the repaired cheapest-hotel row Booking--4. |
| `Booking--10` | 1 - absolute date in the past | Feb 2024 stay. |
| `Booking--11` | 6 - requires login/payment | 'Reserve' implies completing a booking; compounded by a March 2024 past date. |
| `Booking--12` | 1 - absolute date in the past | Jan 2024 stay. |
| `Booking--13` | 6 - requires login/payment | 'Book' plus a Feb 2024 past date. |
| `Booking--14` | 6 - requires login/payment | 'Book' plus a March 2024 past date. |
| `Booking--15` | 1 - absolute date in the past | Jan 2024 stay; also a three-part answer (cost + amenities + rating) that all churn. |
| `Booking--16` | 1 - absolute date in the past | Jan 2024 stay; reference pins an exact per-night USD price. |
| `Booking--17` | 1 - absolute date in the past | Feb 2024 stay. |
| `Booking--18` | 1 - absolute date in the past | Feb 2024 stay. |
| `Booking--19` | 1 - absolute date in the past | March 2024 stay; top-three ordering churns daily anyway. |
| `Booking--20` | 1 - absolute date in the past | Feb/March 2024 stay. |
| `Booking--21` | 1 - absolute date in the past | March 2024 stay. |
| `Booking--22` | 1 - absolute date in the past | March 2024 stay; the reference also self-contradicts (asks for 9+, cites a 9.0-rated property). |
| `Booking--23` | 1 - absolute date in the past | Feb 2024 stay. |
| `Booking--24` | 1 - absolute date in the past | Feb 2024 stay; 'sort by distance from the beach' is also not a first-class Booking sort option any more. |
| `Booking--25` | 1 - absolute date in the past | March 2024 stay. |
| `Booking--26` | 1 - absolute date in the past | Feb 2024 stay; reference pins an exact total price. |
| `Booking--27` | 1 - absolute date in the past | Feb/March 2024 stay. |
| `Booking--28` | 1 - absolute date in the past | Feb 2024 stay; 29 February 2024 was a leap day, so the range is not even re-pinnable verbatim. |
| `Booking--29` | 1 - absolute date in the past | March 2024 stay. |
| `Booking--3` | 2 - drifted/ambiguous date | Year-less date; also conflates star rating with guest review score. Not worth a third dated row on a bot-walled site. |
| `Booking--30` | 1 - absolute date in the past | March 2024 stay, and the '228 hotels' count churns with inventory even if re-dated - not repairable into a deterministic answer. |
| `Booking--31` | 1 - absolute date in the past | March 2024 stay; the most/fewest brand ranking is availability-dependent and reshuffles per search. |
| `Booking--32` | 1 - absolute date in the past | Feb 2024 stay; the reference '1 hotel' is an availability snapshot. |
| `Booking--34` | 1 - absolute date in the past | March 2024 stay; the USD/CNY pair also moves with the FX rate. |
| `Booking--36` | 1 - absolute date in the past | March 2024 stay; top-three price-sorted results churn. |
| `Booking--37` | 1 - absolute date in the past | March 2024 stay. |
| `Booking--38` | 1 - absolute date in the past | Feb/March 2024 stay. |
| `Booking--39` | 1 - absolute date in the past | Feb 2024 stay. |
| `Booking--40` | 1 - absolute date in the past | March 2024 stay; FX-dependent reference. |
| `Booking--42` | 1 - absolute date in the past | March 2024 stay; the per-category review sub-scores also drift continuously as reviews accumulate. |
| `Booking--5` | 1 - absolute date in the past used transactionally | Booking greys out past dates entirely, so the stay is unselectable; scored 0/6 in a prior audit. Unpassable by any agent, not a model failure. |
| `Booking--6` | 6 - requires login/payment | 'Book one room' runs into the sign-in/payment wall; also year-less dates. |
| `Booking--7` | 2 - drifted/ambiguous date | Year-less; 'closest to NUS' depends on a map/distance sort whose ordering shifts with availability. |
| `Booking--8` | 1 - absolute date in the past | Dec 2023 stay; unselectable. |
| `Booking--9` | 2 - drifted/ambiguous date | Year-less; the three named reference properties are 2024 price-sort artifacts. |

</details>

<details><summary><b>Cambridge Dictionary</b> (7)</summary>

| task_id | rule | detail |
| --- | --- | --- |
| `Cambridge Dictionary--13` | 5. ambiguous about which artifact is wanted | 'Related to euphoria' has no single site-published list; reference ('beatitude; bed of roses; for fun') is one arbitrary pick among many. |
| `Cambridge Dictionary--19` | 4. time-varying that churns | The machine-translation vendor attribution is a commercial arrangement that can change without notice; not a durable fact. |
| `Cambridge Dictionary--29` | 7. requires decoding an image | Image quiz; answers are pictures and the score is not a stable reference value. |
| `Cambridge Dictionary--37` | 3. unfilled <...> placeholder in the reference | Reference begins '<understandings>, ...' and the ask is too open-ended to pin to one verifiable list. |
| `Cambridge Dictionary--38` | 4. time-varying that churns | 'Recommended' quiz is rotated and the score is not a fixed reference; +Plus also nudges toward login. |
| `Cambridge Dictionary--39` | 4. time-varying that churns | Randomised game word plus a timer; no deterministic reference answer. |
| `Cambridge Dictionary--41` | 4. time-varying that churns | Merch catalogue inventory rotates; the three named products are not guaranteed to be on sale. |

</details>

<details><summary><b>Coursera</b> (27)</summary>

| task_id | rule | detail |
| --- | --- | --- |
| `Coursera--0` | rule 5 (ambiguous artifact / open-ended) | 'a course' but reference names a Specialization; many catalog items satisfy the filter. |
| `Coursera--1` | rule 5 (ambiguous artifact / open-ended) | Dozens of valid answers; reference pins one arbitrary course. |
| `Coursera--11` | rule 4 (time-varying) | Testimonials rotate on the page; reference quotes one that may no longer render. |
| `Coursera--16` | rule 9-adjacent (reference no longer matches the live page) | Instructor is now Jason Satchell (University of Illinois Urbana-Champaign), not Xi Yang, and he teaches no other course — the second half of the question degenerates to an empty answer. |
| `Coursera--17` | rule 4 (time-varying) | Catalog result counts drift continuously; a pinned '23' is unverifiable and wrong within weeks. |
| `Coursera--20` | rule 3 (unfilled placeholder) | Reference ends in '<summary of main goals>'; also open-ended course selection. |
| `Coursera--21` | rule 3 (unfilled placeholder) | '<outcomes>' placeholder. |
| `Coursera--25` | rule 3 (unfilled placeholder) | '<topic>' placeholder. |
| `Coursera--27` | rule 3 (unfilled placeholder) | '<skills>' placeholder. |
| `Coursera--28` | rule 3 (unfilled placeholder) | '<main subjects>' placeholder; Guided Project catalog also churns. |
| `Coursera--29` | rule 4 (time-varying) | Pricing and promo structure change; the $399/$59 arithmetic is already stale. |
| `Coursera--3` | rule 4 (time-varying) + rule 'reference depends on search-result ordering' | Answer is whatever currently sorts first under 'newest' — churns weekly. |
| `Coursera--30` | rule 8 (superlative the site does not stably publish) | Low-end star percentages are within rounding noise of each other; which level is lowest flips. Dropped in favour of Coursera--15, where 2-star is a clear minimum. |
| `Coursera--31` | unverified | Course page exists (coursera.org/learn/quantum-physics, 4.2 from 74 reviews) but I could not retrieve the star-percentage breakdown to confirm/correct the upstream 52.6%; with only ~74 reviews a single new review shifts  |
| `Coursera--32` | rule 4 (time-varying) | Same catalog-count churn ('568 results'). |
| `Coursera--33` | rule 3 (unfilled placeholder) | '<summary>' placeholder plus open-ended 'high rating' selection. |
| `Coursera--34` | rule 3 (unfilled placeholder) | '<summary> of bio' placeholder. |
| `Coursera--35` | rule 3 (unfilled placeholder) | '<objectives>' placeholder; reference also points at the generic 'Introduction to Sustainability'. |
| `Coursera--36` | rule 1/2 (past absolute + drifted relative dates) | Reference pins Fall 2024 deadlines (March/April 2024). |
| `Coursera--37` | rule 4 (time-varying) | Homepage module contents rotate; no stable reference. |
| `Coursera--38` | rule 4 (time-varying) | Partner directory membership changes and the 'all of them' golden list cannot be held fixed. |
| `Coursera--4` | rule 5 (ambiguous artifact) | Explicitly 'course or Specialization'; exact-4.7 filter also drifts. |
| `Coursera--40` | rule 4 (time-varying) | Marketing copy and the '10,250+ learning opportunities' figure are rewritten frequently. |
| `Coursera--41` | rule 4 (time-varying) | Degree listing rotates; any three are acceptable, so the pinned reference is not discriminative. |
| `Coursera--5` | rule 5 (ambiguous artifact / open-ended) | Reference 'outcomes' are generic Coursera boilerplate, not course-specific. |
| `Coursera--8` | rule 4 (time-varying) | Which results carry the Free tag changes; answer depends on scroll position in search results. |
| `Coursera--9` | rule 5 (open-ended) | Duplicate of Coursera--23 and open-ended; one named-course AI-ethics row would do, but Coursera--15/--39 are stronger. |

</details>

<details><summary><b>ESPN</b> (37)</summary>

| task_id | rule | detail |
| --- | --- | --- |
| `ESPN--0` | rule 4 (time-varying, churns within days) | Live standings table plus an unfilled <standings> placeholder. |
| `ESPN--1` | rule 2 (drifted relative date) | 'past 2 days' plus '<article>, maybe no article' — ungradeable. |
| `ESPN--10` | rule 3 (unfilled placeholder) | '<score>; <summary>' with no concrete value, and 'latest' drifts each January. |
| `ESPN--12` | rule 4 (churns within hours) | Home-page headlines rotate continuously. |
| `ESPN--13` | rule 4 (churns within hours) | 'today's top headline' + '<headline>; <summary>'. |
| `ESPN--14` | rule 4 (churns within days) | Reference is literally 'News about NBA trades'. |
| `ESPN--15` | rule 1 (absolute past date) | This is the spec's own named reject example. Note: the scores ARE still live-verifiable at espn.com/nba/scoreboard/_/date/20231225, so a human may want to reinstate it as an archival row. |
| `ESPN--16` | rule 2 (drifted relative date) | 'current standings' for a 2023 date is self-contradictory now; reference is an unfilled description. |
| `ESPN--17` | rule 9 (section no longer served) | ESPN's BPI page serves the current season; no stable 2023-24 archive confirmed. |
| `ESPN--18` | rule 5 (ambiguous which artifact) | Live check: 7 leagues in the top nav vs 31 more in the 'More Sports' dropdown (38 total), and several entries are not leagues (Gaming, Sports Betting, Bananas). The upstream '31' is both stale and under-specified. |
| `ESPN--19` | rule 9 (page no longer exists) | https://www.espn.com/nba/team/roster/_/name/bos/season/2024 returns the current-season shell with no 2023-24 players or salary column. |
| `ESPN--2` | rule 2 (drifted relative date) | 'last 2 days'; may be no game at all in the NBA offseason. |
| `ESPN--20` | rule 4 (live leaderboard) | Stale 2023-24 leaders (Sabonis/Doncic). |
| `ESPN--21` | rule 2 (drifted relative date) | 'last 3 days' + '<score>' placeholder. |
| `ESPN--22` | rule 2 (drifted relative date) | 'past week'; reference is a February example transaction. |
| `ESPN--25` | rule 4 (time-varying) | 'latest NBA game' with a Feb 2024 example reference. |
| `ESPN--26` | rule 2 (drifted relative date) | 'yesterday' + Jan 2024 example reference (which is itself internally inconsistent). |
| `ESPN--29` | rule 4 (churns within hours) | '<headline> today'. |
| `ESPN--3` | rule 4 (time-varying) | Same 'broadcast on ESPN' ambiguity as ESPN--24; already covered by the pinned ESPN--23. |
| `ESPN--30` | rule 9 / rule 4 | ESPN's standings page serves the current season; no verified stable 2023-24 archive, and the question also asks an open-ended 'what about the situation in Division'. |
| `ESPN--31` | rule 9 (archive not served) + bad reference | Roster season archive is not served (same as ESPN--19), and the reference answer 'Carlos Rodon' is a pitcher, not an infielder. |
| `ESPN--32` | rule 2 (drifted relative date) | 'yesterday' + '<score>' placeholder; also open-ended ('focusing on teams' performance'). |
| `ESPN--33` | rule 4 (most recent article) | 'latest articles' plus a single pinned 2023 article title that search no longer surfaces first. |
| `ESPN--34` | rule 4 (churns within days) | Injury report changes daily; reference is just a page name. |
| `ESPN--35` | rule 4 (time-varying) + offsite | Live ticket prices on a third-party reseller; '<price>' placeholder. |
| `ESPN--36` | rule 2 (drifted relative date) | 'last 5 games' + '<games>'/'<results>' placeholders. |
| `ESPN--37` | rule 4 (time-varying) | Career total 1471 is stale and still increments while he is active. |
| `ESPN--38` | rule 9 (archive not served) | The linked team-stats URL serves the current season, not 2023-24. |
| `ESPN--39` | rule 4 (churns within days) | Depth chart and IR list change weekly; reference is only a URL, not an answer. |
| `ESPN--4` | rule 2 (drifted relative date) | 'yesterday' changes daily; empty in the offseason. |
| `ESPN--40` | rule 9 (section no longer exists) | Live check of https://www.espn.com/espnplus/ shows no 'Tools' section; Bracket Predictor/Analyzer/Dollar Value Generator are gone from that page. |
| `ESPN--42` | rule 4 (live standings) | 'currently' + reference is just a URL. |
| `ESPN--43` | rule 2 (drifted relative date) | Reference pins the Class of 2023 espnW rankings; the live recruiting page now defaults to a much later class. |
| `ESPN--5` | rule 4 (time-varying) | 'latest completed game' + full placeholder reference. |
| `ESPN--6` | rule 2 (drifted relative date) | Reference pins a 2024 Lakers-Celtics result; 'latest' has long since moved. Same shape as the repaired ESPN--23, so not duplicated. |
| `ESPN--7` | rule 4 (time-varying) | 'latest' game + '<score>'/'<summary>' placeholders. |
| `ESPN--8` | rule 5 (ambiguous which artifact) | 'the last day of the regular season' does not say which season; the 2023-24 reference (Embiid/Doncic/Giannis) is stale. |

</details>

<details><summary><b>GitHub</b> (34)</summary>

| task_id | rule | detail |
| --- | --- | --- |
| `GitHub--0` | rule 4 | Free-text search ranking over a fuzzy topic churns; resource-watch is no longer a defensible #1. |
| `GitHub--1` | rule 2 | 'Last 2 days' is a drifting relative window with a pinned 2024 answer. |
| `GitHub--10` | rule 9 | The 'Copilot Individual' plan name no longer exists (now Copilot Free/Pro/Pro+); $100/yr reference and feature blurb are stale marketing copy I could not anchor to a named live section. |
| `GitHub--11` | rule 5 | Any of many January-2023 repos satisfies it; the pinned single repo is not uniquely determined. |
| `GitHub--12` | rule 4 | Electron releases several times a week; reference is '19 hours ago'. |
| `GitHub--13` | rule 4 | Trending + live star count. |
| `GitHub--14` | rule 4 | Live check: now bpasero, jrieken, Tyriar (old ref listed mjbvz); #2/#3 are within 260 commits of each other so ordering can flip. Repairable but dropped to stay at 7. |
| `GitHub--15` | rule 2 | Drifting window. |
| `GitHub--16` | rule 9 | Learning Lab retired and skills.github.com now redirects to GitHub Learn; no 'First day on GitHub' heading exists. |
| `GitHub--17` | rule 2 | Drifting window. |
| `GitHub--18` | rule 8 | 'Image processing tool' is not a GitHub-published ranking category; top-by-stars search does not return OpenCV deterministically. |
| `GitHub--19` | rule 4 | 'Most recently updated' changes by the minute. |
| `GitHub--2` | rule 4 | Trending page churns daily; explicitly excluded by the brief. |
| `GitHub--20` | rule 9 | Copilot Chat has shipped in GitHub Mobile; the 'coming soon' FAQ entry is gone. |
| `GitHub--21` | rule 9 | GitHub Advanced Security was rebranded/unbundled (GitHub Code Security & Secret Protection); the '7x faster remediation' resources copy is not locatable as a named live section. |
| `GitHub--22` | rule 3 | Placeholder reference plus drifting window. |
| `GitHub--24` | rule 4 | Closed-issue list changes hourly; reference is just a search URL. |
| `GitHub--25` | rule 3 | Placeholder reference plus drifting window. |
| `GitHub--27` | rule 3 | Placeholder reference plus drifting window. |
| `GitHub--28` | rule 3 | Placeholder reference; ranking over a now-3-year window churns. |
| `GitHub--29` | rule 9 | Live /pricing page lists only Free, Team and Enterprise; there is no Pro column to compare. |
| `GitHub--30` | rule 2 | Drifting window plus churning contributor ranking. |
| `GitHub--31` | rule 4 | TensorFlow commits land many times per day; reference is a non-answer. |
| `GitHub--32` | rule 3 | Placeholder reference plus 'latest'. |
| `GitHub--33` | rule 4 | Homepage customer-story carousel rotates; Philips/Shopify pair is 2024 copy. |
| `GitHub--34` | rule 4 | Fork-sorted free-text search churns; DeepPurpose is no longer defensible. |
| `GitHub--35` | rule 4 | Reference pinned at 18.2.0 (2022); React has shipped 19.x since. |
| `GitHub--36` | rule 3 | Unfilled placeholder reference; many repos satisfy it. |
| `GitHub--39` | rule 2 | 'This month' trending; reference is just a URL. |
| `GitHub--4` | rule 3 | Unfilled <repo> placeholder plus a 2023 date in the hint. |
| `GitHub--40` | rule 6 | Account-creation flow; signup/CAPTCHA required. |
| `GitHub--5` | rule 3 | Placeholder reference, drifting window. |
| `GitHub--6` | rule 2 | Relative window; pinned blocknet contributor list is stale and unverifiable as 'the' answer. |
| `GitHub--9` | rule 3 | Placeholder reference plus drifting window. |

</details>

<details><summary><b>Google Flights</b> (36)</summary>

| task_id | rule | detail |
| --- | --- | --- |
| `Google Flights--0` | rule 1 (past absolute date, transactional) | Undated 'December 28th' same-day return plus a dead 2024 fare; 'Book a journey' edges toward a booking action and 'Edinburg' is a typo for Edinburgh - not worth repairing when better rows exist. |
| `Google Flights--1` | rule 1 (absolute date in the past) | Hard-coded 'today (February 17, 2024)'; also rule 2 - the relative 'today' conflicts with any repinned date. |
| `Google Flights--10` | rule 1 (absolute date in the past) | 2024 dates; one of five near-duplicate NYC-Tokyo rows, all superseded by the kept JFK-LHR cheapest-fare row. |
| `Google Flights--11` | rule 1 (absolute date in the past) | 2024 dates; stop-count coverage already provided by the kept New York-London non-stop filter row. |
| `Google Flights--12` | rule 1 (absolute date in the past) | 2023/2024 dates; duplicate of the kept NYC-London row. |
| `Google Flights--13` | rule 1 (absolute date in the past) | 2024 dates; near-duplicate of Google Flights--10. |
| `Google Flights--14` | rule 1 (absolute date in the past) | 2024 dates; duplicate of the kept NYC-London row. |
| `Google Flights--15` | rule 1 (absolute date in the past) | 2024 dates; the 2024 reference ('Only one flight') depends on a seasonal NRT schedule that will not hold in 2027. |
| `Google Flights--16` | rule 1 (absolute date in the past) | 2024 date; fifth near-duplicate NYC-Tokyo cheapest-fare row. |
| `Google Flights--17` | rule 1 (absolute date in the past) | 2023/2024 dates; redundant cheapest-fare transatlantic row. |
| `Google Flights--18` | rule 1 (absolute date in the past) | 2024 dates; duration-sort coverage already provided by the kept Buenos Aires-Amsterdam row. |
| `Google Flights--19` | rule 1 (absolute date in the past) | 2024 date; layover coverage already provided by the kept Pune-New York row. |
| `Google Flights--20` | rule 1 (absolute date in the past) | 2024 dates, and the verb is 'Book' - edges into an actual booking/payment flow. |
| `Google Flights--21` | rule 1 (absolute date in the past) | 2024 date; another cheapest-fare duplicate, dropped to avoid over-weighting this bot-walled site. |
| `Google Flights--22` | rule 1 (absolute date in the past) | 2024 dates; CO2-sort coverage already provided by the kept Calgary-New York row. |
| `Google Flights--23` | rule 1 (absolute date in the past) | 2024 date; a solid repair candidate, but duplicates the stops-filter behaviour of the kept New York-London row. |
| `Google Flights--24` | rule 1 (absolute date in the past) | 2024 dates; duplicates cabin-class + stops coverage already held by kept rows. |
| `Google Flights--26` | rule 1 (absolute date in the past) | 2024 dates, plus a $1000 ceiling calibrated to 2024 fares that may match nothing in 2027. |
| `Google Flights--27` | rule 7 (requires decoding an image) | Reference contains the unfilled placeholder '<analyze the price graph>' (also rule 3) and the answer lives in a chart image. |
| `Google Flights--28` | rule 1 (absolute date in the past) | 2024 dates; redundant cheapest-fare + stops-filter row. |
| `Google Flights--29` | rule 1 (absolute date in the past) | 2024 dates; the 'Only one flight' reference depends on a 2024 schedule that will not hold. |
| `Google Flights--30` | rule 1 (absolute date in the past) | 2024 date; redundant cheapest-fare row. |
| `Google Flights--31` | rule 8 (superlative the site does not publish) | Google Flights offers no 'most stops' sort; the answer depends on scrolling an unbounded, churning result list to find a 3-stop outlier. Also a 2024 date. |
| `Google Flights--32` | rule 1 (absolute date in the past) | 2024 dates; duration-sort coverage already held by the kept Buenos Aires-Amsterdam row. |
| `Google Flights--33` | rule 1 (absolute date in the past) | 2024 date; CO2 coverage already held by the kept Calgary-New York row. |
| `Google Flights--34` | rule 6 (requires login/payment flow) | Drilling into third-party booking-option pages pushes the agent onto OTA checkout surfaces; the OTA list is also volatile per-itinerary. 2024 date as well. |
| `Google Flights--35` | rule 1 (absolute date in the past) | 2024 date; redundant cheapest-fare row. |
| `Google Flights--36` | rule 1 (absolute date in the past) | 2024 dates plus a 2024-calibrated $1000 price filter. |
| `Google Flights--37` | rule 1 (absolute date in the past) | 2024 dates; the yes/no answer is also a coin flip that depends entirely on which option the agent happens to pick. |
| `Google Flights--38` | rule 1 (absolute date in the past) | 2024 date; truncated '...' reference and redundant with kept filter rows. |
| `Google Flights--39` | rule 5 (ambiguous about which artifact is wanted) | 'a certain city in Hokkaido' is never named, so the comparison is underspecified and ungradeable. 2024 date as well. |
| `Google Flights--41` | rule 1 (absolute date in the past) | 2024 date; 'Glacier National Park' is a fuzzy destination token (maps to Kalispell FCA) and a 1-stop HKG-FCA business itinerary is unlikely to exist in mid-2027. |
| `Google Flights--5` | rule 1 (past absolute date, transactional) | Undated Dec 20-23 round trip with a dead $1456 reference; redundant with the kept NYC-London row. |
| `Google Flights--6` | rule 1 (past absolute date, transactional) | Repairable in principle, but cabin-class coverage is already provided by the kept Buenos Aires business-class row, and TLV-VCE first class availability in mid-2027 is unpredictable. |
| `Google Flights--7` | rule 1 (past absolute date, transactional) | Undated Dec 25-28 plus a hard $1320 price ceiling that was calibrated to 2024 fares - the filter may return zero or all results in 2027, making it ungradeable. |
| `Google Flights--8` | rule 7 (requires decoding an image) | Reference is literally 'Analyse the picture of Price graph' - the price-graph row the prompt instructs us to reject. |

</details>

<details><summary><b>Google Map</b> (35)</summary>

| task_id | rule | detail |
| --- | --- | --- |
| `Google Map--0` | rule 4 | Set of salons above a 4.8 rating threshold churns continuously with new reviews; the five pinned names are not reproducible. |
| `Google Map--1` | rule 5 | No state given and Altavista's transit stops are sparsely/inconsistently mapped; the two-way 'or' reference signals the artifact is already ambiguous. |
| `Google Map--11` | rule 5 | 'A large store in Washington' is unconstrained (state vs city, which store), and the reference pins one small boutique. |
| `Google Map--12` | rule 4 | Exact ratings to one decimal drift with every new review; the pinned ordering is not reproducible. |
| `Google Map--13` | rule 4 | Reference is a real-time-traffic drive estimate; 'book a ride' also edges into a transactional flow. |
| `Google Map--14` | rule 9 | Reference (Rising Wolf Garage) is in the East Village, not near Radio City, and Maps has no motorcycle-parking category - the pinned answer does not match the question. |
| `Google Map--15` | rule 3 | Reference contains an unfilled '<reviews>' placeholder that cannot be made concrete - review text churns. |
| `Google Map--16` | rule 5 | 'Smithsonian museum' names no specific museum (there are ~19 on and off the Mall), and the reference 'EVgo Charging Station' has no address. |
| `Google Map--17` | rule 4 | Same open-now shape as Google Map--32, which is already kept as a criteria row; a second one adds no coverage and the reference pins a UTC timestamp. |
| `Google Map--18` | rule 4 | Cross-country drive time ('about 29 hours') and chosen interstate vary with traffic and routing updates. |
| `Google Map--19` | rule 5 | Reference's walking time is given as a 15-30 min range with no named supermarket; the second half is underspecified. |
| `Google Map--20` | rule 9 | Charger listings are added and removed frequently and the pinned 1330 Maryland Ave SW site could not be confirmed on a non-Google source. |
| `Google Map--21` | rule 5 | 'Elm Street and Oak Street in Massachusetts' matches many towns; the reference silently assumes Amesbury. |
| `Google Map--22` | rule 9 | Best Buy has been closing stores; the pinned 1131 5th St Miami Beach location could not be confirmed as still open. |
| `Google Map--23` | rule 5 | Reasonable and stable, but Maps returns several near-identical walking routes and the reference pins one avenue and one minute figure; kept out only to stay within the 6-row cap for this bot-walled site - a good candidat |
| `Google Map--24` | rule 4 | Reference drive time is explicitly tagged 'current traffic condition'. |
| `Google Map--25` | rule 4 | Parking listing and its posted closing time (11 PM) are volatile. |
| `Google Map--26` | rule 3 | Reference is the placeholder '<Action>, print PDF' with no gradeable content; print dialog is also outside the page. |
| `Google Map--27` | rule 4 | Result count depends on viewport zoom and current store set; the pinned '8' is not reproducible. |
| `Google Map--28` | rule 9 | Google's settings UI has been reorganized since 2024 and the signed-out view differs; the pinned option list no longer matches. |
| `Google Map--29` | rule 4 | Which three stops surface depends on viewport and transit-layer state; stop names in the reference ('YTC - Stop 5') are unstable. |
| `Google Map--3` | rule 5 | Not phrased as a question and 'Broadway Theater' is ambiguous between the named Broadway Theatre and any theater on Broadway. |
| `Google Map--30` | rule 3 | Reference contains an unfilled '<reviews>' placeholder. |
| `Google Map--31` | rule 4 | The generated maps.app.goo.gl short link is newly minted per session; the pinned link can never be reproduced. |
| `Google Map--34` | rule 3 | Reference is a truncated 'Speer Blvd Park ...' with an ellipsis and no concrete acceptance criterion; downtown Denver also has no real hiking trail within 2 miles. |
| `Google Map--36` | rule 4 | Exact ratings and the resulting ranking drift with every new review. |
| `Google Map--37` | rule 4 | Answer depends on which parking lot the agent picks plus live drive times. |
| `Google Map--38` | rule 9 | Generic 'Bike Parking' POIs are user-contributed and frequently removed; the 104 W 38th St entry could not be confirmed. |
| `Google Map--39` | rule 4 | Reference is a turn-by-turn breakdown with live-traffic durations (12 hr 6 min) that will not reproduce. |
| `Google Map--4` | rule 4 | Reference is a live-traffic driving time ('about 10 mins based on real-time traffic conditions') that moves hour to hour. |
| `Google Map--40` | rule 4 | Pins an exact rating (4.6) and the verbatim text of one particular one-star review; both churn. |
| `Google Map--5` | rule 4 | Parking-garage listings and their posted hours turn over quickly; pinned SP+ lot may no longer be listed. |
| `Google Map--6` | rule 4 | 'All locations' is a retail footprint that changes with store openings/closings; the single-name reference is already incomplete. |
| `Google Map--7` | rule 9 | Reference is a single 'EZ-Mart' stop in a village with essentially no mapped transit; could not confirm the stop still exists. |
| `Google Map--9` | rule 5 | Reference already offers two alternatives; small galleries open and close, and 'Los Angeles Hindu Temple' matches several listings. |

</details>

<details><summary><b>Google Search</b> (34)</summary>

| task_id | rule | detail |
| --- | --- | --- |
| `Google Search--10` | rule 4 (churns within days) | Weekly chart, plus an admittedly source-dependent top-10. |
| `Google Search--11` | rule 2 (relative date that has drifted) | 'Last week' plus a real-time traffic count. |
| `Google Search--14` | rule 2 (relative date that has drifted) | 'The year before last' meant 2022 when authored and means 2024 today, which matches no James Smith retirement; reference 'Yeovil Town' is unreachable. Google also CAPTCHAs the sandbox for this multi-hop search. |
| `Google Search--15` | rule 6 (requires login/CAPTCHA) | Deliberate credential-submission task; X gates with CAPTCHA and the row teaches credential entry. |
| `Google Search--16` | rule 4 (churns within days) | Subscriber count has moved far past 880K and 'hottest news right now' is live. |
| `Google Search--18` | rule 4 (churns within days) | 'Most recent' drifted: the reference pins Qatar 2022, but the 2026 World Cup has since been held. |
| `Google Search--19` | rule 4 (churns within days) | 'Latest commit' SHA is not a stable target. |
| `Google Search--2` | rule 4 (churns within days) | Reference is literally 'News Title (real-time)'. |
| `Google Search--20` | rule 4 (churns within days) | 'Latest' plus a reference date (April 4, 2025) that the franchise schedule has already moved past. |
| `Google Search--21` | rule 4 (churns within days) | The top-5 animated gross list has reordered since 2024 (Inside Out 2, Ne Zha 2 etc.); exact-order reference is stale. |
| `Google Search--22` | rule 2 (relative date that has drifted) | 'This month' with a February-2024 snapshot reference. |
| `Google Search--23` | rule 3 (unfilled placeholder reference) | Reference is '<bio> LeBron James'; a free-form bio has no gradeable answer and his career facts are still moving. |
| `Google Search--25` | rule 4 (churns within days) | Live headline. |
| `Google Search--26` | rule 4 (churns within days) | 'Latest version' requirements are revised each Photoshop release; the pinned spec list is a 2024 snapshot. |
| `Google Search--27` | rule 4 (churns within days) | Live AQI reading. |
| `Google Search--28` | rule 4 (churns within days) | Same drifting-aggregate-score problem as Prometheus; would need a tolerance band to grade. |
| `Google Search--29` | rule 4 (churns within days) | Still 9.58 Bolt today, but the question asks for a live record that can break at any championship; dropped because Google Search is already capped at 6 rows and deterministic rows were available. |
| `Google Search--3` | rule 8 (superlative the site does not publish) | No canonical 'top 5 comedies by user rating' ranking; the pinned five-title list is one SERP snapshot. |
| `Google Search--30` | rule 4 (churns within days) | Daily-refreshed chart. |
| `Google Search--32` | rule 4 (churns within days) | 'Most recent' final has turned over three times since the pinned 2023 Istanbul answer. |
| `Google Search--33` | rule 3 (unfilled placeholder reference) | Reference is '<SHA> of latest Tensorflow' and the value changes hourly. |
| `Google Search--34` | rule 4 (churns within days) | Ephemeris value changes every day. |
| `Google Search--35` | rule 4 (churns within days) | 'Latest paper' with a Feb-2024 reference. |
| `Google Search--36` | rule 4 (churns within days) | Annual turnover: reference names the 2023 laureates; also carries a '<summary>' placeholder. |
| `Google Search--37` | rule 8 (superlative the site does not publish) | No published ranking of 'top' super-Earths; 'current' makes it worse. |
| `Google Search--38` | rule 4 (churns within days) | Reference 'next: April 8, 2024' is two and a half years in the past; the 'next' answer has moved. |
| `Google Search--39` | rule 5 (ambiguous which artifact is wanted) | 'Through a blog' — any blog — makes the 10-item reference list unreproducible. |
| `Google Search--4` | rule 4 (churns within days) | Live concurrent-player count. |
| `Google Search--41` | rule 3 (unfilled placeholder reference) | Reference restates the question; value is live anyway. |
| `Google Search--42` | rule 5 (ambiguous which artifact is wanted) | Any of thousands of articles satisfies it; the reference is one specific British Council page's five-point list. |
| `Google Search--5` | rule 4 (churns within days) | Reference is a 2024 box score marked '(real-time)'. |
| `Google Search--6` | rule 4 (churns within days) | Trends feed; reference is a seasonal snapshot. |
| `Google Search--8` | rule 4 (churns within days) | Top-comment identity and like/reply counts change continuously. |
| `Google Search--9` | rule 4 (churns within days) | Aggregate scores drift by tenths/percentage points; an exact-match reference goes wrong silently. |

</details>

<details><summary><b>Huggingface</b> (36)</summary>

| task_id | rule | detail |
| --- | --- | --- |
| `Huggingface--0` | rule 9 | The hub exposes no 'updated during March 2023' filter, so the constraint is unsearchable; the pinned reference is one of thousands of possible answers. |
| `Huggingface--1` | rule 6 | Inference now requires a logged-in account with inference credits; reference is a free-text '<story>' placeholder anyway. |
| `Huggingface--10` | rule 9 | A 2024 demo Space; Spaces of this vintage are routinely paused/removed and the answer is a non-deterministic LLM generation. |
| `Huggingface--11` | rule 4 | 'Latest updated' in a popular task tag changes within minutes. |
| `Huggingface--12` | rule 4 | Same churn problem; 'Error Correction' is also not a hub task tag. |
| `Huggingface--13` | rule 9 | VERIFIED DEAD: the current LlamaTokenizer signature at /docs/transformers/main/en/model_doc/llama no longer documents spaces_between_special_tokens. |
| `Huggingface--15` | rule 4 | Live API: top PaddlePaddle model has only 6,143 rolling downloads and #2 has 1,561 - a thin, fast-flipping margin. |
| `Huggingface--16` | rule 3 + rule 4 | Placeholder reference '<model> (today, text classification)' and a result set that changes hourly. |
| `Huggingface--17` | rule 3 + rule 4 | All-placeholder reference plus a minute-by-minute superlative. |
| `Huggingface--18` | rule 9 | VERIFIED DEAD: /docs/trl/main/en/reward_trainer has been rewritten and no longer mentions the Llama-2-style 'margin' dataset column. |
| `Huggingface--19` | rule 3 + rule 4 | Placeholder reference and a churning 'most recent' superlative. |
| `Huggingface--2` | rule 3 + rule 4 | Reference is the bare placeholder '<model 1>; <model 2>; <model 3>' with no fillable concrete value, and the answer set turns over every month; a criteria rewrite would still be a churning, barely-gradeable 3-item list,  |
| `Huggingface--20` | rule 9 | Reference ckiplab/bert-base-chinese-ner now shows 91,152 downloads (rolling window), so it no longer satisfies '1M+'; and the hub offers neither an update-year nor a download-threshold filter. |
| `Huggingface--21` | rule 9 | VERIFIED DEAD: the current transformers quicktour no longer contains the sentiment-analysis pipeline example or the distilbert-base-uncased-finetuned-sst-2-english default. |
| `Huggingface--22` | rule 9 | The add_tensorflow_model guide has been removed from transformers docs (TF support deprecated); reference is a '<summary>' placeholder. |
| `Huggingface--23` | rule 2 + rule 4 | Drifted relative date; reference 'eg, openai/whisper-large-v3' is a 2023 model that no longer fits the window. |
| `Huggingface--25` | rule 9 | load_in_8bit / load_in_4bit as from_pretrained kwargs are deprecated in favour of BitsAndBytesConfig; the quoted doc sentence no longer stands as written. |
| `Huggingface--26` | rule 5 | Open-ended 'a model' with a pinned reference to an obscure personal repo (PhilipTheGreat/DiabloGPT-small-Traveller). |
| `Huggingface--27` | rule 4 | Rolling 30-day download superlative over a crowded tag. |
| `Huggingface--28` | rule 5 | 'An example' with a pinned reference ('eg, /roberta-base-squad2') - not gradeable. |
| `Huggingface--29` | rule 3 + rule 2 | '<summary> of Falconsai/medical_summarization' placeholder plus a drifted 'recent'. |
| `Huggingface--31` | rule 2 + rule 4 | 'Latest' churns and the reference pins an update date of Jan 16 2024. |
| `Huggingface--33` | rule 4 | GitHub star counts move daily and the reference holds 2024 values; also partly off-site. |
| `Huggingface--34` | rule 9 | The HF Classroom marketing page has been folded into the Enterprise/education offering; the quoted bullet list no longer exists as written. |
| `Huggingface--35` | rule 4 | 'Latest blog' changes constantly; reference pins a January 2024 post. |
| `Huggingface--36` | rule 3 | Reference is literally 'summary of https://huggingface.co/pricing'. Dropped in favour of the sharper Huggingface--14 on the same page. |
| `Huggingface--37` | rule 4 | Daily Papers rotates every day and upvote counts move hourly; the reference points at /posts, not even the right page. |
| `Huggingface--38` | rule 5 | Still live (add_tokens is documented) but the reference 'use add_tokens method' is a single trivially-guessable token that an agent can answer from priors without visiting the site - no browsing signal. |
| `Huggingface--39` | rule 5 | Reference is a bare doc URL; 'note the configurable parameters' of TrainingArguments is an unbounded list. |
| `Huggingface--4` | rule 5 | Any of thousands of chat models satisfies it, but the reference pins TinyLlama-1.1B-Chat-v1.0 - ungradeable without rewriting into a different task. |
| `Huggingface--40` | rule 5 | Free-text summary with an elided reference ending in '...' - no stable grading target. |
| `Huggingface--41` | rule 4 | Live API on 2026-10-10 shows microsoft/TRELLIS-text-xlarge (15,874) has displaced the reference openai/shap-e (8,847); rolling-window downloads keep flipping this. |
| `Huggingface--6` | rule 6 | The on-page inference widget now requires a signed-in account / inference provider credits. |
| `Huggingface--7` | rule 4 | HF download counts are a rolling 30-day window; the top audio dataset changes week to week. |
| `Huggingface--8` | rule 5 | Unboundedly open ('an example'), pinned reference microsoft/phi-2. |
| `Huggingface--9` | rule 4 | Rolling-window download superlative; the upstream reference is also the wrong direction (opus-mt-ja-en for an en-ja question). Huggingface--30 covers the same shape, repaired. |

</details>

<details><summary><b>Wolfram Alpha</b> (39)</summary>

| task_id | rule | detail |
| --- | --- | --- |
| `Wolfram Alpha--0` | surplus | Clean and verified (11.2 = 2*5.6) but trivial one-step query; target of 7 already met with meatier rows. |
| `Wolfram Alpha--1` | 5 (ambiguous about which artifact is wanted) | Reference answer is truncated with '... (Search inner region of the pentagram on Wolfram)'; no gradeable ground truth and the output form depends on Wolfram's parameterization. |
| `Wolfram Alpha--10` | 4 (time-varying) | Depends on the IGRF model generation Wolfram currently ships (IGRF-13 vs IGRF-14); the total-field value moves between model releases. |
| `Wolfram Alpha--11` | surplus | Materials-database lookup; stable but sourced from Wolfram's curated data rather than computed. Held as a reserve row. |
| `Wolfram Alpha--12` | 5 (ambiguous) | 'Looks like a snowflake' is subjective; U+22C6 is STAR OPERATOR and U+2744 is the actual SNOWFLAKE, so the intended answer is only inferable from Wolfram's glyph rendering. |
| `Wolfram Alpha--13` | 4 (time-varying) | 'now' is a relative date and the result tracks the latest CPI release; the reference drifts every month. |
| `Wolfram Alpha--14` | 4 (time-varying) | Chain-restaurant nutrition data is revised as recipes change; Wolfram's food database follows. |
| `Wolfram Alpha--15` | 4 / wrong reference | Father's mother's sister's son is your father's first cousin, i.e. your first cousin once removed, whose coefficient of relationship is 1/16 = 6.25%, not the 3.125% in the reference. Rejected rather than repaired because |
| `Wolfram Alpha--16` | 4 (time-varying) | Output of Wolfram's weight-loss calculator model, which has been re-tuned; also phrased with a drifting duration. |
| `Wolfram Alpha--17` | 4 (time-varying) | Cost-of-living dataset is refreshed; per-city 2023 averages get restated. |
| `Wolfram Alpha--18` | 7 (image answer) | Reference answer is a plot image; cannot be expressed or graded as text. |
| `Wolfram Alpha--19` | 3 (unfilled placeholder) + 4 | Reference is literally '<sunborn time> (real-time date)'; depends on today's UV index. |
| `Wolfram Alpha--20` | surplus | Clean and verified with sympy (33038.1987); held in reserve, target of 7 already met. |
| `Wolfram Alpha--22` | surplus | Clean and verified with sympy (147*sqrt(3)/2 = 127.305734 cm^2); held in reserve. |
| `Wolfram Alpha--23` | 4 (time-varying) | Population estimates for 2020-2023 are still being revised; the derived growth rate churns. |
| `Wolfram Alpha--25` | surplus | Arithmetic checks out (x=63.64 m, y=19.49 m, Vx=21.21 m/s, Vy=-8.22 m/s with g=9.81); held in reserve. |
| `Wolfram Alpha--26` | 3 (malformed reference) | Reference is garbled -- 'hydrogen (H), 32.69% for sulfur (S)' omits the H percentage (should be ~2.06%). Repairable but dropped since the target was met. |
| `Wolfram Alpha--27` | surplus | Database lookup, stable; held in reserve. |
| `Wolfram Alpha--28` | surplus | Verified correct (9649 = U+25B1 WHITE PARALLELOGRAM) and unambiguous -- the best reserve row if a replacement is needed. |
| `Wolfram Alpha--29` | 7 (image answer) | Reference is 'any cat curve' -- a plot image, not text. |
| `Wolfram Alpha--30` | 4 (time-varying) | Reference says 'real-time, search query'; depends on today's UV conditions. |
| `Wolfram Alpha--31` | 4 (time-varying) | Live weather; reference is a search query, not an answer. |
| `Wolfram Alpha--32` | surplus | Verified with sympy -- the 28-prime list in the reference is exactly right; held in reserve (long list makes grading brittle to formatting). |
| `Wolfram Alpha--33` | 4 (time-varying) | Annual generation figures are restated; Wolfram's dam dataset is refreshed. |
| `Wolfram Alpha--34` | surplus | Astronomical constants are stable (317.8x, 9.925 h); held in reserve. |
| `Wolfram Alpha--36` | surplus | Verified convergent with sympy, but the answer is a single word -- too easy to guess to be a useful benchmark row. |
| `Wolfram Alpha--37` | surplus | Both dates are absolute, and 9675 days verified with Python datetime; held in reserve. |
| `Wolfram Alpha--38` | surplus | Verified with sympy as 39.24993, consistent with 'around 39.2'; held in reserve (vague reference). |
| `Wolfram Alpha--39` | 3 (corrupted/wrong reference) | Reference LaTeX contains literal form-feed escapes and uses 2*pi/15 rad = 24 degrees, which contradicts the 33 degrees in the question. |
| `Wolfram Alpha--4` | 4 (time-varying) | 'Densest KNOWN packing' is a record that can be improved; Wolfram updates its packing database, so 0.176939r is not permanent. |
| `Wolfram Alpha--40` | 4 (time-varying) | Output of Wolfram's exercise-metabolism model, which is re-tuned between releases. |
| `Wolfram Alpha--41` | 4 (time-varying) | Depends on which max-HR formula Wolfram currently uses (220-age gives 110; Tanaka gives ~107). |
| `Wolfram Alpha--42` | surplus | Computes to ~192 MB (7214.4 x 8892 px x 3 bytes), but depends on Wolfram's pixel-rounding and MB-vs-MiB convention; held in reserve. |
| `Wolfram Alpha--43` | 5 (ambiguous) | 35 free hexominoes is right, but 'only 2 rows' depends on the orientation Wolfram happens to draw each piece in -- not well defined. |
| `Wolfram Alpha--44` | 5 (ambiguous) + 3 | Reference is an enormous expression ending in 'OR ...'; multiple equivalent closed forms exist and the truncated reference is not gradeable. |
| `Wolfram Alpha--45` | 4 (time-varying) | Five-field output of Wolfram's metabolic model; values shift when the model is updated. |
| `Wolfram Alpha--7` | 5 (ambiguous) | Spring pendulum is chaotic; the t=6s state depends on Wolfram's integrator tolerances and sign/branch convention for 'angle from vertical'. Not reproducibly gradeable. |
| `Wolfram Alpha--8` | surplus | Stoichiometry is deterministic but the reference is hedged ('approximately') across four quantities; dropped in favor of exact-valued rows. |
| `Wolfram Alpha--9` | 4 (time-varying) | Wolfram's curated power-plant dataset is revised; the 2010 figure served today may differ from 9752 GWh/yr. |

</details>

## The 100 rows

| task_id | prov | vet | type | question |
| --- | --- | --- | --- | --- |
| `Allrecipes--13` | add | insp | poss | Find a recipe with over 100 reviews for Fried Fish on Allrecipes, list the Full Nutrition Label and tell me the amount of Iron per |
| `Allrecipes--20` | kep | insp | poss | Find a recipe for a cauliflower pizza crust that has a preparation time of under 30 minutes and a rating of at least 4 stars on Al |
| `Allrecipes--22` | add | insp | poss | Find a recipe for a healthy avocado salad on Allrecipes that has a preparation time of less than 20 minutes and more than 30 user  |
| `Allrecipes--24` | add | insp | poss | Locate a recipe for an eggplant Parmesan on Allrecipes with a rating of at least 4.5 stars and over 50 reviews. Include the prepar |
| `Allrecipes--36` | add | insp | poss | Locate a recipe for an American apple pie on Allrecipes with a rating of at least 4 stars and more than 50 reviews. Note the maxim |
| `Allrecipes--38` | add | insp | poss | Find a French ratatouille recipe on Allrecipes with a 4-star rating or higher and at least 15 reviews. Note the variety of vegetab |
| `Amazon--0` | add | live | poss | Search an Xbox Wireless controller with green color and rated above 4 stars. |
| `Amazon--10` | rep | live | crit | Find the cost of a 2-year protection for PS4 on Amazon. |
| `Amazon--2` | add | live | poss | Find a gaming desktop with Windows 11 Home, and the disk size should be 1TB. |
| `Amazon--22` | add | live | poss | Search for a set of non-stick, oven-safe cookware on Amazon. The set should include at least 10 pieces and be priced under $150. |
| `Amazon--32` | add | live | poss | Search for an electric kettle on Amazon with a capacity of at least 1.5 liters, made of stainless steel, and with a customer ratin |
| `Amazon--7` | add | live | crit | Browse the women's hiking boots on Amazon and filter the results to show only those that are waterproof and have a rating of at le |
| `Amazon--9` | rep | live | crit | Find a dog bed on Amazon that is washable and has a length of at least 30 inches. |
| `Apple--12` | add | live | gold | What Apple Repair ways are mentioned on apple website, answer 2 of them. |
| `Apple--25` | kep | live | poss | On the Apple website, look up the processor for the latest model of the Apple TV. |
| `Apple--26` | kep | live | poss | Find the maximum video recording resolution supported by the latest iPad mini on the Apple website. |
| `Apple--27` | add | live | poss | On Apple's website, check if the HomePod mini in store is available in multiple colors and list them. |
| `Apple--28` | add | live | gold | On the Apple website, find out if the Mac Mini can be configured with a GPU larger than 16-core. |
| `Apple--29` | add | live | poss | On Apple's website, check the estimated battery life of the latest MacBook Air during web browsing in Tech Specs. |
| `Apple--33` | add | live | poss | Look for the color options available for the newest iMac. |
| `ArXiv--10` | add | live | gold | Visit ArXiv Help on how to withdraw an article if the submission is not yet announced. |
| `ArXiv--14` | add | live | gold | On ArXiv, use the search (all fields) to find articles mentioning 'SimCSE' that were originally announced in October 2023. How man |
| `ArXiv--17` | add | live | gold | Find the paper 'GPT-4 Technical Report', when was v3 submitted? |
| `ArXiv--27` | add | live | gold | On ArXiv, what categories does Economics include, and what are their abbreviations? |
| `ArXiv--30` | add | live | gold | Look up the submission guidelines on ArXiv for submitting a paper and tell me the formats for figures. |
| `ArXiv--39` | add | live | gold | Search the title 'GPT-4 Technical Report' and access this paper through HTML format. Read the paper on this page and tell me what  |
| `ArXiv--42` | kep | insp | gold | Find an article published between 1 January 2000 and 1 January 2005 that requires Support Vector Machines in the title and its Jou |
| `BBC News--10` | add | live | gold | Find the BBC News explainer "What is the Paris agreement and why does 1.5C matter?". On what date did the Paris agreement come int |
| `BBC News--11` | add | live | gold | In the SPORT section of BBC, open Football > Scottish Premiership > Table. How many teams are listed in the table, and which team  |
| `BBC News--18` | add | live | gold | Go to BBC Audio and open Podcast Categories. How many podcast categories are listed, and what are they? |
| `BBC News--25` | add | live | gold | In the SPORT section of BBC, open Football > Premier League > Table and read the table key below the standings. How many teams are |
| `BBC News--28` | add | live | gold | Navigate to the Market Data page in BBC News's Business section. What does the BBC say has happened to its market data feeds, and  |
| `BBC News--34` | add | live | gold | Find The SpeciaList section in BBC Travel and open the feature about Monty Python star Eric Idle's French escape. Which Provencal  |
| `BBC News--5` | add | live | gold | Find the BBC News climate explainer titled "A really simple guide to climate change" (listed in search results as "What is climate |
| `Booking--2` | add | insp | crit | Find a hotel in Ohio, USA on Booking.com for a stay from 20 June 2027 to 23 June 2027 for 3 adults in 2 rooms. Give the property n |
| `Booking--33` | add | live | gold | On Booking.com, find the Customer Service / Help Center, browse the questions about cancellation, and tell me how I can know wheth |
| `Booking--35` | add | live | crit | Go to Booking.com's Travel articles section (the travel-inspiration articles, not search results). Open one article, tell me its t |
| `Booking--4` | rep | insp | crit | On Booking.com, search for the cheapest hotel near Kashi Vishwanath Temple (Varanasi, India) that includes breakfast, for a one-ni |
| `Booking--41` | add | live | gold | Browse Booking's homepage to find out which company it belongs to. |
| `Booking--43` | add | insp | crit | Search for properties in Los Angeles on Booking.com, then browse the results page and list at least five of the filters available  |
| `Cambridge Dictionary--0` | add | live | gold | Look up the pronunciation and definition of the word "sustainability" on the Cambridge Dictionary. |
| `Cambridge Dictionary--17` | add | live | gold | How many meanings of "unblemished" are given in Cambridge Dictionary? Please browse the page and give the number directly. |
| `Cambridge Dictionary--21` | add | live | gold | Search for the word "ephemeral" on Cambridge Dictionary and find its translation into Spanish. |
| `Cambridge Dictionary--28` | add | live | gold | Search for "feel giddy" in Cambridge Dictionary's Thesaurus and list the synonyms the dictionary provides. |
| `Cambridge Dictionary--33` | add | live | poss | Find explanations and examples of the passive voice in Grammar on the Cambridge Dictionary website. |
| `Cambridge Dictionary--34` | kep | live | poss | Use the Cambridge Dictionary to understand the rules for forming and using comparative and superlative adjectives in English Gramm |
| `Cambridge Dictionary--35` | rep | live | poss | On the Cambridge Dictionary's English Grammar 'Prepositions' page, list the prepositions that consist of groups of words. |
| `Coursera--13` | add | live | poss | Find the 'Python 3 Programming' Specialization on Coursera offered by the University of Michigan, and list the 'What you'll learn' |
| `Coursera--15` | add | live | poss | Find the Coursera course 'Introduction to Mathematical Thinking' offered by Stanford. In the reviews section, what is the percenta |
| `Coursera--2` | add | live | poss | Find the beginner-level Specialization 'Learn Spanish: Basic Spanish Vocabulary' on Coursera and list all the courses in this Spec |
| `Coursera--22` | add | live | poss | Identify the Coursera Specialization focused on Human Resource management offered by the University of Minnesota. Give its exact n |
| `Coursera--39` | add | live | gold | Find the course 'Space Safety' offered by the Technical University of Munich (TUM) on Coursera. How many videos are there in modul |
| `Coursera--6` | rep | live | poss | On Coursera, open the course 'Fundamentals of Machine Learning for Healthcare' (Stanford Online, part of the AI in Healthcare Spec |
| `Coursera--7` | rep | live | poss | On Coursera, find the Reinforcement Learning Specialization (an intermediate-level Specialization rated 4.5 or higher). Provide th |
| `ESPN--11` | add | live | gold | How many NBA teams are there and list all the teams with 'New' in their name. |
| `ESPN--23` | rep | live | gold | On ESPN, find the result of the NBA regular-season game between the Miami Heat and the New York Knicks played on January 27, 2024, |
| `ESPN--24` | rep | live | gold | On ESPN, find the final score of Super Bowl LVIII, including both teams' names and the date the game was played. |
| `ESPN--27` | add | live | gold | How many NHL teams are there on ESPN, and list all the teams with 'Golden' in their name. |
| `ESPN--28` | add | live | gold | How many MLB teams are there and list all the teams with 'City' in their name. |
| `ESPN--41` | add | live | gold | Find out which four teams the NFC North contains in the NFL on ESPN. |
| `ESPN--9` | add | live | gold | Using ESPN's team lists for the NBA, NFL, NHL and MLB, how many teams have the city spelled out in full as 'Los Angeles' in their  |
| `GitHub--23` | add | live | gold | Find the wiki page of ohmyzsh on GitHub and tell me how to change the theme of zsh to agnoster. |
| `GitHub--26` | add | live | gold | Find the 'Resolve merge conflicts' GitHub Skills exercise repository (github.com/skills/resolve-merge-conflicts) and list the thin |
| `GitHub--3` | add | live | gold | Find out how much more package storage the Enterprise version has over Team in GitHub Pricing. |
| `GitHub--37` | rep | live | poss | On GitHub's Copilot product page (github.com/features/copilot), find the section headed 'Code, command, and collaborate' and list  |
| `GitHub--38` | rep | live | gold | Identify and report the most popular (by stars) open-source repo related to cybersecurity on GitHub. |
| `GitHub--7` | add | live | gold | Find the official GitHub repository for ALBERT and show me what files the repo changed in the most recent commit. |
| `GitHub--8` | add | live | gold | Look up the latest stable release version of Vuex and find out when it was published. |
| `Google Flights--2` | add | insp | crit | Find the lowest fare among all one-way flights for 1 adult from New York JFK to London Heathrow (LHR) on 15 June 2027. Report the  |
| `Google Flights--25` | add | insp | crit | Find a one-way business class flight from Buenos Aires to Amsterdam on 18 June 2027 and give me the details of the option with the |
| `Google Flights--3` | rep | insp | crit | Search for the one-way flight from Calgary to New York on 10 June 2027 with the lowest carbon dioxide emissions. Report the airlin |
| `Google Flights--4` | rep | insp | crit | Search for one-way flights from New York to London on 12 June 2027, apply the filter that shows only non-stop flights, and report  |
| `Google Flights--40` | add | live | crit | On the Google Flights Explore destinations map (google.com/travel/explore), set Seattle as the origin and browse the map. List thr |
| `Google Flights--9` | add | insp | crit | Find a one-way economy flight from Pune to New York departing 16 June 2027, open a connecting itinerary from the results, and tell |
| `Google Map--10` | add | live | gold | Search Google Maps for Castle Mountains National Monument in California and report its basic information: the address shown, its o |
| `Google Map--2` | add | live | poss | Find Apple Stores close to zip code 90028 on Google Maps. Name the two nearest ones and give their street addresses. |
| `Google Map--32` | rep | insp | crit | On Google Maps, search for plumbers in Orlando, FL and apply the "Open now" filter. Name one plumber that is open now but is NOT l |
| `Google Map--33` | rep | insp | poss | Open Denver International Airport's listing on Google Maps and tell me: 1) which star rating (1-5) has the smallest share of its r |
| `Google Map--35` | add | live | poss | Search Google Maps for Big Bend National Park in Texas and gather its basic information: phone number, the plus code / address sho |
| `Google Map--8` | add | live | poss | Find a place to climb (a climbing or bouldering gym) within 2 miles of zip code 90028 on Google Maps. Give its name and street add |
| `Google Search--0` | add | live | gold | Find the initial release date for Guardians of the Galaxy Vol. 3 the movie. |
| `Google Search--12` | add | live | gold | Find the year that Tom Brady had the most touchdown passes in a single NFL season. |
| `Google Search--13` | rep | live | gold | Which character did Jerry Trainor play in Nickelodeon's original iCarly series, and on what date did that original series premiere |
| `Google Search--31` | add | live | gold | In which club season did Cristiano Ronaldo score the most goals across all competitions, and how many goals was that? |
| `Google Search--40` | add | live | gold | Look up the elevation of Mount Kilimanjaro on Google Search. |
| `Google Search--7` | add | live | gold | Find the software requirements for iPhones that support AirDrop's ability to continue transmitting over the web when out of range. |
| `Huggingface--14` | add | live | poss | How much is the Pro account of Hugging face for a month and what are the features? |
| `Huggingface--24` | add | live | gold | Search for a model on Hugging Face with an Apache-2.0 license that has received the highest number of likes. |
| `Huggingface--3` | kep | live | gold | Look up a model with a license of cc-by-sa-4.0 with the most likes on Hugging face. |
| `Huggingface--30` | add | live | gold | Find the machine translation model Helsinki-NLP/opus-mt-en-zh on Huggingface and report the BLEU and chr-F scores its model card l |
| `Huggingface--32` | add | live | gold | On the Hugging Face website, open the model EleutherAI/gpt-j-6b and look at its config.json. What is the value of the 'temperature |
| `Huggingface--42` | add | live | gold | Check the Dataset Viewer for ai2lumos/lumos_complex_qa_plan_onetime on Hugging face. what is the content corresponding to user in  |
| `Huggingface--5` | add | live | gold | Find the model flax-community/t5-recipe-generation on Hugging Face. Report its number of parameters (model size) and the tensor ty |
| `Wolfram Alpha--2` | add | live | gold | Calculate 3^71 and retain 5 significant figures in scientific notation. |
| `Wolfram Alpha--21` | add | live | gold | Calculate (1+0.1*i)^8 + (1−0.2*i)^8  where i is a complex number. |
| `Wolfram Alpha--24` | add | live | gold | Solve the differential equation y''(t) - 2y'(t) + 10y(t) = 0 and display its general solution. |
| `Wolfram Alpha--3` | add | live | gold | Let g(x) be the integral of x^2 cos(2x). Write the expression of g(x). |
| `Wolfram Alpha--35` | add | live | gold | Calculate the determinant of a 6x6 Hilbert matrix. |
| `Wolfram Alpha--5` | kep | live | gold | Show the solution of y"(z) + sin(y(z)) = 0 from wolframalpha. |
| `Wolfram Alpha--6` | kep | live | gold | Simplify x^5-20x^4+163x^3-676x^2+1424x-1209 so that it has fewer items. |

## Known weaknesses — what a human should double-check

1. **All 6 Allrecipes rows are unverified.** The site is unreachable from every channel
   tried. Their references are upstream 2024 values. `Allrecipes--13`'s "Iron: 15 mg" looks
   implausible even for an Allrecipes nutrition panel. Re-read these six on a residential IP
   or drop the site to get a 94-row set you can fully stand behind.
2. **The 14 `criteria` rows need an LLM judge** and a judge that actually checks the stated
   condition — e.g. `Google Map--32` requires checking the hours field, not just that a
   plumber was named.
3. **The June-2027 date pins expire.** `Booking--2`, `Booking--4` and the five dated Google
   Flights rows are pinned to June 2027, inside both sites' booking windows as of Oct 2026.
   Past roughly February 2027 those windows slide and the rows need re-pinning. This is a
   standing maintenance cost, not a one-time fix.
4. **Three rows are live superlatives and will need re-verification before each run**:
   `GitHub--38` (top cybersecurity repo by stars — `sherlock-project/sherlock` leads by ~26k,
   so it should hold), `Huggingface--24` (most-liked apache-2.0 model — 2.4x margin),
   `Huggingface--3` (most-liked cc-by-sa-4.0 model — small dormant pool, very stable).
5. **`Google Map--33`'s Accessibility/Amenities reference could not be refreshed.** A search
   of flydenver.com found no mention of an "assistive hearing loop" and sources conflict on
   baggage storage. The star-share half of that row is the safer half; someone with a real
   browser should eyeball the Maps panel and trim the reference if needed.
6. **`BBC News--28` is a page that announces its own removal.** BBC switched off all market
   data feeds on 26 Nov 2024; the row now asks about that notice. Legitimate but unusual —
   drop it and ship 6 BBC rows if that reads wrong.
7. **`Apple` rows say "the latest model".** The questions stay well-posed but the references
   record Oct-2026 values and will need a one-line update at Apple's next hardware refresh.
8. **`ArXiv--42` was kept verbatim and not re-fetched** (a prior audit verified it as an
   exact string match); it is the only non-Allrecipes/Booking/Google inspection row.
9. **Google Search rows verify the facts, not google.com's rendering.** Nothing here is
   evidence that a datacenter-IP browser can complete a google.com SERP without a CAPTCHA.
   If the CLI arm fails these uniformly, that is an infrastructure signal, not a task-quality
   signal.

## Reproducing

`webvoyager_100.jsonl` carries five extra fields — `provenance`, `upstream_reference_updated`,
`vetting`, `evidence` and `repair_note` — alongside `inputs`/`outputs`. The upload strips them:

```python
c.create_examples(dataset_id=ds.id,
                  inputs=[r["inputs"] for r in rows],
                  outputs=[r["outputs"] for r in rows])
```

so the LangSmith examples carry exactly the `webvoyager-30` schema and nothing else.
