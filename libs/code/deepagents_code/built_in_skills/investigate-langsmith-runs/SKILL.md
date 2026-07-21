---
name: investigate-langsmith-runs
description: "Workflow for investigating, analyzing, or inspecting traces or runs in LangSmith without blowing up model context or cost. Use this skill when the user asks to investigate/analyze/inspect traces or runs in LangSmith, debug a project's runs, audit token or cost usage across a time window, or find errors across runs. Trigger on phrases like \"investigate traces\", \"analyze runs\", \"inspect the LangSmith project\", \"why did this trace cost so much\", or \"look at runs in the last 24h\"."
license: MIT
compatibility: designed for deepagents-code
---

# Investigate LangSmith Runs

## Overview

Run-metadata tools such as `langsmith_fetch_runs` return large JSON payloads (30KB+ per page). Pulling many overlapping pages into context compounds prompt tokens turn over turn and can drive a single investigation to tens of millions of tokens and tens of dollars. This skill codifies a compact, cursor-paged, aggregate-locally workflow so an equivalent investigation stays within a few small pages.

## Best Practices

- **Fetch ONE compact page, then aggregate locally.** After each page compute counts, token/cost sums, and an error/success split yourself — do not ask the tool to re-return the full data.
- **Page forward with the returned cursor, never re-fetch a window.** Advance using the `next_cursor` the previous call returned; never re-issue a fetch with an overlapping or identical `min_start_time`/`max_start_time` window.
- **Request compact fields, not full run JSON.** Ask only for `{id, name, status, total_tokens, total_cost, start_time}`. Never request full metadata for 100 runs at once — the result-size cap will truncate it and you will have burned the call for nothing.
- **Stop as soon as the question is answered.** You almost never need the entire 24h window run-by-run.

## Process

1. Scope the query: pick the smallest time window and filter (project, error status, name) that can answer the question.
2. Fetch a single compact page (small `limit`, compact fields only).
3. Aggregate that page locally: total runs seen, summed tokens, summed cost, error vs. success counts, and any outliers (e.g. the single most expensive run).
4. Decide with the stop condition below whether another page is genuinely needed.
5. If needed, fetch the NEXT page using the returned `next_cursor` — never a new overlapping window — and fold its aggregates into the running totals.
6. Report the aggregates and any specific run IDs worth drilling into. Only fetch a full single run's detail when a specific ID has been identified as worth inspecting.

## Stop Condition

Stop paging when any of these is true:

- The user's question is answered by the aggregates so far.
- A page returns no `next_cursor` (end of results).
- You have fetched a few pages and the aggregates have stabilized — do not walk the entire window run-by-run.

## Common Pitfalls

- **Re-issuing the same or overlapping window** instead of advancing with the cursor. This is the primary cause of runaway cost — forbidden.
- **Requesting full run JSON for a large page.** It gets truncated by the result-size cap and wastes the call; request compact fields.
- **Backward scanning the whole 24h window** run-by-run when a few aggregated pages would answer the question.
- **Re-fetching data you already have in context** rather than aggregating locally from the page you already pulled.
