---
name: datadog-notebook-authoring
description: "Authoritative Datadog Notebooks API schema for authoring notebook JSON that the `pup notebooks create` / `pup notebooks update` CLI accepts. Use this skill whenever the user asks you to create, update, or edit a Datadog notebook via `pup notebooks create` or `pup notebooks update`, or otherwise hand-write a Datadog notebook payload. Trigger on phrases like \"create a Datadog notebook\", \"pup notebooks create\", \"pup notebooks update\", \"make a Datadog notebook\", or \"update the notebook in Datadog\"."
---

# Datadog Notebook Authoring

The Datadog Notebooks API rejects malformed payloads with HTTP 400 validation
errors. Generating invalid JSON and retrying with partial corrections wastes
tokens and never recovers. Validate every payload against the constraints below
**before** the first `pup notebooks create` call.

## Schema constraints

1. **Root `type`.** The root object's `type` MUST be `"notebooks"` (plural).
   The API rejects `type: "notebook"` with `Invalid type. Expected "notebooks".`
   The payload is wrapped as `{ "data": { "type": "notebooks", "attributes": { ... } } }`.

2. **`graph_size` only on graph cells.** `graph_size` is valid ONLY on graph
   cell types (`timeseries`, `toplist`, `bar_chart`, etc.). NEVER emit
   `graph_size` on a `markdown` cell — the API rejects it.

3. **`response_format`.** Must be exactly one of `["timeseries", "scalar", "tabular"]`.
   `"toplist"` is NOT a valid `response_format` — it is a cell/visualization
   type, not a response format.

4. **`tags` keys allowlist.** Tag keys must come from: `team`,
   `llm-observability`, `ai`, `ai_generated`, `ai_edited`, `human_edited`. Do
   NOT invent keys like `feedback`, `ingestion`, or `pr-<n>`.

## Workflow

1. Generate the notebook JSON, then check it against all four constraints above
   BEFORE running `pup notebooks create` / `pup notebooks update`.
2. On any HTTP 400, parse the FULL validation error and fix EVERY reported field
   in a single next attempt. Do not retry a near-identical payload.
3. After 2 failed create attempts, STOP retrying. Surface the remaining
   validation errors to the developer and ask for guidance instead of burning
   further tokens.

## Known-good example

```json
{
  "data": {
    "type": "notebooks",
    "attributes": {
      "name": "Service health overview",
      "cells": [
        {
          "type": "notebook_cells",
          "attributes": {
            "definition": {
              "type": "markdown",
              "text": "## Service health"
            }
          }
        },
        {
          "type": "notebook_cells",
          "attributes": {
            "definition": {
              "type": "timeseries",
              "requests": [
                {
                  "response_format": "timeseries",
                  "queries": [
                    {
                      "name": "q1",
                      "data_source": "metrics",
                      "query": "avg:system.cpu.user{*}"
                    }
                  ]
                }
              ]
            },
            "graph_size": "m"
          }
        }
      ],
      "time": { "live_span": "1h" },
      "tags": ["team:platform", "ai_generated"]
    }
  }
}
```

Note: the markdown cell omits `graph_size`; the timeseries cell includes it. The
`response_format` is `timeseries`, and both tag keys are on the allowlist.
