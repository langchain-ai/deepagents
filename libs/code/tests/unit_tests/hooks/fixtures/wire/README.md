# Hooks wire-contract fixtures

These JSON files pin the **external** Hooks v2 wire contract — the stdin payload
shape handlers receive, the registry policy matrix that drives matcher selection
and exit-code / plain-output behavior, and the domain decisions produced when
representative handler exits are reduced.

## When a test fails

A failing diff means one of:

1. **Deliberate contract change** — update the matching fixture in the same PR
   and call out the wire/policy change in the PR description.
2. **Accidental regression** — restore the previous shape; do not loosen the
   assertion.

Do not edit these files to silence a failure without understanding whether the
external contract changed.

## Layout

| Path | Pins |
| --- | --- |
| `inputs/<Event>.json` | Serialized wire input for one lifecycle event (external field names) |
| `registry_policies.json` | Per-event owner, matcher field, timeouts, and exit/plain/aggregation policies |
| `outputs/reduction_cases.json` | Handler exit scenarios → reduced domain decision snapshots |

Volatile values (`prompt_id`, path fields) are normalized to placeholders in the
test helper before comparison so fixtures stay stable across environments.
