# WebVoyager-100, three arms

Dataset `webvoyager-100` (vetted; see `WEBVOYAGER_100.md`), 1 repetition, Jev
judge with the hedge and placeholder fixes.

| arm | correct | blocked | steps med | secs med | cost | graded |
|---|---|---|---|---|---|---|
| **MCP** | **0.890** | **0.130** | 13 | **36** | $17.87 | 100/101 |
| CLI | 0.848 | 0.152 | 8 | 71 | $11.65 | 99/104 |
| REPL | 0.808 | 0.212 | 10 | 41 | **$4.45** | 99/106 |

## The MCP correctness finding was an artifact

An earlier MCP run on this dataset scored **0.576** with a 0.455 blocked rate.
It was contaminated: 39 of 99 runs never reached a website, returning
`429 You've exceeded your max concurrent sessions limit (limit 25, currently 25)`.
Blocked rates of 0.455 on GitHub, Apple and Huggingface -- sites that bot-wall
nobody -- should have been read as an infrastructure signal immediately.

The cause is a session leak specific to this arm. The MCP agent calls
`browserbase_start` per run and never calls `browserbase_end`, and the endpoint
URL carries `keepAlive=true` (added to stop SSE streams dropping mid-task), so
sessions are released by nothing at all. Thirteen tasks produced eighteen live
sessions against a cap of 25. Running the arm "alone" does not help: it starves
itself.

This run used concurrency 3 plus a reaper releasing sessions older than 180s,
and recorded **zero** session-limit errors across 101 runs.

**`MCP_CORRECTNESS_ANALYSIS.md` is superseded on its headline claim.** Its
per-run failure taxonomy was drawn from runs on the 30-task set and may still
describe real `act`/`extract` behaviour, but "MCP scores lower on correctness"
does not survive a clean run.

## What does survive

Cost and step inflation, which were measured independently of the session
problem and reproduce here: MCP spends 4x REPL per suite and takes the most
steps. `MCP_COST_LATENCY.md`'s mechanism -- `browserbase_navigate` returning
~20KB of serialised CDP connection object carrying no page content -- is
unaffected.

The honest shape is **MCP buys accuracy with tokens**, not "MCP is costly and
worse".

## Caveat: the arms did not run under equal conditions

MCP ran at concurrency 3 with a reaper. CLI and REPL ran at concurrency 6,
simultaneously, competing with each other for the same Browserbase pool (REPL
uses it; CLI drives local Chromium and does not). REPL's higher blocked rate may
itself be contention rather than a property of the arm.

Before this table is quoted anywhere, CLI and REPL should be rerun one at a
time under the conditions MCP got.
