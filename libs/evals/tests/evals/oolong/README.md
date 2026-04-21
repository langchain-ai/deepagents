# Oolong evals

Three dataset modules — `trec_coarse`, `multinli`, `metaphors` —
exercise long-context aggregation and classification against a seeded
`/context.txt` file.

## Running

One pytest invocation per dataset. Each dataset module parametrizes
over the `(runner, task)` grid, so a single run covers every runner
(`baseline`, `rlm`, `swarm`, `shell`) against every task. LangSmith
stores one dataset per Oolong source dataset with runners co-located
as separate runs within one experiment — the `runner` field on
outputs lets the UI slice/compare without forking rows.

```bash
export LANGSMITH_API_KEY=...
export LANGSMITH_TRACING=true

# All three datasets:
./scripts/run_oolong.sh

# One dataset:
./scripts/run_oolong.sh trec_coarse

# Smoke: 3 tasks × 3 datasets × 4 runners:
OOLONG_MAX_PER_DATASET=3 ./scripts/run_oolong.sh
```

LangSmith layout for a full sweep:

```
dataset: deepagents-py-oolong-trec-coarse
  experiment: <ts>
    runs: baseline×task, rlm×task, swarm×task, shell×task (per task)
dataset: deepagents-py-oolong-multinli
  experiment: <ts>
    …
```

## Runners

- `baseline` — plain `create_deep_agent`, no REPL or subagent tricks.
  Matches the JS `getDefaultRunner()`.
- `rlm` — REPL + PTC with a recursive compiled `general-purpose`
  chain at depth `RLM_MAX_DEPTH`. Each level dispatches to a deeper
  peer via `tools.task` inside `eval` + `Promise.all`.
- `swarm` — REPL + PTC + the `swarm` skill served from a
  `FilesystemBackend` mounted at `/skills/` via a `CompositeBackend`
  (everything else, including `/context.txt`, goes to `StateBackend`).
  A query-level nudge tells the model to reach for `runSwarm` when a
  question can be decomposed. The skill directory is a symlink to
  `examples/repl_swarm/skills/`.
- `shell` — agent backed by a per-task LangSmith sandbox (template
  `deepagents-cli`). `/context.txt` is uploaded to the sandbox before
  the agent runs so shell commands (`cat /context.txt`, `grep`) find
  the file directly. Requires `LANGSMITH_API_KEY` and the
  `deepagents-cli` template in the account.

## Knobs (match the JS harness)

- `OOLONG_MAX_PER_DATASET` (default `10`) — cap examples per source
  dataset. `0` disables the cap (~1,300 examples across all datasets,
  most of which the three modules here ignore).
- `OOLONG_CONTEXT_LEN` — filter to a single bucket: `1024`, `4096`,
  `32768`, `131072`. Unset → all buckets. Most models can't fit the
  `131072` bucket; in practice you'll sweep the two smallest buckets
  first.

## JS parity notes

The Python scorer is a literal port of `scoring.ts` — same
`canonicalPrediction` pipeline, same `0.75 ** |diff|` numeric partial
credit, same comparison-phrase list. Unit tests in
`tests/unit_tests/test_oolong_scoring.py` guard the port.

One deliberate deviation from the paper's original Python harness
(inherited from the JS port): `canonical_prediction` takes the last
non-empty line *before* splitting on `:`. The paper's
`synth_attempt_answer_parse` splits on `:` only. We keep JS parity so
Python numbers line up with what the team has been reading on the JS
dashboards. If you need paper-to-paper comparability, rerun against
the upstream harness in `oolong_benchmark/src/eval/eval_helpers.py`.
