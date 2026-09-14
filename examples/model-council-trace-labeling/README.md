# Model-council trace labeling with the code interpreter

This example demonstrates the proposed large-trace workflow:

1. Export complete LangSmith traces into a local, dedicated workspace.
2. Attach that workspace to a Deep Agent with `FilesystemBackend`.
3. Attach `CodeInterpreterMiddleware` with `read_file` and `write_file` PTC
   bridges.
4. Load the manifest and vote schema from a small typed config bridge, then read
   trace bodies inside the REPL with `tools.readFile`.
5. Fan out one structured `task()` call per **trace × judge model**, in bounded
   parallel batches.
6. Compute a deterministic majority vote and persist both result files from the
   same REPL program with `tools.writeFile`.

Trace bodies, the response schema, and full vote records stay out of the
orchestrator model's context. Each judge receives the trace body directly from
the REPL as explicitly delimited, untrusted evidence.

## Setup

Install the project and the LangSmith CLI:

```bash
cd examples/model-council-trace-labeling
uv sync
curl -sSL https://raw.githubusercontent.com/langchain-ai/langsmith-cli/main/scripts/install.sh | sh
```

Set the API keys required by the models you choose. Model identifiers are CLI
arguments so the example does not bake in provider-specific model versions:

```bash
export JUDGE_MODEL_A='provider-a:current-model-id'
export JUDGE_MODEL_B='provider-b:current-model-id'
# Set the corresponding provider API keys too.
```

Use at least one judge. Different providers or model families make the council
less correlated; repeated calls to the same model are also valid for a smoke test.

## Try the synthetic traces

The bundled fixtures contain one obviously good trajectory and one incorrect,
unresolved trajectory:

```bash
uv run python run_council.py \
  --seed-samples \
  --judge-model "$JUDGE_MODEL_A" \
  --judge-model "$JUDGE_MODEL_B" \
  --print-workflow
```

The command prints the exact JavaScript handed to the `eval` tool. Its trace
paths and vote schema are supplied at runtime by `tools.getCouncilConfig`, not
copied into the prompt. Results land in:

```text
.workspace/results/labels.jsonl
.workspace/results/summary.md
```

## Export and label real LangSmith traces

`langsmith trace export --full` writes one JSONL file per complete trace, including
its child runs and input/output payloads:

```bash
export LANGSMITH_API_KEY='...'
./export_traces.sh my-tracing-project .workspace/traces 20

uv run python run_council.py \
  --workspace .workspace \
  --max-traces 20 \
  --judge-model "$JUDGE_MODEL_A" \
  --judge-model "$JUDGE_MODEL_B"
```

Add more `--judge-model` flags for an N-model council. The generated workflow
batches dispatches in groups of 10 instead of launching all calls at once. Final
labels use strict majority voting; ties become `0` (reject).

## What demonstrates code mode

`run_council.py` generates a concrete program shaped like this and asks the
orchestrator to execute it once in the sandboxed REPL:

```javascript
const config = await tools.getCouncilConfig({});
const traces = await Promise.all(config.tracePaths.map(async (tracePath) => ({
  tracePath,
  body: await tools.readFile({ file_path: tracePath, offset: 0, limit: 1000 }),
})));
const jobs = traces.flatMap((trace) =>
  config.judges.map((judge) => ({ ...trace, judge }))
);
// Dispatch jobs in batches through task({ responseSchema: config.voteSchema }).
// Aggregate strict-majority labels, then persist without another model turn.
await tools.writeFile({ file_path: "/results/labels.jsonl", content: labelsJsonl });
```

The real program batches judge calls, explicitly delimits each trace body as
untrusted evidence, calculates vote totals, writes both outputs, and returns only
a compact path/count summary. Use `--print-workflow` to inspect it in full.

## Security and operational notes

- Trace content is untrusted. Both the orchestrator and judges are instructed not
  to follow commands embedded in traces.
- `FilesystemBackend(..., virtual_mode=True)` confines virtual paths to the chosen
  workspace. Judges are read-only; the orchestrator can write only under
  `/results` and cannot modify `/traces`.
- REPL `task()` dispatches and PTC file calls occur inside an already-running
  `eval` call and bypass parent-level per-call HITL. Gate the `eval` tool itself,
  or put approval middleware on each declarative judge, if every dispatch needs
  approval. The underlying filesystem tools still enforce the path permissions.
- The REPL has a 32-subagent concurrency ceiling. This example uses batches of 10
  and rejects runs requiring more than its configured 256-call budget, including
  trace reads and result writes.
- A council vote is a heuristic label, not proof that a trace is safe or correct.
  Sample and audit accepted traces before using them for training.
- Exported traces can contain user data, secrets, or regulated data. Keep
  `.workspace` private, apply your retention policy, and do not commit it.
