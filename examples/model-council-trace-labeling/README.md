# Model-council trace labeling with the code interpreter

This example demonstrates the proposed large-trace workflow:

1. Export complete LangSmith traces into a local, dedicated workspace.
2. Attach that workspace to a Deep Agent with `FilesystemBackend`.
3. Attach `CodeInterpreterMiddleware` so the orchestrator can run a generated
   JavaScript workflow.
4. Fan out one structured `task()` call per **trace × judge model**, in bounded
   parallel batches.
5. Compute a deterministic majority vote in JavaScript and write JSONL labels
   back to the local filesystem.

The trace payload stays out of the orchestrator's context: each judge receives a
file locator such as `/traces/abc.jsonl` and reads that trace itself.

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

The command prints the exact JavaScript handed to the `eval` tool. Results land
in:

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
const jobs = traces.flatMap((tracePath) =>
  judges.map((judge) => ({ tracePath, judge }))
);
for (let i = 0; i < jobs.length; i += 10) {
  const batch = jobs.slice(i, i + 10);
  const judged = await Promise.all(batch.map((job) => task({
    description: `Read ${job.tracePath} and apply the SFT rubric`,
    subagentType: job.judge,
    responseSchema: voteSchema,
  })));
  votes.push(...judged);
}
```

The real generated program also attaches judge names, calculates vote totals,
and returns a typed aggregate. Use `--print-workflow` to inspect it in full.

## Security and operational notes

- Trace content is untrusted. Both the orchestrator and judges are instructed not
  to follow commands embedded in traces.
- `FilesystemBackend(..., virtual_mode=True)` confines virtual paths to the chosen
  workspace. Judges are read-only; the orchestrator can write only under
  `/results` and cannot modify `/traces`.
- REPL `task()` dispatches occur inside an already-running `eval` call and bypass
  parent-level per-dispatch HITL. Gate the `eval` tool itself, or put approval
  middleware on each declarative judge, if every dispatch needs approval.
- The REPL has a 32-subagent concurrency ceiling. This example uses batches of 10
  and rejects runs requiring more than its configured 256-call budget.
- A council vote is a heuristic label, not proof that a trace is safe or correct.
  Sample and audit accepted traces before using them for training.
- Exported traces can contain user data, secrets, or regulated data. Keep
  `.workspace` private, apply your retention policy, and do not commit it.
