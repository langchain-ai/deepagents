# Evaluating a deployed agent, with native cost

How to get a LangSmith experiment whose rows carry real Cost and Tokens when the
agent under test is a **deployment** rather than in-process code. Written after
getting it wrong several times.

## The problem

LangSmith computes the native Cost/Tokens columns from **LLM spans inside the
experiment row's own trace**. Evaluate a local chain and this is free: the model
runs in process, its spans nest under the target run, the columns populate.

A Managed Deep Agent only exists as a deployment. `aevaluate`'s target can only
make an HTTP call and wait, so the row it creates holds no LLM spans:

```
experiment row (AsyncTarget)  ──HTTP──▶  deployment run (browser-agent)
  tokens 0 / cost None                     tokens 129,431 / cost $0.0728
```

Nothing is lost -- the numbers are on the deployment's run. They are just not on
the row the UI prices.

## What does not work

**Distributed tracing headers alone.** `RunTree.to_headers()` emits
`langsmith-trace` and `baggage`, and `get_client(headers=...)` forwards them. The
deployment still starts its own root trace; the row stays at zero.

**Headers plus `langsmith_tracing`.** Better, but not one row. The deployment's
trace lands *in* the experiment session carrying real tokens -- and appears as a
**second root run** next to `AsyncTarget`, both bound to the same `example_id`:

```
browser-agent  tok=105,161  cost=$0.0588   example=70c2b54e
AsyncTarget    tok=0        cost=None      example=70c2b54e
```

Two rows per task. The cost is visible, but the eval row is still empty and the
view reads as duplicated.

**Custom feedback metrics.** Reading the deployment run by id and posting
`tokens_k`/`cost_usd` as feedback does surface numbers, as custom columns rather
than native ones. It also has a trap: LangSmith caps a feedback score at
±99,999.9999, a browsing task burns 100k-500k tokens, and an over-range score
422s the **entire multipart batch** -- silently dropping every other metric in it
with no error in the UI.

## What works

An experiment is just a project with `reference_dataset_id` set. A run that is in
that project and carries a `reference_example_id` **is** a row. So skip
`aevaluate` and build it directly:

```python
ls = Client()
ds = ls.read_dataset(dataset_name="webvoyager-15")
project = f"webvoyager-cli-{int(time.time())}"
ls.create_project(project_name=project, reference_dataset_id=ds.id)

run = await client.runs.create(
    thread_id, assistant_id,
    input={"messages": [{"role": "user", "content": prompt}]},
    langsmith_tracing={"project_name": project, "example_id": str(example.id)},
)
```

Then wait for completion, grade however you like, and attach the grades as
feedback **on the deployment run** -- find it by its `reference_example_id`:

```python
runs = {str(r.reference_example_id): r.id
        for r in ls.list_runs(project_id=sess.id, is_root=True)
        if r.reference_example_id}
ls.create_feedback(runs[str(example.id)], key="correctness", score=1, comment=grade)
```

Result: one row per example, native Cost and Tokens, custom grades alongside.

```
root rows: 1
name=browser-agent  tok=129,431  cost=$0.0728
feedback={correctness: 0.0, seconds: 215.0, steps: 18.0}
```

## What I had wrong

Worth recording, because each wrong conclusion was held confidently.

1. **"`langsmith_tracing` is accepted and ignored."** False. It works, both
   fields. I checked `reference_example_id` on **root** runs in the
   **deployment's** project; the routed spans were children in a **different**
   project. An absence looked for in the wrong place is not evidence.
2. **"Headers are ignored."** Unproven. They were tested without
   `langsmith_tracing` and the conclusion generalised from one configuration.
3. **"Both were passed on every `runs.create`."** False. The runner had neither;
   an earlier edit had been lost in a rewrite. The claim was never checked
   against the file it described.

The pattern in all three: asserting a negative from a single observation without
confirming the observation could have shown a positive.

## Other things worth knowing

- **Reference outputs are not optional.** The dataset was first created with a
  placeholder output, so the judge graded from the question alone and the
  Reference Output column was meaningless. WebVoyager publishes
  `data/reference_answer.json`; most entries are `type: "possible"`, not
  `"golden"`, because the underlying data is real-time. A judge must treat those
  as a guide to the shape of a right answer, not a string to match.
- **Separate stale labels from wrong answers.** One task's reference names a
  different tournament than the live page shows. Without a distinct grade for
  that, a correct agent is scored as a failure. A `STALE` verdict keeps the
  number honest.
- **Separate transport failures from bad agents.** An MCP stream that dies mid
  task (`MCPError(-32000, 'SSE stream ended without a response')`) produces a
  run with many steps, high cost, and no answer -- indistinguishable from an
  agent flailing unless it is graded separately.
- **`runs.wait` over `runs.create` + poll.** `create` returns background-run
  metadata immediately; `wait` creates and waits. The polling loop here predates
  that and should be simplified.
