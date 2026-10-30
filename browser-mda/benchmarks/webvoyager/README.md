# WebVoyager benchmark

Runs [WebVoyager](https://github.com/MinorJerry/WebVoyager) tasks against a
running `browser-agent` (local `mda dev` or a deployment) and grades the answers.

## Fetch the dataset

Not vendored -- pull it from upstream:

```bash
curl -sO https://raw.githubusercontent.com/MinorJerry/WebVoyager/main/data/WebVoyager_data.jsonl
```

643 tasks across 15 sites. `GAIA_web.jsonl` (90 tasks) is the harder follow-on.

## Run

```bash
export MDA_URL=https://<your-deployment>.us.langgraph.app   # or http://127.0.0.1:2024
export LANGSMITH_API_KEY=...
uv run --with langgraph-sdk python run_eval.py --per-site 1 --concurrency 5 --out results.jsonl
```

Stratified sample: `--per-site N` takes N tasks from each of the 15 sites with a
fixed seed, so runs are comparable.

`--recursion-limit` defaults to 1000, i.e. effectively uncapped -- the wall-clock
`--timeout` is the real guard. Do not lower it to bound cost: a step ceiling
cuts off a slow-but-converging run exactly like a stuck one, and the result is
indistinguishable from a failure. Omitting the config entirely is worse, not
better, since LangGraph then applies its default of 25.

## Grade

```bash
export ANTHROPIC_API_KEY=...        # routed via the LangSmith gateway
uv run --with httpx python judge.py results.jsonl graded.jsonl
```

Grades SUCCESS / BLOCKED / FAILURE, where BLOCKED means a site bot-walled the
agent -- worth tracking apart from reasoning failures, since it says nothing
about the agent.

## Reading the numbers honestly

- **The judge is weaker than the official protocol.** WebVoyager grades with a
  vision model over the agent's screenshots; this grades the final text only. It
  catches vague or non-answering replies, but not a confident answer the page
  never supported. Scores are an upper bound and are not comparable to published
  WebVoyager results.
- **Single-sample LLM grading is noisy.** Re-grading the same results has flipped
  individual tasks. Treat a few points of difference between runs as noise, and
  sample more tasks before believing a change helped.
- **Site blocks are IP-dependent.** Allrecipes, Amazon, Booking and Google
  variously bot-wall datacenter IPs, and that varies by day and by sandbox.
