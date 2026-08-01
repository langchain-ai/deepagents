# RubricMiddleware with LangSmith tracing

A runnable version of the `RubricMiddleware` end-to-end tests, driven by real
models. The agent drafts an engineering brief, a grader model scores it against
a rubric, and the middleware feeds the failing criteria back to the agent until
every criterion is verifiably satisfied or the iteration budget runs out.

The trace shows what the tests can only assert on: the grader payload for each
pass, the frozen criterion checklist replayed on later passes, and the revision
prompts injected back into the agent.

## Setup

```bash
cp .env.example .env   # then fill in your keys
```

`.env` is gitignored. `ANTHROPIC_API_KEY` and `LANGSMITH_API_KEY` are required;
`LANGSMITH_PROJECT` is optional and defaults to `deepagents-rubric-example`.
The script also finds a `.env` higher up the tree, so an existing repo-root one
works without copying anything. To point at a specific file instead:

```bash
python rubric_agent.py --env-file ../../libs/evals/.env
```

## Run

```bash
uv run --with deepagents --with "langchain[anthropic]" --with python-dotenv \
    python rubric_agent.py
```

Or, from a checkout with the core package already installed:

```bash
cd ../../libs/deepagents && uv run python ../../examples/rubric_middleware/rubric_agent.py
```

## What to look for

The script prints every grader verdict as it arrives, then a summary:

- **`criteria: N frozen after the first pass`** — the criterion list the first
  grading pass derived from the rubric prose. Later passes are held to exactly
  this list, so the criterion set cannot shrink mid-run.
- **`(downgraded: grading was incomplete)`** — a `satisfied` verdict that did
  not account for every criterion, even after one corrective retry. The
  middleware rewrites it to `needs_revision` rather than ending the loop on an
  unbacked pass.
- **revision prompts** — each includes the failing criteria with their gaps,
  the criteria that already pass, and an instruction not to regress them.

The rubric is deliberately demanding, so a first-pass `satisfied` is unlikely;
expect two or three iterations. Raise `MAX_ITERATIONS` in the script to give the
agent more room.
