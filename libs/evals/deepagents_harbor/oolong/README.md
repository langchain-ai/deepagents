# OOLONG → Harbor dataset (proof-of-concept)

Generates a [Harbor](https://www.harborframework.com) dataset from the
[OOLONG-synth](https://huggingface.co/datasets/oolongbench/oolong-synth)
long-context aggregation benchmark (Bertsch et al., 2025). This is the Harbor
counterpart to the pytest + LangSmith eval in PR #4213 — same data, same
official scorer, different substrate (Docker-isolated task dirs instead of
in-process pytest).

## Why a generator (not hand-authored tasks)

OOLONG is parameterized by `(dataset, context_len)` buckets of ~50 examples
each. Hand-authoring task dirs doesn't scale and would discard the targeted HF
loader the original eval already has. So tasks are **generated**: pull a bucket,
emit one self-contained Harbor task dir per row.

## Layout

```
oolong/
├── loader.py                 # HF datasets-server /filter fetch → OolongExample (vendored from PR #4213)
├── official_scorer.py        # upstream OOLONG scorer, verbatim (MIT)
├── generate_oolong_tasks.py  # bucket → Harbor task dirs + dataset.toml + metric.py
└── dataset/                  # generated output (a Harbor dataset)
    ├── dataset.toml
    ├── metric.py             # mean `score` across tasks
    └── oolong-synth-<subset>-<ctxlen>-<id>/
        ├── task.toml         # schema_version 1.3; no-network agent + verifier
        ├── instruction.md    # the OOLONG question + "write answer to /app/answer.txt"
        ├── environment/
        │   ├── Dockerfile    # python:3.13-slim; bakes python-dateutil + /context.txt
        │   └── context.txt   # the long-context document (the eval input)
        ├── solution/solve.sh # oracle: writes the gold answer (for solvability checks)
        └── tests/
            ├── test.sh            # runs score.py, always emits a reward
            ├── score.py           # grades /app/answer.txt → /logs/verifier/reward.json
            ├── official_scorer.py # vendored copy (verifier is self-contained)
            └── datapoint.json     # gold + answer_type + ids — agent never sees this
```

## Generate

```bash
# from libs/evals/ — regenerate the committed dataset (trec_coarse @ 65536, 50 tasks)
make oolong-tasks

# or call the generator directly for a custom bucket / smaller smoke set
python3 deepagents_harbor/oolong/generate_oolong_tasks.py \
    --dataset trec_coarse --context-len 65536 --n-examples 0
```

The committed dataset is the `trec_coarse` **65536-token** bucket — the
north-star bucket PR #4213 evaluates (`1024`/`8192` are only useful as cheap
plumbing smokes). `--n-examples 0` emits the full ~50-row bucket. Fetched rows
are cached under `.cache/` (gitignored), so re-runs are offline. Stdlib-only —
no agent stack needed to generate.

## The agent-agnostic design

The agent is an **agent configuration, not task content** — as in PR #4213,
where the arm is runtime metadata, never an input. The generated task only says
"read `/app/context.txt`, write your answer to `/app/answer.txt`," so any Harbor
agent can run it via `harbor run --agent ... --model ...`. We ship the
code-interpreter (RLM) agent (`oolong_code_interpreter`); the plain
no-code-interpreter baseline is just the existing `bare_deepagent` graph run
against the same dataset, so there's no separate plain graph.

## Scoring

The verifier runs the **official** `synth_process_response` over the agent's
`/app/answer.txt` and writes `{"score": <0..1>}` to
`/logs/verifier/reward.json` (real-valued — NUMERIC partial credit preserved).
`metric.py` averages `score` across the dataset.

## Verified (oracle)

```bash
# from libs/evals/, using an env with harbor installed (--n-tasks 1 = one task from the dataset)
harbor run -p deepagents_harbor/oolong/dataset --n-tasks 1 \
    -a oracle -e docker -k 1
# → Mean: 1.000
```

The oracle (`solve.sh`) writes the gold answer; the full build → agent →
verifier → reward loop scores `1.0`. A wrong/empty answer scores `0.0`.

## Running the code-interpreter (RLM) agent

`oolong_code_interpreter` is a graph in the shared
`deepagents_harbor/langgraph_project/langgraph.json`, with its factory in the
standalone `oolong_graph.py` (imports only `deepagents`; lazy-imports
`langchain_quickjs`). It fans out `general-purpose` `task` subagents from inside
a QuickJS `eval` program (`CodeInterpreterMiddleware`) and aggregates in JS.

```bash
# from libs/evals/, after: make stage-harbor-local-deps
harbor run \
  -p deepagents_harbor/oolong/dataset --n-tasks 1 \
  --agent langgraph \
  --ak project_path=deepagents_harbor/langgraph_project \
  --ak config=langgraph.json \
  --ak graph=oolong_code_interpreter \
  --model anthropic:claude-sonnet-4-6 \
  --ae 'ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}' \
  --ae 'UV_PRERELEASE=allow' \
  --env docker -k 1
# (agent reads /app/context.txt, writes /app/answer.txt; verifier grades it)
```

Verified end-to-end in Docker: the agent invokes the `eval` tool and scores 1.0
on the POC task. For a non-code-interpreter baseline, run `--ak graph=bare_deepagent`
against the same dataset.

The subagent model defaults to the root model; override with
`--ak sub_model=openai:gpt-5-mini` (the paper's asymmetric setup) or
`OOLONG_SUB_MODEL`.

### Two shared-integration fixes this required

Running any langgraph graph in Harbor was broken by two issues, both fixed here
so the single `langgraph.json` works for every graph:

- **quickjs staging gap** — `deepagents-code` now declares a path dependency on
  `../partners/quickjs`, which `make stage-harbor-local-deps` didn't stage; the
  in-container editable install of `deepagents-code` failed. Fixed by staging
  `partners/quickjs` into `.local_deps/`.
- **fireworks pre-release** — `langchain-fireworks>=1.4.2` requires
  `fireworks-ai>=1.2.0a71` (a pre-release) that `uv` won't select by default.
  Fixed by passing `UV_PRERELEASE=allow` as an agent env var (`--ae`), which the
  installed-agent setup merges into its in-container `uv pip install`. It's added
  to the Makefile's shared `HARBOR_AGENT_ENV_ARGS`, so every Harbor run gets it.

**Why the dataset defaults to `--network public`:** Harbor's `langgraph` agent
builds its venv *inside* the container during agent setup (it pip-installs
there), so the environment needs network at setup time. The hardened
`--network no-network` variant (agent-phase allowlist to the LLM host only) is
kept for once the agent's deps are pre-baked into the image so setup needs no
network.

## Not yet done (next steps)

- Generate the full bucket / multiple `(dataset, context_len)` buckets.
- Pre-bake agent deps so the hardened `--network no-network` allowlist works.
- When PR #4213 merges, dedupe `loader.py` / `official_scorer.py` against the
  canonical copies there.
```
