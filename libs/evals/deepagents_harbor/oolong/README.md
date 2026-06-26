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
# from libs/evals/
python3 deepagents_harbor/oolong/generate_oolong_tasks.py \
    --dataset trec_coarse --context-len 1024 --n-examples 1
```

`--n-examples 0` emits the full bucket. The fetched rows are cached under
`.cache/` (gitignored), so re-runs are offline. Stdlib-only — no agent stack
needed to generate.

## The arm-agnostic design

The two OOLONG arms (plain subagents vs. code-interpreter/RLM) are **agent
configurations, not task content** — exactly as in PR #4213, where the arm is
runtime metadata, never an input. So the generated task only says "read
`/context.txt`, write your answer to `/app/answer.txt`." Which arm runs is a
`harbor run --agent ... --model ...` choice. Comparing arms = two experiments
over this one dataset, lined up by Harbor's job/metric machinery (the analogue
of the PR's "both arms share one LangSmith example").

## Scoring

The verifier runs the **official** `synth_process_response` over the agent's
`/app/answer.txt` and writes `{"score": <0..1>}` to
`/logs/verifier/reward.json` (real-valued — NUMERIC partial credit preserved).
`metric.py` averages `score` across the dataset.

## Verified (oracle)

```bash
# from libs/evals/, using an env with harbor installed
harbor run -p deepagents_harbor/oolong/dataset/oolong-synth-trec_coarse-1024-10000000 \
    -a oracle -e docker -k 1
# → Mean: 1.000
```

The oracle (`solve.sh`) writes the gold answer; the full build → agent →
verifier → reward loop scores `1.0`. A wrong/empty answer scores `0.0`.

## Running the two arms

Both arms from PR #4213 are graphs in the shared
`deepagents_harbor/langgraph_project/langgraph.json`, with factories in the
standalone `oolong_graph.py` (imports only `deepagents`; the RLM arm
lazy-imports `langchain_quickjs`):

| Arm | `--ak graph=` | Substrate |
| --- | --- | --- |
| Plain | `oolong_plain` | fan-out to `general-purpose` `task` subagents |
| RLM | `oolong_code_interpreter` | fan-out + aggregation inside a QuickJS `eval` (`CodeInterpreterMiddleware`) |

```bash
# from libs/evals/, after: make stage-harbor-local-deps
harbor run \
  -p deepagents_harbor/oolong/dataset/oolong-synth-trec_coarse-1024-10000000 \
  --agent langgraph \
  --ak project_path=deepagents_harbor/langgraph_project \
  --ak config=langgraph.json \
  --ak graph=oolong_plain \            # or: oolong_code_interpreter
  --model anthropic:claude-sonnet-4-6 \
  --ae 'ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}' \
  --ae 'UV_PRERELEASE=allow' \
  --env docker -k 1
# → Mean: 1.000   (agent reads /app/context.txt, writes /app/answer.txt; verifier grades it)
```

Both arms are verified end-to-end in Docker (each scores 1.0 on the POC task;
the RLM arm invokes the `eval` tool). Run them as two experiments over the same
dataset for the side-by-side comparison that is the point of the original eval.

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

- Generate the full bucket / multiple `(dataset, context_len)` buckets and run
  both arms as two experiments for the side-by-side comparison.
- Pre-bake agent deps so the hardened `--network no-network` allowlist works.
- When PR #4213 merges, dedupe `loader.py` / `official_scorer.py` against the
  canonical copies there.
```
