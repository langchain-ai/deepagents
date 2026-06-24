# OOLONG long-context aggregation evals

Plain **subagents** vs. **code-interpreter (RLM)** on the OOLONG-synth
long-context aggregation benchmark (Zhang et al. 2025), wired into LangSmith for
side-by-side comparison.

- `benchmarks.py` — dataset loader, agent factories, and the `run_oolong_case`
  entry point. Pulls any `(dataset, context_len)` bucket straight from
  HuggingFace (`oolongbench/oolong-synth`), so new eval sets need **no new
  code** — just new env vars.
- `official_scorer.py` — the **official** OOLONG scorer, vendored verbatim from
  [abertsch72/oolong](https://github.com/abertsch72/oolong) (MIT). Don't edit;
  re-vendor if upstream changes.
- `test_rlm_vs_subagents.py` — the pytest module (one case per example).

## The two arms

| Arm | `OOLONG_ARM` | What it does |
| --- | --- | --- |
| Subagents | `plain` | Deep agent delegates chunk reads/classification to `task` subagents. |
| RLM | `code_interpreter` | Deep agent uses the QuickJS `eval` tool for aggregation (+ `task` subagents for reads). |

The arm is **not** a pytest parameter — it's read from `OOLONG_ARM` at runtime.
Both arms run the *same* examples with the *same* logged inputs, so they share
one dataset example and line up in the LangSmith compare view. The arm travels
as **run metadata** + the experiment name, never as an input (putting it in the
inputs would split the example by arm and break the comparison).

## Prerequisites

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_WORKSPACE_ID=495db4ef-0b85-4f00-8deb-326a2ec006ec   # OSS workspace
# LANGSMITH_API_KEY is already in your env.
# Gateway auth workaround (empty ANTHROPIC_API_KEY fails SDK validation):
export ANTHROPIC_API_KEY="<langsmith service key>"
```

Run from `libs/evals/`.

## Choosing an eval set

Everything is driven by env vars — mix and match to pull any bucket from HF:

| Env var | Default | Meaning |
| --- | --- | --- |
| `OOLONG_DATASET` | `trec_coarse` | HF subset (`trec_coarse`, `agnews`, `spam`, …). |
| `OOLONG_CONTEXT_LEN` | `65536` | Token bucket (`1024`…`4194304`; `65536`/`131072` most useful). |
| `OOLONG_N_EXAMPLES` | `5` | Examples to run; `0` = the full 50-task bucket. |
| `OOLONG_ARM` | `plain` | `plain` (subagents) or `code_interpreter` (RLM). |

`trec_coarse` is in the HF `validation` split; other subsets are in `test`
(handled automatically).

### Dataset (test-suite) naming convention

Name the LangSmith dataset after the bucket so different `(dataset, context_len)`
combinations don't collide into one dataset:

```
LANGSMITH_TEST_SUITE=oolong-<dataset>-<context_len>   # e.g. oolong-trec_coarse-65536
```

## Running both arms (one comparison)

Run each arm in its own session, **same** `LANGSMITH_TEST_SUITE`, **different**
`LANGSMITH_EXPERIMENT`. The first session creates the dataset + examples; the
second reuses the same examples (identical inputs), so both experiments line up.

```bash
# Arm 1 — subagents
LANGSMITH_TRACING=true \
LANGSMITH_WORKSPACE_ID=495db4ef-0b85-4f00-8deb-326a2ec006ec \
LANGSMITH_TEST_SUITE=oolong-trec_coarse-65536 \
LANGSMITH_EXPERIMENT=oolong-subagents-sonnet-4-6-$(date +%F) \
OOLONG_ARM=plain \
ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" \
uv run --no-sync pytest tests/evals/oolong/test_rlm_vs_subagents.py \
  --model claude-sonnet-4-6 -q

# Arm 2 — RLM (code interpreter)
LANGSMITH_TRACING=true \
LANGSMITH_WORKSPACE_ID=495db4ef-0b85-4f00-8deb-326a2ec006ec \
LANGSMITH_TEST_SUITE=oolong-trec_coarse-65536 \
LANGSMITH_EXPERIMENT=oolong-rlm-sonnet-4-6-$(date +%F) \
OOLONG_ARM=code_interpreter \
ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" \
uv run --no-sync pytest tests/evals/oolong/test_rlm_vs_subagents.py \
  --model claude-sonnet-4-6 -q
```

Each run prints its LangSmith compare URL. Open either and add the other
experiment to see them side by side.

### Fast smoke run

Shrink the work to prove the plumbing without paying for 65K-token contexts:

```bash
OOLONG_N_EXAMPLES=2 OOLONG_CONTEXT_LEN=8192 LANGSMITH_TEST_SUITE=oolong-trec_coarse-8192 ...
```

## What gets logged

Each example is a faithful, self-contained copy of the HF record (so the full
content is inspectable in the LangSmith UI), with the north-star feedback set:

- **inputs**: every HF field — identifiers (`dataset`, `task_id`, `task_type`,
  `task_group`, `answer_type`, `context_len`, `input_subset`,
  `context_window_id`, `num_labels`) and content (`question`,
  `context_window_text`, `context_window_text_with_labels`). Inputs are
  display-only: the agent reads the document from its workspace (`/context.txt`),
  not from the logged inputs.
- **reference output**: `{"answer": "['less common than']"}` (raw gold).
- **run outputs**: the official per-example record — `attempted_parse`,
  `parse_confidence`, `score`, `gold_answer` (+ `final_text`).
- **feedback**: `score` — **the only OOLONG metric** (0-1, with NUMERIC partial
  credit; the paper reports its mean) — plus `agent_steps` and
  `tool_call_requests` (harness efficiency telemetry, the point of plain-vs-RLM;
  not OOLONG metrics).
- **run metadata**: `arm`, `sub_model`, `model`, … (kept off the inputs so both
  arms share one example).
- **latency**: tracked automatically by LangSmith per run.

## Data loading

`load_oolong_examples` fetches only the matching `(dataset, context_len)` rows
from the HuggingFace **datasets-server `/filter`** endpoint (paginated,
server-side `where`) and caches them to `.cache/oolong/*.jsonl` — *not*
`datasets.load_dataset`, which would download the whole split parquet (~2 GB
`validation` / ~10 GB `test`) just to keep ~50 rows. A bucket is ~0.3 MB at 1024
tokens, ~35 MB at the 131K paper bucket. The cache is gitignored; delete it to
re-fetch.

## Scoring

`score` comes from the official `synth_process_response` (vendored verbatim in
`official_scorer.py`): exact match, COMPARISON phrase containment, NUMERIC
partial credit (`0.75 ** |Δ|`), and DATE equality via `dateutil`. It is the only
OOLONG metric; we deliberately don't log a binary `correct` (not in the spec; it
would discard NUMERIC partial credit).
