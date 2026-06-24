# OOLONG long-context aggregation benchmark

Runs [OOLONG](https://arxiv.org/abs/2511.02817) (Bertsch et al., 2025) long-context
aggregation tasks through a Deep Agent, comparing two configurations on the **same**
examples:

- **`plain`** — a deep agent that delegates chunk-level analysis to general-purpose
  subagents via the `task` tool. The long context is seeded as `/context.txt` in the
  agent workspace.
- **`code_interpreter`** — a deep agent with the QuickJS `CodeInterpreterMiddleware`
  (an `eval` tool for JS aggregation), plus `task` subagents for chunk reads. This is
  the "RLM"-style arm from Zhang et al. 2025.

Files:

- `oolong_benchmarks.py` — dataset loader, official scorer, agent factories, runner.
- `test_oolong_rlm_vs_subagents.py` — the pytest module (parameterizes over examples).

---

## Quickstart (smoke test)

One example, the paper's `trec_coarse` dataset, smallest context bucket:

```bash
cd libs/evals

OOLONG_DATASET=trec_coarse OOLONG_CONTEXT_LEN=1024 OOLONG_N_EXAMPLES=1 OOLONG_ARM=plain \
LANGSMITH_TRACING=true LANGSMITH_TEST_SUITE=oolong-smoke \
LANGSMITH_EXPERIMENT=oolong-trec_coarse-gpt5mini-$(date +%Y-%m-%d) \
uv run --group test pytest tests/evals/test_oolong_rlm_vs_subagents.py \
  --model openai:gpt-5-mini --sub-model openai:gpt-5-mini -v --tb=short
```

This downloads only the 50 rows of that one bucket (~0.3 MB), not the whole split.

---

## Prerequisites

- **LangSmith tracing** must be on or the suite aborts: set `LANGSMITH_TRACING=true`
  (any of `LANGSMITH_TRACING` / `LANGSMITH_TRACING_V2` / `LANGCHAIN_TRACING_V2`) and a
  valid `LANGSMITH_API_KEY`.
- **`--model` is required** (the root/orchestrator model).
- **Model credentials** for the providers you use (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
  …). When routing through the LangSmith model gateway, set the provider key to your
  LangSmith service key value.
- `LANGSMITH_TEST_SUITE` names the LangSmith project/dataset; `LANGSMITH_EXPERIMENT`
  names the run. Use a descriptive experiment name, e.g. `{suite}-{model}-{date}`.
- Set `LANGSMITH_WORKSPACE_ID` to the deepagents OSS workspace so runs land there.

---

## Knobs

### Environment variables

| Variable             | Default   | Description                                                              |
| -------------------- | --------- | ------------------------------------------------------------------------ |
| `OOLONG_ARM`         | `plain`   | Which arm to run: `plain` or `code_interpreter`.                          |
| `OOLONG_DATASET`     | `agnews`  | Input subset. Use `trec_coarse` for the paper's exact dataset.           |
| `OOLONG_CONTEXT_LEN` | `8192`    | Token-length bucket (e.g. `1024`, `8192`, `32768`, `65536`, `131072`).   |
| `OOLONG_N_EXAMPLES`  | `5`       | Examples to run, balanced across task groups. `0` (or empty) = full bucket (50). |

### CLI options (pytest)

| Option                | Default              | Description                                       |
| --------------------- | -------------------- | ------------------------------------------------- |
| `--model`             | (required)           | Root/orchestrator model id, e.g. `openai:gpt-5`.  |
| `--sub-model`         | `openai:gpt-5-mini`  | Sub-model for `task` subagents (paper uses mini). |

> The **arm is not a CLI/parametrize dimension** — it's read from `OOLONG_ARM` at
> runtime so both arms produce the *same* example identity and line up for a
> side-by-side compare in LangSmith. Run each arm in its own invocation.

---

## Recipes

### Compare the two arms side-by-side

Run each arm in its own session with its own experiment name. Inputs are identical
across arms, so both experiments map to the same dataset examples and compare cleanly.

```bash
cd libs/evals
COMMON="OOLONG_DATASET=trec_coarse OOLONG_CONTEXT_LEN=131072 OOLONG_N_EXAMPLES=0 \
  LANGSMITH_TRACING=true LANGSMITH_TEST_SUITE=oolong-rlm-vs-subagents"

# Arm 1: plain subagents
env $COMMON OOLONG_ARM=plain \
  LANGSMITH_EXPERIMENT=oolong-plain-gpt5-$(date +%Y-%m-%d) \
  uv run --group test pytest tests/evals/test_oolong_rlm_vs_subagents.py \
    --model openai:gpt-5 --sub-model openai:gpt-5-mini -q

# Arm 2: code interpreter (RLM)
env $COMMON OOLONG_ARM=code_interpreter \
  LANGSMITH_EXPERIMENT=oolong-rlm-gpt5-$(date +%Y-%m-%d) \
  uv run --group test pytest tests/evals/test_oolong_rlm_vs_subagents.py \
    --model openai:gpt-5 --sub-model openai:gpt-5-mini -q
```

### Paper-matching run (`trec_coarse`, 131K, 50 tasks)

Set `OOLONG_CONTEXT_LEN=131072` and `OOLONG_N_EXAMPLES=0` (full 50-task bucket):

```bash
OOLONG_DATASET=trec_coarse OOLONG_CONTEXT_LEN=131072 OOLONG_N_EXAMPLES=0 OOLONG_ARM=plain \
LANGSMITH_TRACING=true LANGSMITH_TEST_SUITE=oolong-paper \
LANGSMITH_EXPERIMENT=oolong-trec_coarse-131k-gpt5-$(date +%Y-%m-%d) \
uv run --group test pytest tests/evals/test_oolong_rlm_vs_subagents.py \
  --model openai:gpt-5 -q
```

### Default analog dataset (`agnews`)

`agnews` (test split) is a same-shape 4-way classification analog kept as the cheap
default. Just omit `OOLONG_DATASET`:

```bash
OOLONG_CONTEXT_LEN=8192 OOLONG_N_EXAMPLES=10 OOLONG_ARM=plain \
LANGSMITH_TRACING=true LANGSMITH_TEST_SUITE=oolong-agnews \
LANGSMITH_EXPERIMENT=oolong-agnews-8k-gpt5mini-$(date +%Y-%m-%d) \
uv run --group test pytest tests/evals/test_oolong_rlm_vs_subagents.py \
  --model openai:gpt-5-mini -q
```

### Sweep context lengths

```bash
for ctx in 1024 8192 32768 131072; do
  OOLONG_DATASET=trec_coarse OOLONG_CONTEXT_LEN=$ctx OOLONG_N_EXAMPLES=0 OOLONG_ARM=plain \
  LANGSMITH_TRACING=true LANGSMITH_TEST_SUITE=oolong-ctx-sweep \
  LANGSMITH_EXPERIMENT=oolong-trec-${ctx}-gpt5-$(date +%Y-%m-%d) \
  uv run --group test pytest tests/evals/test_oolong_rlm_vs_subagents.py --model openai:gpt-5 -q
done
```

---

## Datasets, splits & context buckets

OOLONG-synth ([`oolongbench/oolong-synth`](https://huggingface.co/datasets/oolongbench/oolong-synth))
is partitioned across two HuggingFace splits by source dataset:

| Subset        | Split        | Notes                                                       |
| ------------- | ------------ | ---------------------------------------------------------- |
| `trec_coarse` | `validation` | The paper's dataset. 50 tasks per context bucket.         |
| `agnews`      | `test`       | Default analog (same task-group structure, 50/bucket).    |
| others (`spam`, `imdb`, `yahoo`, …) | `test` | Additional source datasets.            |

The split is resolved automatically per subset (`trec_coarse` → `validation`, else
`test`), so you only set `OOLONG_DATASET`. Context buckets exist at `1024`, `4096`,
`8192`, `16384`, `32768`, `65536`, `131072` (and larger). The paper reports `131072`.

Task groups: `counting`, `user`, `timeline` (which groups are present varies by
subset/bucket; `OOLONG_N_EXAMPLES` balances across whatever groups exist).

---

## Data loading & caching

Rows are fetched **targeted** via the HuggingFace datasets-server `/filter` endpoint —
only the rows matching `dataset` + `context_len`, server-side — and cached to JSONL.
This avoids `datasets.load_dataset`, which downloads the entire split parquet:

| Split        | Whole-split download | Targeted (50 rows)        |
| ------------ | -------------------- | ------------------------- |
| `validation` | ~2.0 GB / 4.3 GB mem | ~0.3 MB @ 1024, ~35 MB @ 131K |
| `test`       | ~10 GB / 19 GB mem   | proportional to bucket    |

Cache location: `tests/evals/.cache/oolong/{split}__{subset}__ctx{N}.jsonl`
(git-ignored). Delete a file to force a re-fetch.

---

## Scoring & metrics

The scorer is a faithful port of the official OOLONG harness
([`abertsch72/oolong` `src/eval/eval_helpers.py`](https://github.com/abertsch72/oolong/blob/main/src/eval/eval_helpers.py):
`synth_attempt_answer_parse` + `synth_process_response`), so results are directly
comparable to the paper:

1. **Parse** — split the model answer on `:`, take the last segment, strip
   `*`/`[`/`]` artifacts; normalize long comparison answers to the phrase.
2. **Exact match** → `1.0`.
3. **`ANSWER_TYPE.COMPARISON`** → substring match for
   "more common" / "less common" / "same frequency".
4. **`ANSWER_TYPE.NUMERIC`** → partial credit `0.75 ** |gold − pred|`.
5. **`ANSWER_TYPE.DATE`** → flexible `dateutil` parse, then equality.

### LangSmith feedback & outputs

Scoring is **soft** (logged as feedback, never `pytest.fail`) so a wrong answer still
syncs its example and stays in the side-by-side compare view.

OOLONG defines exactly **one** metric per example — `score` — and the paper reports its
mean. We log that and nothing that competes with it:

| Feedback key         | Meaning                                                              |
| -------------------- | ------------------------------------------------------------------- |
| `score`              | **The OOLONG metric** (0–1, numeric partial credit folded in). The paper reports the mean of this. |
| `agent_steps`        | Number of agent steps — harness efficiency telemetry, *not* part of OOLONG. |
| `tool_call_requests` | Total tool calls — harness efficiency telemetry, *not* part of OOLONG. |

We deliberately **do not** log a binary "correctness": it is not in the OOLONG spec, is
redundant for LABEL/COMPARISON/DATE (already 0/1), and would discard partial credit on
NUMERIC tasks.

Per-example **outputs** mirror the official record for debuggability:
`attempted_parse` (parsed prediction), `parse_confidence` (`low`/`med`/`high`/`vhigh`),
`score`, and `gold_answer`.

> To reproduce a paper number, aggregate the **mean `score`** over the bucket.

> Note: the pytest terminal summary prints a `correctness: N.NN` line — that is the
> harness-wide **pytest pass-rate**, unrelated to the OOLONG score. Because this suite
> scores softly (never fails), it is always `1.00` here; ignore it and read `score`.

---

## Notes

- The official parser strips markdown/brackets and normalizes comparison phrases only
  when the answer contains a `:` (e.g. `Label: X`, `Answer: N`). The arm system prompts
  and the OOLONG questions themselves steer the model toward that format.
- `marks`: this module is tagged `eval_category("long_context_aggregation")` and
  `eval_tier("hillclimb")`, so it runs under `--eval-category long_context_aggregation`.

## Citation

```bibtex
@article{bertsch2025oolong,
  title={Oolong: Evaluating Long Context Reasoning and Aggregation Capabilities},
  author={Bertsch, Amanda and Pratapa, Adithya and Mitamura, Teruko and Neubig, Graham and Gormley, Matthew R.},
  journal={arXiv preprint arXiv:2511.02817},
  year={2025}
}
```
