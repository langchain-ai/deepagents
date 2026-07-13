# OOLONG-synth (Harbor dataset)

A Harbor dataset for the [OOLONG-synth](https://huggingface.co/datasets/oolongbench/oolong-synth)
long-context aggregation benchmark (Bertsch et al., 2025). Each task seeds a long
document at `/app/context.txt` and asks the agent to write its answer to
`/app/answer.txt`; the verifier grades it with the official OOLONG scorer and
writes `/logs/verifier/reward.json`. `metric.py` averages the per-task `score`.

## The data is pulled, not committed

The per-task directories (`oolong-synth-*/`, including each `environment/context.txt`)
are **not** committed — they are git-ignored and regenerated on demand from the
HuggingFace datasets-server. Only `dataset.toml`, `metric.py`, and `bucket.toml`
live in git.

`bucket.toml` records the bucket this directory represents (`subset` /
`context_len` / `n_examples`), so the directory always reproduces the same
dataset. It is the single source of truth for what gets fetched — this dir is the
north-star `trec_coarse` subset at the 65536-token bucket (the full ~50-example
set). To evaluate another bucket, add a **new** dataset directory with its own
`bucket.toml` (rather than mutating this one).

CI populates the tasks before running Harbor; the workflow's "Populate local
dataset corpus" step runs, for this dataset:

```bash
python -m harbor_adapters.oolong.main --populate datasets/oolong-synth
```

which reads `bucket.toml` and fetches the matching rows.

## Run it locally

```bash
# from libs/evals/
make oolong-populate                       # fetch the bucket + write task dirs
harbor run --path datasets/oolong-synth --n-tasks 1 -a oracle -e docker -k 1
# → Mean: 1.000  (the oracle writes the gold answer)
```

The adapter source (HF loader, official scorer, task generator) lives in
`libs/evals/harbor_adapters/oolong/`.

## Arms: RLM vs. baseline

The task is agent-agnostic, so the arm is a `harbor run` choice, not task content:

- **Code-interpreter (RLM)** — the `oolong_code_interpreter` graph fans out
  `general-purpose` subagents from inside a QuickJS `eval` program and aggregates
  in JavaScript. In the manual Harbor workflow, select it with `agent_impl=oolong-rlm`.
- **Baseline** — the existing `bare_deepagent` graph against the same dataset.
  Select it with `agent_impl=bare`.

Both dispatch with `dataset_path=datasets/oolong-synth`. Locally, after
`make stage-harbor-local-deps` + `make oolong-populate`, run either graph with
`harbor run --agent langgraph --ak graph=oolong_code_interpreter` (or
`graph=bare_deepagent`) against `--path datasets/oolong-synth`. The subagent model
defaults to the root model; override with `OOLONG_SUB_MODEL` for the paper's
asymmetric setup.
