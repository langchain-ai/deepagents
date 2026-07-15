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
`trec_coarse` subset at the 65536-token bucket (the full ~50-example set). To
evaluate another bucket, add a **new** dataset directory (see below) rather than
mutating this one.

CI populates the tasks before running Harbor; the workflow's "Populate local
dataset corpus" step runs, for this dataset:

```bash
python -m harbor_adapters.oolong.main --populate datasets/oolong-synth
```

which reads `bucket.toml` and fetches the matching rows.

## Adding another bucket

Each dataset directory is one `(subset, context_len)` bucket. To add another:

1. Copy this directory to `libs/evals/datasets/oolong-synth-<name>/` — the name
   must keep the `oolong-synth` prefix (see below). Keep `metric.py` and
   `.gitignore` as-is (the task dirs are git-ignored, so there is nothing else to
   copy), and edit `dataset.toml`'s `name` so it is distinct.
2. Edit `bucket.toml` to the new `subset` / `context_len` / `n_examples`.

That's it — run it with `dataset_path=datasets/oolong-synth-<name>`. No workflow
change is needed: the "Populate local dataset corpus" step in
`.github/workflows/_harbor_run.yml` dispatches every `datasets/oolong-synth*` path
to the OOLONG populate adapter, which reads that directory's `bucket.toml`. Each
directory is its own reproducible dataset (Harbor/LangSmith names it from the path).

## Run it locally

```bash
# from libs/evals/ — fetch the bucket + write the (git-ignored) task dirs
python -m harbor_adapters.oolong.main --populate datasets/oolong-synth
harbor run --path datasets/oolong-synth --n-tasks 1 -a oracle -e docker -k 1
# → Mean: 1.000  (the oracle writes the gold answer)
```

The adapter source (HF loader, official scorer, task generator) lives in
`libs/evals/harbor_adapters/oolong/`.

## Arms

The task is agent-agnostic, so the arm is a `harbor run` choice, not task content.
Both arms use the same generic Deep Agent; they differ only by the code interpreter:

- **Baseline** — the `bare_deepagent` graph. Select it with `agent_impl=bare`.
- **Code interpreter** — the `bare_ci_deepagent` graph: identical to `bare` plus
  the `CodeInterpreterMiddleware` (a QuickJS `eval` tool). Nothing is tailored to
  OOLONG, so `bare` vs `bare-ci` isolates the code interpreter's effect. Select it
  with `agent_impl=bare-ci`.

Both dispatch with `dataset_path=datasets/oolong-synth`. Locally, after
`make stage-harbor-local-deps` + `python -m harbor_adapters.oolong.main --populate
datasets/oolong-synth`, run either graph with
`harbor run --agent langgraph --ak graph=bare_ci_deepagent` (or
`graph=bare_deepagent`) against `--path datasets/oolong-synth`.
