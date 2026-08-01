# DRBench source data

This directory vendors the per-task configuration files of ServiceNow's
[DRBench](https://github.com/ServiceNow/drbench) enterprise deep-research
benchmark, pinned at commit `0d699ecf6aa96b1de378595b432e9b16a82f0ed9`:

- `tasks/<TASK_ID>/task.json` — DR question, persona, company profile, date
  (upstream `drbench/data/tasks/<TASK_ID>/config/task.json`)
- `tasks/<TASK_ID>/env.json` — the task's file manifest: source path, the app it
  is served from, and whether it carries an insight or is a distractor
- `tasks/<TASK_ID>/eval.json` — the ground-truth insights the report is scored
  against; consumed only by the verifier, never exposed to the agent
- `tasks/<TASK_ID>/info.json` — industry, domain, and the upstream difficulty label
- `tasks/<TASK_ID>/dr_question.json` — question provenance and sub-questions

Reproduced verbatim so our verifier scores the same insights the upstream
`insights_recall` metric does — see `../adapter.py` and `../templates/judge.py`.

The document corpus itself (`files/`, ~87 MB of PDF/DOCX/XLSX/PPTX/JSONL) is
**not** vendored. It is fetched from the pinned upstream tree by
`python -m harbor_adapters.drbench.main --populate <dataset_dir>`, which writes
it to each task's git-ignored `environment/files/`.

The source repository is licensed under Apache-2.0. Its unmodified `LICENSE`
file is included alongside this attribution. The upstream repository does not
provide a `NOTICE` file.
