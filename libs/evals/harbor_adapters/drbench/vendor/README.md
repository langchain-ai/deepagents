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

Reproduced verbatim so our verifier scores the same ground truth the upstream
metrics do — see `../adapter.py` and `../templates/judge.py`.

- `subsets/{minival,val,sanity}.jsonl` — upstream's own task subsets
  (upstream `drbench/data/subsets/`), copied verbatim. `minival.jsonl` is the
  15-task set the paper calls **MinEval** ("restricted to 15 tasks for efficient
  ablation studies"); the paper names the subset but not its task ids, so this
  file is the only authoritative list. It is what
  `.github/scripts/evals/lite_tasks.py` pins the `research` lite profile to, and a
  test asserts the two match — otherwise "lite is upstream's MinEval" would be a
  claim in a comment that nothing checks. Vendored rather than read from the
  installed package because `drbench` exists only inside the verifier image, never
  in the dev or CI environment.

Also here:

- `image_digests.json` — each task's upstream image resolved to an immutable
  `sha256:` digest. The tasks run in **app mode**, so the documents are served by
  upstream's per-task container rather than laid down on disk, and there is no
  corpus to vendor or fetch. Pinning by digest matters because the upstream tags
  are mutable, live in a personal namespace (`ghcr.io/mmunozm`), and upstream
  publishes no version tags at all — so a re-push would otherwise change eval
  results with no change on our side. Regenerate with
  `python -m harbor_adapters.drbench.main --refresh-digests`.

Note the images are published for **arm64 only** ("amd64 images are coming soon"
per upstream's README), which is why this category needs an arm64 runner and
`sandbox_env: docker`.

The source repository is licensed under Apache-2.0. Its unmodified `LICENSE`
file is included alongside this attribution. The upstream repository does not
provide a `NOTICE` file.
