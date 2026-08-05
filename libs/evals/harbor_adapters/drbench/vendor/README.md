# DRBench source data

Pins for ServiceNow's [DRBench](https://github.com/ServiceNow/drbench)
enterprise deep-research benchmark, at commit
`0d699ecf6aa96b1de378595b432e9b16a82f0ed9`.

**The per-task configuration files are not vendored here.** `../adapter.py` fetches
them from that commit at generation time, with a blobless, depth-1, sparse checkout
that takes only the five config files per task (~2.4 MiB) and skips the ~69 MiB
document corpus app mode never needs. A git commit hash is a hash of the content, so
the fetch is pinned exactly as firmly as a committed copy would be — without putting
500 verbatim upstream files, and 100 gold `solve.sh` answer keys derived from them, in
this repository. Build the dataset with `make dataset` from `libs/evals`.

What each task contributes, for reference when reading `../adapter.py`:

- `config/task.json` — DR question, persona, company profile, date
- `config/env.json` — the file manifest: source path, the app it is served from, and
  whether the file carries an insight or is a distractor
- `config/eval.json` — the ground-truth insights the report is scored against;
  consumed only by the verifier, never exposed to the agent
- `info.json` — industry, domain, and the upstream difficulty label
- `dr_question.json` — question provenance and sub-questions

What *is* vendored, because neither is derivable from a task config:

- `subsets/{minival,val,sanity}.jsonl` — upstream's own task subsets
  (upstream `drbench/data/subsets/`), copied verbatim. `minival.jsonl` is the
  15-task set the paper calls **MinEval** ("restricted to 15 tasks for efficient
  ablation studies"); the paper names the subset but not its task ids, so this
  file is the only authoritative list. It is what
  `.github/scripts/evals/lite_tasks.py` pins the `research` lite profile to, and a
  test asserts the two match — otherwise "lite is upstream's MinEval" would be a
  claim in a comment that nothing checks. Vendored rather than read from the
  installed package because `drbench` exists only inside the verifier image, never
  in the dev or CI environment. `val.jsonl` lists all 100 scored tasks and is what
  `adapter.available_task_ids()` reads, so `--all` and the workflow's prep phase
  resolve the task list with no network and before any task directory exists.

- `task_labels.json` — each task's `difficulty`, `industry`, and `domain`, read from
  upstream's `info.json` at the pinned commit. Committed because these labels are what
  make the `research` full profile's 30-task set checkable as a *proportional* sample of
  the 100 (6/7/17 against 20/23/57), and that assertion lives in
  `.github/scripts/tests/`, which never builds the dataset. Labels only — no question,
  document, or ground-truth content, so this is not the task duplication the fetch
  replaced. `make dataset-check` re-reads the configs and fails if the record disagrees,
  so it cannot drift when `UPSTREAM_SHA` is bumped. Regenerate with
  `python -m harbor_adapters.drbench.main --refresh-labels`.

- `image_digests.json` — each task's upstream image resolved to an immutable
  `sha256:` digest. The tasks run in **app mode**, so the documents are served by
  upstream's per-task container rather than laid down on disk, and there is no
  corpus to vendor or fetch. The image *name* is derivable — upstream tags each
  task's image with the task id (`ghcr.io/mmunozm/drbench-services:DR0001`) — so
  what this file actually carries is the digest, the one thing no git tree can.
  That matters because the upstream tags are mutable, live in a personal namespace
  (`ghcr.io/mmunozm`), and upstream publishes no version tags at all, so a re-push
  would otherwise change eval results with no change on our side. Regenerate with
  `python -m harbor_adapters.drbench.main --refresh-digests`.

Note the images are published for **arm64 only** ("amd64 images are coming soon"
per upstream's README), which is why this category needs an arm64 runner and
`sandbox_env: docker`. The generated compose file sets no `platform:`: upstream ships
a single-entry arm64 OCI index, so letting Docker match the host makes an amd64 runner
fail immediately at pull rather than emulate under qemu and blow the build timeout.

The source repository is licensed under Apache-2.0. Its unmodified `LICENSE`
file is included alongside this attribution. The upstream repository does not
provide a `NOTICE` file.
