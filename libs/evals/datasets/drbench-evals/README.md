# Enterprise deep-research evals (DRBench, app mode)

100 Harbor tasks generated from ServiceNow's [DRBench](https://github.com/ServiceNow/drbench)
([paper](https://arxiv.org/abs/2510.00172)), the enterprise deep-research benchmark.

Each task hands the agent a company profile, a persona, and an open-ended research question
whose answer is split between the company's own systems and public information on the open
web. The company's documents are served by **running applications** — Nextcloud, Mattermost,
Roundcube/IMAP, and a file browser — so the agent has to navigate them over the network. It
writes a cited report; the verifier scores how many of the benchmark's ground-truth insights
it recovered, whether it avoided the planted distractors, whether its citations support its
claims, and how good the report is.

This is the `research` category of the unified evals workflow. It is opt-in.

## Running it

```
categories: research
sandbox_env: docker              # required — see below
runner_label: ubuntu-24.04-arm   # required — see below
profile: full
include_tasks: DR0001            # a single task, for a smoke test
concurrency: 1
```

**Two settings are not optional.** Upstream publishes the per-task images for **arm64 only**
("amd64 images are coming soon"), so the tasks need an arm64 runner. And a runner's
architecture only matters when the containers run *on* the runner, which means
`sandbox_env: docker` — with the default LangSmith sandbox the containers run off-runner and
the label has no effect.

`profile: lite` will not work for a single task: it intersects `include_tasks` against the
frozen lite list. Use `profile: full`.

## Runtime shape

Two compose services per task:

| Service | What it is |
|---|---|
| `main` | Where Harbor installs and runs the agent. Holds **no** task data. The verifier runs elsewhere — see [Scoring](#scoring). |
| `drbench` | Upstream's per-task image, pinned by digest. Boots its own supervisord with this task's documents already loaded. |

The agent reaches the apps by compose service name:

| App | Endpoint | Access |
|---|---|---|
| Nextcloud | `http://drbench:8081` | HTTP Basic; WebDAV `PROPFIND /remote.php/dav/files/<user>/` |
| Mattermost | `http://drbench:8082` | `POST /api/v4/users/login` → token in the `Token` header |
| Roundcube | `http://drbench:8085` | HTTP |
| IMAP | `drbench:1143` | `imaplib` |
| File browser | `http://drbench:8090` | HTTP |
| Health | `http://drbench:8099/health` | 200 only when every service is up |

Inside `main` the agent has `curl`, `extract-text` (the documents are PDF/DOCX/XLSX/PPTX/JSONL,
so anything downloaded is binary), `imaplib`, and a Tavily-backed `web_search` — the workflow
forwards `TAVILY_API_KEY` to this category only. `network_mode = "public"`, because 45 of the
613 gold insights are `external_fact` entries that exist only on the open web.

### Why two services

The agent's container is kept empty deliberately. Upstream's image contains
`/drbench/task/env.json`, which carries a `qa_type` **per document** — an explicit
insight-vs-distractor label for every file. An agent with filesystem access could read it and
skip the research entirely. Ground truth (`eval.json`) is not in the image, so this is not a
full answer leak, but it would defeat the distractor design. Two services remove the file from
the agent's reach structurally rather than by deletion.

### Readiness

`compose up --wait` only waits for containers to be *running*, and the image declares no
`HEALTHCHECK`, so it returns long before the apps are usable. The real gate is
`[environment].healthcheck` in `task.toml`, which Harbor runs in `main` **before it even
installs the agent**, polling `/health` until it returns 200.

## Credentials: two regimes

Which login works depends on the task, and it is upstream's doing rather than a choice here.
`task.toml` records which regime a task is in as `credential_regime`.

| Regime | Tasks | Login |
|---|---|---|
| `persona` | 15 | The persona's username with password `my_drbench_pwd`. DR0001's documents sit under Nextcloud's `emily.patel`. |
| `default` | 85 | Each app's built-in login — Nextcloud and file browser `admin` / `admin_pwd`, Mattermost `admin@drbench.com` / `mm_admin_pwd`, mail `current.user` / `current_user_pwd`. |

The 85 arise because their persona's `password` is `null` upstream, so DRBench's credential
override returns early and every app keeps its own login. Verified by unpacking the shipped
images: DR0016's documents are under Nextcloud's `admin` user and its mailbox is
`current.user`, not the persona. These are synthetic logins baked into a public image, not
secrets.

## Scoring

`tests/judge.py` calls **upstream's own metrics** — it installs `drbench` at the pinned commit
and hands the report to `drbench.score_report.score_report`. Claim extraction, citation
normalization, chunk retrieval, and every judging prompt are upstream's code, not a
reimplementation.

| Metric | Upstream class | Note |
|---|---|---|
| `insights_recall` | `QASimilarityV2` | Fraction of gold insights the report lets you derive. |
| `distractor_recall` | `DistractorRecall` | **Higher is worse** — the report swallowed planted material. |
| `factuality` | `CitationFactuality` | Per cited claim: resolve the cited source, chunk, rank by embedding similarity, judge. |
| `report_quality` | `ReportQuality` | Five criteria scored 1–10, averaged and divided by 10. |

The headline `reward` is the **harmonic mean** of `insights_recall`, `1 − distractor_recall`,
`factuality`, and `report_quality`. That is the paper's own aggregate (arXiv 2510.00172,
Table 2: *Insight Recall, Factuality, Distractor Avoidance, Report Quality, Harmonic Mean*),
which also defines distractor avoidance as `1 − distractor recall`. Upstream's released code
computes the four metrics but not the mean, so the combination happens in `judge.py`. The only
deviation from the paper is a 0.01 floor per component, so one zero craters the score without
erasing all ranking signal. All four components are written alongside it in `reward.json`, and
`/logs/verifier/drbench_metrics.json` carries the breakdown — read that when diagnosing a score.

### The verifier runs in its own environment

`[verifier].environment_mode = "separate"`, so Harbor builds a second image from `tests/` and
starts it only **after** the agent environment has been torn down. That is what makes installing
`drbench` safe: the package ships both the gold `eval.json` **and** the whole document corpus as
package data, and none of it may exist while the agent is running.

Two consequences:

- **`tests/case.json` holds no answers.** It is just the task id plus the upstream commit; the
  verifier looks up ground truth in the installed package. Nothing in the task directory
  contains a gold insight except `solution/solve.sh`, which is the oracle and is uploaded only
  by Harbor's `OracleAgent`.
- **The verifier never touches the app stack.** Cited documents are resolved from the corpus, so
  email, chat, file-browser, and Nextcloud sources all resolve as plain files. An earlier
  version re-fetched them over WebDAV and could therefore not resolve email or chat citations
  at all.

### Citations have to be resolvable

Scoring resolves each citation back to a source, and one it cannot resolve counts as
unsupported however accurate the claim. `instruction.md` therefore specifies the exact forms:
a file name for a document, a full URL for a web page,
`RoundCube-<sender address>-<recipient address>-<Subject>` for an email, and
`MatterMost-<channel>-<team>-<user>` for a chat message. Sender address and subject are matched
character for character, and a display name resolves nothing — every pattern in upstream's
`normalize_email_citation` requires an `@`.

## Building the dataset

**No task directory is committed.** All 100 are generated from upstream's configs at the
commit pinned as `UPSTREAM_SHA` in `harbor_adapters/drbench/adapter.py`, so this directory
holds only its `README.md`, `dataset.toml`, and `.gitignore` until you build it:

```bash
cd libs/evals
make dataset          # == python -m harbor_adapters.drbench.main --populate datasets/drbench-evals
```

That fetches upstream with a blobless, depth-1, **sparse** checkout — the 5 config files
per task (~2.4 MiB), skipping `drbench/data/tasks/*/files/` (~69 MiB), because in app mode
upstream's per-task image serves the documents. The whole build takes about two seconds and
is cached under `harbor_adapters/drbench/.upstream/`.

CI runs the same command in every research shard before `harbor run --path`, and the prep
job runs it before enumerating tasks to shard.

Two reasons the tasks are generated rather than committed: `solution/solve.sh` is the
benchmark's answer key (its gold insights), which does not belong in a public repository;
and 1,100 files that are a pure function of one commit hash add nothing to a diff.

`make dataset-check` builds twice and diffs, which is how CI proves generation is
deterministic now that the output is no longer reviewable in a PR.

To re-pin the images after upstream republishes them (the only step that talks to a registry):

```bash
python -m harbor_adapters.drbench.main --refresh-digests
```

See [`../../harbor_adapters/drbench/vendor/README.md`](../../harbor_adapters/drbench/vendor/README.md)
for what is vendored, the pinned upstream commit, and attribution.

## Operational notes

- **Disk is the binding constraint.** A per-task image is ~1.22 GiB compressed, roughly 3–4
  GiB extracted, against a runner's ~14 GB. Per-task images barely share layers (only ~158 MiB
  of DR0001 is shared with `:latest`, because it was committed on an earlier base), and Harbor
  never prunes — `down --rmi local` leaves pulled images behind. Run `concurrency: 1` and prune
  between trials; sharding wide beats stacking deep.
- **Images are pinned by digest** (`vendor/image_digests.json`) because the upstream tags are
  mutable, live in a personal namespace, and upstream publishes no version tags at all.
- **`force_build` is nearly a no-op** on the docker sandbox: it only switches a task declaring
  both `docker_image` and a Dockerfile over to building the Dockerfile.
- This is the only category not running on amd64 LangSmith sandboxes, so its numbers are not
  hardware-comparable to the others. Fine for an absolute DRBench score.
