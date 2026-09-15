# Labels

The label taxonomy for this repo, and which automation owns each label.
Written for future agents: before adding, renaming, or reading a label, find it
here first.

The file map for this folder is [`LAYOUT.md`](./LAYOUT.md); repo-wide CI
conventions are in root [`AGENTS.md`](../AGENTS.md).

## The model

> Issue or PR work type via `type:*` + one `package:*` + optional `topic:*` and
> `integration:*` + provenance via `org:*` + optional `priority:*` + one PR
> `size:*` + temporary `triage:*`, `auto:*`, and `ci:*` state.

| Prefix | Purpose | Applied by |
| --- | --- | --- |
| `type:*` | Issue/PR work type and PR breaking marker | Issue forms + PR title labeler + maintainers |
| `package:*` | Repository package | Labeler + maintainers |
| `topic:*` | Technical subject spanning packages | Maintainers |
| `integration:*` | External sandbox or service under `libs/partners/` | Labeler + maintainers |
| `org:*` | Author provenance | Automation |
| `priority:*` | Priority | Maintainers |
| `size:*` | PR diff size | Automation |
| `triage:*` | Issue management state | Maintainers + agents |
| `auto:*` | State owned by automation | Automation |
| `ci:*` | Human acknowledgement or override of a check | Maintainers only |

Rules that are easy to get wrong:

- **PR type labels mirror the Conventional Commit title.** The labeler derives
  the work type and optional breaking marker from the title, and `package:*`
  and `integration:*` from its scope. Labels support triage; release-please
  still reads Conventional Commits to determine releases. `release(...)`
  titles receive `auto:release-pr` for release and stale-PR automation.
- An issue carries exactly one `type:*`, normally one `package:*`, and any
  number of `topic:*`.
- `priority:*` has three levels. **Backlog is the absence of a priority
  label** — the retired `p3`/`p4` are not migrated onto `priority:backlog`.
- `ci:*` is never applied by automation. Every one of them is read by a gate.
- Every label must have a description.

## Automatic labels

### `size:*` — `pr_labeler.yml`

`size: XS` (<50 changed lines), `size: S` (<200), `size: M` (<500),
`size: L` (<1000), `size: XL` (rest). Mutually exclusive; stale ones removed.
Thresholds are `sizeThresholds` in
[`scripts/labeling/pr-labeler-config.json`](./scripts/labeling/pr-labeler-config.json).
`uv.lock` (`excludedFiles`) and `docs/` (`excludedPaths`) do not count.

### `package:*` and `integration:*` — `pr_labeler.yml`, `auto-label-by-package.yml`

`scopeToLabel` maps title scopes and `fileRules` maps path prefixes onto the
same labels:

| Label | Scopes | Paths |
| --- | --- | --- |
| `package:deepagents` | `sdk`, `deepagents` | `libs/deepagents/` |
| `package:dcode` | `code`, `deepagents-code` | `libs/code/` |
| `package:acp` | `acp`, `deepagents-acp` | `libs/acp/` |
| `package:talon` | `talon`, `deepagents-talon` | `libs/talon/` |
| `package:evals` | `evals`, `harbor` | `libs/evals/`, `libs/harbor/` |
| `package:examples` | `examples` | `examples/` |
| `integration:daytona` | `daytona`, `langchain-daytona` | `libs/partners/daytona/` |
| `integration:modal` | `modal`, `langchain-modal` | `libs/partners/modal/` |
| `integration:quickjs` | `quickjs`, `langchain-quickjs` | `libs/partners/quickjs/` |
| `integration:runloop` | `runloop`, `langchain-runloop` | `libs/partners/runloop/` |
| `integration:vercel` | `vercel`, `langchain-vercel-sandbox` | `libs/partners/vercel/` |
| `integration:langsmith` | `langsmith-sandbox` | — |

Additive: these are never removed on edit. `skipExcludedFiles: true` means a
lockfile-only change does not pull in a package label.

`pr_labeler.yml` first rewrites package-component title scopes to their
canonical PR scope (`deepagents`→`sdk`, `deepagents-code`→`code`, …) via
`scopeAliases` and `h.canonicalizeTitleScopes()`, skipping `release(...)`
titles whose scope is a canonical version record. Nothing in the workflow
hardcodes a scope.

`pr_scope_file_check.yml` reuses these two maps as package identity, so a label
rename changes that gate's behavior too — see
[`scripts/labeling/check_pr_scope_files.py`](./scripts/labeling/check_pr_scope_files.py).
`package:evals` deliberately covers both `libs/evals/` and `libs/harbor/`;
neither is a release-please component, so the merge costs the gate nothing.

### `topic:*` — maintainers

`topic:async-subagents`, `topic:backends`, `topic:filesystem`,
`topic:harness`, `topic:mcp`, `topic:memory`, `topic:middleware`,
`topic:models`, `topic:multimodal`, `topic:performance`, `topic:prompts`,
`topic:sandboxes`, `topic:skills`, `topic:streaming`, `topic:subagents`,
`topic:tracing`. No automation applies these. `topic:async-subagents` stays
distinct from `topic:subagents` because async execution has its own
implementation and operational concerns.

### `org:*` — `pr_labeler.yml` (PRs), `tag-external-issues.yml` (issues)

| Label | Meaning |
| --- | --- |
| `org:external` | author is not an active `langchain-ai` member |
| `org:internal` | author is a member, or a Bot |
| `org:open-swe` | PR from an `open-swe/` branch (`branchRules`) |

Applied on `opened` only, using the `ORG_MEMBERSHIP_APP_*` GitHub App token
(org membership is private). A non-404 membership error fails the step rather
than defaulting to external.

### `priority:*` — `sync_priority_labels.yml`

`priority:urgent` > `priority:high` > `priority:backlog`, mutually exclusive,
copied from issues linked by `Closes/Fixes/Resolves #N` onto the PR; highest
across linked issues wins. The workflow also strips the retired `p0`–`p4` from
PRs (`STALE_PRIORITY_LABELS`) without mapping them onto a new priority — drop
that list once no open item carries one.

### `auto:*` — lifecycle and release automation

| Label | Applied by | Meaning |
| --- | --- | --- |
| `auto:pending-deletion` | `close_old_prs.yml` (day 14 warning) | PR closes at day 30 unless exempted |
| `auto:waiting-on-author` | maintainer, cleared by `waiting_on_author_reply.yml` | closes the item 10 days later; `auto:*` because workflows own its removal and timeout |
| `auto:missing-issue-link` | `require_issue_link.yml` | external PR had no approved, assigned issue link; PR was closed |
| `auto:new-contributor` | `pr_labeler.yml` | external author, 0 merged PRs (PRs only) |
| `auto:trusted-contributor` | `pr_labeler.yml`, `tag-external-issues.yml` | external author, ≥`trustedThreshold` (5) merged PRs **in this repo** |
| `auto:release-pr` | PR title labeler + `release-please.yml` via `h.labelPR()` | package release PR (`releaseLabel` in the config) |
| `auto:release-pending` | release-please itself | release PR open, not yet tagged |
| `auto:release-tagged` | `release.yml` after tagging | release tagged |

`clear_pending_deletion.yml` drops `auto:pending-deletion` the moment
`ci:keep-open` lands. Thresholds (14/30 days) and the release exemption
(`RELEASE_LABELS`) live in
[`scripts/labeling/close-old-prs.js`](./scripts/labeling/close-old-prs.js).

> `auto:release-pending` / `auto:release-tagged` are release-please's own
> lifecycle labels, set via the `label` / `release-label` keys in
> [`release-please-config.json`](../release-please-config.json). release-please
> requires every configured pending label, so listing old and new names in
> config would require both rather than accept either. Before release-please
> runs, automation adds `auto:release-pending` to open PRs carrying the legacy
> `autorelease: pending` label. The publish guard reads either pending name;
> the publisher recognizes either pending/tagged name and removes both pending
> labels when applying `auto:release-tagged`. No manual label rename or pending
> label cleanup is needed. Let workflows running the old code finish before
> merging the migration; an already-running workflow cannot adopt this handling.

### `triage:*` — maintainers and agents

`triage:duplicate`, `triage:help-wanted`, `triage:needs-investigation`,
`triage:unable-to-reproduce`. Four durable states on purpose: missing
information is requested in a comment rather than tracked as another label
lifecycle.

### `type:*` — issue forms, PR title labeler, and maintainers

[`ISSUE_TEMPLATE/bug-report.yml`](./ISSUE_TEMPLATE/) applies `type:bug` and
`feature-request.yml` applies `type:feature`. Maintainers can also assign
`type:spike`, `type:chore`, or `type:docs` to issues. These replace GitHub Issue
Types so the repo owns the names and descriptions.

For PRs, `typeToLabel` in `pr-labeler-config.json` maps the title's commit type:

| Commit type | Label |
| --- | --- |
| `feat` | `type:feature` |
| `fix` | `type:bug` |
| `docs` | `type:docs` |
| `hotfix` | `type:hotfix` |
| `style` | `type:style` |
| `refactor` | `type:refactor` |
| `perf` | `type:performance` |
| `test` | `type:test` |
| `build` | `type:build` |
| `ci` | `type:ci` |
| `chore` | `type:chore` |
| `revert` | `type:revert` |
| `release` | `auto:release-pr` |

The `!` marker adds `type:breaking` (`breakingLabel` in the config) alongside
the work type. A recognized title edit replaces stale managed type labels and
removes the breaking label when `!` is dropped; unrecognized titles preserve
the previous classification. Live labeling, backfill, and release PR labeling
share this behavior. Scope/file labels remain additive. Newly created type
labels receive descriptions from `labelDescriptions` in the same config.

> A label in an issue form's `labels:` list that does not exist on the repo is
> **silently skipped** — GitHub applies nothing and reports nothing. Create the
> label before merging a form change.

## `ci:*` — human overrides

Read, never applied by automation. Each unblocks a gate that is otherwise red.

| Label | Gate it bypasses |
| --- | --- |
| `ci:skip-title-lint` | `pr_lint.yml` Conventional Commits title check |
| `ci:allow-scope-mismatch` | `pr_scope_file_check.yml` (title scope vs changed package dirs) |
| `ci:allow-lockfile-release` | `release_please_scope_check.yml` lockfile scope check |
| `ci:ack-markdown` | `markdown_file_check.yml` (non-`docs` PR adds `.md` files) |
| `ci:ack-readme` | `project_readme_check.yml` (non-`docs` PR edits a project README) |
| `ci:ack-release-deps` | `check_release_deps.yml`, `check_sdk_pin.yml` dependency freshness |
| `ci:dcode-skip-sdk-pin` | `check_sdk_pin.yml` SDK pin check; `release-please.yml` then dispatches with `dangerous-skip-sdk-pin-check=true` (a workflow input, not a label) |
| `ci:skip-curated-notes` | the curated release-notes gate (`release_notes_check.yml` via `scripts/release/release-notes.js`) |
| `ci:skip-issue-link` | `require_issue_link.yml`; also added by `reopen_on_assignment.yml` when a maintainer assigns the issue |
| `ci:skip-ripgrep` | strict ripgrep install failure on a release PR (`_test.yml`, surfaced by `ripgrep_timeout_comment.yml`) |
| `ci:allow-warnings` | warnings-as-errors in `_test.yml` (runs pytest with `-W default`) |
| `ci:bypass-fork-main` | `block_fork_main_prs.yml` |
| `ci:keep-open` | the `close_old_prs.yml` sweep; also set by a `!keep-open` comment (`keep_open_on_comment.yml`) |

Applying `ci:skip-title-lint`, `ci:allow-scope-mismatch`, or
`ci:allow-lockfile-release` triggers a sticky comment
(`release_fanout_bypass_warn.yml`, `pr_lint.yml`) spelling out the
release-please consequence. Those warnings are advisory and never fail.

## Mechanics worth knowing

- **Labels are created on demand.** `h.ensureLabel(name)` in
  [`scripts/labeling/pr-labeler.js`](./scripts/labeling/pr-labeler.js) does a
  get-then-create, so an applied label appears the first time it is used,
  colored `labelColor` from the config. A label that is only *read* (every
  `ci:*`) never auto-creates — **create those by hand**, or the gate offers a
  bypass nobody can select.
- **A batch containing one nonexistent label 422s entirely.** `pr_labeler.yml`
  calls `ensureLabel` for every label in `toAdd` before a single `addLabels`.
  Keep that ordering.
- **Events from the default `GITHUB_TOKEN` do not trigger other workflows.** A
  label applied with `secrets.GITHUB_TOKEN` fires no `labeled` event, so
  anything that must wake a downstream workflow (`org:external` →
  `require_issue_link.yml`) uses the App token. Same reason
  `keep_open_on_comment.yml` duplicates `clear_pending_deletion.yml`'s cleanup
  inline.
- **All PR labeling goes through one workflow on purpose.** `pr_labeler.yml`
  replaced four concurrent workflows that raced on label mutations. Add PR
  label logic inside it.
- **Bypass labels are read from the live API**, not the event payload, wherever
  a re-run must see a label added after the fact (see the `gh api` read in
  [`workflows/_test.yml`](./workflows/_test.yml)).

## Changing the taxonomy

1. **New package**: add a `scopeToLabel` key and a `fileRules` prefix in
   `pr-labeler-config.json`, add the scope to `pr_lint.yml`, and add the issue
   form option plus its `mapping` entry in `auto-label-by-package.yml`. Check
   `check_pr_scope_files.py`'s tests — it reads both maps as package identity.
2. **New `ci:*` gate**: read the label from the live API in the gate workflow,
   create the label by hand, and add a row above.
3. **Backfill**: open PRs via `pr_labeler_backfill.yml` (`workflow_dispatch`,
   `max_items`); open issues via `tag-external-issues.yml`'s dispatch job.
4. **Tests** live in [`scripts/tests/labeling/`](./scripts/tests/labeling/) and
   run in CI under `pytest .github/scripts/tests` ("Validate Release Options").
   Keep reusable label logic in `pr-labeler.js` and workflow steps thin. The
   labeler tests also execute the live and backfill scripts against a fake API.

Note for agents: a PR adding a Markdown file is red under
`markdown_file_check.yml` unless titled `docs(...)` or carrying `ci:ack-markdown`.
