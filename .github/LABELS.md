# Labels

## The model

> Issue or PR work type via `type:*` + one `package:*` + optional `topic:*` and `integration:*` + provenance via `org:*` + optional `priority:*` + one PR `size:*` + temporary `triage:*`, `auto:*`, and `ci:*` state.

| Prefix | Purpose | Applied by |
| --- | --- | --- |
| `type:*` | Issue/PR work type and PR breaking marker | Issue forms + PR title labeler + maintainers |
| `package:*` | Repository package | Labeler + maintainers |
| `topic:*` | Technical subject spanning packages | Labeler (paths + model classification) + maintainers |
| `integration:*` | External sandbox or service under `libs/partners/` | Labeler + maintainers |
| `org:*` | Author provenance | Automation |
| `priority:*` | Priority | Maintainers + automation (issue default and PR propagation) |
| `size:*` | PR diff size | Automation |
| `triage:*` | Issue management state | Maintainers + agents |
| `auto:*` | State owned by automation | Automation |
| `ci:*` | Human acknowledgement or override of a check | Maintainers (two applied on their instruction) |

Rules that are easy to get wrong:

- **PR type labels mirror the Conventional Commit title.** The labeler derives the work type and optional breaking marker from the title, and `package:*` and `integration:*` from its scope. Labels support triage; release-please still reads Conventional Commits to determine releases. `release(...)` titles receive `auto:release-pr` for release and stale-PR automation.
- An issue carries exactly one `type:*`, normally one `package:*`, and any number of `topic:*`.
- `priority:*` has three levels on issues, with `priority:backlog` as the default. Only `priority:urgent` and `priority:high` propagate to linked PRs; backlog leaves a PR without a priority label. Retired `p0`–`p4` labels are stripped from PRs without mapping them to a new priority.
- **A `ci:*` label always represents a human decision, but two of them are written by automation on a maintainer's instruction:** `ci:keep-open` (`keep_open_on_comment.yml`, on a `!keep-open` comment) and `ci:skip-issue-link` (`require_issue_link.yml`, when a maintainer bypasses the gate). The labeler never applies a `ci:*`, and every one of them is read by a gate. See the `ci:*` table for which are read-only.
- Every label must have a description.

## Automatic labels

### `size:*` — `pr_labeler.yml`

`size: XS` (<50 changed lines), `size: S` (<200), `size: M` (<500), `size: L` (<1000), `size: XL` (rest). Mutually exclusive; stale ones removed. Thresholds are `sizeThresholds` in [`scripts/labeling/pr-labeler-config.json`](./scripts/labeling/pr-labeler-config.json). `uv.lock` (`excludedFiles`) and `docs/` (`excludedPaths`) do not count.

### `package:*` and `integration:*` — `pr_labeler.yml`, `auto-label-by-package.yml`

`scopeToLabel` maps title scopes and `fileRules` maps path prefixes onto the same labels:

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

Package and integration labels are additive: title edits do not remove them. `pr_labeler.yml` normalizes package aliases such as `deepagents` → `sdk` before labeling. `pr_scope_file_check.yml` uses the same mappings to validate that PR title scopes match changed packages.

### `topic:*` — `pr_labeler.yml`, `auto-label-by-package.yml`, maintainers

`topic:async-subagents`, `topic:backends`, `topic:filesystem`, `topic:harness`, `topic:mcp`, `topic:memory`, `topic:middleware`, `topic:models`, `topic:multimodal`, `topic:performance`, `topic:prompts`, `topic:sandboxes`, `topic:skills`, `topic:streaming`, `topic:subagents`, `topic:tracing`. Any number may apply.

Two signals feed them, both additive. A topic is never removed, so a maintainer's hand-applied topic survives. Issue classification runs on `opened` only, so a removed topic is not re-added by a later edit:

- **Changed modules** (`topicFileRules`): a PR touching `middleware/subagents.py` gets `topic:subagents` (and `topic:middleware`, since the whole middleware dir maps too); `backends/sandbox.py` gets `topic:backends` and `topic:sandboxes`; `mcp_*.py` gets `topic:mcp`. Rules name a *module*, not a package, so the label means the diff actually touched that subject.
- **Issue wording**: `openai/gpt-oss-20b`, an open-weight production model available on Groq's Developer plan, classifies an issue's title and body against the cached `.github/topic-labels.json` manifest. PRs receive automatic topic labels from changed modules only.

`topic:async-subagents` stays distinct from `topic:subagents` because async execution has its own implementation and operational concerns. Path rules distinguish the implementations; wording classifications come from the model.

A daily workflow adds newly discovered repository topics to `.github/topic-labels.json` and opens or refreshes a PR when it changes. Existing choices are preserved because labels are created on demand; a topic absent from the repository may simply not have been used yet. A response containing no topic labels fails the sync without changing the manifest. To retire a topic, remove it explicitly from the manifest, the repository labels, and any `topicFileRules` entries that could recreate it.

Runtime labeling reads the local manifest, so it makes no extra label-list API call. Add a `topicFileRules` entry only when deterministic path-based matching is also useful; model output is filtered against the manifest before labels are applied.

### `org:*` — `pr_labeler.yml` (PRs), `tag-external-issues.yml` (issues)

| Label | Meaning |
| --- | --- |
| `org:external` | author is not an active `langchain-ai` member |
| `org:internal` | author is a member, or a Bot |
| `org:open-swe` | PR from an `open-swe/` branch (`branchRules`) |

`org:external` is applied on `opened` only, using the `ORG_MEMBERSHIP_APP_*` GitHub App token, because org membership is private. A non-404 membership error fails the step rather than defaulting to external. The other two come from the default-token step, so they fire no `labeled` event.

### `priority:*` — `auto-label-by-package.yml`, `sync_priority_labels.yml`

`priority:urgent` > `priority:high` > `priority:backlog`, mutually exclusive.

**Every new issue gets `priority:backlog`** from the "Apply default priority" step in `auto-label-by-package.yml`, which runs on `opened` before the Area sync so an issue filed without the form is still prioritized. It skips an issue that already carries a `priority:*`, so a re-run never overwrites an escalation.

`sync_priority_labels.yml` copies a priority from issues linked by `Closes/Fixes/Resolves #N` onto the PR, highest across linked issues winning — but **only `priority:urgent` and `priority:high` propagate** (`PROPAGATED_PRIORITY_LABELS`). Backlog is every issue's default, so copying it would label nearly every PR while saying nothing; a PR carrying a backlog label from an earlier run gets it stripped. The workflow also strips the retired `p0`–`p4` from PRs (`STALE_PRIORITY_LABELS`) without mapping them onto a new priority — drop that list once no open item carries one.

### `auto:*` — lifecycle and release automation

| Label | Applied by | Meaning |
| --- | --- | --- |
| `auto:pending-deletion` | `close_old_prs.yml` (day 14 warning) | PR closes at day 30 unless exempted |
| `auto:waiting-on-author` | maintainer, cleared by `waiting_on_author_reply.yml` and by the `waiting_on_author.yml` sweep when it detects an author reply | closes the item 10 days later; `auto:*` because workflows own its removal and timeout |
| `auto:missing-issue-link` | `require_issue_link.yml` | external PR had no approved, assigned issue link; PR was closed |
| `auto:new-contributor` | `pr_labeler.yml` | external author, 0 merged PRs (PRs only) |
| `auto:trusted-contributor` | `pr_labeler.yml`, `tag-external-issues.yml` | external author, ≥`trustedThreshold` (5) merged PRs **in this repo** |
| `auto:release-pr` | PR title labeler + `release-please.yml` via `h.labelPR()` | package release PR (`releaseLabel` in the config) |
| `auto:release-pending` | release-please itself | release PR open, not yet tagged |
| `auto:release-tagged` | `release.yml` after tagging | release tagged |

`clear_pending_deletion.yml` drops `auto:pending-deletion` the moment `ci:keep-open` lands. Thresholds (14/30 days) and the release exemption (`RELEASE_LABELS`) live in [`scripts/labeling/close-old-prs.js`](./scripts/labeling/close-old-prs.js).

> During migration, release workflows accept legacy `autorelease: pending` and `autorelease: tagged` labels. Before release-please runs, automation adds `auto:release-pending` to open PRs that still use the legacy pending label. Publishing then removes both pending-label variants and applies `auto:release-tagged`. Let workflows running the old code finish before merging.

### `triage:*` — maintainers and agents

`triage:duplicate`, `triage:help-wanted`, `triage:needs-investigation`, `triage:unable-to-reproduce`. Four durable states on purpose: missing information is requested in a comment rather than tracked as another label lifecycle.

### `type:*` — issue forms, PR title labeler, and maintainers

[`ISSUE_TEMPLATE/bug-report.yml`](./ISSUE_TEMPLATE/) applies `type:bug` and `feature-request.yml` applies `type:feature`. Maintainers can also assign `type:spike`, `type:chore`, or `type:docs` to issues. These replace GitHub Issue Types so the repo owns the names and descriptions.

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
| `release` | `auto:release-pr` (from `releaseLabel`, not `typeToLabel` — see [`auto:*`](#auto---lifecycle-and-release-automation)) |

The `!` marker immediately before `:` (for example, `feat(sdk)!:` or `feat!:`) adds `type:breaking` (`breakingLabel` in the config) alongside the work type. The label parser does not recognize `feat!(sdk):`. A recognized title edit replaces stale managed type labels and removes the breaking label when `!` is dropped; unrecognized titles preserve the previous classification. Live labeling, backfill, and release PR labeling share this behavior. Scope/file labels remain additive. Newly created type labels receive descriptions from `labelDescriptions` in the same config.

> A label in an issue form's `labels:` list that does not exist on the repo is **silently skipped** — GitHub applies nothing and reports nothing. Create the label before merging a form change.

## `ci:*` — human overrides

Each unblocks a gate that is otherwise red. Every one is read-only and must be created by hand except `ci:skip-issue-link` and `ci:keep-open`, which automation also applies on a maintainer's instruction (noted in their rows below). See [Mechanics](#mechanics-worth-knowing).

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
| `ci:skip-issue-link` | `require_issue_link.yml`, when a maintainer bypasses the gate; read (never applied) by `reopen_on_assignment.yml` |
| `ci:skip-ripgrep` | strict ripgrep install failure on a release PR (`_test.yml`, surfaced by `ripgrep_timeout_comment.yml`) |
| `ci:allow-warnings` | warnings-as-errors in `_test.yml` (runs pytest with `-W default`) |
| `ci:bypass-fork-main` | `block_fork_main_prs.yml` |
| `ci:keep-open` | the `close_old_prs.yml` sweep; also set by a `!keep-open` comment (`keep_open_on_comment.yml`) |

## Mechanics worth knowing

- Labels applied by `pr_labeler.yml` are created on demand, taking their description from `labelDescriptions` and their color from `labelColors` (keyed by taxonomy prefix) in `pr-labeler-config.json`, so a label created on demand matches the ones already on the repo. Labels that workflows only read, including most `ci:*` labels, must already exist in the repository.
- `labelColors` defines the color palette. `close-old-prs.js` and `normalize-release-labels.js` resolve colors from the config. `sync_priority_labels.yml` and `require_issue_link.yml` have no repository checkout from which to load the shared helper, so they inline colors. The palette test in `pr-labeler.test.js` rejects inline hex values outside the configured palette; it does not verify that each value matches the label's prefix.
- All PR label changes belong in `pr_labeler.yml` to avoid workflows racing to update the same labels.
- Workflows that must trigger follow-up label events use the GitHub App token because events created by the default `GITHUB_TOKEN` do not start other workflows.

## Changing the taxonomy

1. **New package**: add a `scopeToLabel` key and a `fileRules` prefix in `pr-labeler-config.json`, add the scope to `pr_lint.yml`, and add the issue form option plus its `mapping` entry in `auto-label-by-package.yml`. Check `check_pr_scope_files.py`'s tests — it reads both maps as package identity.
2. **New `ci:*` gate**: read the label from the live API in the gate workflow, create the label by hand, and add a row above.
3. **Backfill**: use `pr_labeler_backfill.yml` for open PRs and the dispatch job in `tag-external-issues.yml` for open issues.

`markdown_file_check.yml` flags new Markdown files in non-documentation PRs so reviewers explicitly acknowledge unexpected documentation changes, including files added by coding agents. After reviewing the file, apply `ci:ack-markdown`.
