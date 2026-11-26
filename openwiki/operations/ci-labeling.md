---
type: operations guide
title: CI Gates and Labels
description: Operational map of merge-facing CI, acknowledgement labels, release publication, release notes, and workflow trust boundaries. Explains what blocks a merge or release, where labels apply, and how credentials are scoped.
tags: [ci, github-actions, labels, releases, security]
verified:
  - by: openwiki/0.4.2
    at: 2026-11-25T08:04:06.996Z
sources:
  - id: openwiki-source-4e1a48a53f7b34ae16b8d91f
    resource: repo://.github/LABELS.md
  - id: openwiki-source-9d81aa681a56a98960013750
    resource: repo://.github/scripts/checks/check_lockfiles_pre_commit.py
  - id: openwiki-source-c0c09f6fe83cf1ab6089bb24
    resource: repo://.github/scripts/checks/ci_gate.py
  - id: openwiki-source-f59dcf85e4ad8ca3f9f6d5de
    resource: repo://.github/scripts/checks/curated_apply_only.py
  - id: openwiki-source-07ac367f59bbed1812fef155
    resource: repo://.github/scripts/labeling/check_pr_scope_files.py
  - id: openwiki-source-3943bedb80b27cdf64ea8bff
    resource: repo://.github/scripts/release/build_release_notes.py
  - id: openwiki-source-d0bc66880aeeb0817674e1c3
    resource: repo://.github/scripts/release/check_lockfile_release_scope.py
  - id: openwiki-source-c01179865033a68574568db7
    resource: repo://.github/scripts/release/check_open_release_fanout.py
  - id: openwiki-source-92f0f267bd28e544f8eba170
    resource: repo://.github/scripts/tests/checks/test_check_lockfiles_pre_commit.py
  - id: openwiki-source-276657d0dcc1b71922f5e1f4
    resource: repo://.github/scripts/tests/labeling/test_check_pr_scope_files.py
  - id: openwiki-source-74e4ee6ae7c1e3e1008eb655
    resource: repo://.github/scripts/tests/release/test_build_release_notes.py
  - id: openwiki-source-a3e73f18ddc0439259e73c17
    resource: repo://.github/scripts/tests/release/test_check_lockfile_release_scope.py
  - id: openwiki-source-bdbb800adcdf22c6c535f06d
    resource: repo://.github/scripts/tests/release/test_check_open_release_fanout.py
  - id: openwiki-source-75ab344a500d6a7392581a42
    resource: repo://.github/scripts/tests/workflows/test_ci_workflow.py
  - id: openwiki-source-5ec08fb431e595bde89de502
    resource: repo://.github/scripts/tests/workflows/test_inherited_ci_diagnostics.py
  - id: openwiki-source-ce9e844e8d33dbc3e766d8f1
    resource: repo://.github/scripts/tests/workflows/test_workflow_secret_scoping.py
  - id: openwiki-source-8d4ac162fca0a57f00bb83b7
    resource: repo://.github/SECRETS.md
  - id: openwiki-source-4d9cccca7700db7220ec055e
    resource: repo://.github/workflows/_test.yml
  - id: openwiki-source-fa3e3af42f9e8d6ed58a473e
    resource: repo://.github/workflows/check_lockfiles.yml
  - id: openwiki-source-164e2da859b5277df81c7d94
    resource: repo://.github/workflows/ci.yml
  - id: openwiki-source-4d4bdb4ee2c259e9a0b5b8f7
    resource: repo://.github/workflows/pr_scope_file_check.yml
  - id: openwiki-source-504736637a54f8573036c2df
    resource: repo://.github/workflows/release_fanout_bypass_warn.yml
  - id: openwiki-source-9f478a7a77ce5bdf50ca09e4
    resource: repo://.github/workflows/release_please_fanout_watch.yml
  - id: openwiki-source-7f3822065cb7dc086752e9d5
    resource: repo://.github/workflows/release_please_scope_check.yml
  - id: openwiki-source-4d1d392666be6dfdd7a91a2e
    resource: repo://.github/workflows/release.yml
generated: { by: "openwiki/0.4.2", at: "2026-11-25T08:04:06.996Z" }
---

# CI Gates and Labels

`🔧 CI` owns the aggregate `✅ CI Success` check. Separate PR workflows enforce scope, release-fan-out, trailer, Markdown, README, and dependency policies; they block only when branch protection requires their individual check. Labels are recorded human acknowledgements read by specific workflows, not authorization controls. Likewise, an `environment:` selection is deployment configuration—not proof that secrets, reviewers, branch policy, or an external trusted publisher are configured. See [Development](development.md), [Security](security.md), [Testing guide](../testing/testing-guide.md), and [the source map](../architecture/source-map.md).

## Merge CI and its aggregate gate

`ci.yml` runs for pull requests, merge groups, and pushes to `main`. It always calls the reusable lockfile check. On non-push events it separately runs text hygiene over the PR or merge-group range; package lint/test work is path-selected for PRs and merge groups, but every package job runs on `main`. The filters deliberately include CI infrastructure and editable-SDK dependencies.

```mermaid
flowchart TD
    Trigger["PR merge group or main push"] --> Lock["Check all lockfiles"]
    Trigger --> Hygiene["Run text hygiene on non-push"]
    Trigger --> Select["Select package work"]
    Lock --> Gate["CI Success gate"]
    Hygiene --> Gate
    Select --> Tests["Selected lint and test jobs"]
    Tests --> Gate
    Gate --> Check["CI Success check"]
    Check --> Protection["Branch protection decides merge"]
```

*The workflow emits the aggregate check; repository branch-protection configuration determines whether it is a merge requirement.*

`ci_success` runs even after dependencies finish unsuccessfully. Lockfiles must succeed; hygiene must succeed when scheduled (its push-time `skipped` result is expected). The trusted-base gate script judges the remaining non-skipped jobs. The sole failure waiver is `lint-talon` or `test-talon` on a pull request that does not change `libs/talon/**`; the workflow reports the waiver and required follow-up. All other failures block. Merge groups remain strict; only push cancellations are tolerated, because concurrency replacement is expected there.

The gate script is sparse-checked out from `github.base_ref || github.sha` without persisted credentials. Thus, after bootstrap, a PR cannot replace the logic that decides its own aggregate result. A curated release-bot changelog-only apply is a narrow exception to normal package work: only a conclusive successful parent `✅ CI Success` produces a passing inherited gate; an inherited failure remains failed. Diagnostics are read-only and cannot alter that decision. Re-run **all** jobs after fixing the parent state, because a failed-job-only rerun retains the earlier detection output.

### Lockfiles, reusable tests, and limited acknowledgements

The lockfile workflow performs a credential-less checkout and checks every discovered package when called without explicit paths. For explicit paths, its checker finds owning package directories, follows local `[tool.uv.sources]` consumers, and reports the package-specific `uv lock --check` remediation. Do not make lockfile validation advisory or path-bypass it.

```bash
make -C libs hygiene
make -C libs lock-check
```

The reusable `_test.yml` validates JSON matrix inputs before expanding a non-fail-fast matrix, configures the local `uv_setup` action, synchronizes the test group, runs platform-specific tests, and verifies a clean worktree. It reads bypass labels from the live GitHub API only for PR runs and fails closed by retaining strict behavior if the lookup fails. `ci:allow-warnings` supplies `-W default`; `ci:skip-ripgrep` is restricted to release-sensitive PR runs, where it can tolerate a failed ripgrep install and record the lost coverage rather than silently claiming it passed. Neither affects push or merge-group enforcement.

## Labels acknowledge consequences

The `ci:*` taxonomy is a maintainer-facing acknowledgement/override contract. Most labels must already exist in the repository because readers do not create them. Applying one changes a named check's result; it does not grant credentials, weaken other checks, or eliminate the release consequence.

| Label | Acknowledges | Result and remaining consequence |
| --- | --- | --- |
| `ci:allow-scope-mismatch` | A PR title scope does not cover changed package directories | The scope check passes with a sticky explanation; release-please may still create a release PR for each touched component. |
| `ci:allow-lockfile-release` | Intentional multi-component fan-out | The fan-out check passes with an informational note; its historical name predates `uv.lock` exclusion, and each component can still receive a release PR. |
| `ci:allow-warnings` | Warnings-as-errors on a PR test run | `_test.yml` runs pytest with `-W default`; merge-group and push runs stay strict. |
| `ci:skip-ripgrep` | A release-PR ripgrep installation failure | The missing rg-gated coverage is recorded; normal push and merge-group checks remain strict. |
| `ci:ack-release-deps` | Review of stale dependency advice on a release PR | It does not relax built-wheel metadata validation or PyPI-only installation. |

### Scope and fan-out controls

`🔍 PR scope file check` compares title scopes with package identities derived from the base revision's labeler configuration and an API-derived changed-file list. It ignores lockfile churn and allows `release(...)` only for a fully allowlisted set of release artifacts. It fails closed on incomplete file enumeration, invalid detector output, or configuration errors. The detector/configuration checkout is from the base SHA without persisted credentials; PR title and changed-file facts are inputs. This prevents a PR from editing the detector or config to self-bypass, but does not by itself make the workflow unchangeable: branch protection and review of workflow edits remain necessary.

`🔍 Release-please scope check` uses the same pattern for release fan-out. Its trusted-base detector reads managed component paths, excluded paths, and bump-worthy types from release-please configuration. A bump-worthy PR that changes release-relevant files in two or more components blocks unless `ci:allow-lockfile-release` acknowledges it. Excluded paths such as `tests/` and `uv.lock` do not count. The companion bypass-warning workflow and the post-merge fan-out watch are advisory safety nets, not merge gates.

## Release dispatch and publication boundary

`🚀 Package Release` is dispatch-only rather than reusable because PyPI Trusted Publishing does not support the reusable-workflow path. It is normally triggered by release automation after a release PR merge, but can also be manually dispatched. The normal path requires an explicit resolvable release SHA reachable from `main` and verifies that the selected package's `pyproject.toml` version equals the supplied version. `dangerous-nonmain-release` is the explicit backport/hotfix exception: it can use the dispatch SHA when no SHA is supplied and skips the normal main/version equality checks, while still requiring the target to be reachable from the dispatched commit.

```mermaid
flowchart TD
    Dispatch["Release dispatch"] --> Build["Build validated release SHA"]
    Build --> Checks["Pre-release checks"]
    Build --> Notes["Generate release notes"]
    Checks --> TestPyPI["TestPyPI publish"]
    TestPyPI --> PyPI["PyPI publish"]
    PyPI --> Tag["Create GitHub release and tag"]
    Notes --> Tag
    Tag --> Labels["Update release PR lifecycle label"]
    PyPI --> Pin["Best effort SDK pin-bump dispatch"]
```

*Artifacts flow from a read-only build job through checksum verification into the minimal OIDC publishing jobs; notes are intentionally fail-open.*

The build job has read-only repository permissions, checks out the validated SHA without persisted credentials, records distribution checksums, and uploads `dist`. The TestPyPI and PyPI jobs download that artifact, compare its checksums with the build output, and have only `id-token: write`; they neither check out source nor receive broad repository permissions. Both select `environment: release`, but this is not itself an authorization guarantee: GitHub environment protection and each package's external PyPI/TestPyPI trusted-publisher registration must be configured separately. The focused secret-scoping test enforces the YAML invariants that these two jobs use exactly that environment and permission set.

Pre-release checks run from the resolved SHA with no dependency cache. `dangerous-skip-sdk-pin-check` only bypasses the Code SDK-pin comparison. `dangerous-skip-ripgrep-check` still attempts installation and makes a degraded result explicit when rg-gated artifact tests could not run. The disabled release integration-test step retains package-conditional credential wiring as a future contract, but it currently runs with `if: false`; the release environment intentionally has no provider credentials for it.

After successful PyPI publication, `mark-release` creates the GitHub tag/release with the validated SHA and transitions a pending release PR to `auto:release-tagged`. It first looks up the PR associated with the commit, then uses an exact title plus pending-label fallback; a normal release fails the final job if that lifecycle update cannot be completed. A successful `deepagents` publish may dispatch the Code SDK pin-bump using a GitHub App token with `actions: write`; that post-publish convenience is best effort and cannot make the release fail.

## Release notes and recovery

The `release-notes` job checks out full history and tags at the release SHA, but checks out `build_release_notes.py` from the workflow revision into `_workflow`. It passes the already resolved package, version, working directory, SHA, and prerelease state to the helper, keeping the release body and GitHub prerelease flag consistent while allowing the helper to be fixed independently of an older release commit.

The helper is dependency-free Python for local recovery. It resolves a predecessor reachable from the release SHA: stable releases prefer the previous patch then the newest reachable stable tag; prereleases prefer an earlier same-base prerelease, then the base tag, then a stable tag. It reads the changelog at the release SHA, renders a package-scoped and bounded git log, and uses bounded GitHub CLI lookups to credit contributors while warning when attribution is incomplete. Run it from a full clone with tags when reconstructing a body:

```bash
python .github/scripts/release/build_release_notes.py \
  --package "${PKG_NAME}" --version "${VERSION}" \
  --sha "${RELEASE_SHA}" --repo "${REPOSITORY}" \
  --actor "${ACTOR}" --base-branch "${BASE_BRANCH}" \
  --out /tmp/release-body.md

gh release edit "${PKG_NAME}==${VERSION}" --repo "${REPOSITORY}" \
  --notes-file /tmp/release-body.md
```

Release notes deliberately fail open: a notes failure does not prevent a successful pre-release check and PyPI publication from creating the GitHub release, but the result is an empty release body and `mark-release` writes a recovery command to the job summary. Do not add the notes result as a prerequisite to publishing.

## Focused verification

Run the narrow tests after changing this area:

```bash
python -m pytest .github/scripts/tests/checks/test_ci_gate.py
python -m pytest .github/scripts/tests/checks/test_check_lockfiles_pre_commit.py
python -m pytest .github/scripts/tests/labeling/test_check_pr_scope_files.py
python -m pytest .github/scripts/tests/release/test_check_lockfile_release_scope.py
python -m pytest .github/scripts/tests/release/test_build_release_notes.py
python -m pytest .github/scripts/tests/workflows/test_ci_workflow.py
python -m pytest .github/scripts/tests/workflows/test_workflow_secret_scoping.py
```

The workflow tests cover aggregate-gate and inherited-CI behavior. The release-note tests construct histories to exercise predecessor selection, changelog extraction, bounded logs, and incomplete attribution diagnostics; secret-scoping tests ensure publishing jobs retain their minimal OIDC permission and selected environment.
