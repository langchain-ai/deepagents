# Fork Customizations

> Upstream: [langchain-ai/deepagents](https://github.com/langchain-ai/deepagents)
> Fork maintained by: @ashsolei
> Last reviewed: 2026-04-08
> Fork type: **active-dev**
> Sync cadence: **monthly**

## Purpose of Fork

LangChain deep-agent runtime with iAiFy-slimmed CI footprint.

## Upstream Source

| Property | Value |
|---|---|
| Upstream | [langchain-ai/deepagents](https://github.com/langchain-ai/deepagents) |
| Fork org | AiFeatures |
| Fork type | active-dev |
| Sync cadence | monthly |
| Owner | @ashsolei |

## Carried Patches

Local commits ahead of `upstream/main` at last review:

- `95ca92eb chore: sync CLAUDE.md and copilot-instructions docs`
- `ebf90f3f chore(deps): bump cbor2 from 5.8.0 to 5.9.0 in /libs/cli (#1)`
- `8656f71d docs: update FORK-CUSTOMIZATIONS.md with upstream source`
- `2cce95dd docs: add FORK-CUSTOMIZATIONS.md per enterprise fork governance`
- `d527fa1e ci: add copilot-setup-steps.yml for Copilot Workspace`
- `cebe284a chore: add CLAUDE.md`
- `6fb8c0d3 chore: add copilot-instructions.md`
- `26036478 chore: add Copilot Coding Agent setup steps`
- `0497ae0f chore: remove misplaced agent files from .github/copilot/agents/`
- `6b85e016 chore: deploy core custom agents from AgentHub`
- `857bb625 chore: deploy core Copilot agents from AgentHub`
- `94c633cc docs: add FORK-CUSTOMIZATIONS.md`
- `55339067 chore: remove workflow tag-external-issues.yml — enterprise cleanup`
- `d902243c chore: remove workflow require_issue_link.yml — enterprise cleanup`
- `3d65bfa2 chore: remove workflow release.yml — enterprise cleanup`
- `6a485999 chore: remove workflow release-please.yml — enterprise cleanup`
- `eb4369c3 chore: remove workflow pr_lint.yml — enterprise cleanup`
- `0bb24686 chore: remove workflow pr_labeler.yml — enterprise cleanup`
- `40842ddc chore: remove workflow evals.yml — enterprise cleanup`
- `2e4fee9f chore: remove workflow deepagents-example.yml — enterprise cleanup`
- `2631ae59 chore: remove workflow ci.yml — enterprise cleanup`
- `a3c26d84 chore: remove workflow check_versions.yml — enterprise cleanup`
- `f8e0afb5 chore: remove workflow check_lockfiles.yml — enterprise cleanup`
- `6296964d chore: remove workflow check_extras_sync.yml — enterprise cleanup`
- `190370df chore: remove workflow auto-label-by-package.yml — enterprise cleanup`
- ... (2 more commits ahead of `upstream/main`)

## Supported Components

- Root governance files (`.github/`, `CLAUDE.md`, `AGENTS.md`, `FORK-CUSTOMIZATIONS.md`)
- Enterprise CI/CD workflows imported from `Ai-road-4-You/enterprise-ci-cd`

## Out of Support

- All upstream source directories are tracked as upstream-of-record; local edits to core source are discouraged.

## Breaking-Change Policy

1. On upstream sync, classify per `governance/docs/fork-governance.md`.
2. Breaking API/license/security changes auto-classify as `manual-review-required`.
3. Owner triages within 5 business days; conflicts are logged to the `fork-sync-failure` issue label.
4. Revert local customizations only after stakeholder sign-off.

## Sync Strategy

This fork follows the [Fork Governance Policy](https://github.com/Ai-road-4-You/governance/blob/main/docs/fork-governance.md)
and the [Fork Upstream Merge Runbook](https://github.com/Ai-road-4-You/governance/blob/main/docs/runbooks/fork-upstream-merge.md).

- **Sync frequency**: monthly
- **Conflict resolution**: Prefer upstream; reapply iAiFy customizations on a sync branch
- **Automation**: [`Ai-road-4-You/fork-sync`](https://github.com/Ai-road-4-You/fork-sync) workflows
- **Failure handling**: Sync failures create issues tagged `fork-sync-failure`

## Decision: Continue, Rebase, Refresh, or Replace

| Option | Current Assessment |
|---|---|
| Continue maintaining fork | yes - active iAiFy product scope |
| Full rebase onto upstream | feasible on request |
| Fresh fork (discard local changes) | not acceptable without owner review |
| Replace with upstream directly | not possible (local product value) |

## Maintenance

- **Owner**: @ashsolei
- **Last reviewed**: 2026-04-08
- **Reference runbook**: `ai-road-4-you/governance/docs/runbooks/fork-upstream-merge.md`
