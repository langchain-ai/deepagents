# release-please runner

`.github/workflows/release-please.yml` runs release-please from this directory instead of using `googleapis/release-please-action`. The action ships a prebuilt bundle, so it can't carry fixes that upstream hasn't released. Installing the npm package with pnpm lets us pin the version and patch it.

| File | Purpose |
| --- | --- |
| `package.json`, `pnpm-lock.yaml` | Pin `release-please` and pnpm (`packageManager`) to exact versions. |
| `pnpm-workspace.yaml`, `patches/` | pnpm [`patchedDependencies`](https://pnpm.io/settings#patcheddependencies): the patch is applied on install, and an install fails if it no longer applies. |
| `run.mjs` | Loads the manifest, opens or updates release PRs, and writes the same `pr` / `prs` outputs as the action. It checks that each change in the patch is applied before calling GitHub. |

## What the patch changes

pnpm keeps one patch file per package version, so every change lives in `patches/release-please@<version>.patch`.

| Change | Why | Upstream |
| --- | --- | --- |
| `build/src/util/commit-exclude.js` | Lets `exclude-paths` name a single file (e.g. `libs/code/uv.lock`), not only a directory. | [googleapis/release-please#2868](https://github.com/googleapis/release-please/pull/2868) |
| `build/src/util/tag-name.js` | Parses tags whose separator is longer than one character. Our tags use `==` (`deepagents-code==0.2.7`); unpatched, every tag parses as component `deepagents-code=`, so release lookup pages through up to 400 releases, logs a warning for each, and falls back to exact tag lookups. | None yet |

Remove a change once the pinned release-please includes the fix.

## Running locally

```bash
pnpm install --dir .github/release-please --frozen-lockfile --ignore-scripts
GITHUB_TOKEN="$(gh auth token)" GITHUB_REPOSITORY=langchain-ai/deepagents DRY_RUN=true \
  node .github/release-please/run.mjs
```

`DRY_RUN=true` prints the release PRs that would be opened or updated and writes nothing to GitHub. It reads from GitHub, so a run takes about as long as the workflow step.

## Changing the patch

```bash
cd .github/release-please
pnpm patch release-please@<version>   # prints a directory with the current patch applied
# edit files under build/src/ in that directory
pnpm patch-commit <directory>         # rewrites the patch, pnpm-workspace.yaml, and the lockfile
```

## Upgrading release-please

1. Bump the version in `package.json` and run `pnpm install --ignore-scripts`. pnpm fails because the patch names the old version.
2. Run `pnpm patch release-please@<new version>`, reapply the changes that upstream doesn't include yet, and `pnpm patch-commit`. Delete the old patch file and its `pnpm-workspace.yaml` entry.
3. Compare a `DRY_RUN=true` run against the open `release(...)` PRs.
