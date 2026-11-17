// Run release-please's manifest release-PR flow and write the same `pr` / `prs`
// outputs as googleapis/release-please-action. See README.md for why this
// replaces the action.
//
// Environment:
//   GITHUB_TOKEN       token used for all GitHub API calls (required)
//   GITHUB_REPOSITORY  `owner/repo` (required)
//   GITHUB_OUTPUT      step output file; set by Actions
//   DRY_RUN            when `true`, print the release PRs that would be opened
//                      or updated instead of writing anything to GitHub

import { appendFileSync } from 'node:fs';
import { randomUUID } from 'node:crypto';
import { createRequire } from 'node:module';

import { GitHub, Manifest, VERSION } from 'release-please';

const require = createRequire(import.meta.url);
const { CommitExclude } = require('release-please/build/src/util/commit-exclude');
const { TagName } = require('release-please/build/src/util/tag-name');

const CONFIG_FILE = 'release-please-config.json';
const MANIFEST_FILE = '.release-please-manifest.json';

// Fail before touching GitHub if the install did not apply a patch. Each check
// exercises the patched behavior directly rather than grepping source.
function assertPatched() {
  const commit = { sha: 'x', message: 'chore: relock', files: ['pkg/uv.lock'] };
  const kept = new CommitExclude({ pkg: { excludePaths: ['pkg/uv.lock'] } })
    .excludeCommits({ pkg: [commit] });
  if (kept.pkg.length !== 0) {
    throw new Error('the commit-exclude.js change in the release-please patch is not applied: exclude-paths ignores file paths');
  }

  const tag = TagName.parse('deepagents-code==0.2.7');
  if (tag?.component !== 'deepagents-code' || tag.separator !== '==') {
    throw new Error('the tag-name.js change in the release-please patch is not applied: `==` tags parse with the wrong component');
  }
}

function setOutput(name, value) {
  const file = process.env.GITHUB_OUTPUT;
  if (!file) return;
  const text = typeof value === 'string' ? value : JSON.stringify(value);
  const delimiter = `ghadelimiter_${randomUUID()}`;
  appendFileSync(file, `${name}<<${delimiter}\n${text}\n${delimiter}\n`);
}

async function main() {
  assertPatched();
  console.log(`Running release-please version: ${VERSION}`);

  const { GITHUB_TOKEN: token, GITHUB_REPOSITORY: repository } = process.env;
  if (!token) throw new Error('GITHUB_TOKEN is not set');
  const [owner, repo] = (repository || '').split('/');
  if (!owner || !repo) throw new Error(`GITHUB_REPOSITORY must be owner/repo, got ${JSON.stringify(repository)}`);

  const github = await GitHub.create({ owner, repo, token });
  const manifest = await Manifest.fromManifest(
    github,
    github.repository.defaultBranch,
    CONFIG_FILE,
    MANIFEST_FILE,
  );

  // The action also runs `createReleases()` first. It is a no-op here:
  // `skip-github-release` is set for every package, so each strategy returns no
  // releases and no labels change (release.yml creates releases and flips
  // labels). Skipping it saves a manifest load and a merged-PR search.

  if (process.env.DRY_RUN === 'true') {
    const candidates = await manifest.buildPullRequests();
    const summary = candidates.map(pr => ({
      title: pr.title.toString(),
      branch: pr.headRefName,
      body: pr.body.toString(),
    }));
    console.log(JSON.stringify(summary, null, 2));
    return;
  }

  const prs = (await manifest.createPullRequests()).filter(pr => pr !== undefined);
  setOutput('prs_created', String(prs.length > 0));
  if (prs.length) {
    setOutput('pr', prs[0]);
    setOutput('prs', prs);
  }
}

main().catch(err => {
  console.log(`::error::release-please failed: ${err.message}`);
  process.exitCode = 1;
});
