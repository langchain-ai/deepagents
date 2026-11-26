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
import { execFileSync } from 'node:child_process';

import { GitHub, Manifest, VERSION } from 'release-please';

import { fetchOverrideBodies, gitMergeCommits } from './history.mjs';

const require = createRequire(import.meta.url);
const { CommitExclude } = require('release-please/build/src/util/commit-exclude');
const { TagName } = require('release-please/build/src/util/tag-name');

const CONFIG_FILE = 'release-please-config.json';
const MANIFEST_FILE = '.release-please-manifest.json';

// Fail before touching GitHub if the install did not apply a patch. Each check
// exercises the patched behavior directly rather than grepping source.
async function assertPatched() {
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

  await assertConcurrentLookups();
}

// Each stubbed lookup waits until all three lookups and the candidate build
// have started. Run one after another, the first never finishes.
async function assertConcurrentLookups() {
  const message = 'the manifest.js change in the release-please patch is not applied: pull request lookups run one after another';
  if (typeof Manifest.prototype.hasMergedReleasePullRequest !== 'function') throw new Error(message);
  const started = new Set();
  let release;
  const allStarted = new Promise(resolve => { release = resolve; });
  const start = async name => {
    started.add(name);
    if (started.size === 4) release();
    await allStarted;
  };
  const stub = {
    logger: { warn() {} },
    sequentialCalls: true,
    buildPullRequests: async () => { await start('build'); return [{}]; },
    findMergedReleasePullRequests: async function* () { await start('merged'); },
    findOpenReleasePullRequests: async () => { await start('open'); return []; },
    findSnoozedReleasePullRequests: async () => { await start('snoozed'); return []; },
    createOrUpdatePullRequest: async () => undefined,
  };
  stub.hasMergedReleasePullRequest = Manifest.prototype.hasMergedReleasePullRequest;
  let timer;
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error(message)), 1000);
  });
  try {
    await Promise.race([Manifest.prototype.createPullRequests.call(stub), timeout]);
  } finally {
    clearTimeout(timer);
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
  await assertPatched();
  console.log(`Running release-please version: ${VERSION}`);

  const { GITHUB_TOKEN: token, GITHUB_REPOSITORY: repository } = process.env;
  if (!token) throw new Error('GITHUB_TOKEN is not set');
  const [owner, repo] = (repository || '').split('/');
  if (!owner || !repo) throw new Error(`GITHUB_REPOSITORY must be owner/repo, got ${JSON.stringify(repository)}`);

  const github = await GitHub.create({ owner, repo, token });
  const targetBranch = github.repository.defaultBranch;

  // Walk history from a local clone instead of release-please's GraphQL
  // iterator: 10 commits per request makes the walk most of the run, it fails
  // on GitHub timeouts, and its 500-commit cap hides commits of packages that
  // haven't released in longer than that. Fetch first so the walk starts at the
  // same branch tip the API reads.
  const ref = `refs/remotes/origin/${targetBranch}`;
  execFileSync('git', ['fetch', '--quiet', '--filter=blob:none', 'origin', `+refs/heads/${targetBranch}:${ref}`],
    { stdio: 'inherit' });
  const overrideBodies = await fetchOverrideBodies({ owner, repo, token });
  console.log(`Found ${overrideBodies.size} merged PRs with BEGIN_COMMIT_OVERRIDE`);
  github.mergeCommitIterator = (branch, options = {}) => {
    if (branch !== targetBranch) throw new Error(`Unexpected commit walk on ${branch}`);
    return gitMergeCommits(ref, { targetBranch: branch, overrideBodies, maxResults: options.maxResults });
  };

  const manifest = await Manifest.fromManifest(
    github,
    github.repository.defaultBranch,
    CONFIG_FILE,
    MANIFEST_FILE,
  );
  // The default cap (500) is a cost guard for the GraphQL walk. Locally the
  // walk is cheap, and release-please stops once it reaches every package's
  // last release commit.
  manifest.commitSearchDepth = Number.MAX_SAFE_INTEGER;

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
