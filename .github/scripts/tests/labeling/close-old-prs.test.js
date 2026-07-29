const assert = require('node:assert/strict');
const test = require('node:test');

const { run, closeBody, ageInDays, COMMENT_MARKER } = require('../../labeling/close-old-prs.js');

function httpError(message, status) {
  const error = new Error(message);
  error.status = status;
  return error;
}

function makeCore() {
  return {
    failed: null,
    infos: [],
    warnings: [],
    info(message) { this.infos.push(message); },
    warning(message) { this.warnings.push(message); },
    setFailed(message) { this.failed = message; },
  };
}

function makeGithub({
  items = [],
  labeledItems = [],
  comments = new Map(),
  live = new Map(),
  labelExists = true,
  createLabelError = null,
  iteratorError = null,
  maxPagesBeforeError = null,
  incompleteResults = false,
  getErrors = new Map(),
  removeLabelErrors = new Map(),
} = {}) {
  const calls = {
    createLabel: [],
    addLabels: [],
    removeLabel: [],
    createComment: [],
    updateComment: [],
    close: [],
    queries: [],
    get: [],
  };
  // When labels are presumed present, unknown names still succeed so tests do
  // not have to seed every label the workflow may ensure.
  const presentLabels = new Set(labelExists ? ['do-not-close', 'pending-deletion'] : []);

  const github = {
    rest: {
      issues: {
        getLabel: async ({ name }) => {
          if (!labelExists && !presentLabels.has(name)) {
            throw httpError('missing label', 404);
          }
          return { data: { name } };
        },
        createLabel: async params => {
          calls.createLabel.push(params);
          if (createLabelError) throw createLabelError;
          presentLabels.add(params.name);
        },
        addLabels: async params => {
          calls.addLabels.push(params);
        },
        removeLabel: async params => {
          calls.removeLabel.push(params);
          const error = removeLabelErrors.get(params.issue_number);
          if (error) throw error;
        },
        listComments: async ({ issue_number }) => ({
          data: (comments.get(issue_number) ?? []).map(comment => ({
            created_at: '2026-04-22T00:00:00Z',
            ...comment,
          })),
        }),
        createComment: async params => { calls.createComment.push(params); },
        updateComment: async params => { calls.updateComment.push(params); },
      },
      pulls: {
        get: async ({ pull_number }) => {
          calls.get.push(pull_number);
          const error = getErrors.get(pull_number);
          if (error) throw error;
          const pr = live.get(pull_number) ?? {};
          return {
            data: {
              created_at: pr.created_at ?? '2026-04-01T00:00:00Z',
              draft: pr.draft ?? false,
              head: {
                ref: pr.headRef ?? 'contributor/feature',
                repo: { full_name: pr.headRepo ?? 'contributor/deepagents' },
              },
              labels: (pr.labels ?? []).map(name => ({ name })),
              state: pr.state ?? 'open',
              user: {
                login: pr.authorLogin ?? 'contributor',
                type: pr.authorType ?? 'User',
              },
            },
          };
        },
        update: async ({ pull_number }) => { calls.close.push(pull_number); },
      },
      search: { issuesAndPullRequests: async () => {} },
    },
    paginate: async (method, params) => (await method(params)).data,
  };

  github.paginate.iterator = async function* iterator(_method, params) {
    calls.queries.push(params);
    // Primary open-PR scan vs pending-deletion sweep use different queries.
    const labeledQuery = typeof params.q === 'string' && params.q.includes('label:"');
    const source = labeledQuery ? labeledItems : items;
    const pages = [];
    for (let index = 0; index < source.length; index += 100) {
      pages.push(source.slice(index, index + 100));
    }
    if (pages.length === 0) pages.push([]);

    for (const [index, page] of pages.entries()) {
      if (!labeledQuery && maxPagesBeforeError !== null && index >= maxPagesBeforeError) {
        throw iteratorError;
      }
      page.incomplete_results = !labeledQuery && incompleteResults;
      yield { data: page };
    }
    if (!labeledQuery && iteratorError && maxPagesBeforeError === null) throw iteratorError;
  };

  return { github, calls };
}

const context = { repo: { owner: 'langchain-ai', repo: 'deepagents' } };
const now = new Date('2026-05-08T00:00:00Z');
const workflowBot = { login: 'github-actions[bot]', type: 'Bot' };

// A genuine release-please PR. `overrides` spoils exactly one provenance
// attribute at a time so each conjunct in isReleasePr is independently
// load-bearing for some test — otherwise a dropped guard passes CI unnoticed.
function releasePleasePr(labels, overrides = {}) {
  return {
    authorLogin: 'github-actions[bot]',
    authorType: 'Bot',
    headRef: 'release-please--branches--main--components--deepagents',
    headRepo: 'langchain-ai/deepagents',
    labels,
    ...overrides,
  };
}

test('warns after 14 days and closes after 30 days from opening', async () => {
  const comments = new Map([
    [102, [{ id: 77, body: `${COMMENT_MARKER}\nwarning`, user: workflowBot }]],
  ]);
  const live = new Map([
    [102, { labels: ['pending-deletion'] }],
    [103, { labels: ['do-not-close'] }],
    [104, { draft: true }],
  ]);
  const { github, calls } = makeGithub({
    items: [
      { number: 101, created_at: '2026-04-23T00:00:00Z' },
      { number: 102, created_at: '2026-04-08T00:00:00Z' },
      { number: 103, created_at: '2026-04-08T00:00:00Z' },
      { number: 104, created_at: '2026-04-08T00:00:00Z' },
    ],
    comments,
    live,
  });
  const core = makeCore();

  const summary = await run({ github, context, core, options: { now } });

  assert.deepEqual(summary, {
    checked: 4, warned: 1, closed: 1, skipped: 2, skippedRelease: 0,
    staleCleared: 0, sweepFailure: null, incomplete: false, truncated: false, errors: [],
  });
  assert.equal(calls.createComment.length, 1);
  assert.equal(calls.createComment[0].issue_number, 101);
  assert.match(calls.createComment[0].body, /open for at least 14 days/);
  assert.deepEqual(
    calls.addLabels.map(call => [call.issue_number, call.labels]),
    [[101, ['pending-deletion']]],
  );
  assert.equal(calls.updateComment.length, 1);
  assert.equal(calls.updateComment[0].comment_id, 77);
  assert.match(calls.updateComment[0].body, /open for at least 30 days/);
  assert.deepEqual(calls.close, [102]);
  assert.deepEqual(calls.removeLabel, [{
    owner: 'langchain-ai',
    repo: 'deepagents',
    issue_number: 102,
    name: 'pending-deletion',
  }]);
  assert.equal(core.failed, null);
});

test('skips genuine release-please PRs without warning or closing', async () => {
  const { github, calls } = makeGithub({
    items: [{ number: 105, created_at: '2026-04-08T00:00:00Z' }],
    live: new Map([[105, releasePleasePr(['release'])]]),
  });
  const core = makeCore();

  const summary = await run({ github, context, core, options: { now } });

  assert.equal(summary.skipped, 1);
  assert.equal(summary.skippedRelease, 1);
  assert.equal(calls.createComment.length, 0);
  assert.equal(calls.updateComment.length, 0);
  assert.equal(calls.addLabels.length, 0);
  assert.equal(calls.removeLabel.length, 0);
  assert.deepEqual(calls.close, []);
  assert.ok(core.infos.some(message => message.includes('release PR; skipping')));
  assert.deepEqual(core.warnings, []);
});

// The regression this exemption exists for: a release PR already past
// closeDays *and* already carrying a bot warning from an earlier run is
// close-eligible on every axis except the release check. Also the only
// coverage for stripping pending-deletion via the processPr release branch,
// and the only positive assertion that 'autorelease: pending' is exempting.
test('does not close a release PR already past the close threshold', async () => {
  const comments = new Map([
    [909, [{ id: 95, body: `${COMMENT_MARKER}\nwarning`, user: workflowBot }]],
  ]);
  const { github, calls } = makeGithub({
    items: [{ number: 909, created_at: '2026-04-01T00:00:00Z' }],
    comments,
    live: new Map([[909, releasePleasePr(['pending-deletion', 'autorelease: pending'])]]),
  });

  const summary = await run({ github, context, core: makeCore(), options: { now } });

  assert.equal(summary.closed, 0);
  assert.equal(summary.skippedRelease, 1);
  assert.deepEqual(calls.close, []);
  assert.equal(calls.updateComment.length, 0);
  assert.deepEqual(calls.removeLabel, [{
    owner: 'langchain-ai',
    repo: 'deepagents',
    issue_number: 909,
    name: 'pending-deletion',
  }]);
});

// One test per provenance conjunct. Each spoils exactly one attribute, so a
// guard that stops being enforced fails precisely one of these rather than
// hiding behind the others.
const provenanceDenials = [
  ['a fork head repository', { headRepo: 'attacker/deepagents' }, 'repo'],
  ['a non-release branch name', {
    headRef: 'chore/a-branch-name-longer-than-the-release-please-prefix-string',
  }, 'branch'],
  ['a bare branch prefix with no component', {
    headRef: 'release-please--branches--main--components--',
  }, 'branch'],
  ['a human impersonating the bot login', { authorType: 'User' }, 'author'],
  ['a different bot', { authorLogin: 'dependabot[bot]' }, 'author'],
];

for (const [description, overrides, expectedReason] of provenanceDenials) {
  test(`denies the release exemption for ${description}`, async () => {
    const { github, calls } = makeGithub({
      items: [{ number: 106, created_at: '2026-04-08T00:00:00Z' }],
      live: new Map([[106, releasePleasePr(['release'], overrides)]]),
    });
    const core = makeCore();

    const summary = await run({ github, context, core, options: { now } });

    assert.equal(summary.warned, 1);
    assert.equal(summary.skippedRelease, 0);
    assert.equal(calls.createComment.length, 1);
    assert.deepEqual(calls.addLabels, [{
      owner: 'langchain-ai',
      repo: 'deepagents',
      issue_number: 106,
      labels: ['pending-deletion'],
    }]);
    assert.deepEqual(calls.close, []);
    // A release label without provenance is either spoofing or real drift;
    // both must be audible rather than silently falling through to warn.
    assert.ok(
      core.warnings.some(message =>
        message.includes('failed provenance') && message.includes(expectedReason)),
      `expected a provenance warning mentioning "${expectedReason}", got ${JSON.stringify(core.warnings)}`,
    );
  });
}

test('matches the head repository case-insensitively', async () => {
  const { github, calls } = makeGithub({
    items: [{ number: 111, created_at: '2026-04-08T00:00:00Z' }],
    live: new Map([[111, releasePleasePr(['release'], { headRepo: 'LangChain-AI/DeepAgents' })]]),
  });
  const core = makeCore();

  const summary = await run({ github, context, core, options: { now } });

  assert.equal(summary.skippedRelease, 1);
  assert.deepEqual(calls.close, []);
  assert.equal(calls.createComment.length, 0);
  assert.deepEqual(core.warnings, []);
});

// A genuine release-please PR whose label merely *contains* a release label as
// a substring must not be exempt — guards against loosening Set.has() into a
// prefix/substring match.
test('still warns a release-please PR whose label only resembles a release label', async () => {
  const { github, calls } = makeGithub({
    items: [{ number: 107, created_at: '2026-04-23T00:00:00Z' }],
    live: new Map([[107, releasePleasePr(['release-notes'])]]),
  });

  const summary = await run({ github, context, core: makeCore(), options: { now } });

  assert.equal(summary.warned, 1);
  assert.equal(summary.skippedRelease, 0);
  assert.equal(calls.createComment.length, 1);
  assert.deepEqual(calls.addLabels, [{
    owner: 'langchain-ai',
    repo: 'deepagents',
    issue_number: 107,
    labels: ['pending-deletion'],
  }]);
  assert.deepEqual(calls.close, []);
});

test('sweep clears pending-deletion from a release PR', async () => {
  const { github, calls } = makeGithub({
    labeledItems: [{ number: 108, created_at: '2026-04-01T00:00:00Z' }],
    live: new Map([[108, releasePleasePr(['pending-deletion', 'release'])]]),
  });

  const summary = await run({ github, context, core: makeCore(), options: { now } });

  assert.equal(summary.staleCleared, 1);
  assert.deepEqual(calls.removeLabel, [{
    owner: 'langchain-ai',
    repo: 'deepagents',
    issue_number: 108,
    name: 'pending-deletion',
  }]);
  assert.equal(calls.createComment.length, 0);
  assert.deepEqual(calls.close, []);
});

test('sweep keeps pending-deletion on a contributor PR labeled release', async () => {
  const { github, calls } = makeGithub({
    labeledItems: [{ number: 110, created_at: '2026-04-01T00:00:00Z' }],
    live: new Map([[110, { labels: ['pending-deletion', 'release'] }]]),
  });

  const summary = await run({ github, context, core: makeCore(), options: { now } });

  assert.equal(summary.staleCleared, 0);
  assert.equal(calls.removeLabel.length, 0);
});

// The label search index lags this run's own mutations, so a PR whose label
// processPr just removed can still come back from the sweep query. Removing
// nothing must not be counted or logged as a removal.
test('sweep does not count a PR whose label is already gone', async () => {
  const { github, calls } = makeGithub({
    labeledItems: [{ number: 112, created_at: '2026-04-01T00:00:00Z' }],
    live: new Map([[112, { draft: true, labels: [] }]]),
  });
  const core = makeCore();

  const summary = await run({ github, context, core, options: { now } });

  assert.equal(summary.staleCleared, 0);
  assert.equal(calls.removeLabel.length, 0);
  assert.ok(!core.infos.some(message => message.includes('Cleared pending-deletion from PR #112')));
  assert.equal(core.failed, null);
});

// A sweep that dies partway must not look like a sweep with nothing to do.
test('fails the run when the pending-deletion sweep errors', async () => {
  const { github, calls } = makeGithub({
    labeledItems: [
      { number: 113, created_at: '2026-04-01T00:00:00Z' },
      { number: 114, created_at: '2026-04-01T00:00:00Z' },
    ],
    live: new Map([
      [113, { state: 'closed', labels: ['pending-deletion'] }],
      [114, { state: 'closed', labels: ['pending-deletion'] }],
    ]),
    removeLabelErrors: new Map([[114, httpError('forbidden', 403)]]),
  });
  const core = makeCore();

  const summary = await run({ github, context, core, options: { now } });

  // #113 was cleared before #114 threw; the partial progress is still reported.
  assert.equal(summary.staleCleared, 1);
  assert.equal(calls.removeLabel.length, 2);
  assert.match(summary.sweepFailure, /sweep failed after clearing 1 label\(s\) \(HTTP 403\)/);
  assert.match(core.failed, /sweep failed after clearing 1 label\(s\) \(HTTP 403\)/);
});

// Unlike an error, the cap is not a failure: the sweep is idempotent and the
// next daily run picks up the remainder, so it warns without failing the run.
test('hitting the sweep cap warns but does not fail the run', async () => {
  const { github } = makeGithub({
    labeledItems: [
      { number: 116, created_at: '2026-04-01T00:00:00Z' },
      { number: 117, created_at: '2026-04-01T00:00:00Z' },
    ],
    live: new Map([
      [116, { state: 'closed', labels: ['pending-deletion'] }],
      [117, { state: 'closed', labels: ['pending-deletion'] }],
    ]),
  });
  const core = makeCore();

  const summary = await run({ github, context, core, options: { now, maxItems: 1 } });

  assert.equal(summary.staleCleared, 1);
  assert.equal(summary.sweepFailure, null);
  assert.equal(core.failed, null);
  assert.ok(core.warnings.some(message => message.includes('Reached maxItems cap (1) while sweeping')));
});

test('a 404 on label removal is tolerated but not counted as cleared', async () => {
  const { github, calls } = makeGithub({
    labeledItems: [{ number: 115, created_at: '2026-04-01T00:00:00Z' }],
    live: new Map([[115, { state: 'closed', labels: ['pending-deletion'] }]]),
    removeLabelErrors: new Map([[115, httpError('label not found', 404)]]),
  });
  const core = makeCore();

  const summary = await run({ github, context, core, options: { now } });

  assert.equal(calls.removeLabel.length, 1);
  assert.equal(summary.staleCleared, 0);
  assert.equal(summary.sweepFailure, null);
  assert.equal(core.failed, null);
});

test('warns an old PR that was never warned instead of closing it', async () => {
  const { github, calls } = makeGithub({
    items: [{ number: 201, created_at: '2026-04-01T00:00:00Z' }],
  });

  const summary = await run({ github, context, core: makeCore(), options: { now } });

  // Warn-first: a 37-day-old PR with no prior bot warning is warned now and
  // only becomes eligible to close on a later run.
  assert.equal(summary.warned, 1);
  assert.equal(summary.closed, 0);
  assert.equal(calls.createComment[0].issue_number, 201);
  assert.match(calls.createComment[0].body, /open for at least 14 days/);
  assert.match(calls.createComment[0].body, /warning is at least 16 days old/);
  assert.deepEqual(calls.addLabels, [{
    owner: 'langchain-ai',
    repo: 'deepagents',
    issue_number: 201,
    labels: ['pending-deletion'],
  }]);
  assert.deepEqual(calls.close, []);
});

test('gives backlogged PRs the full warning interval before closing', async () => {
  const comments = new Map([
    [202, [{
      id: 78,
      body: `${COMMENT_MARKER}\nwarning`,
      user: workflowBot,
      created_at: '2026-05-07T00:00:00Z',
    }]],
    [203, [{
      id: 79,
      body: `${COMMENT_MARKER}\nwarning`,
      user: workflowBot,
      created_at: '2026-04-22T00:00:00Z',
    }]],
  ]);
  const { github, calls } = makeGithub({
    items: [
      { number: 202, created_at: '2026-04-01T00:00:00Z' },
      { number: 203, created_at: '2026-04-01T00:00:00Z' },
    ],
    comments,
  });

  const summary = await run({ github, context, core: makeCore(), options: { now } });

  assert.equal(summary.skipped, 1);
  assert.equal(summary.closed, 1);
  assert.deepEqual(calls.close, [203]);
});

test('does not duplicate warning comments on daily runs', async () => {
  const comments = new Map([
    [301, [{ id: 88, body: `${COMMENT_MARKER}\nwarning`, user: workflowBot }]],
  ]);
  const { github, calls } = makeGithub({
    items: [{ number: 301, created_at: '2026-04-23T00:00:00Z' }],
    comments,
    live: new Map([[301, { labels: ['pending-deletion'] }]]),
  });

  const summary = await run({ github, context, core: makeCore(), options: { now } });

  assert.equal(summary.skipped, 1);
  assert.equal(calls.createComment.length, 0);
  assert.equal(calls.updateComment.length, 0);
  assert.equal(calls.addLabels.length, 0);
  assert.deepEqual(calls.close, []);
});

test('backfills pending-deletion on already-warned PRs', async () => {
  const comments = new Map([
    [305, [{ id: 91, body: `${COMMENT_MARKER}\nwarning`, user: workflowBot }]],
  ]);
  const { github, calls } = makeGithub({
    items: [{ number: 305, created_at: '2026-04-23T00:00:00Z' }],
    comments,
  });

  const summary = await run({ github, context, core: makeCore(), options: { now } });

  assert.equal(summary.skipped, 1);
  assert.equal(calls.createComment.length, 0);
  assert.deepEqual(calls.addLabels, [{
    owner: 'langchain-ai',
    repo: 'deepagents',
    issue_number: 305,
    labels: ['pending-deletion'],
  }]);
});

test('ignores marker comments posted by PR participants when warning', async () => {
  const comments = new Map([
    [302, [{
      id: 89,
      body: `${COMMENT_MARKER}\nforged warning`,
      user: { login: 'contributor', type: 'User' },
    }]],
  ]);
  const { github, calls } = makeGithub({
    items: [{ number: 302, created_at: '2026-04-23T00:00:00Z' }],
    comments,
  });

  const summary = await run({ github, context, core: makeCore(), options: { now } });

  assert.equal(summary.warned, 1);
  assert.equal(calls.createComment.length, 1);
  assert.equal(calls.createComment[0].issue_number, 302);
  assert.equal(calls.updateComment.length, 0);
});

test('warns a PR exactly at the warning threshold', async () => {
  const { github, calls } = makeGithub({
    // 2026-04-24 is exactly 14 days before `now`.
    items: [{ number: 310, created_at: '2026-04-24T00:00:00Z' }],
  });

  const summary = await run({ github, context, core: makeCore(), options: { now } });

  assert.equal(summary.warned, 1);
  assert.equal(calls.createComment.length, 1);
  assert.match(calls.createComment[0].body, /open for at least 14 days/);
  assert.deepEqual(calls.close, []);
});

test('does not close a warned PR the day before the close threshold', async () => {
  const comments = new Map([
    [311, [{ id: 90, body: `${COMMENT_MARKER}\nwarning`, user: workflowBot }]],
  ]);
  const { github, calls } = makeGithub({
    // 2026-04-09 is 29 days before `now`: already warned, but not yet old enough.
    items: [{ number: 311, created_at: '2026-04-09T00:00:00Z' }],
    comments,
  });

  const summary = await run({ github, context, core: makeCore(), options: { now } });

  assert.equal(summary.skipped, 1);
  assert.equal(summary.closed, 0);
  assert.equal(calls.updateComment.length, 0);
  assert.deepEqual(calls.close, []);
});

test('skips young PRs without any per-PR API calls', async () => {
  const { github, calls } = makeGithub({
    items: [{ number: 401, created_at: '2026-05-01T00:00:00Z' }],
  });

  const summary = await run({ github, context, core: makeCore(), options: { now } });

  assert.equal(summary.skipped, 1);
  assert.deepEqual(calls.get, []);
  assert.equal(calls.createComment.length, 0);
  assert.deepEqual(calls.close, []);
});

test('skips PRs that are no longer open', async () => {
  const { github, calls } = makeGithub({
    items: [{ number: 501, created_at: '2026-04-01T00:00:00Z' }],
    live: new Map([[501, { state: 'closed' }]]),
  });

  const summary = await run({ github, context, core: makeCore(), options: { now } });

  assert.equal(summary.skipped, 1);
  assert.equal(summary.closed, 0);
  assert.equal(calls.createComment.length, 0);
  assert.deepEqual(calls.close, []);
});

test('skips a PR that 404s on the live re-fetch', async () => {
  const { github, calls } = makeGithub({
    items: [{ number: 510, created_at: '2026-04-01T00:00:00Z' }],
    getErrors: new Map([[510, httpError('not found', 404)]]),
  });
  const core = makeCore();

  const summary = await run({ github, context, core, options: { now } });

  // A deleted/transferred PR is a routine skip, not an error — it must not
  // land in summary.errors or fail the run.
  assert.equal(summary.skipped, 1);
  assert.equal(summary.errors.length, 0);
  assert.equal(calls.createComment.length, 0);
  assert.deepEqual(calls.close, []);
  assert.equal(core.failed, null);
});

test('does not rewrite an identical close comment', async () => {
  const body = closeBody({ closeDays: 30, bypassLabel: 'do-not-close' });
  const comments = new Map([[601, [{ id: 5, body, user: workflowBot }]]]);
  const { github, calls } = makeGithub({
    items: [{ number: 601, created_at: '2026-04-01T00:00:00Z' }],
    comments,
  });

  const summary = await run({ github, context, core: makeCore(), options: { now } });

  assert.equal(summary.closed, 1);
  assert.equal(calls.updateComment.length, 0);
  assert.equal(calls.createComment.length, 0);
  assert.deepEqual(calls.close, [601]);
});

test('warns instead of closing when only a participant marker exists', async () => {
  const comments = new Map([
    [602, [{
      id: 6,
      body: `${COMMENT_MARKER}\nforged warning`,
      user: { login: 'contributor', type: 'User' },
    }]],
  ]);
  const { github, calls } = makeGithub({
    items: [{ number: 602, created_at: '2026-04-01T00:00:00Z' }],
    comments,
  });

  const summary = await run({ github, context, core: makeCore(), options: { now } });

  // A forged participant marker is not a real warning, so a 37-day-old PR is
  // treated as unwarned: it gets a genuine bot warning and is neither closed
  // nor has the forged comment overwritten.
  assert.equal(summary.warned, 1);
  assert.equal(summary.closed, 0);
  assert.equal(calls.createComment.length, 1);
  assert.equal(calls.createComment[0].issue_number, 602);
  assert.equal(calls.updateComment.length, 0);
  assert.deepEqual(calls.close, []);
});

test('fails the run after an isolated transient per-PR error', async () => {
  const comments = new Map([
    [702, [{ id: 70, body: `${COMMENT_MARKER}\nwarning`, user: workflowBot }]],
  ]);
  const { github, calls } = makeGithub({
    items: [
      { number: 701, created_at: '2026-04-08T00:00:00Z' },
      { number: 702, created_at: '2026-04-08T00:00:00Z' },
    ],
    comments,
    getErrors: new Map([[701, httpError('temporary outage', 503)]]),
  });
  const core = makeCore();

  const summary = await run({ github, context, core, options: { now } });

  assert.equal(summary.checked, 2);
  assert.equal(summary.closed, 1);
  assert.equal(summary.errors.length, 1);
  assert.equal(summary.errors[0].number, 701);
  assert.equal(summary.errors[0].status, 503);
  assert.equal(summary.errors[0].transient, true);
  assert.match(core.warnings[0], /PR #701 failed \(HTTP 503, transient\)/);
  assert.match(core.failed, /#701: temporary outage/);
  assert.deepEqual(calls.close, [702]);
});

test('fails the run on a non-transient per-PR error', async () => {
  const { github, calls } = makeGithub({
    items: [{ number: 710, created_at: '2026-04-08T00:00:00Z' }],
    getErrors: new Map([[710, httpError('forbidden', 403)]]),
  });
  const core = makeCore();

  const summary = await run({ github, context, core, options: { now } });

  assert.equal(summary.errors.length, 1);
  assert.equal(summary.errors[0].transient, false);
  assert.match(core.warnings[0], /PR #710 failed \(HTTP 403, fatal\)/);
  assert.match(core.failed, /#710/);
  assert.deepEqual(calls.close, []);
});

test('fails the run when every processed PR errors, even transiently', async () => {
  const { github } = makeGithub({
    items: [
      { number: 720, created_at: '2026-04-08T00:00:00Z' },
      { number: 721, created_at: '2026-04-08T00:00:00Z' },
    ],
    getErrors: new Map([
      [720, httpError('outage', 503)],
      [721, httpError('outage', 502)],
    ]),
  });
  const core = makeCore();

  const summary = await run({ github, context, core, options: { now } });

  assert.equal(summary.errors.length, 2);
  assert.ok(summary.errors.every(error => error.transient));
  assert.match(core.failed, /#720: outage/);
  assert.match(core.failed, /#721: outage/);
});

test('removes pending-deletion when a PR gains do-not-close', async () => {
  const { github, calls } = makeGithub({
    items: [{ number: 321, created_at: '2026-04-23T00:00:00Z' }],
    live: new Map([[321, { labels: ['pending-deletion', 'do-not-close'] }]]),
  });

  const summary = await run({ github, context, core: makeCore(), options: { now } });

  assert.equal(summary.skipped, 1);
  assert.deepEqual(calls.removeLabel, [{
    owner: 'langchain-ai',
    repo: 'deepagents',
    issue_number: 321,
    name: 'pending-deletion',
  }]);
  assert.equal(calls.addLabels.length, 0);
  assert.deepEqual(calls.close, []);
});

test('removes pending-deletion when a PR becomes a draft', async () => {
  const { github, calls } = makeGithub({
    items: [{ number: 322, created_at: '2026-04-23T00:00:00Z' }],
    live: new Map([[322, { draft: true, labels: ['pending-deletion'] }]]),
  });

  const summary = await run({ github, context, core: makeCore(), options: { now } });

  assert.equal(summary.skipped, 1);
  assert.deepEqual(calls.removeLabel, [{
    owner: 'langchain-ai',
    repo: 'deepagents',
    issue_number: 322,
    name: 'pending-deletion',
  }]);
});

test('sweep clears pending-deletion on closed or draft PRs missed by open search', async () => {
  const { github, calls } = makeGithub({
    items: [],
    labeledItems: [
      { number: 330, created_at: '2026-04-01T00:00:00Z' },
      { number: 331, created_at: '2026-04-01T00:00:00Z' },
      { number: 332, created_at: '2026-04-01T00:00:00Z' },
    ],
    live: new Map([
      [330, { state: 'closed', labels: ['pending-deletion'] }],
      [331, { draft: true, labels: ['pending-deletion'] }],
      [332, { labels: ['pending-deletion'] }],
    ]),
  });

  const summary = await run({ github, context, core: makeCore(), options: { now } });

  assert.equal(summary.staleCleared, 2);
  assert.deepEqual(
    calls.removeLabel.map(call => call.issue_number).sort((a, b) => a - b),
    [330, 331],
  );
  // Still-open non-exempt PRs keep the label.
  assert.ok(!calls.removeLabel.some(call => call.issue_number === 332));
});

test('creates the bypass and pending-deletion labels when they do not exist', async () => {
  const { github, calls } = makeGithub({ items: [], labelExists: false });

  const summary = await run({ github, context, core: makeCore(), options: { now } });

  assert.equal(calls.createLabel.length, 2);
  assert.deepEqual(
    calls.createLabel.map(call => [call.name, call.color]),
    [
      ['do-not-close', '0e8a16'],
      ['pending-deletion', 'fbca04'],
    ],
  );
  assert.equal(summary.checked, 0);
});

test('confirms the label exists after a create-label 422 race', async () => {
  const { github, calls } = makeGithub({
    items: [],
    labelExists: false,
    createLabelError: httpError('already exists', 422),
  });
  // Simulate a concurrent run that created the label between our getLabel (404)
  // and createLabel (422): the confirming getLabel now succeeds per name.
  const seen = new Set();
  github.rest.issues.getLabel = async ({ name }) => {
    if (!seen.has(name)) {
      seen.add(name);
      throw httpError('missing label', 404);
    }
    return { data: { name } };
  };

  const summary = await run({ github, context, core: makeCore(), options: { now } });

  assert.equal(calls.createLabel.length, 2);
  assert.equal(summary.checked, 0);
});

test('surfaces the original 422 when the label is genuinely absent', async () => {
  const { github } = makeGithub({
    items: [],
    labelExists: false,
    createLabelError: httpError('Validation Failed: name is invalid', 422),
  });

  await assert.rejects(
    run({ github, context, core: makeCore(), options: { now } }),
    /name is invalid/,
  );
});

test('keeps collected pages and fails the run when search pagination fails', async () => {
  const { github } = makeGithub({
    items: Array.from({ length: 101 }, (_, index) => ({
      number: 800 + index,
      created_at: '2026-05-07T00:00:00Z',
    })),
    iteratorError: httpError('search timeout', 503),
    maxPagesBeforeError: 1,
  });
  const core = makeCore();

  const summary = await run({ github, context, core, options: { now } });

  assert.equal(summary.checked, 100);
  assert.equal(summary.incomplete, true);
  assert.match(core.warnings[0], /Search failed after collecting 100 PR\(s\)/);
  assert.match(core.failed, /did not complete/);
});

test('fails the run when search returns successful but incomplete results', async () => {
  const { github } = makeGithub({
    items: [{ number: 850, created_at: '2026-05-07T00:00:00Z' }],
    incompleteResults: true,
  });
  const core = makeCore();

  const summary = await run({ github, context, core, options: { now } });

  assert.equal(summary.checked, 1);
  assert.equal(summary.incomplete, true);
  assert.match(core.failed, /did not complete/);
});

test('honors maxItems truncation', async () => {
  const { github } = makeGithub({
    items: Array.from({ length: 3 }, (_, index) => ({
      number: 900 + index,
      created_at: '2026-05-07T00:00:00Z',
    })),
  });

  const core = makeCore();
  const summary = await run({ github, context, core, options: { now, maxItems: 2 } });

  assert.equal(summary.checked, 2);
  assert.equal(summary.incomplete, false);
  // Hitting the cap is surfaced (flag + warning) rather than looking like a
  // complete sweep, but does not fail the run since it self-corrects.
  assert.equal(summary.truncated, true);
  assert.match(core.warnings[0], /Reached maxItems cap \(2\)/);
  assert.equal(core.failed, null);
});

test('uses all non-draft open PRs and rejects invalid thresholds', async () => {
  const { github, calls } = makeGithub();
  await run({ github, context, core: makeCore(), options: { now } });

  assert.equal(calls.queries[0].q, 'repo:langchain-ai/deepagents is:pr is:open draft:false');
  assert.equal(calls.queries[0].sort, 'created');
  assert.equal(calls.queries[0].order, 'asc');

  await assert.rejects(
    run({
      github,
      context,
      core: makeCore(),
      options: { now, warningDays: 30, closeDays: 30 },
    }),
    /warningDays \(30\) must be less than closeDays \(30\)/,
  );
});

test('rejects non-integer threshold configuration', async () => {
  const { github } = makeGithub({ items: [] });

  await assert.rejects(
    run({ github, context, core: makeCore(), options: { now, maxItems: 0 } }),
    /maxItems must be a positive integer/,
  );
  await assert.rejects(
    run({ github, context, core: makeCore(), options: { now, maxItems: '10O' } }),
    /maxItems must be a positive integer/,
  );
});

test('ageInDays throws on an unparseable created date', () => {
  assert.throws(() => ageInDays('not-a-date', now), /Unparseable created date/);
});

test('records an unparseable created date as a fatal per-PR error', async () => {
  const { github, calls } = makeGithub({
    items: [{ number: 1001, created_at: 'not-a-date' }],
  });
  const core = makeCore();

  const summary = await run({ github, context, core, options: { now } });

  // A malformed date must surface loudly, not silently skip the PR forever.
  assert.equal(summary.errors.length, 1);
  assert.equal(summary.errors[0].number, 1001);
  assert.equal(summary.errors[0].transient, false);
  assert.match(core.failed, /#1001/);
  assert.deepEqual(calls.close, []);
});
