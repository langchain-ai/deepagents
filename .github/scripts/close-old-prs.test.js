const assert = require('node:assert/strict');
const test = require('node:test');

const { run, closeBody } = require('./close-old-prs.js');

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
  comments = new Map(),
  live = new Map(),
  labelExists = true,
  createLabelError = null,
  iteratorError = null,
  maxPagesBeforeError = null,
  getErrors = new Map(),
} = {}) {
  const calls = {
    createLabel: [],
    createComment: [],
    updateComment: [],
    close: [],
    queries: [],
    get: [],
  };
  let labelPresent = labelExists;

  const github = {
    rest: {
      issues: {
        getLabel: async () => {
          if (!labelPresent) throw httpError('missing label', 404);
          return { data: { name: 'do-not-close' } };
        },
        createLabel: async params => {
          calls.createLabel.push(params);
          if (createLabelError) throw createLabelError;
          labelPresent = true;
        },
        listComments: async ({ issue_number }) => ({ data: comments.get(issue_number) ?? [] }),
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
              labels: (pr.labels ?? []).map(name => ({ name })),
              state: pr.state ?? 'open',
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
    const pages = [];
    for (let index = 0; index < items.length; index += 100) {
      pages.push(items.slice(index, index + 100));
    }
    if (pages.length === 0) pages.push([]);

    for (const [index, page] of pages.entries()) {
      if (maxPagesBeforeError !== null && index >= maxPagesBeforeError) {
        throw iteratorError;
      }
      yield { data: page };
    }
    if (iteratorError && maxPagesBeforeError === null) throw iteratorError;
  };

  return { github, calls };
}

const context = { repo: { owner: 'langchain-ai', repo: 'deepagents' } };
const now = new Date('2026-05-08T00:00:00Z');

test('warns after 14 days and closes after 30 days from opening', async () => {
  const comments = new Map([
    [102, [{ id: 77, body: '<!-- old-pr-auto-close -->\nwarning' }]],
  ]);
  const live = new Map([
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
    checked: 4, warned: 1, closed: 1, skipped: 2, incomplete: false, errors: [],
  });
  assert.equal(calls.createComment.length, 1);
  assert.equal(calls.createComment[0].issue_number, 101);
  assert.match(calls.createComment[0].body, /open for at least 14 days/);
  assert.equal(calls.updateComment.length, 1);
  assert.equal(calls.updateComment[0].comment_id, 77);
  assert.match(calls.updateComment[0].body, /open for at least 30 days/);
  assert.deepEqual(calls.close, [102]);
  assert.equal(core.failed, null);
});

test('closes an old PR even when the warning comment is missing', async () => {
  const { github, calls } = makeGithub({
    items: [{ number: 201, created_at: '2026-04-01T00:00:00Z' }],
  });

  const summary = await run({ github, context, core: makeCore(), options: { now } });

  assert.equal(summary.closed, 1);
  assert.equal(calls.createComment[0].issue_number, 201);
  assert.match(calls.createComment[0].body, /being closed automatically/);
  assert.deepEqual(calls.close, [201]);
});

test('does not duplicate warning comments on daily runs', async () => {
  const comments = new Map([
    [301, [{ id: 88, body: '<!-- old-pr-auto-close -->\nwarning' }]],
  ]);
  const { github, calls } = makeGithub({
    items: [{ number: 301, created_at: '2026-04-23T00:00:00Z' }],
    comments,
  });

  const summary = await run({ github, context, core: makeCore(), options: { now } });

  assert.equal(summary.skipped, 1);
  assert.equal(calls.createComment.length, 0);
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

test('does not rewrite an identical close comment', async () => {
  const body = closeBody({ closeDays: 30, bypassLabel: 'do-not-close' });
  const comments = new Map([[601, [{ id: 5, body }]]]);
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

test('continues after a single PR failure and fails the run at the end', async () => {
  const getErrors = new Map([[701, httpError('temporary outage', 503)]]);
  const { github, calls } = makeGithub({
    items: [
      { number: 701, created_at: '2026-04-08T00:00:00Z' },
      { number: 702, created_at: '2026-04-08T00:00:00Z' },
    ],
    getErrors,
  });
  const core = makeCore();

  const summary = await run({ github, context, core, options: { now } });

  assert.equal(summary.checked, 2);
  assert.equal(summary.closed, 1);
  assert.equal(summary.errors.length, 1);
  assert.equal(summary.errors[0].number, 701);
  assert.equal(summary.errors[0].status, 503);
  assert.match(core.warnings[0], /PR #701 failed \(HTTP 503\)/);
  assert.match(core.failed, /#701/);
  assert.deepEqual(calls.close, [702]);
});

test('creates the bypass label when it does not exist', async () => {
  const { github, calls } = makeGithub({ items: [], labelExists: false });

  const summary = await run({ github, context, core: makeCore(), options: { now } });

  assert.equal(calls.createLabel.length, 1);
  assert.equal(calls.createLabel[0].name, 'do-not-close');
  assert.equal(summary.checked, 0);
});

test('confirms the label exists after a create-label 422 race', async () => {
  const { github, calls } = makeGithub({
    items: [],
    labelExists: false,
    createLabelError: httpError('already exists', 422),
  });
  // Simulate a concurrent run that created the label between our getLabel (404)
  // and createLabel (422): the confirming getLabel now succeeds.
  let confirmed = false;
  github.rest.issues.getLabel = async () => {
    if (!confirmed) {
      confirmed = true;
      throw httpError('missing label', 404);
    }
    return { data: { name: 'do-not-close' } };
  };

  const summary = await run({ github, context, core: makeCore(), options: { now } });

  assert.equal(calls.createLabel.length, 1);
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

test('honors maxItems truncation', async () => {
  const { github } = makeGithub({
    items: Array.from({ length: 3 }, (_, index) => ({
      number: 900 + index,
      created_at: '2026-05-07T00:00:00Z',
    })),
  });

  const summary = await run({
    github,
    context,
    core: makeCore(),
    options: { now, maxItems: 2 },
  });

  assert.equal(summary.checked, 2);
  assert.equal(summary.incomplete, false);
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
