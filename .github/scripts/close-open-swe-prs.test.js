const assert = require('node:assert/strict');
const test = require('node:test');

const { run } = require('./close-open-swe-prs.js');

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
    info(message) {
      this.infos.push(message);
    },
    warning(message) {
      this.warnings.push(message);
    },
    setFailed(message) {
      this.failed = message;
    },
  };
}

function makeGithub({
  items,
  comments = new Map(),
  live = new Map(),
  labelExists = true,
  createLabelError = null,
  iteratorError = null,
  getErrors = new Map(),
  maxPagesBeforeError = null,
} = {}) {
  const calls = {
    createLabel: [],
    createComment: [],
    updateComment: [],
    close: [],
    searchQueries: [],
  };

  let getLabelCalls = 0;
  const github = {
    rest: {
      issues: {
        getLabel: async () => {
          getLabelCalls += 1;
          if (!labelExists) throw httpError('missing label', 404);
          return { data: { name: 'do-not-close' } };
        },
        createLabel: async params => {
          calls.createLabel.push(params);
          if (createLabelError) throw createLabelError;
          labelExists = true;
        },
        listComments: async params => ({
          data: comments.get(params.issue_number) ?? [],
        }),
        createComment: async params => {
          calls.createComment.push(params);
          return { data: { id: 1000 + params.issue_number } };
        },
        updateComment: async params => {
          calls.updateComment.push(params);
          return { data: { id: params.comment_id } };
        },
      },
      pulls: {
        get: async params => {
          const error = getErrors.get(params.pull_number);
          if (error) throw error;

          const state = live.get(params.pull_number) ?? {};
          return {
            data: {
              draft: state.draft ?? false,
              state: state.state ?? 'open',
              updated_at: state.updated_at ?? '2026-04-30T00:00:00Z',
              labels: (state.labels ?? []).map(name => ({ name })),
            },
          };
        },
        update: async params => {
          calls.close.push(params.pull_number);
          return { data: { number: params.pull_number, state: params.state } };
        },
      },
      search: {
        issuesAndPullRequests: async () => {},
      },
    },
    paginate: async (method, params) => {
      const result = await method(params);
      return result.data;
    },
  };

  github.paginate.iterator = async function* iterator(_method, params) {
    calls.searchQueries.push(params.q);
    const pages = [];
    for (let index = 0; index < (items ?? []).length; index += 100) {
      pages.push((items ?? []).slice(index, index + 100));
    }

    for (const [index, page] of pages.entries()) {
      if (maxPagesBeforeError !== null && index >= maxPagesBeforeError) {
        throw iteratorError;
      }
      yield { data: { items: page } };
    }
    if (iteratorError) throw iteratorError;
  };

  return { github, calls, getLabelCalls: () => getLabelCalls };
}

const context = { repo: { owner: 'langchain-ai', repo: 'deepagents' } };
const now = new Date('2026-05-08T00:00:00Z');

test('reminds inactive PRs, closes aged reminders, and skips bypass/draft/closed PRs', async () => {
  const comments = new Map([
    [
      102,
      [{
        id: 77,
        body: '<!-- open-swe-auto-close -->\nold',
        created_at: '2026-04-30T00:00:00Z',
        updated_at: '2026-04-30T00:00:00Z',
      }],
    ],
  ]);
  const live = new Map([
    [101, { updated_at: '2026-04-30T00:00:00Z' }],
    [102, { updated_at: '2026-04-30T00:00:00Z' }],
    [103, { labels: ['do-not-close'], updated_at: '2026-04-10T00:00:00Z' }],
    [104, { draft: true, updated_at: '2026-04-10T00:00:00Z' }],
    [105, { state: 'closed', updated_at: '2026-04-10T00:00:00Z' }],
  ]);
  const { github, calls } = makeGithub({
    items: [
      { number: 101, created_at: '2026-04-01T00:00:00Z', updated_at: '2026-04-30T00:00:00Z' },
      { number: 102, created_at: '2026-04-01T00:00:00Z', updated_at: '2026-04-30T00:00:00Z' },
      { number: 103, created_at: '2026-04-01T00:00:00Z', updated_at: '2026-04-10T00:00:00Z' },
      { number: 104, created_at: '2026-04-01T00:00:00Z', updated_at: '2026-04-10T00:00:00Z' },
      { number: 105, created_at: '2026-04-01T00:00:00Z', updated_at: '2026-04-10T00:00:00Z' },
    ],
    comments,
    live,
  });
  const core = makeCore();

  const summary = await run({ github, context, core, options: { now } });

  assert.deepEqual(summary, { checked: 5, reminded: 1, closed: 1, skipped: 3, errors: [] });
  assert.match(calls.searchQueries[0], /draft:false/);
  assert.equal(calls.createComment.length, 1);
  assert.equal(calls.createComment[0].issue_number, 101);
  assert.match(calls.createComment[0].body, /at least 7 day\(s\)/);
  assert.equal(calls.updateComment.length, 1);
  assert.equal(calls.updateComment[0].comment_id, 77);
  assert.match(calls.updateComment[0].body, /Closing automatically/);
  assert.deepEqual(calls.close, [102]);
  assert.equal(core.failed, null);
});

test('refreshes reminder instead of closing when PR had activity after prior reminder', async () => {
  const comments = new Map([
    [
      201,
      [{
        id: 88,
        body: '<!-- open-swe-auto-close -->\nold',
        created_at: '2026-04-20T00:00:00Z',
        updated_at: '2026-04-20T00:00:00Z',
      }],
    ],
  ]);
  const live = new Map([
    [201, { updated_at: '2026-05-01T00:00:00Z' }],
  ]);
  const { github, calls } = makeGithub({
    items: [{ number: 201, created_at: '2026-04-01T00:00:00Z', updated_at: '2026-05-01T00:00:00Z' }],
    comments,
    live,
  });

  const summary = await run({ github, context, core: makeCore(), options: { now } });

  assert.equal(summary.reminded, 1);
  assert.equal(summary.closed, 0);
  assert.equal(calls.updateComment.length, 1);
  assert.match(calls.updateComment[0].body, /at least 7 day\(s\)/);
  assert.deepEqual(calls.close, []);
});

test('continues after a single PR failure and fails the workflow at the end', async () => {
  const getErrors = new Map([[301, httpError('temporary outage', 503)]]);
  const { github, calls } = makeGithub({
    items: [
      { number: 301, created_at: '2026-04-01T00:00:00Z', updated_at: '2026-04-30T00:00:00Z' },
      { number: 302, created_at: '2026-04-01T00:00:00Z', updated_at: '2026-04-30T00:00:00Z' },
    ],
    getErrors,
  });
  const core = makeCore();

  const summary = await run({ github, context, core, options: { now } });

  assert.equal(summary.checked, 2);
  assert.equal(summary.reminded, 1);
  assert.equal(summary.errors.length, 1);
  assert.equal(summary.errors[0].number, 301);
  assert.match(core.warnings[0], /PR #301 failed/);
  assert.match(core.failed, /#301/);
  assert.equal(calls.createComment[0].issue_number, 302);
});

test('confirms label exists after create-label 422', async () => {
  const { github, calls, getLabelCalls } = makeGithub({
    items: [],
    labelExists: false,
    createLabelError: httpError('already exists', 422),
  });
  let confirmed = false;
  github.rest.issues.getLabel = async () => {
    getLabelCalls();
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

test('throws when create-label 422 cannot be confirmed', async () => {
  const { github } = makeGithub({
    items: [],
    labelExists: false,
    createLabelError: httpError('invalid label', 422),
  });

  await assert.rejects(
    run({ github, context, core: makeCore(), options: { now } }),
    /missing label/,
  );
});

test('rejects invalid reminder and close day configuration', async () => {
  const { github } = makeGithub({ items: [] });

  await assert.rejects(
    run({
      github,
      context,
      core: makeCore(),
      options: { now, reminderDays: 14, closeDays: 14 },
    }),
    /reminderDays \(14\) must be less than closeDays \(14\)/,
  );
});

test('keeps collected search pages when later pagination fails', async () => {
  const { github } = makeGithub({
    items: [
      { number: 501, created_at: '2026-04-01T00:00:00Z', updated_at: '2026-04-30T00:00:00Z' },
      ...Array.from({ length: 100 }, (_, index) => ({
        number: 600 + index,
        created_at: '2026-05-07T00:00:00Z',
        updated_at: '2026-05-07T00:00:00Z',
      })),
    ],
    iteratorError: httpError('search timeout', 503),
    maxPagesBeforeError: 1,
  });
  const core = makeCore();

  const summary = await run({ github, context, core, options: { now } });

  assert.equal(summary.checked, 100);
  assert.match(core.warnings[0], /Search failed after collecting 100 PR\(s\)/);
});

test('honors maxItems truncation', async () => {
  const { github } = makeGithub({
    items: Array.from({ length: 3 }, (_, index) => ({
      number: 701 + index,
      created_at: '2026-05-07T00:00:00Z',
      updated_at: '2026-05-07T00:00:00Z',
    })),
  });

  const summary = await run({
    github,
    context,
    core: makeCore(),
    options: { now, maxItems: 2 },
  });

  assert.equal(summary.checked, 2);
});
