const assert = require('node:assert/strict');
const test = require('node:test');
const fs = require('node:fs');
const path = require('node:path');

const closeOldPrs = require('../../labeling/close-old-prs.js');

const {
  run,
  applyBypassLabel,
  clearPendingDeletion,
  closeBody,
  ageInDays,
  COMMENT_MARKER,
  RELEASE_PLEASE_BRANCH_PREFIX,
  DEFAULT_BYPASS_LABEL,
  DEFAULT_PENDING_DELETION_LABEL,
} = closeOldPrs;

const REPO_ROOT = path.resolve(__dirname, '../../../..');
const CLEAR_PENDING_DELETION_YML = path.join(
  REPO_ROOT, '.github/workflows/clear_pending_deletion.yml',
);
const KEEP_OPEN_ON_COMMENT_YML = path.join(
  REPO_ROOT, '.github/workflows/keep_open_on_comment.yml',
);

// clear_pending_deletion.yml destructures helpers out of close-old-prs.js at
// runtime. Nothing else connects the two: a helper that is not exported
// destructures to undefined, and the step dies with a TypeError on every run
// after its label removal has already succeeded — a job that goes red on
// success, which is exactly the shape of failure maintainers learn to ignore.
test('clear_pending_deletion.yml only requires exported symbols', () => {
  const workflow = fs.readFileSync(CLEAR_PENDING_DELETION_YML, 'utf8');
  const requires = [...workflow.matchAll(
    /const \{([^}]*)\} = require\('\.\/\.github\/scripts\/labeling\/close-old-prs\.js'\)/g,
  )];
  assert.ok(requires.length > 0, 'expected the workflow to require close-old-prs.js');

  for (const [, destructured] of requires) {
    const names = destructured
      .split(',')
      .map(entry => entry.split(':')[0].trim())
      .filter(Boolean);
    assert.ok(names.length > 0, 'expected at least one destructured name');
    for (const name of names) {
      assert.ok(
        Object.hasOwn(closeOldPrs, name),
        `clear_pending_deletion.yml requires ${name}, which close-old-prs.js does not export`,
      );
    }
  }
});

// The workflow's `if:` expression cannot call into JS, so these two label names
// are necessarily duplicated there. Drift makes the workflow a permanent no-op:
// the trigger condition simply stops matching, with no failing run to say so.
test('clear_pending_deletion.yml gates on the label names this script uses', () => {
  const workflow = fs.readFileSync(CLEAR_PENDING_DELETION_YML, 'utf8');
  const condition = workflow.match(/if: >-\n([\s\S]*?)\n {4}runs-on:/);
  assert.ok(condition, 'expected to find the job-level if: expression');

  assert.match(
    condition[1],
    new RegExp(`github\\.event\\.label\\.name == '${DEFAULT_BYPASS_LABEL}'`),
    `expected the trigger to gate on ${DEFAULT_BYPASS_LABEL}`,
  );
  assert.match(
    condition[1],
    new RegExp(`labels\\.\\*\\.name, '${DEFAULT_PENDING_DELETION_LABEL}'`),
    `expected the trigger to require ${DEFAULT_PENDING_DELETION_LABEL}`,
  );
});

// A job-level permissions block replaces the workflow-level one rather than
// merging, so every scope the job needs must be listed on the job itself.
// Omitting contents:read fails the checkout before the script ever runs.
test('clear_pending_deletion.yml grants the scopes its steps need', () => {
  const workflow = fs.readFileSync(CLEAR_PENDING_DELETION_YML, 'utf8');
  const jobPermissions = workflow.match(
    /runs-on: ubuntu-latest\n[\s\S]*?permissions:\n([\s\S]*?)\n\n/,
  );
  assert.ok(jobPermissions, 'expected job-level permissions');

  for (const scope of ['contents: read', 'issues: write', 'pull-requests: write']) {
    assert.match(jobPermissions[1], new RegExp(scope));
  }
});

// keep_open_on_comment.yml is the comment-driven entry point for the same
// bypass: it requires applyBypassLabel out of close-old-prs.js, and nothing
// else connects the two. Same failure shape as clear_pending_deletion.yml —
// a missing export destructures to undefined and the job dies on every run.
test('keep_open_on_comment.yml only requires exported symbols', () => {
  const workflow = fs.readFileSync(KEEP_OPEN_ON_COMMENT_YML, 'utf8');
  const requires = [...workflow.matchAll(
    /const \{([^}]*)\} = require\('\.\/\.github\/scripts\/labeling\/close-old-prs\.js'\)/g,
  )];
  assert.ok(requires.length > 0, 'expected the workflow to require close-old-prs.js');

  for (const [, destructured] of requires) {
    const names = destructured
      .split(',')
      .map(entry => entry.split(':')[0].trim())
      .filter(Boolean);
    assert.ok(names.length > 0, 'expected at least one destructured name');
    for (const name of names) {
      assert.ok(
        Object.hasOwn(closeOldPrs, name),
        `keep_open_on_comment.yml requires ${name}, which close-old-prs.js does not export`,
      );
    }
  }
});

// The trigger phrase and the actor gate live only in the workflow's `if:` —
// there is no JS constant to share, and a workflow expression cannot call one.
// Drift is invisible at runtime: rename the phrase and keep-open comments
// silently stop doing anything; loosen the actor gate and any passerby can
// exempt a PR from cleanup. Pin both. The label name is deliberately absent:
// the workflow takes it from DEFAULT_BYPASS_LABEL via applyBypassLabel.
test('keep_open_on_comment.yml gates on the phrase and maintainer association', () => {
  const workflow = fs.readFileSync(KEEP_OPEN_ON_COMMENT_YML, 'utf8');
  const condition = workflow.match(/if: >-\n([\s\S]*?)\n {4}runs-on:/);
  assert.ok(condition, 'expected to find the job-level if: expression');

  assert.match(
    condition[1],
    /github\.event\.issue\.pull_request/,
    'expected the trigger to require a PR conversation comment',
  );
  assert.match(
    condition[1],
    /contains\(github\.event\.comment\.body, '!keep-open'\)/,
    'expected the trigger to gate on the keep-open phrase',
  );
  assert.match(
    condition[1],
    /author_association/,
    'expected the trigger to gate on the commenter',
  );
  for (const association of ['MEMBER', 'OWNER', 'COLLABORATOR']) {
    assert.match(
      condition[1],
      new RegExp(`"${association}"`),
      `expected the actor gate to include ${association}`,
    );
  }
});

// A job-level permissions block replaces the workflow-level one rather than
// merging, so every scope the job needs must be listed on the job itself.
test('keep_open_on_comment.yml grants the scopes its steps need', () => {
  const workflow = fs.readFileSync(KEEP_OPEN_ON_COMMENT_YML, 'utf8');
  const jobPermissions = workflow.match(
    /runs-on: ubuntu-latest\n[\s\S]*?permissions:\n([\s\S]*?)\n\n/,
  );
  assert.ok(jobPermissions, 'expected job-level permissions');

  for (const scope of ['contents: read', 'issues: write', 'pull-requests: write']) {
    assert.match(jobPermissions[1], new RegExp(scope));
  }
});

// applyBypassLabel is the workflow's only mutation. These two arms are its
// whole contract: apply when absent (which emits the `labeled` event that
// clear_pending_deletion.yml listens for), no-op when present so a maintainer
// racing the bot does not cause a duplicate event.
test('applyBypassLabel adds the default bypass label when absent', async () => {
  const { github, calls } = makeGithub();
  github.rest.issues.get = async () => ({ data: { labels: [] } });

  const applied = await applyBypassLabel({
    github, owner: 'langchain-ai', repo: 'deepagents', issueNumber: 42,
  });

  assert.equal(applied, true);
  assert.deepEqual(calls.addLabels, [{
    owner: 'langchain-ai',
    repo: 'deepagents',
    issue_number: 42,
    labels: [DEFAULT_BYPASS_LABEL],
  }]);
});

test('applyBypassLabel no-ops when the label is already present', async () => {
  const { github, calls } = makeGithub();
  github.rest.issues.get = async () => ({
    data: { labels: [{ name: DEFAULT_BYPASS_LABEL }, { name: 'other' }] },
  });

  const applied = await applyBypassLabel({
    github, owner: 'langchain-ai', repo: 'deepagents', issueNumber: 42,
  });

  assert.equal(applied, false);
  assert.deepEqual(calls.addLabels, []);
});

// clearPendingDeletion is the shared "PR is exempt now" cleanup behind both
// clear_pending_deletion.yml (manual do-not-close) and keep_open_on_comment.yml
// (whose GITHUB_TOKEN-authored `labeled` event GitHub suppresses, so the
// cleanup has to run inline). Its contract: remove pending-deletion, tolerate
// its absence, and always attempt to minimize the stale warning.
test('clearPendingDeletion removes the label and minimizes the warning', async () => {
  const { github, calls } = makeGithub({
    comments: new Map([[42, [{
      id: 1,
      body: COMMENT_MARKER,
      user: { login: 'github-actions[bot]', type: 'Bot' },
    }]]]),
  });
  const core = makeCore();

  const removed = await clearPendingDeletion({
    github, core, owner: 'langchain-ai', repo: 'deepagents', issueNumber: 42,
  });

  assert.equal(removed, true);
  assert.deepEqual(calls.removeLabel, [{
    owner: 'langchain-ai',
    repo: 'deepagents',
    issue_number: 42,
    name: DEFAULT_PENDING_DELETION_LABEL,
  }]);
  assert.equal(calls.graphql.length, 1);
  assert.match(calls.graphql[0].query, /minimizeComment/);
});

// keep_open_on_comment.yml reaches this branch routinely — a !keep-open PR
// may never have carried pending-deletion. A 404 must not fail the run, and
// the warning minimization must still happen: the label and the comment are
// applied separately, so one can exist without the other.
test('clearPendingDeletion tolerates an absent label and still minimizes', async () => {
  const { github, calls } = makeGithub({
    removeLabelErrors: new Map([[42, httpError('gone', 404)]]),
    comments: new Map([[42, [{
      id: 1,
      body: COMMENT_MARKER,
      user: { login: 'github-actions[bot]', type: 'Bot' },
    }]]]),
  });
  const core = makeCore();

  const removed = await clearPendingDeletion({
    github, core, owner: 'langchain-ai', repo: 'deepagents', issueNumber: 42,
  });

  assert.equal(removed, false);
  assert.equal(core.warnings.length, 1);
  assert.equal(calls.graphql.length, 1);
});

test('clearPendingDeletion propagates non-404 removal failures', async () => {
  const { github } = makeGithub({
    removeLabelErrors: new Map([[42, httpError('forbidden', 403)]]),
  });

  await assert.rejects(
    clearPendingDeletion({
      github, core: makeCore(), owner: 'langchain-ai', repo: 'deepagents', issueNumber: 42,
    }),
    /forbidden/,
  );
});

// RELEASE_PLEASE_BRANCH_PREFIX is a hardcoded literal, not derived, and the
// same string is duplicated in five other workflows. Nothing else notices when
// the config that determines the real branch name changes — and here the
// consequence is closing real release PRs, so pin the coupling explicitly.
test('the release branch prefix matches release-please-config.json', () => {
  const config = JSON.parse(fs.readFileSync(
    path.join(REPO_ROOT, 'release-please-config.json'),
    'utf8',
  ));

  // `separate-pull-requests: false` would drop the `--components--<pkg>`
  // segment entirely, so the prefix would never match.
  assert.equal(
    config['separate-pull-requests'], true,
    'separate-pull-requests drives the --components-- branch segment',
  );
  assert.ok(
    RELEASE_PLEASE_BRANCH_PREFIX.endsWith('--components--'),
    `expected the prefix to end in --components--, got ${RELEASE_PLEASE_BRANCH_PREFIX}`,
  );

  // A `target-branch` override (or a renamed default branch) changes the
  // `<base>` segment; absent one, release-please branches off main.
  assert.equal(
    config['target-branch'], undefined,
    'a target-branch override changes the <base> segment of the branch name',
  );
  assert.equal(
    RELEASE_PLEASE_BRANCH_PREFIX, 'release-please--branches--main--components--',
  );

  // Drafts are warned and closed like any other PR, so if release PRs ever
  // stop opening as drafts nothing here breaks: the provenance check above is
  // their only protection either way. This pin keeps the "stay drafts for
  // most of their life" assumption the comments in close-old-prs.js make
  // verifiable against the config.
  assert.equal(
    config['draft-pull-request'], true,
    'release PRs are expected to open as drafts',
  );
});

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
    errors: [],
    info(message) { this.infos.push(message); },
    warning(message) { this.warnings.push(message); },
    error(message) { this.errors.push(message); },
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
  graphqlError = null,
  onCreateComment = null,
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
    graphql: [],
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
            node_id: `node-${comment.id}`,
            ...comment,
          })),
        }),
        createComment: async params => {
          calls.createComment.push(params);
          // Minimization re-lists the comments, so the posted warning must
          // be visible there — as it would be for a real second API call.
          const posted = comments.get(params.issue_number) ?? [];
          posted.push({
            id: 'new',
            node_id: 'node-new',
            body: params.body,
            user: { login: 'github-actions[bot]', type: 'Bot' },
          });
          comments.set(params.issue_number, posted);
          onCreateComment?.(params);
        },
        updateComment: async params => { calls.updateComment.push(params); },
      },
      pulls: {
        get: async ({ pull_number }) => {
          calls.get.push(pull_number);
          const error = getErrors.get(pull_number);
          if (error) throw error;
          const pr = live.get(pull_number) ?? {};
          // hasOwn rather than `??` so a fixture can express an explicitly
          // absent provenance field. GitHub really does return `head.repo:
          // null` when a fork PR's source repository was deleted, and `?? `
          // defaults would silently rewrite that into a valid-looking value.
          const field = (name, fallback) =>
            Object.hasOwn(pr, name) ? pr[name] : fallback;
          const headRepo = field('headRepo', 'contributor/deepagents');
          return {
            data: {
              created_at: field('created_at', '2026-04-01T00:00:00Z'),
              draft: field('draft', false),
              head: {
                ref: field('headRef', 'contributor/feature'),
                repo: headRepo === null ? null : { full_name: headRepo },
              },
              labels: (pr.labels ?? []).map(name => ({ name })),
              state: field('state', 'open'),
              user: {
                login: field('authorLogin', 'contributor'),
                type: field('authorType', 'User'),
              },
            },
          };
        },
        update: async ({ pull_number }) => { calls.close.push(pull_number); },
      },
      search: { issuesAndPullRequests: async () => {} },
    },
    paginate: async (method, params) => (await method(params)).data,
    graphql: async (query, variables) => {
      calls.graphql.push({ query, variables });
      if (graphqlError) throw graphqlError;
      // Mirror the real mutation's response shape rather than `{}`, so code
      // that starts reading isMinimized cannot pass here on an undefined.
      return { minimizeComment: { minimizedComment: { isMinimized: true } } };
    },
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

// Asserts both which comments were minimized and that the mutation itself is
// well-formed. The variables alone are not enough: a typo in the field name or
// the wrong classifier is a GraphQL validation error at runtime, and
// minimizeMarkerComment deliberately swallows those, so a broken mutation would
// otherwise look identical to a working one in every test here.
function assertMinimized(calls, nodeIds) {
  assert.deepEqual(calls.graphql.map(call => call.variables.id), nodeIds);
  for (const { query } of calls.graphql) {
    assert.match(query, /minimizeComment\(input: \{subjectId: \$id, classifier: OUTDATED\}\)/);
  }
}

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
    // Drafts get no exemption: #104 is warned like any other 30-day-old PR.
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
    checked: 4, warned: 2, closed: 1, skipped: 1, skippedRelease: 0,
    skippedRaced: 0,
    staleCleared: 0, sweepNotFound: 0, sweepTruncated: false, sweepFailure: null,
    incomplete: false, truncated: false, errors: [],
  });
  assert.equal(calls.createComment.length, 2);
  assert.equal(calls.createComment[0].issue_number, 101);
  assert.equal(calls.createComment[1].issue_number, 104);
  assert.match(calls.createComment[0].body, /open for at least 14 days/);
  assert.match(calls.createComment[1].body, /open for at least 14 days/);
  assert.deepEqual(
    calls.addLabels.map(call => [call.issue_number, call.labels]),
    [[101, ['pending-deletion']], [104, ['pending-deletion']]],
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
  // Ordinary contributor PRs must stay silent. Without this, dropping the
  // release-label condition from the provenance warning would emit a spurious
  // "failed provenance" line for every PR in the repo, every day.
  assert.deepEqual(core.warnings, []);
});

test('does not add pending-deletion when do-not-close is applied during warning', async () => {
  const live = new Map([[106, { labels: [] }]]);
  const { github, calls } = makeGithub({
    items: [{ number: 106, created_at: '2026-04-23T00:00:00Z' }],
    live,
    onCreateComment: ({ issue_number }) => {
      live.set(issue_number, { labels: ['do-not-close'] });
    },
  });
  const core = makeCore();

  const summary = await run({ github, context, core, options: { now } });

  assert.equal(summary.warned, 0);
  assert.equal(summary.skipped, 1);
  // Counted apart from a plain skip: the warning comment did post, so this PR
  // is visibly warned while carrying no label.
  assert.equal(summary.skippedRaced, 1);
  assert.deepEqual(calls.addLabels, []);
  assert.deepEqual(calls.get, [106, 106]);
  // The mid-warning race leaves no pending-deletion behind, so the
  // clear_pending_deletion workflow never fires for this PR. The daily bypass
  // branch would retry the minimization tomorrow; doing it here is what
  // retracts the just-posted warning promptly instead of a day late.
  assertMinimized(calls, ['node-new']);
  assert.ok(core.infos.some(message => message.includes('gained do-not-close')));
  assert.equal(core.failed, null);
  // minimizeMarkerComment downgrades every failure to a warning or an error, so
  // without these a broken minimize path would look exactly like a healthy one.
  assert.deepEqual(core.warnings, []);
  assert.deepEqual(core.errors, []);
});

// Models a label race by returning a different label set on each pulls.get.
// `onCreateComment` cannot express this on paths that post no comment, and the
// boundary re-check is defined by making a *second* fetch, so the fixture has to
// distinguish the two calls. Throws a named error rather than a bare TypeError
// if the code makes more fetches than the fixture describes.
function liveLabelSequence(...labelSets) {
  const remaining = [...labelSets];
  return {
    get labels() {
      if (remaining.length === 0) {
        throw new Error('pulls.get called more times than the fixture describes');
      }
      return remaining.shift();
    },
  };
}

// Same race as the warning path, one branch later: an already-warned PR (no
// label yet, e.g. warned before the label existed) gains do-not-close between
// the initial live fetch and the backfill boundary. The label must stay off
// and the stale warning must be minimized — the PR never carries
// pending-deletion, so clear_pending_deletion.yml never fires for it.
test('minimizes the warning when do-not-close lands before the label backfill', async () => {
  const comments = new Map([
    [107, [{ id: 92, node_id: 'node-92', body: `${COMMENT_MARKER}\nwarning`, user: workflowBot }]],
  ]);
  const { github, calls } = makeGithub({
    items: [{ number: 107, created_at: '2026-04-23T00:00:00Z' }],
    comments,
    live: new Map([[107, liveLabelSequence([], ['do-not-close'])]]),
  });
  const core = makeCore();

  const summary = await run({ github, context, core, options: { now } });

  assert.equal(summary.skipped, 1);
  // Plain skip, not skippedRaced: this path posts no comment, so bailing leaves
  // the PR exactly as it was found.
  assert.equal(summary.skippedRaced, 0);
  assert.deepEqual(calls.addLabels, []);
  assertMinimized(calls, ['node-92']);
  assert.equal(core.failed, null);
  assert.deepEqual(core.warnings, []);
  assert.deepEqual(core.errors, []);
});

// The same boundary, but with the label already applied — so the removal arm of
// refreshLabelsUnlessBypassed actually runs. Reachable in production whenever a
// PR was warned and labeled on an earlier day, passes the initial bypass check,
// and gains do-not-close mid-run.
test('removes an already-applied pending-deletion at the label boundary', async () => {
  const comments = new Map([
    [108, [{ id: 94, node_id: 'node-94', body: `${COMMENT_MARKER}\nwarning`, user: workflowBot }]],
  ]);
  const { github, calls } = makeGithub({
    items: [{ number: 108, created_at: '2026-04-23T00:00:00Z' }],
    comments,
    live: new Map([[108, liveLabelSequence(
      ['pending-deletion'],
      ['pending-deletion', 'do-not-close'],
    )]]),
  });
  const core = makeCore();

  const summary = await run({ github, context, core, options: { now } });

  assert.equal(summary.skipped, 1);
  assert.deepEqual(calls.addLabels, []);
  assert.deepEqual(
    calls.removeLabel.map(call => [call.issue_number, call.name]),
    [[108, 'pending-deletion']],
  );
  assertMinimized(calls, ['node-94']);
  assert.equal(core.failed, null);
});

// The close is the widest and most consequential boundary: unlike a spurious
// label it is never reverted. A maintainer applying do-not-close after the
// initial fetch but before the close must stop it, and must not leave the
// warning rewritten into a close notice for a PR that stays open.
test('does not close when do-not-close lands before the close boundary', async () => {
  const comments = new Map([
    [109, [{ id: 95, node_id: 'node-95', body: `${COMMENT_MARKER}\nwarning`, user: workflowBot }]],
  ]);
  const { github, calls } = makeGithub({
    items: [{ number: 109, created_at: '2026-04-08T00:00:00Z' }],
    comments,
    live: new Map([[109, liveLabelSequence(
      ['pending-deletion'],
      ['pending-deletion', 'do-not-close'],
    )]]),
  });
  const core = makeCore();

  const summary = await run({ github, context, core, options: { now } });

  assert.equal(summary.closed, 0);
  assert.equal(summary.skipped, 1);
  assert.deepEqual(calls.close, []);
  // The comment must still read as the warning, not the close notice.
  assert.deepEqual(calls.updateComment, []);
  assert.deepEqual(
    calls.removeLabel.map(call => [call.issue_number, call.name]),
    [[109, 'pending-deletion']],
  );
  assertMinimized(calls, ['node-95']);
  assert.ok(core.infos.some(message => message.includes('gained do-not-close; skipping close')));
  assert.equal(core.failed, null);
});

// A PR deleted or transferred between the initial fetch and the boundary is the
// same benign condition the initial fetch already tolerates. Treating it as
// fatal here would turn one deleted PR into a red daily sweep.
test('treats a 404 at the label boundary as benign', async () => {
  const comments = new Map([
    [110, [{ id: 96, node_id: 'node-96', body: `${COMMENT_MARKER}\nwarning`, user: workflowBot }]],
  ]);
  let fetches = 0;
  const { github, calls } = makeGithub({
    items: [{ number: 110, created_at: '2026-04-23T00:00:00Z' }],
    comments,
    live: new Map([[110, { get labels() {
      fetches += 1;
      if (fetches > 1) throw httpError('Not Found', 404);
      return [];
    } }]]),
  });
  const core = makeCore();

  const summary = await run({ github, context, core, options: { now } });

  assert.equal(summary.skipped, 1);
  assert.deepEqual(summary.errors, []);
  assert.deepEqual(calls.addLabels, []);
  assert.equal(core.failed, null);
  assert.ok(core.infos.some(message => message.includes('not found at the label boundary')));
});

test('minimizes the warning comment on an already-bypassed PR', async () => {
  const comments = new Map([
    [123, [{ id: 93, node_id: 'node-93', body: `${COMMENT_MARKER}\nwarning`, user: workflowBot }]],
  ]);
  const { github, calls } = makeGithub({
    items: [{ number: 123, created_at: '2026-04-08T00:00:00Z' }],
    comments,
    live: new Map([[123, { labels: ['do-not-close'] }]]),
  });
  const core = makeCore();

  const summary = await run({ github, context, core, options: { now } });

  assert.equal(summary.skipped, 1);
  assertMinimized(calls, ['node-93']);
  assert.equal(core.failed, null);
  assert.deepEqual(core.warnings, []);
  assert.deepEqual(core.errors, []);
});

// The entire design premise of minimizeMarkerComment is that a minimize failure
// must not disturb the label decisions around it. Nothing verified that, so
// removing the try/catch passed the whole suite.
test('a fatal minimize failure is escalated but does not fail the run', async () => {
  const comments = new Map([
    [124, [{ id: 97, node_id: 'node-97', body: `${COMMENT_MARKER}\nwarning`, user: workflowBot }]],
  ]);
  const { github, calls } = makeGithub({
    items: [{ number: 124, created_at: '2026-04-08T00:00:00Z' }],
    comments,
    live: new Map([[124, { labels: ['do-not-close', 'pending-deletion'] }]]),
    graphqlError: httpError('Resource not accessible by integration', 403),
  });
  const core = makeCore();

  const summary = await run({ github, context, core, options: { now } });

  // The label decisions still land, and the run stays green.
  assert.equal(summary.skipped, 1);
  assert.deepEqual(summary.errors, []);
  assert.equal(core.failed, null);
  assert.deepEqual(
    calls.removeLabel.map(call => [call.issue_number, call.name]),
    [[124, 'pending-deletion']],
  );
  // 403 means the token lacks the scope: permanent, so it must not read as a
  // routine warning that will sort itself out tomorrow.
  assert.deepEqual(core.warnings, []);
  assert.equal(core.errors.length, 1);
  assert.match(core.errors[0], /HTTP 403, fatal/);
  assert.match(core.errors[0], /Resource not accessible by integration/);
});

test('a transient minimize failure is only a warning', async () => {
  const comments = new Map([
    [125, [{ id: 98, node_id: 'node-98', body: `${COMMENT_MARKER}\nwarning`, user: workflowBot }]],
  ]);
  const { github } = makeGithub({
    items: [{ number: 125, created_at: '2026-04-08T00:00:00Z' }],
    comments,
    live: new Map([[125, { labels: ['do-not-close'] }]]),
    graphqlError: httpError('Bad gateway', 502),
  });
  const core = makeCore();

  const summary = await run({ github, context, core, options: { now } });

  assert.equal(summary.skipped, 1);
  assert.equal(core.failed, null);
  // The daily bypass branch retries this tomorrow, so it self-heals.
  assert.deepEqual(core.errors, []);
  assert.equal(core.warnings.length, 1);
  assert.match(core.warnings[0], /HTTP 502, transient/);
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

// A label that merely *contains* a release label as a substring is not a
// release label — guards against loosening Set.has() into a prefix/substring
// match. Uses a contributor PR because that is where the distinction bites:
// RELEASE_LABELS decides whether a provenance failure is reported as possible
// drift, and 'release-notes' must not trip it. (A genuine release PR is exempt
// on provenance regardless of its labels, so it cannot test this.)
test('does not report drift for a label that only resembles a release label', async () => {
  const { github, calls } = makeGithub({
    items: [{ number: 107, created_at: '2026-04-23T00:00:00Z' }],
    live: new Map([[107, { labels: ['release-notes'] }]]),
  });
  const core = makeCore();

  const summary = await run({ github, context, core, options: { now } });

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
  assert.deepEqual(core.warnings, []);
});

// The bug the provenance-only exemption exists to prevent. `release` is
// applied by a continue-on-error step in release-please.yml and RELEASING.md
// documents hand-editing `autorelease: pending`, so a genuine release PR can
// hold neither label. Gating the exemption on labels would warn it, then close
// it 16 days later on a green run.
test('exempts a release-please PR whose release labels are missing', async () => {
  const { github, calls } = makeGithub({
    items: [{ number: 118, created_at: '2026-04-01T00:00:00Z' }],
    live: new Map([[118, releasePleasePr([])]]),
  });
  const core = makeCore();

  const summary = await run({ github, context, core, options: { now } });

  assert.equal(summary.skippedRelease, 1);
  assert.equal(summary.warned, 0);
  assert.equal(calls.createComment.length, 0);
  assert.deepEqual(calls.addLabels, []);
  assert.deepEqual(calls.close, []);
  assert.equal(core.failed, null);
  // Exempt, but the missing label is itself a problem: per RELEASING.md a
  // stuck/absent `autorelease: pending` blocks future release PRs.
  assert.ok(
    core.warnings.some(message =>
      message.includes('provenance but no release label')),
    `expected a missing-label warning, got ${JSON.stringify(core.warnings)}`,
  );
});

// head.repo is null (not merely different) when a fork PR's source repository
// was deleted. The exemption must fail closed, and say which field was absent
// rather than rendering a bare "undefined" that reads like a mismatch.
test('denies the release exemption when the head repository is absent', async () => {
  const { github, calls } = makeGithub({
    items: [{ number: 119, created_at: '2026-04-08T00:00:00Z' }],
    live: new Map([[119, releasePleasePr(['release'], { headRepo: null })]]),
  });
  const core = makeCore();

  const summary = await run({ github, context, core, options: { now } });

  assert.equal(summary.skippedRelease, 0);
  assert.equal(summary.warned, 1);
  assert.deepEqual(calls.close, []);
  assert.ok(
    core.warnings.some(message =>
      message.includes('failed provenance') && message.includes('repo <absent>')),
    `expected an absent-repo provenance warning, got ${JSON.stringify(core.warnings)}`,
  );
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
  const core = makeCore();

  const summary = await run({ github, context, core, options: { now } });

  assert.equal(summary.staleCleared, 0);
  assert.equal(calls.removeLabel.length, 0);
  // The sweep classifies only; processPr owns the drift/anomaly reporting for a
  // given PR. Without warnOnAnomaly:false a title-spoofed PR appearing in both
  // searches emits the identical warning twice per run, raising the noise floor
  // the real drift signal has to clear.
  assert.deepEqual(core.warnings, []);
});

// A PR returned by both searches in one run: the open scan warns it, then the
// (index-lagged) label query returns it again. The second pass must not
// re-report the anomaly the first pass already handled.
test('does not double-report a PR returned by both searches', async () => {
  const { github } = makeGithub({
    items: [{ number: 120, created_at: '2026-04-08T00:00:00Z' }],
    labeledItems: [{ number: 120, created_at: '2026-04-08T00:00:00Z' }],
    live: new Map([[120, { labels: ['release'] }]]),
  });
  const core = makeCore();

  await run({ github, context, core, options: { now } });

  const driftWarnings = core.warnings.filter(message =>
    message.includes('failed provenance'));
  assert.equal(driftWarnings.length, 1, JSON.stringify(core.warnings));
});

// A 404 per PR is routine, but an unlogged `continue` makes a sweep that drops
// every PR indistinguishable from one with nothing to do.
test('sweep counts and logs PRs that vanished before it read them', async () => {
  const { github, calls } = makeGithub({
    labeledItems: [{ number: 121, created_at: '2026-04-01T00:00:00Z' }],
    getErrors: new Map([[121, httpError('not found', 404)]]),
  });
  const core = makeCore();

  const summary = await run({ github, context, core, options: { now } });

  assert.equal(summary.staleCleared, 0);
  assert.equal(summary.sweepNotFound, 1);
  assert.equal(summary.sweepFailure, null);
  assert.equal(calls.removeLabel.length, 0);
  assert.equal(core.failed, null);
  assert.ok(core.infos.some(message =>
    message.includes('PR #121 not found while sweeping')));
});

// The label search index lags this run's own mutations, so a PR whose label
// processPr just removed can still come back from the sweep query. Removing
// nothing must not be counted or logged as a removal.
test('sweep does not count a PR whose label is already gone', async () => {
  const { github, calls } = makeGithub({
    labeledItems: [{ number: 112, created_at: '2026-04-01T00:00:00Z' }],
    live: new Map([[112, { labels: [] }]]),
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
  assert.equal(summary.sweepTruncated, true);
  assert.equal(core.failed, null);
  assert.ok(core.warnings.some(message => message.includes('Reached maxItems cap (1) while sweeping')));
});

// The sweep defers capped work to the next run only if its order is stable;
// GitHub search's default relevance order can return the same over-cap subset
// forever, starving a specific PR rather than deferring it.
test('sweeps oldest-first so the cap defers work instead of starving it', async () => {
  const { github, calls } = makeGithub({
    labeledItems: [{ number: 122, created_at: '2026-04-01T00:00:00Z' }],
    live: new Map([[122, { state: 'closed', labels: ['pending-deletion'] }]]),
  });

  await run({ github, context, core: makeCore(), options: { now } });

  const sweepQuery = calls.queries.find(query => query.q.includes('label:"'));
  assert.equal(sweepQuery.sort, 'created');
  assert.equal(sweepQuery.order, 'asc');
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

test('warns a draft PR past the warning threshold', async () => {
  const { github, calls } = makeGithub({
    items: [{ number: 322, created_at: '2026-04-23T00:00:00Z' }],
    live: new Map([[322, { draft: true, labels: [] }]]),
  });

  const summary = await run({ github, context, core: makeCore(), options: { now } });

  assert.equal(summary.warned, 1);
  assert.equal(summary.skipped, 0);
  assert.equal(calls.createComment.length, 1);
  assert.equal(calls.createComment[0].issue_number, 322);
  assert.match(calls.createComment[0].body, /open for at least 14 days/);
  assert.deepEqual(calls.addLabels, [{
    owner: 'langchain-ai',
    repo: 'deepagents',
    issue_number: 322,
    labels: ['pending-deletion'],
  }]);
  assert.deepEqual(calls.close, []);
});

test('closes a draft PR past the close threshold with an old-enough warning', async () => {
  const comments = new Map([
    [323, [{ id: 96, body: `${COMMENT_MARKER}\nwarning`, user: workflowBot }]],
  ]);
  const { github, calls } = makeGithub({
    items: [{ number: 323, created_at: '2026-04-01T00:00:00Z' }],
    comments,
    live: new Map([[323, { draft: true, labels: ['pending-deletion'] }]]),
  });

  const summary = await run({ github, context, core: makeCore(), options: { now } });

  assert.equal(summary.closed, 1);
  assert.deepEqual(calls.close, [323]);
  assert.equal(calls.createComment.length, 0);
  assert.match(calls.updateComment[0].body, /open for at least 30 days/);
  assert.deepEqual(calls.removeLabel, [{
    owner: 'langchain-ai',
    repo: 'deepagents',
    issue_number: 323,
    name: 'pending-deletion',
  }]);
});

test('sweep clears pending-deletion on closed PRs missed by open search', async () => {
  const { github, calls } = makeGithub({
    items: [],
    labeledItems: [
      { number: 330, created_at: '2026-04-01T00:00:00Z' },
      { number: 331, created_at: '2026-04-01T00:00:00Z' },
      { number: 332, created_at: '2026-04-01T00:00:00Z' },
    ],
    live: new Map([
      [330, { state: 'closed', labels: ['pending-deletion'] }],
      // A draft is still a close candidate, so its label is not stale.
      [331, { draft: true, labels: ['pending-deletion'] }],
      [332, { labels: ['pending-deletion'] }],
    ]),
  });

  const summary = await run({ github, context, core: makeCore(), options: { now } });

  assert.equal(summary.staleCleared, 1);
  assert.deepEqual(
    calls.removeLabel.map(call => call.issue_number),
    [330],
  );
  // Still-open non-exempt PRs — drafts included — keep the label.
  assert.ok(!calls.removeLabel.some(call => call.issue_number === 331));
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

test('uses all open PRs and rejects invalid thresholds', async () => {
  const { github, calls } = makeGithub();
  await run({ github, context, core: makeCore(), options: { now } });

  assert.equal(calls.queries[0].q, 'repo:langchain-ai/deepagents is:pr is:open');
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
