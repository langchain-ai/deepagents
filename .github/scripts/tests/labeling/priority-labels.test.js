const assert = require('node:assert/strict');
const test = require('node:test');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const WORKFLOW = 'sync_priority_labels.yml';
const REPO_ROOT = path.resolve(__dirname, '../../../..');

// github-script resolves `require('./.github/...')` from the workspace root
// (the workflow checks the repo out first). Mirror that inside the sandbox so
// these tests exercise the real shared helper rather than a stub.
const sandboxRequire = spec => require(path.join(REPO_ROOT, spec));

function workflow() {
  return fs.readFileSync(path.join(__dirname, '../../../workflows', WORKFLOW), 'utf8');
}

// Execute the checked-in github-script body against API doubles. The same
// extraction as lifecycle-workflows.test.js, so the tests exercise the real
// workflow source rather than a copy that can drift from it.
function runStep(step, globals) {
  const source = workflow().split(`- name: ${step}\n`)[1];
  assert.ok(source, `Missing step: ${step}`);
  const lines = source.split('          script: |\n')[1].split('\n');
  const end = lines.findIndex(line => line.trim() && !line.startsWith('            '));
  const body = (end === -1 ? lines : lines.slice(0, end))
    .map(line => line.slice(12)).join('\n')
    .replace('${{ inputs.max_items }}', '100');
  return vm.runInNewContext(`(async () => {\n${body}\n})()`, {
    console: { log() {} }, ...globals,
  });
}

// A stateful double: labels are read back after mutation, so a test cannot
// pass by asserting on a call that had no effect.
function api({ prs = [], issues = {} } = {}) {
  const prLabels = new Map(prs.map(pr => [pr.number, new Set(pr.labels)]));
  const known = new Set(['priority:urgent', 'priority:high', 'priority:backlog']);
  const rest = {
    issues: {
      get: async ({ issue_number }) => {
        const labels = issues[issue_number];
        if (!labels) throw Object.assign(new Error('Not Found'), { status: 404 });
        return { data: { labels: labels.map(name => ({ name })) } };
      },
      listLabelsOnIssue: async ({ issue_number }) =>
        [...(prLabels.get(issue_number) ?? issues[issue_number] ?? [])].map(name => ({ name })),
      removeLabel: async ({ issue_number, name }) => {
        const set = prLabels.get(issue_number);
        if (!set?.has(name)) throw Object.assign(new Error('Label does not exist'), { status: 404 });
        set.delete(name);
      },
      addLabels: async ({ issue_number, labels }) => {
        for (const name of labels) {
          assert.ok(known.has(name), `label ${name} must exist before applying`);
          prLabels.get(issue_number).add(name);
        }
      },
      getLabel: async ({ name }) => {
        if (!known.has(name)) throw Object.assign(new Error('Missing'), { status: 404 });
      },
      createLabel: async ({ name }) => known.add(name),
    },
    pulls: { list: async () => prs },
    search: { issuesAndPullRequests: async () => ({ data: { items: prs } }) },
  };
  const github = { rest, paginate: (method, options) => method(options) };
  return { github, labelsOn: num => [...(prLabels.get(num) ?? [])].sort() };
}

const failed = [];
const core = { warning() {}, setFailed(message) { failed.push(message); } };

function backfill(state) {
  failed.length = 0;
  return runStep('Backfill priority labels on open PRs', {
    ...state, core, context: { repo: { owner: 'owner', repo: 'repo' } },
  });
}

function priorityEvent(state, pr, event) {
  const isPr = ['opened', 'edited'].includes(event);
  return runStep(isPr ? 'Sync priority label to PR' : 'Propagate priority label to linked PRs', {
    ...state,
    core: { warning() {}, setFailed(message) { throw new Error(message); } },
    context: {
      repo: { owner: 'owner', repo: 'repo' },
      payload: {
        action: event, pull_request: pr,
        issue: { number: 200, labels: [{ name: 'priority:urgent' }] },
        label: { name: event === 'unlabeled' ? 'priority:urgent' : 'priority:backlog' },
      },
    },
  });
}

for (const event of ['opened', 'edited', 'labeled', 'unlabeled']) {
  for (const labels of [[], ['priority:backlog', 'p2']]) {
    test(`${event} clears old PR priorities without propagating backlog (existing: ${labels})`, async () => {
      const pr = { number: 20, body: 'Fixes #200', labels: [...labels, 'package:deepagents'] };
      const state = api({ prs: [pr], issues: { 200: ['priority:backlog'] } });
      await priorityEvent(state, pr, event);
      assert.deepEqual(state.labelsOn(20), ['package:deepagents']);
    });
  }

  test(`${event} propagates the highest escalation across all linked issues`, async () => {
    const pr = {
      number: 20, body: 'Fixes #200, fixes #201, fixes #202',
      labels: ['priority:backlog', 'priority:high', 'package:deepagents'],
    };
    const state = api({
      prs: [pr],
      issues: { 200: ['priority:backlog'], 201: ['priority:high'], 202: ['priority:urgent'] },
    });
    await priorityEvent(state, pr, event);
    assert.deepEqual(state.labelsOn(20), ['package:deepagents', 'priority:urgent']);
  });
}

// The regression this PR fixed. A PR carrying only a retired name made
// `currentPriority` and `targetLabel` both null, so the equality check
// short-circuited before the removal loop and `p2` survived the one job whose
// purpose is to strip it. Deleting the `&& !stale.length` guard fails here.
test('backfill strips a retired priority label when no new priority applies', async () => {
  const state = api({
    prs: [{ number: 7, body: 'Fixes #100', labels: ['p2', 'package:deepagents'] }],
    issues: { 100: ['type:bug'] },
  });
  await backfill(state);
  assert.deepEqual(state.labelsOn(7), ['package:deepagents']);
  assert.deepEqual(failed, []);
});

test('backfill strips a retired label while applying the linked issue priority', async () => {
  const state = api({
    prs: [{ number: 8, body: 'Closes #101', labels: ['p0'] }],
    issues: { 101: ['priority:high'] },
  });
  await backfill(state);
  assert.deepEqual(state.labelsOn(8), ['priority:high']);
});

test('backfill never applies a retired name even when the issue carries one', async () => {
  const state = api({
    prs: [{ number: 9, body: 'Resolves #102', labels: [] }],
    issues: { 102: ['p1'] },
  });
  await backfill(state);
  assert.deepEqual(state.labelsOn(9), [], 'retired names are strippable, never appliable');
});

test('backfill takes the highest priority across several linked issues', async () => {
  const state = api({
    prs: [{ number: 10, body: 'Fixes #103 and fixes #104', labels: ['priority:backlog'] }],
    issues: { 103: ['priority:backlog'], 104: ['priority:urgent'] },
  });
  await backfill(state);
  assert.deepEqual(state.labelsOn(10), ['priority:urgent']);
});

test('backfill leaves a PR carrying the correct priority untouched', async () => {
  const state = api({
    prs: [{ number: 11, body: 'Fixes #105', labels: ['priority:high'] }],
    issues: { 105: ['priority:high'] },
  });
  await backfill(state);
  assert.deepEqual(state.labelsOn(11), ['priority:high']);
});

test('backfill keeps priority labels mutually exclusive', async () => {
  const state = api({
    prs: [{ number: 12, body: 'Fixes #106', labels: ['priority:urgent', 'priority:backlog'] }],
    issues: { 106: ['priority:high'] },
  });
  await backfill(state);
  assert.deepEqual(state.labelsOn(12), ['priority:high']);
});

// `priority:backlog` is every new issue's default, so propagating it would
// label nearly every PR while saying nothing. Only escalations travel.
test('backfill does not propagate the default backlog priority to a PR', async () => {
  const state = api({
    prs: [{ number: 20, body: 'Fixes #200', labels: [] }],
    issues: { 200: ['priority:backlog'] },
  });
  await backfill(state);
  assert.deepEqual(state.labelsOn(20), [], 'backlog must not reach the PR');
});

test('backfill strips a backlog label a PR already carries', async () => {
  const state = api({
    prs: [{ number: 21, body: 'Fixes #201', labels: ['priority:backlog'] }],
    issues: { 201: ['priority:backlog'] },
  });
  await backfill(state);
  assert.deepEqual(state.labelsOn(21), [], 'a previously propagated backlog is cleared');
});

test('backfill still propagates an escalation over a backlog issue', async () => {
  const state = api({
    prs: [{ number: 22, body: 'Fixes #202 and fixes #203', labels: ['priority:backlog'] }],
    issues: { 202: ['priority:backlog'], 203: ['priority:high'] },
  });
  await backfill(state);
  assert.deepEqual(state.labelsOn(22), ['priority:high']);
});

test('backfill ignores a PR with no issue link', async () => {
  const state = api({ prs: [{ number: 13, body: 'no link here', labels: ['priority:high'] }] });
  await backfill(state);
  assert.deepEqual(state.labelsOn(13), ['priority:high'], 'an unlinked PR is not reconciled');
});

test('backfill fails the run when a PR throws, after processing the rest', async () => {
  const state = api({
    prs: [
      { number: 14, body: 'Fixes #107', labels: ['p3'] },
      { number: 15, body: 'Fixes #108', labels: [] },
    ],
    issues: { 108: ['priority:urgent'] },
  });
  state.github.rest.issues.get = async ({ issue_number }) => {
    if (issue_number === 107) throw Object.assign(new Error('boom'), { status: 500 });
    return { data: { labels: [{ name: 'priority:urgent' }] } };
  };
  await backfill(state);
  assert.equal(failed.length, 1, 'a failed PR must not leave the run green');
  assert.match(failed[0], /1 PR\(s\) failed/);
  assert.deepEqual(state.labelsOn(15), ['priority:urgent'], 'the loop continues past a failure');
});

// The three jobs each keep their own copy of these constants, and the
// `sync-to-prs` trigger repeats the union as a `fromJSON` literal. A retired
// name missing from the gate means an `unlabeled` event for it never fires and
// the stale copy on the linked PR is never cleared.
test('the three jobs and the trigger gate agree on the priority label lists', () => {
  const source = workflow();
  const lists = name => [...source.matchAll(new RegExp(`const ${name} = (\\[[^\\]]*\\]);`, 'g'))]
    .map(m => JSON.parse(m[1].replace(/'/g, '"')));

  const current = lists('PRIORITY_LABELS');
  const stale = lists('STALE_PRIORITY_LABELS');
  const propagated = lists('PROPAGATED_PRIORITY_LABELS');
  assert.equal(current.length, 3, 'every job must declare PRIORITY_LABELS');
  assert.equal(stale.length, 3, 'every job must declare STALE_PRIORITY_LABELS');
  assert.equal(propagated.length, 3, 'every job must declare PROPAGATED_PRIORITY_LABELS');
  for (const list of current) assert.deepEqual(list, current[0]);
  for (const list of stale) assert.deepEqual(list, stale[0]);
  for (const list of propagated) assert.deepEqual(list, propagated[0]);
  for (const name of propagated[0]) {
    assert.ok(current[0].includes(name), `${name} is propagated but not a current priority`);
  }
  assert.ok(!propagated[0].includes('priority:backlog'),
    'the default backlog priority must not propagate to PRs');

  const gate = JSON.parse(source.match(/fromJSON\('(\[[^)]*\])'\)/)[1]);
  assert.deepEqual([...gate].sort(), [...current[0], ...stale[0]].sort(),
    'the sync-to-prs gate must list every current and retired priority name');
});

test('no job can apply a retired priority name', () => {
  const source = workflow();
  for (const match of source.matchAll(/const PRIORITY_LABELS = (\[[^\]]*\]);/g)) {
    const applied = JSON.parse(match[1].replace(/'/g, '"'));
    for (const name of applied) {
      assert.match(name, /^priority:/, `${name} is appliable and must be a current name`);
    }
  }
});

// ── Default priority on a new issue (auto-label-by-package.yml) ──────────
// Runs the checked-in step body, same VM extraction as above, so the test
// exercises the real workflow source.
function runDefaultPriorityStep(globals) {
  const source = fs.readFileSync(
    path.join(__dirname, '../../../workflows/auto-label-by-package.yml'), 'utf8',
  ).split('- name: Apply default priority\n')[1];
  assert.ok(source, 'Missing step: Apply default priority');
  const lines = source.split('          script: |\n')[1].split('\n');
  const end = lines.findIndex(line => line.trim() && !line.startsWith('            '));
  const body = (end === -1 ? lines : lines.slice(0, end)).map(l => l.slice(12)).join('\n');
  return vm.runInNewContext(`(async () => {\n${body}\n})()`, {
    console: { log() {} }, require: sandboxRequire, ...globals,
  });
}

function issueApi({ labels = [], known = ['priority:backlog'] } = {}) {
  const present = new Set(labels), exists = new Set(known);
  const calls = { created: [], added: [] };
  return {
    calls,
    labels: () => [...present].sort(),
    globals: {
      core: { info() {}, warning() {} },
      context: { repo: { owner: 'owner', repo: 'repo' },
                 payload: { issue: { number: 7, labels: labels.map(name => ({ name })) } } },
      github: { paginate: method => method(), rest: { issues: {
        get: async () => ({ data: { labels: [...present].map(name => ({ name })) } }),
        getLabel: async ({ name }) => {
          if (!exists.has(name)) throw Object.assign(new Error('Missing'), { status: 404 });
        },
        createLabel: async ({ name, color }) => { calls.created.push([name, color]); exists.add(name); },
        addLabels: async ({ labels: names }) => {
          calls.added.push(...names); names.forEach(n => present.add(n));
        },
      } } },
    },
  };
}

test('a new issue with no priority gets the backlog default', async () => {
  const a = issueApi();
  await runDefaultPriorityStep(a.globals);
  assert.deepEqual(a.calls.added, ['priority:backlog']);
  assert.deepEqual(a.labels(), ['priority:backlog']);
});

test('an issue that already carries a priority is left alone', async () => {
  for (const existing of ['priority:high', 'priority:urgent', 'priority:backlog']) {
    const a = issueApi({ labels: [existing] });
    a.globals.context.payload.issue.labels = [];
    await runDefaultPriorityStep(a.globals);
    assert.deepEqual(a.calls.added, [], `${existing} must not be overwritten`);
    assert.deepEqual(a.labels(), [existing]);
  }
});

test('the default priority label is created with the prefix color when absent', async () => {
  const { labelColors } = require('../../labeling/pr-labeler.js').loadConfig();
  const a = issueApi({ known: [] });
  await runDefaultPriorityStep(a.globals);
  assert.deepEqual(a.calls.created, [['priority:backlog', labelColors['priority:']]]);
  assert.deepEqual(a.calls.added, ['priority:backlog']);
});

// The workflow fires on [opened, edited]. Neither step removes a label, so
// both must be gated: an ungated re-run re-adds a topic or priority that a
// maintainer removed, and the removal can never stick.
test('the issue steps that only add labels run on opened alone', () => {
  const source = fs.readFileSync(
    path.join(REPO_ROOT, '.github/workflows/auto-label-by-package.yml'), 'utf8',
  );
  assert.match(source, /on:\n  issues:\n    types: \[opened, edited\]/);
  for (const step of ['Apply default priority', 'Apply topic labels']) {
    const declaration = source.split(`- name: ${step}\n`)[1];
    assert.ok(declaration, `Missing step: ${step}`);
    assert.match(
      declaration.split('\n')[0].trim() || declaration.split('\n')[0],
      /^if: github\.event\.action == 'opened'$/,
      `${step} must be gated on the opened action`,
    );
  }
});

function runPackageStep(globals) {
  const source = fs.readFileSync(
    path.join(REPO_ROOT, '.github/workflows/auto-label-by-package.yml'), 'utf8',
  ).split('- name: Sync package labels\n')[1];
  assert.ok(source, 'Missing step: Sync package labels');
  const lines = source.split('          script: |\n')[1].split('\n');
  const end = lines.findIndex(line => line.trim() && !line.startsWith('            '));
  const body = (end === -1 ? lines : lines.slice(0, end)).map(l => l.slice(12)).join('\n');
  return vm.runInNewContext(`(async () => {\n${body}\n})()`, {
    console: { log() {} }, ...globals,
  });
}

test('package labeling skips issues without an Area section', async () => {
  await runPackageStep({
    context: {
      repo: { owner: 'owner', repo: 'repo' }, issue: { number: 7 },
      payload: { issue: { body: 'Freeform issue opened without a form.' } },
    },
    github: { rest: { issues: { get: async () => assert.fail('labels must not be read') } } },
  });
});

// ── Topic labels on an issue (auto-label-by-package.yml) ─────────────────
function runTopicStep(globals) {
  const source = fs.readFileSync(
    path.join(REPO_ROOT, '.github/workflows/auto-label-by-package.yml'), 'utf8',
  ).split('- name: Apply topic labels\n')[1];
  assert.ok(source, 'Missing step: Apply topic labels');
  const lines = source.split('          script: |\n')[1].split('\n');
  const end = lines.findIndex(line => line.trim() && !line.startsWith('            '));
  const body = (end === -1 ? lines : lines.slice(0, end)).map(l => l.slice(12)).join('\n');
  return vm.runInNewContext(`(async () => {\n${body}\n})()`, {
    console: { log() {} },
    require: spec => spec.endsWith('topic-classifier.js')
      ? { classifyTopicLabels: globals.classifyTopicLabels, loadTopicLabels: () => [] }
      : sandboxRequire(spec),
    ...globals,
  });
}

function topicApi({ title = '', body = '', labels = [], topics = [] } = {}) {
  const present = new Set(labels), added = [];
  return {
    added,
    globals: {
      classifyTopicLabels: async () => new Set(topics),
      core: { info() {}, warning() {} },
      context: { repo: { owner: 'owner', repo: 'repo' },
                 payload: { issue: { number: 42, title, body, labels: labels.map(name => ({ name })) } } },
      github: { paginate: method => method(), rest: { issues: {
        listLabelsForRepo: async () => [
          { name: 'topic:mcp' }, { name: 'topic:subagents' },
          { name: 'topic:async-subagents' },
        ],
        getLabel: async () => ({}),
        createLabel: async () => ({}),
        addLabels: async ({ labels: names }) => { added.push(...names); names.forEach(n => present.add(n)); },
      } } },
    },
  };
}

test('an issue naming a topic gets the matching topic label', async () => {
  const a = topicApi({
    title: 'async subagents hang on exit', body: 'repro below',
    topics: ['topic:async-subagents', 'topic:subagents'],
  });
  await runTopicStep(a.globals);
  assert.deepEqual(a.added.sort(), ['topic:async-subagents', 'topic:subagents']);
});

test('model topics from the issue text are applied', async () => {
  const a = topicApi({
    title: 'crash on startup', body: 'happens when the MCP server reconnects',
    topics: ['topic:mcp'],
  });
  await runTopicStep(a.globals);
  assert.deepEqual(a.added, ['topic:mcp']);
});

test('an empty model classification adds no topic labels', async () => {
  const a = topicApi({
    title: 'SDK call fails',
    body: '## Area\n\n- [x] deepagents\n- [ ] langsmith-sandbox\n',
  });
  await runTopicStep(a.globals);
  assert.deepEqual(a.added, []);
});

test('a topic label already present is not re-applied', async () => {
  const a = topicApi({ title: 'sandbox teardown leaks', labels: ['topic:sandboxes'] });
  await runTopicStep(a.globals);
  assert.deepEqual(a.added, [], 'no duplicate add, so a hand-applied topic survives an edit');
});

test('model classifications are not guessed from common words', async () => {
  const a = topicApi({ title: 'the model is slow', body: 'streams of output look fine' });
  await runTopicStep(a.globals);
  assert.deepEqual(a.added, [], 'only the model classification should apply topics');
});
