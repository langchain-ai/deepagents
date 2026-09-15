const assert = require('node:assert/strict');
const test = require('node:test');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const WORKFLOW = 'sync_priority_labels.yml';

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
        [...(prLabels.get(issue_number) ?? [])].map(name => ({ name })),
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
  assert.equal(current.length, 3, 'every job must declare PRIORITY_LABELS');
  assert.equal(stale.length, 3, 'every job must declare STALE_PRIORITY_LABELS');
  for (const list of current) assert.deepEqual(list, current[0]);
  for (const list of stale) assert.deepEqual(list, stale[0]);

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
