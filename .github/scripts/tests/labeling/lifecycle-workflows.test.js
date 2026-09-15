const assert = require('node:assert/strict');
const test = require('node:test');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

function workflow(name) {
  return fs.readFileSync(path.join(__dirname, '../../../workflows', name), 'utf8');
}

// Execute the checked-in github-script body with API doubles, without checking
// out PR code or requiring a YAML dependency for this network-free test suite.
function runStep(name, step, globals) {
  const source = workflow(name).split(`- name: ${step}\n`)[1];
  assert.ok(source, `Missing step: ${step}`);
  const lines = source.split('          script: |\n')[1].split('\n');
  const end = lines.findIndex(line => line.trim() && !line.startsWith('            '));
  const body = (end === -1 ? lines : lines.slice(0, end))
    .map(line => line.slice(12)).join('\n');
  return vm.runInNewContext(`(async () => {\n${body}\n})()`, {
    console: { log() {} }, core: { warning() {} }, ...globals,
  });
}

for (const label of [null, 'ci:keep-open', 'do-not-close']) {
  for (const boundary of [0, 1, 2]) {
    test(`waiting-on-author respects ${label} at fetch ${boundary}`, async () => {
      let fetch = 0;
      const closed = [];
      const item = () => ({
        number: 42, state: 'open', updated_at: '2020-01-01T00:00:00Z',
        user: { login: 'contributor', type: 'User' },
        labels: [
          { name: 'auto:waiting-on-author' },
          ...(label && fetch++ >= boundary ? [{ name: label }] : []),
        ],
      });
      const issues = {
        listForRepo: async () => [item()],
        get: async () => ({ data: item() }),
        listEvents: async () => [{
          event: 'labeled', label: { name: 'auto:waiting-on-author' },
          created_at: '2020-01-01T00:00:00Z',
        }],
        listComments: async () => [],
        createComment: async () => {},
        update: async params => closed.push(params.issue_number),
      };
      await runStep('waiting_on_author.yml', 'Close issues and PRs awaiting an author response', {
        context: { repo: { owner: 'langchain-ai', repo: 'deepagents' } },
        github: { rest: { issues }, paginate: (method, params) => method(params) },
      });
      assert.deepEqual(closed, label ? [] : [42]);
    });
  }
}

function issueLinkGate(payload) {
  const condition = workflow('require_issue_link.yml')
    .split('    if: >-\n')[1].split('    runs-on:')[0]
    .replaceAll('github.event.pull_request.labels.*.name', 'labelNames');
  return vm.runInNewContext(condition, {
    github: { event: payload },
    labelNames: payload.pull_request.labels.map(label => label.name),
    contains: (values, value) => values.includes(value),
    fromJSON: JSON.parse,
  });
}

function recoveryFixture(labels, { action = 'edited', assigned = true, reopenFails = false } = {}) {
  const pr = {
    number: 42, state: 'closed', merged: false, body: 'Fixes #123',
    user: { login: 'contributor' }, head: { sha: 'abc' },
    labels: labels.map(name => ({ name })),
  };
  const outputs = {};
  const context = {
    repo: { owner: 'langchain-ai', repo: 'deepagents' },
    payload: {
      action, pull_request: pr, sender: { login: 'maintainer' },
      issue: { number: 123 }, assignee: { login: 'contributor' },
    },
  };
  const github = {
    rest: {
      issues: {
        listLabelsOnIssue: async () => ({ data: pr.labels }),
        get: async () => ({ data: { assignees: assigned ? [pr.user] : [] } }),
        getLabel: async () => ({}),
        addLabels: async ({ labels: names }) => {
          pr.labels.push(...names.map(name => ({ name })));
        },
        removeLabel: async ({ name }) => {
          pr.labels = pr.labels.filter(label => label.name !== name);
        },
        listComments: async () => [],
      },
      pulls: {
        update: async ({ state }) => {
          if (reopenFails) throw new Error('Reopen unavailable');
          pr.state = state;
        },
        get: async () => ({ data: pr }),
      },
      repos: { getCollaboratorPermissionLevel: async () => ({ data: { permission: 'write' } }) },
      actions: { listWorkflowRuns: async () => ({ data: { workflow_runs: [] } }) },
      search: {
        issuesAndPullRequests: async ({ q }) => {
          // GitHub's comma-separated label qualifier matches any listed label.
          const names = q.match(/label:(\S+)/)[1].replaceAll('"', '').split(',');
          const items = pr.labels.some(label => names.includes(label.name)) ? [pr] : [];
          return { data: { total_count: items.length, items } };
        },
      },
    },
    paginate: (method, params) => method(params),
  };
  return {
    pr, outputs, context, github,
    core: { warning() {}, setOutput: (key, value) => { outputs[key] = value; } },
  };
}

for (const labels of [
  ['org:external', 'auto:missing-issue-link'],
  ['external', 'missing-issue-link'],
  ['org:external', 'missing-issue-link'],
  ['external', 'org:external', 'missing-issue-link', 'auto:missing-issue-link'],
]) {
  test(`body edit reopens a PR carrying ${labels.join(', ')}`, async () => {
    const fixture = recoveryFixture(labels);
    assert.equal(issueLinkGate(fixture.context.payload), true);
    await runStep('require_issue_link.yml', 'Check for issue link and assignee', fixture);
    assert.equal(fixture.outputs['has-link'], 'true');
    assert.equal(fixture.outputs['is-assigned'], 'true');
    await runStep('require_issue_link.yml', 'Remove auto:missing-issue-link label and reopen PR', fixture);
    assert.equal(fixture.pr.state, 'open');
    assert.ok(fixture.pr.labels.every(label => !label.name.includes('missing-issue-link')));
  });

  test(`issue assignment reopens a PR carrying ${labels.join(', ')}`, async () => {
    const fixture = recoveryFixture(labels);
    await runStep('reopen_on_assignment.yml', 'Find and reopen matching PRs', fixture);
    assert.equal(fixture.pr.state, 'open');
    assert.ok(fixture.pr.labels.every(label => !label.name.includes('missing-issue-link')));
  });
}

test('legacy body edits still require assignment and skip merged PRs', async () => {
  const fixture = recoveryFixture(['external', 'missing-issue-link'], { assigned: false });
  assert.equal(issueLinkGate(fixture.context.payload), true);
  await runStep('require_issue_link.yml', 'Check for issue link and assignee', fixture);
  assert.equal(fixture.outputs['is-assigned'], 'false');
  assert.equal(fixture.pr.state, 'closed');
  fixture.pr.merged = true;
  assert.equal(issueLinkGate(fixture.context.payload), false);
});

for (const label of ['trusted-contributor', 'auto:trusted-contributor', 'bypass-issue-check', 'ci:skip-issue-link']) {
  test(`issue-link enforcement preserves the ${label} bypass`, () => {
    const fixture = recoveryFixture(['external', 'missing-issue-link', label]);
    assert.equal(issueLinkGate(fixture.context.payload), false);
  });
}

for (const action of ['reopened', 'unlabeled']) {
  test(`maintainer ${action} override accepts legacy lifecycle labels`, async () => {
    const fixture = recoveryFixture(['external', 'missing-issue-link'], { action });
    if (action === 'unlabeled') {
      fixture.pr.labels = [{ name: 'external' }];
      fixture.context.payload.label = { name: 'missing-issue-link' };
    } else {
      fixture.pr.state = 'open';
    }
    assert.equal(issueLinkGate(fixture.context.payload), true);
    await runStep('require_issue_link.yml', 'Check for issue link and assignee', fixture);
    assert.equal(fixture.pr.state, 'open');
    assert.ok(fixture.pr.labels.some(label => label.name === 'ci:skip-issue-link'));
    assert.ok(fixture.pr.labels.every(label => !label.name.includes('missing-issue-link')));
  });
}

for (const [name, step] of [
  ['require_issue_link.yml', 'Remove auto:missing-issue-link label and reopen PR'],
  ['reopen_on_assignment.yml', 'Find and reopen matching PRs'],
]) {
  test(`${name} preserves recovery labels if reopening fails`, async () => {
    const labels = ['external', 'missing-issue-link', 'auto:missing-issue-link'];
    const fixture = recoveryFixture(labels, { reopenFails: true });
    await assert.rejects(runStep(name, step, fixture), /Reopen unavailable/);
    assert.equal(fixture.pr.state, 'closed');
    assert.deepEqual(fixture.pr.labels.map(label => label.name), labels);
  });
}

test('assignment leaves unrelated and bypassed legacy PRs closed', async () => {
  for (const bypass of [null, 'bypass-issue-check', 'ci:skip-issue-link']) {
    const fixture = recoveryFixture(['external', 'missing-issue-link', ...(bypass ? [bypass] : [])]);
    if (!bypass) fixture.pr.body = 'Fixes #999';
    await runStep('reopen_on_assignment.yml', 'Find and reopen matching PRs', fixture);
    assert.equal(fixture.pr.state, 'closed');
    assert.ok(fixture.pr.labels.some(label => label.name === 'missing-issue-link'));
  }
});
