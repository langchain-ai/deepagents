const assert = require('node:assert/strict');
const test = require('node:test');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { spawnSync } = require('node:child_process');
const normalize = require('../../release/normalize-release-labels.js');

const ROOT = path.resolve(__dirname, '../../../..');
const PENDING = 'auto:release-pending';
const LEGACY = 'autorelease: pending';
const TAGGED = 'auto:release-tagged';
const OLD_TAGGED = 'autorelease: tagged';

function release(number, labels, overrides = {}) {
  return { number, title: 'release(deepagents): 1.2.3', state: 'merged', labels, ...overrides };
}

// Run the real workflow shell against a stateful gh substitute. Its label
// filters implement GitHub's AND semantics, so dual-label-only queries fail
// the old-only/new-only cases rather than being hidden by canned responses.
function fakeGh() {
  const fs = require('node:fs');
  const state = JSON.parse(fs.readFileSync(process.env.RELEASE_TEST_STATE, 'utf8'));
  const args = process.argv.slice(2);
  const value = flag => args[args.indexOf(flag) + 1];
  const values = flag => args.flatMap((arg, index) => arg === flag ? [args[index + 1]] : []);
  const fail = message => { process.stderr.write(message); process.exit(1); };
  if (state.fail === args[1] || state.fail === value('--label')) fail('GitHub unavailable');
  if (args[0] === 'api') {
    process.stdout.write(JSON.stringify(state.commitPr ? [{ number: state.commitPr }] : []));
  } else if (args[0] === 'label' && args[1] === 'create') {
    if (state.repoLabels.includes(args[2]) && !args.includes('--force')) fail('Label already exists');
    state.repoLabels = [...new Set([...state.repoLabels, args[2]])];
    fs.writeFileSync(process.env.RELEASE_TEST_STATE, JSON.stringify(state));
  } else if (args[1] === 'list') {
    if (state.malformed) {
      process.stdout.write('{}');
      return;
    }
    const prs = state.prs.filter(pr => pr.state === value('--state') &&
      values('--label').every(label => pr.labels.includes(label)));
    process.stdout.write(JSON.stringify(prs.map(pr => ({ number: pr.number, title: pr.title }))));
  } else if (args[1] === 'view') {
    const pr = state.prs.find(pr => pr.number === Number(args[2]));
    if (!pr) fail('PR not found');
    process.stdout.write(pr.labels.join('\n'));
  } else if (args[1] === 'edit') {
    const pr = state.prs.find(pr => pr.number === Number(args[2]));
    for (const label of values('--add-label')) {
      if (!state.repoLabels.includes(label)) fail('Label does not exist');
    }
    for (const label of values('--remove-label')) {
      if (!pr.labels.includes(label)) fail('Label was not present');
      pr.labels = pr.labels.filter(name => name !== label);
    }
    pr.labels = [...new Set([...pr.labels, ...values('--add-label')])];
    fs.writeFileSync(process.env.RELEASE_TEST_STATE, JSON.stringify(state));
  } else {
    fail(`Unexpected gh arguments: ${args.join(' ')}`);
  }
}

function runShell(t, script, state) {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'release-labels-'));
  t.after(() => fs.rmSync(directory, { recursive: true, force: true }));
  const statePath = path.join(directory, 'state.json');
  fs.writeFileSync(statePath, JSON.stringify({
    repoLabels: [...new Set(state.prs.flatMap(pr => pr.labels))], ...state,
  }));
  fs.writeFileSync(path.join(directory, 'gh'), `#!/usr/bin/env node\n(${fakeGh.toString()})();`, { mode: 0o755 });
  const result = spawnSync('bash', ['-c', script], {
    encoding: 'utf8',
    env: {
      ...process.env, PATH: `${directory}:${process.env.PATH}`,
      RELEASE_TEST_STATE: statePath, GITHUB_STEP_SUMMARY: path.join(directory, 'summary'),
      REPO: 'owner/repo', PKG_NAME: 'deepagents', VERSION: '1.2.3',
      RELEASE_SHA: 'abc123', IS_DANGEROUS: 'false',
    },
  });
  return { ...result, state: JSON.parse(fs.readFileSync(statePath, 'utf8')) };
}

const publisherYaml = fs.readFileSync(path.join(ROOT, '.github/workflows/release.yml'), 'utf8');
const publisher = publisherYaml.split('      - name: Update release PR label\n')[1]
  .split('        run: |\n')[1].split('\n  # Kick the Code SDK')[0]
  .split('\n').map(line => line.slice(10)).join('\n')
  .replaceAll('${{ github.repository }}', 'owner/repo');
const guardYaml = fs.readFileSync(path.join(ROOT, '.github/workflows/release-please.yml'), 'utf8');
const guard = guardYaml.match(/          list_pending_json\(\) \{[\s\S]*?\n          \}/)[0]
  .split('\n').map(line => line.slice(10)).join('\n');

for (const labels of [[LEGACY], [PENDING], [LEGACY, PENDING]]) {
  for (const commitPr of [1, null]) {
    test(`publish clears ${labels.join(' + ')} via ${commitPr ? 'commit' : 'fallback'} lookup`, t => {
      const result = runShell(t, publisher, { commitPr, prs: [release(1, [...labels, 'package:deepagents'])] });
      assert.equal(result.status, 0, result.stdout + result.stderr);
      assert.deepEqual(result.state.prs[0].labels.sort(), [TAGGED, 'package:deepagents'].sort());
      const retry = runShell(t, publisher, result.state);
      assert.equal(retry.status, 0, retry.stdout + retry.stderr);
      assert.deepEqual(retry.state.prs, result.state.prs);
    });
  }
}

for (const label of [TAGGED, OLD_TAGGED]) {
  for (const commitPr of [1, null]) {
    test(`already tagged with ${label} succeeds via ${commitPr ? 'commit' : 'fallback'}`, t => {
      const result = runShell(t, publisher, { commitPr, prs: [release(1, [label])] });
      assert.equal(result.status, 0, result.stdout + result.stderr);
    });
  }
}

test('publish succeeds when the repository already has the tagged label', t => {
  const result = runShell(t, publisher, {
    commitPr: 1, repoLabels: [PENDING, TAGGED], prs: [release(1, [PENDING])],
  });
  assert.equal(result.status, 0, result.stdout + result.stderr);
  assert.deepEqual(result.state.prs[0].labels, [TAGGED]);
});

test('fallback never updates another release version or an open PR', t => {
  const prs = [release(2, [LEGACY], { title: 'release(deepagents): 1.2.30' }),
    release(3, [PENDING], { state: 'open' })];
  const result = runShell(t, publisher, { prs });
  assert.notEqual(result.status, 0);
  assert.deepEqual(result.state.prs, prs);
});

test('retry clears stale pending labels even when a tagged label is already present', t => {
  const result = runShell(t, publisher, {
    commitPr: 1, prs: [release(1, [PENDING, LEGACY, OLD_TAGGED])],
  });
  assert.equal(result.status, 0, result.stdout + result.stderr);
  assert.deepEqual(result.state.prs[0].labels.sort(), [OLD_TAGGED, TAGGED].sort());
});

for (const fail of ['list', 'view', 'create', 'edit']) {
  test(`publisher fails when GitHub ${fail} fails`, t => {
    const prs = [release(1, [PENDING, LEGACY])];
    const result = runShell(t, publisher, { commitPr: fail === 'list' ? null : 1, prs, fail });
    assert.notEqual(result.status, 0);
    assert.deepEqual(result.state.prs, prs);
  });
}

test('guard finds either pending name, deduplicates both, and ignores open PRs', t => {
  const result = runShell(t, `${guard}\nlist_pending_json`, {
    prs: [release(1, [LEGACY]), release(2, [PENDING]), release(3, [PENDING, LEGACY]),
      release(4, [LEGACY], { state: 'open' }), release(5, [TAGGED])],
  });
  assert.equal(result.status, 0, result.stderr);
  assert.deepEqual(JSON.parse(result.stdout).map(pr => pr.number), [1, 2, 3]);
});

test('guard succeeds with no pending releases', t => {
  const result = runShell(t, `${guard}\nlist_pending_json`, { prs: [] });
  assert.equal(result.status, 0, result.stderr);
  assert.deepEqual(JSON.parse(result.stdout), []);
});

for (const state of [{ fail: PENDING }, { fail: LEGACY }, { malformed: true }]) {
  test(`guard fails closed for ${JSON.stringify(state)}`, t => {
    const result = runShell(t, `${guard}\nlist_pending_json`, { prs: [], ...state });
    assert.notEqual(result.status, 0);
  });
}

function normalizationApi(issues, { labelExists = true, fail, createStatus } = {}) {
  const labels = new Set(labelExists ? [PENDING] : []);
  const api = {
    listForRepo: async options => issues.filter(issue => issue.state === options.state &&
      issue.labels.some(label => (label.name ?? label) === options.labels)),
    getLabel: async () => {
      if (fail === 'read') throw Object.assign(new Error('Forbidden'), { status: 403 });
      if (!labels.has(PENDING)) throw Object.assign(new Error('Missing'), { status: 404 });
    },
    createLabel: async ({ name }) => {
      if (createStatus) {
        throw Object.assign(new Error('Validation failed'), { status: createStatus });
      }
      labels.add(name);
    },
    addLabels: async ({ issue_number, labels: added }) => {
      if (fail === 'write') throw new Error('Write failed');
      assert.ok(added.every(label => labels.has(label)));
      issues.find(issue => issue.number === issue_number).labels.push(...added);
    },
  };
  return { rest: { issues: api }, paginate: (method, options) => method(options) };
}

test('normalization makes legacy PRs discoverable without losing existing labels', async () => {
  const prs = [release(1, [{ name: LEGACY }, 'package:deepagents'], { state: 'open', pull_request: {} }),
    release(2, [LEGACY, PENDING], { state: 'open', pull_request: {} }),
    release(3, [LEGACY], { state: 'open' }),
    release(4, [LEGACY], { state: 'closed', pull_request: {} })];
  const api = normalizationApi(prs, { labelExists: false });
  await normalize(api, 'owner', 'repo');
  assert.deepEqual(prs[0].labels, [{ name: LEGACY }, 'package:deepagents', PENDING]);
  assert.deepEqual(prs[1].labels, [LEGACY, PENDING]);
  assert.deepEqual(prs[2].labels, [LEGACY]);
  assert.deepEqual(prs[3].labels, [LEGACY]);
  await normalize(api, 'owner', 'repo');
  assert.equal(prs[0].labels.length, 3);
});

test('normalization tolerates a create-label 422 race', async () => {
  const prs = [release(1, [LEGACY], { state: 'open', pull_request: {} })];
  const api = normalizationApi(prs, { labelExists: false, createStatus: 422 });
  // The first lookup 404s, the create loses the race, and the re-fetch finds
  // the label a concurrent run just made.
  let lookups = 0;
  api.rest.issues.getLabel = async () => {
    if (lookups++ === 0) throw Object.assign(new Error('Missing'), { status: 404 });
  };
  api.rest.issues.addLabels = async ({ issue_number, labels: added }) => {
    prs.find(issue => issue.number === issue_number).labels.push(...added);
  };
  await normalize(api, 'owner', 'repo');
  assert.equal(lookups, 2, 'the 422 must be re-checked, not assumed to be the race');
  assert.deepEqual(prs[0].labels, [LEGACY, PENDING]);
});

test('normalization surfaces a 422 that is not the race', async () => {
  const prs = [release(1, [LEGACY], { state: 'open', pull_request: {} })];
  // getLabel keeps 404ing, so the label is genuinely absent: the 422 was a
  // real validation failure and must not be mistaken for a concurrent create.
  await assert.rejects(
    normalize(normalizationApi(prs, { labelExists: false, createStatus: 422 }), 'owner', 'repo'),
    /Validation failed/,
  );
  assert.deepEqual(prs[0].labels, [LEGACY]);
});

// The script destructures loadConfig at load time, so the stub has to be in
// place before it is required. Re-requiring in isolation keeps the shared
// `normalize` above bound to the real config.
test('normalization fails loudly when the palette has no auto: prefix', async () => {
  const helperPath = require.resolve('../../labeling/pr-labeler.js');
  const scriptPath = require.resolve('../../release/normalize-release-labels.js');
  const helper = require(helperPath);
  const realLoadConfig = helper.loadConfig;
  helper.loadConfig = () => ({ ...realLoadConfig(), labelColors: {} });
  delete require.cache[scriptPath];
  const prs = [release(1, [LEGACY], { state: 'open', pull_request: {} })];
  try {
    await assert.rejects(
      require(scriptPath)(normalizationApi(prs, { labelExists: false }), 'owner', 'repo'),
      /labelColors\['auto:'\] is missing/,
    );
  } finally {
    helper.loadConfig = realLoadConfig;
    delete require.cache[scriptPath];
  }
  assert.deepEqual(prs[0].labels, [LEGACY], 'no PR is labeled when the palette is broken');
});

for (const fail of ['read', 'write']) {
  test(`normalization stops release-please when label ${fail} fails`, async () => {
    const prs = [release(1, [LEGACY], { state: 'open', pull_request: {} })];
    await assert.rejects(normalize(normalizationApi(prs, { fail }), 'owner', 'repo'));
    assert.deepEqual(prs[0].labels, [LEGACY]);
  });
}
