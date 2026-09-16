const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');

const { syncTopicLabels } = require('../../labeling/sync-topic-labels.js');

test('writes sorted topic labels and excludes other namespaces', async () => {
  const output = path.join(fs.mkdtempSync(path.join(os.tmpdir(), 'topics-')), 'labels.json');
  const github = {
    rest: { issues: { listLabelsForRepo: async () => [
      { name: 'topic:zeta' }, { name: 'priority:high' }, { name: 'topic:alpha' },
    ] } },
    paginate: method => method(),
  };

  const topics = await syncTopicLabels(github, 'owner', 'repo', output);

  assert.deepEqual(topics, ['topic:alpha', 'topic:zeta']);
  assert.equal(fs.readFileSync(output, 'utf8'), '[\n  "topic:alpha",\n  "topic:zeta"\n]\n');
});

const { execFileSync, spawnSync } = require('node:child_process');
const branch = 'automation/sync-topic-labels';
const manifest = '.github/topic-labels.json';
const workflow = fs.readFileSync(
  path.join(__dirname, '../../../workflows/sync_topic_labels.yml'), 'utf8',
);
const publishScript = workflow.split('        run: |\n')[1]
  .split('\n').map(line => line.slice(10)).join('\n');

function repository(t, { existing = false, changed = false, lookupFails = false } = {}) {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'topic-workflow-'));
  t.after(() => fs.rmSync(directory, { recursive: true, force: true }));
  const cwd = path.join(directory, 'checkout');
  const remote = path.join(directory, 'remote.git');
  const bin = path.join(directory, 'bin');
  fs.mkdirSync(cwd);
  fs.mkdirSync(bin);
  const git = (...args) => execFileSync('git', args, { cwd, encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] }).trim();
  git('init', '--bare', remote);
  git('init', '-b', 'main');
  git('config', 'user.name', 'Test');
  git('config', 'user.email', 'test@example.com');
  git('config', 'commit.gpgsign', 'false');
  git('config', 'core.hooksPath', bin);
  fs.mkdirSync(path.join(cwd, '.github'));
  fs.writeFileSync(path.join(cwd, manifest), '[]\n');
  git('add', manifest);
  git('commit', '-m', 'Initial manifest');
  git('remote', 'add', 'origin', remote);
  if (existing) {
    git('checkout', '-b', branch);
    fs.writeFileSync(path.join(cwd, manifest), '["topic:removed"]\n');
    git('commit', '-am', 'Obsolete manifest');
    git('push', 'origin', branch);
    git('checkout', 'main');
  }
  if (changed) fs.writeFileSync(path.join(cwd, manifest), '["topic:added"]\n');
  const stateFile = path.join(directory, 'prs.json');
  fs.writeFileSync(stateFile, JSON.stringify({ open: existing, lookupFails }));
  // Simulate GitHub PR state; Git reads/writes a real local bare remote.
  fs.writeFileSync(path.join(bin, 'gh'), `#!${process.execPath}
const fs = require('node:fs');
const { execFileSync } = require('node:child_process');
const args = process.argv.slice(2);
const file = process.env.PR_STATE;
const state = JSON.parse(fs.readFileSync(file, 'utf8'));
if (args[0] !== 'pr') process.exit(1);
if (args[1] === 'list') {
  if (state.lookupFails) process.exit(1);
  if (state.open) console.log('42');
} else if (args[1] === 'close') {
  if (!state.open || args[2] !== '42') process.exit(1);
  state.open = false;
  if (args.includes('--delete-branch')) {
    execFileSync('git', ['push', 'origin', '--delete', process.env.BRANCH]);
  }
} else if (args[1] === 'create') {
  if (state.open) process.exit(1);
  state.open = true;
} else process.exit(1);
fs.writeFileSync(file, JSON.stringify(state));
`, { mode: 0o755 });
  return {
    git,
    state: () => JSON.parse(fs.readFileSync(stateFile, 'utf8')),
    run: () => spawnSync('bash', ['-c', publishScript], {
      cwd, encoding: 'utf8',
      env: { ...process.env, PATH: `${bin}:${process.env.PATH}`, BRANCH: branch, BASE: 'main', PR_STATE: stateFile },
    }),
  };
}

for (const existing of [false, true]) {
  for (const changed of [false, true]) {
    test(`manifest publication: existing PR=${existing}, changed=${changed}`, t => {
      const repo = repository(t, { existing, changed });
      const base = repo.git('rev-parse', 'main');
      const result = repo.run();
      assert.equal(result.status, 0, result.stdout + result.stderr);
      assert.equal(repo.state().open, changed);
      assert.equal(repo.git('rev-parse', 'main'), changed ? repo.git('rev-parse', 'HEAD') : base);
      const remoteRef = repo.git('ls-remote', '--heads', 'origin', branch);
      if (changed) {
        assert.ok(remoteRef);
        assert.equal(repo.git('show', `${branch}:${manifest}`), '["topic:added"]');
      } else {
        assert.equal(remoteRef, '', 'obsolete automation branch must be removed');
      }
    });
  }
}

test('PR lookup failure fails the workflow without deleting the existing branch', t => {
  const repo = repository(t, { existing: true, lookupFails: true });
  const before = repo.git('ls-remote', '--heads', 'origin', branch);
  assert.notEqual(repo.run().status, 0);
  assert.equal(repo.state().open, true);
  assert.equal(repo.git('ls-remote', '--heads', 'origin', branch), before);
});
