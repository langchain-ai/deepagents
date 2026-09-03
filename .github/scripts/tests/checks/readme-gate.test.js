const assert = require('node:assert/strict');
const test = require('node:test');
const fs = require('node:fs');
const path = require('node:path');

const readmeGate = require('../../checks/readme-gate.js');

const {
  collectChangedPaths,
  enforce,
  isForkPr,
  STICKY_MARKER,
  ACKNOWLEDGMENT_LABEL,
  LISTFILES_CAP,
} = readmeGate;

const REPO_ROOT = path.resolve(__dirname, '../../../..');
const WORKFLOW = path.join(REPO_ROOT, '.github/workflows/project_readme_check.yml');

const OWNER = 'langchain-ai';
const REPO = 'deepagents';

/** Minimal `core` double recording the calls the gate's verdict depends on. */
function fakeCore() {
  const summary = {
    written: [],
    addHeading() { return summary; },
    addRaw(text) { summary.pending = text; return summary; },
    async write() { summary.written.push(summary.pending); return summary; },
  };
  return {
    failed: [],
    warnings: [],
    outputs: {},
    summary,
    setFailed(message) { this.failed.push(message); },
    warning(message) { this.warnings.push(message); },
    setOutput(name, value) { this.outputs[name] = value; },
  };
}

function fakeContext({ files = [], changedFiles = null, fork = false } = {}) {
  return {
    repo: { owner: OWNER, repo: REPO },
    payload: {
      pull_request: {
        number: 42,
        changed_files: changedFiles === null ? files.length : changedFiles,
        head: {
          repo: fork ? { full_name: 'someone-else/deepagents' } : {
            full_name: `${OWNER}/${REPO}`,
          },
        },
      },
    },
  };
}

function fakeGithub({ files = [], labels = [], comments = [], labelsThrow = false } = {}) {
  const calls = { created: [], updated: [], deleted: [] };
  const github = {
    calls,
    async paginate(fn) {
      if (fn === github.rest.pulls.listFiles) return files;
      if (fn === github.rest.issues.listComments) return comments;
      if (fn === github.rest.issues.listLabelsOnIssue) {
        if (labelsThrow) throw new Error('API down');
        return labels;
      }
      throw new Error('unexpected paginate target');
    },
    rest: {
      pulls: { listFiles() {} },
      issues: {
        listComments() {},
        listLabelsOnIssue() {},
        async createComment(args) { calls.created.push(args); },
        async updateComment(args) { calls.updated.push(args); },
        async deleteComment(args) { calls.deleted.push(args); },
      },
    },
  };
  return github;
}

const BLOCKING = { pr_type: 'feat', readmes: ['README.md'] };
const CLEAN = { pr_type: 'feat', readmes: [] };

// --- collectChangedPaths -----------------------------------------------

test('collects filenames and pre-rename paths', async () => {
  const core = fakeCore();
  const files = [
    { filename: 'libs/code/foo.py' },
    { filename: 'libs/code/READ_ME.md', previous_filename: 'libs/code/README.md' },
  ];
  const result = await collectChangedPaths({
    github: fakeGithub({ files }), context: fakeContext({ files }), core,
  });
  // Without previous_filename, `git mv README.md READ_ME.md` bypasses the gate.
  assert.ok(result.paths.includes('libs/code/README.md'));
  assert.equal(result.truncated, false);
  assert.deepEqual(core.failed, []);
});

test('fails closed when the file list is short for an unexplained reason', async () => {
  const core = fakeCore();
  const files = [{ filename: 'a.py' }];
  const result = await collectChangedPaths({
    github: fakeGithub({ files }),
    context: fakeContext({ files, changedFiles: 9 }),
    core,
  });
  assert.equal(result, null);
  assert.equal(core.failed.length, 1);
});

test('fails closed when changed_files is missing from the payload', async () => {
  const core = fakeCore();
  const context = fakeContext();
  delete context.payload.pull_request.changed_files;
  const result = await collectChangedPaths({
    github: fakeGithub(), context, core,
  });
  assert.equal(result, null);
  assert.equal(core.failed.length, 1);
});

test('a PR past the listing cap is reported truncated, not hard-failed', async () => {
  // Hard-failing here would run before both bypasses, leaving a >3000-file PR
  // permanently unmergeable with no escape hatch.
  const core = fakeCore();
  const files = Array.from({ length: LISTFILES_CAP }, (_, i) => ({
    filename: `f${i}.py`,
  }));
  const result = await collectChangedPaths({
    github: fakeGithub({ files }),
    context: fakeContext({ files, changedFiles: LISTFILES_CAP + 5 }),
    core,
  });
  assert.equal(result.truncated, true);
  assert.deepEqual(core.failed, []);
  assert.equal(core.warnings.length, 1);
});

// --- enforce: the blocking decision ------------------------------------

test('blocks a non-docs PR that touches a protected README', async () => {
  const core = fakeCore();
  const github = fakeGithub();
  await enforce({ github, context: fakeContext(), core, result: BLOCKING });
  assert.equal(core.failed.length, 1, 'the gate must fail the check');
  assert.match(core.failed[0], /README\.md/);
  assert.equal(github.calls.created.length, 1);
  assert.ok(github.calls.created[0].body.startsWith(STICKY_MARKER));
});

test('the acknowledgment label clears the block', async () => {
  const core = fakeCore();
  const github = fakeGithub({ labels: [{ name: ACKNOWLEDGMENT_LABEL }] });
  await enforce({ github, context: fakeContext(), core, result: BLOCKING });
  assert.deepEqual(core.failed, [], 'the label must clear the failure');
  assert.match(github.calls.created[0].body, /acknowledged/);
});

test('an unrelated label does not clear the block', async () => {
  const core = fakeCore();
  const github = fakeGithub({ labels: [{ name: 'size:XL' }, { name: 'lgtm' }] });
  await enforce({ github, context: fakeContext(), core, result: BLOCKING });
  assert.equal(core.failed.length, 1);
});

test('a clean result fails nothing and removes a stale sticky', async () => {
  const core = fakeCore();
  const github = fakeGithub({
    comments: [{ id: 7, body: `${STICKY_MARKER}\nold block notice` }],
  });
  await enforce({ github, context: fakeContext(), core, result: CLEAN });
  assert.deepEqual(core.failed, []);
  assert.deepEqual(github.calls.deleted, [
    { owner: OWNER, repo: REPO, comment_id: 7 },
  ]);
});

test('removing the label re-arms the block and rewrites the sticky', async () => {
  const core = fakeCore();
  const github = fakeGithub({
    comments: [{ id: 7, body: `${STICKY_MARKER}\nacknowledged` }],
  });
  await enforce({ github, context: fakeContext(), core, result: BLOCKING });
  assert.equal(core.failed.length, 1);
  assert.equal(github.calls.updated.length, 1);
  assert.match(github.calls.updated[0].body, /not a `docs` PR/);
});

test('a truncated file list blocks unless acknowledged', async () => {
  const core = fakeCore();
  await enforce({
    github: fakeGithub(), context: fakeContext(), core, result: CLEAN, truncated: true,
  });
  assert.equal(core.failed.length, 1, 'an unverifiable PR must not pass');

  const ackCore = fakeCore();
  await enforce({
    github: fakeGithub({ labels: [{ name: ACKNOWLEDGMENT_LABEL }] }),
    context: fakeContext(),
    core: ackCore,
    result: CLEAN,
    truncated: true,
  });
  assert.deepEqual(ackCore.failed, []);
});

test('fails closed when the label lookup throws', async () => {
  const core = fakeCore();
  await enforce({
    github: fakeGithub({ labelsThrow: true }),
    context: fakeContext(),
    core,
    result: BLOCKING,
  });
  assert.equal(core.failed.length, 1);
  assert.match(core.failed[0], /failing closed/);
});

test('fails closed on a malformed detector result', async () => {
  for (const bad of [null, {}, { readmes: 'README.md' }, { readmes: [1] }]) {
    const core = fakeCore();
    await enforce({ github: fakeGithub(), context: fakeContext(), core, result: bad });
    assert.equal(core.failed.length, 1, `expected failure for ${JSON.stringify(bad)}`);
  }
});

// --- fork handling ------------------------------------------------------

test('fork PRs still block, and use the summary instead of a comment', async () => {
  // The token is read-only on forks, so a comment would 403 on creation and
  // again on cleanup — leaving a stale block notice on a since-green PR.
  const core = fakeCore();
  const github = fakeGithub();
  const context = fakeContext({ fork: true });
  assert.equal(isForkPr(context), true);
  await enforce({ github, context, core, result: BLOCKING });
  assert.equal(core.failed.length, 1, 'the gate must still block fork PRs');
  assert.deepEqual(github.calls.created, [], 'no comment attempt on a fork');
  assert.equal(core.summary.written.length, 1);
});

test('a deleted fork repo is treated as a fork', async () => {
  const context = fakeContext();
  context.payload.pull_request.head.repo = null;
  assert.equal(isForkPr(context), true);
});

test('the comment API failing cannot un-fail the gate', async () => {
  const core = fakeCore();
  const github = fakeGithub();
  github.rest.issues.createComment = async () => { throw new Error('403'); };
  await enforce({ github, context: fakeContext(), core, result: BLOCKING });
  assert.equal(core.failed.length, 1);
  assert.equal(core.summary.written.length, 1, 'falls back to the job summary');
});

// --- workflow wiring ----------------------------------------------------

test('the workflow only requires symbols this module exports', () => {
  const workflow = fs.readFileSync(WORKFLOW, 'utf8');
  const requires = [...workflow.matchAll(
    /const \{([^}]*)\} = require\('\.\/\.readme-gate\/readme-gate\.js'\)/g,
  )];
  assert.ok(requires.length > 0, 'expected the workflow to require readme-gate.js');
  for (const [, destructured] of requires) {
    for (const name of destructured.split(',').map(s => s.trim()).filter(Boolean)) {
      assert.ok(
        Object.hasOwn(readmeGate, name),
        `the workflow requires ${name}, which readme-gate.js does not export`,
      );
    }
  }
});
