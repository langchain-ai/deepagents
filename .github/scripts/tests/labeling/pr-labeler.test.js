const assert = require('node:assert/strict');
const test = require('node:test');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const prLabeler = require('../../labeling/pr-labeler.js');

const REPO_ROOT = path.resolve(__dirname, '../../../..');
const PR_LABELER_YML = path.join(REPO_ROOT, '.github/workflows/pr_labeler.yml');

// `core` is supplied by actions/github-script at runtime; init() refuses to
// run without one, so stub the two methods the helpers touch.
const core = { info() {}, warning() {} };

function helpers() {
  return prLabeler.loadAndInit({}, 'langchain-ai', 'deepagents', core).h;
}

test('canonicalizeTitleScopes rewrites package-component scopes', () => {
  const h = helpers();
  assert.deepEqual(h.canonicalizeTitleScopes('fix(deepagents-code): x'), {
    title: 'fix(code): x',
    scopes: 'code',
  });
  assert.deepEqual(h.canonicalizeTitleScopes('feat(deepagents-talon)!: y'), {
    title: 'feat(talon)!: y',
    scopes: 'talon',
  });
});

test('canonicalizeTitleScopes handles the `!` before the parens', () => {
  const h = helpers();
  assert.deepEqual(h.canonicalizeTitleScopes('feat!(deepagents-acp): w'), {
    title: 'feat!(acp): w',
    scopes: 'acp',
  });
});

test('canonicalizeTitleScopes rewrites each scope in a comma list', () => {
  const h = helpers();
  assert.deepEqual(h.canonicalizeTitleScopes('feat(deepagents, talon): y'), {
    title: 'feat(sdk,talon): y',
    scopes: 'sdk,talon',
  });
});

test('canonicalizeTitleScopes leaves release titles alone', () => {
  const h = helpers();
  // A release PR title is a canonical version record — rewriting its scope
  // would break the release-please fan-out that reads it.
  assert.equal(h.canonicalizeTitleScopes('release(deepagents-code): 1.2.0'), null);
  assert.equal(h.canonicalizeTitleScopes('release(deepagents): 1.2.0'), null);
});

test('canonicalizeTitleScopes returns null when there is nothing to do', () => {
  const h = helpers();
  assert.equal(h.canonicalizeTitleScopes('chore: bump'), null, 'unscoped title');
  assert.equal(h.canonicalizeTitleScopes('fix(code): already'), null, 'canonical scope');
  assert.equal(h.canonicalizeTitleScopes(''), null, 'empty title');
  assert.equal(h.canonicalizeTitleScopes(undefined), null, 'missing title');
});

test('every scopeAliases target resolves to a label', () => {
  const { config, h } = prLabeler.loadAndInit({}, 'o', 'r', core);
  for (const [source, target] of Object.entries(config.scopeAliases)) {
    const renamed = h.canonicalizeTitleScopes(`fix(${source}): x`);
    assert.ok(renamed, `alias ${source} should rewrite`);
    assert.ok(
      config.scopeToLabel[target],
      `scopeAliases maps ${source} -> ${target}, which has no scopeToLabel entry`,
    );
  }
});

// pr_labeler.yml calls this helper instead of carrying its own copy of the
// alias map. A rename here would leave the workflow calling an undefined
// function, and github-script swallows that into a failed step rather than an
// obvious config error.
test('pr_labeler.yml consumes the shared helper, not an inline alias map', () => {
  const workflow = fs.readFileSync(PR_LABELER_YML, 'utf8');
  assert.match(workflow, /h\.canonicalizeTitleScopes\(/);
  assert.doesNotMatch(
    workflow,
    /scopeAliases\s*=\s*new Map/,
    'the inline alias map is back — keep it in pr-labeler-config.json',
  );
});

const typeCases = [
  ['feat', 'type:feature'], ['fix', 'type:bug'], ['docs', 'type:docs'],
  ['hotfix', 'type:hotfix'], ['style', 'type:style'], ['refactor', 'type:refactor'],
  ['perf', 'type:performance'], ['test', 'type:test'], ['build', 'type:build'],
  ['ci', 'type:ci'], ['chore', 'type:chore'], ['revert', 'type:revert'],
  ['release', 'auto:release-pr'],
];

for (const [type, label] of typeCases) {
  test(`${type} titles derive ${label} alongside package/integration labels`, () => {
    const result = helpers().matchTitleLabels(`${type}(sdk,daytona): update behavior`);
    assert.deepEqual([...result.labels].sort(), [label, 'package:deepagents', 'integration:daytona'].sort());
  });
}

for (const title of ['feat(sdk)!: incompatible change', 'feat!(sdk): incompatible change', 'feat!: incompatible change']) {
  test(`${title} carries both the feature and breaking labels`, () => {
    const { labels, breaking } = helpers().matchTitleLabels(title);
    assert.ok(labels.has('type:feature'));
    assert.ok(labels.has('type:breaking'));
    assert.equal(breaking, true);
  });
}

test('title edits remove obsolete classifications but preserve unrelated metadata', () => {
  const stale = helpers().getStaleTitleLabels('fix(code): correct behavior', [
    'type:feature', 'type:breaking', 'type:bug', 'package:dcode', 'priority:high',
    'auto:release-pending', 'auto:release-tagged', 'type:spike',
  ]);
  assert.deepEqual(stale, ['type:feature', 'type:breaking']);
});

for (const title of ['', undefined, 'not a conventional title', 'unknown(sdk): change', 'constructor(sdk): change']) {
  test(`unrecognized title ${JSON.stringify(title)} preserves existing classifications`, () => {
    const h = helpers();
    assert.equal(h.matchTitleLabels(title).typeLabel, null);
    assert.deepEqual(h.getStaleTitleLabels(title, ['type:feature', 'type:breaking']), []);
  });
}

function labelerApi(title, labels) {
  const assigned = new Set(labels);
  const known = new Map(labels.map(name => [name, {}]));
  const pr = { number: 12, title, user: { login: 'contributor', type: 'User' }, head: { ref: 'feature-branch' } };
  const issues = {
    listLabelsOnIssue: async () => [...assigned].map(name => ({ name })),
    getLabel: async ({ name }) => {
      if (!known.has(name)) throw Object.assign(new Error('Missing label'), { status: 404 });
    },
    createLabel: async options => known.set(options.name, options),
    removeLabel: async ({ name }) => assigned.delete(name),
    addLabels: async ({ labels: added }) => {
      for (const name of added) {
        assert.ok(known.has(name), `label ${name} must exist before applying`);
        assigned.add(name);
      }
    },
  };
  const pulls = {
    get: async () => ({ data: pr }), list: async () => [pr],
    listFiles: async () => [{ filename: 'libs/code/example.py', additions: 1, deletions: 0 }],
  };
  const github = { rest: { issues, pulls }, paginate: (method, options) => method(options) };
  const h = prLabeler.loadAndInit(github, 'owner', 'repo', core).h;
  h.getContributorInfo = async () => ({ isExternal: false });
  return { assigned, known, pr, github, h };
}

function workflowScript(filename, stepName) {
  const yaml = fs.readFileSync(path.join(REPO_ROOT, '.github/workflows', filename), 'utf8');
  const step = yaml.split(`      - name: ${stepName}\n`)[1].split('\n      - name:')[0];
  return step.split('          script: |\n')[1].split('\n')
    .map(line => line.slice(12)).join('\n').replace('${{ inputs.max_items }}', '100');
}

async function runLabeler(mode, api) {
  if (mode === 'release helper') return api.h.labelPR(api.pr.number);
  const [filename, stepName] = mode === 'live'
    ? ['pr_labeler.yml', 'Apply PR labels']
    : ['pr_labeler_backfill.yml', 'Backfill labels on open PRs'];
  const script = workflowScript(filename, stepName);
  // The script is checked-in workflow code; PR titles remain data in context.
  await vm.runInNewContext(`(async () => { ${script}\n })()`, {
    github: api.github,
    context: { repo: { owner: 'owner', repo: 'repo' }, payload: { pull_request: api.pr, action: 'edited' } },
    require: () => ({ loadAndInit: () => ({ h: api.h }) }),
    core: { ...core, setFailed(message) { throw new Error(message); } },
    process: { env: {} }, console: { log() {} },
  });
}

for (const mode of ['live', 'backfill', 'release helper']) {
  test(`${mode} replaces stale types and breaking labels after a title edit`, async () => {
    const api = labelerApi('refactor(code): simplify the implementation', [
      'type:feature', 'type:breaking', 'package:dcode', 'priority:high',
    ]);
    await runLabeler(mode, api);
    assert.ok(api.assigned.has('type:refactor'));
    assert.ok(!api.assigned.has('type:feature'));
    assert.ok(!api.assigned.has('type:breaking'));
    assert.ok(api.assigned.has('package:dcode'));
    assert.ok(api.assigned.has('priority:high'));
    assert.ok(api.known.get('type:refactor').description);
  });

  test(`${mode} preserves classification when a title has no recognized type`, async () => {
    const api = labelerApi('work in progress', ['type:bug', 'type:breaking']);
    await runLabeler(mode, api);
    assert.ok(api.assigned.has('type:bug'));
    assert.ok(api.assigned.has('type:breaking'));
  });

  test(`${mode} applies the release marker without touching lifecycle labels`, async () => {
    const api = labelerApi('release(deepagents-code): 1.2.0', ['type:chore', 'auto:release-pending']);
    await runLabeler(mode, api);
    assert.ok(api.assigned.has('auto:release-pr'));
    assert.ok(api.assigned.has('auto:release-pending'));
    assert.ok(!api.assigned.has('type:chore'));
  });
}
