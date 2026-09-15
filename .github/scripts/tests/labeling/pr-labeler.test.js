const assert = require('node:assert/strict');
const test = require('node:test');
const fs = require('node:fs');
const path = require('node:path');

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
