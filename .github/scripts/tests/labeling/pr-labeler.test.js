const assert = require('node:assert/strict');
const test = require('node:test');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const prLabeler = require('../../labeling/pr-labeler.js');

const REPO_ROOT = path.resolve(__dirname, '../../../..');
const PR_LABELER_YML = path.join(REPO_ROOT, '.github/workflows/pr_labeler.yml');
const REQUIRE_ISSUE_LINK_YML = path.join(REPO_ROOT, '.github/workflows/require_issue_link.yml');
const TAG_EXTERNAL_ISSUES_YML = path.join(REPO_ROOT, '.github/workflows/tag-external-issues.yml');
const BACKFILL_YML = path.join(REPO_ROOT, '.github/workflows/pr_labeler_backfill.yml');

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

// ── Label descriptions ────────────────────────────────────────────

// Every label ensureLabel() can be asked to create. `fileRules` and
// `branchRules` carry a singular `label`, not a `labels` array.
//
// The org labels are applied by pr_labeler.yml directly rather than derived
// from config (see its `org:external` branch), so they are not reachable from
// any config map and have to be named here.
const WORKFLOW_APPLIED_LABELS = ['org:external', 'org:internal'];

function creatableLabels() {
  const { config, h } = prLabeler.loadAndInit({}, 'o', 'r', core);
  const names = new Set([
    ...Object.values(config.typeToLabel),
    ...Object.values(config.scopeToLabel),
    ...h.sizeLabels,
    ...h.tierLabels,
    ...WORKFLOW_APPLIED_LABELS,
    config.breakingLabel,
    config.releaseLabel,
  ]);
  for (const rule of [...config.fileRules, ...config.branchRules]) {
    if (rule.label) names.add(rule.label);
  }
  return { config, names };
}

// Guards the helper above: if a rule key is ever renamed, the reachability
// set silently empties and both tests below stop meaning anything.
test('rule-derived labels are actually reachable', () => {
  const { names } = creatableLabels();
  assert.ok(names.has('org:open-swe'), 'branchRules label not picked up');
  assert.ok(names.has('package:deepagents'), 'fileRules label not picked up');
  assert.ok(names.size > 25, `expected the full taxonomy, got ${names.size}`);
});

// ensureLabel() writes a description only when it CREATES a label, and it
// creates each one once, lazily. A label that first appears without a
// description keeps the blank forever — `getLabel` succeeds on every later
// run, so nothing patches it. That makes a missing entry permanent in
// practice, which is why this is a test and not a lint.
test('every label the labeler can create has a description', () => {
  const { config, names } = creatableLabels();
  const undescribed = [...names]
    .filter(name => !config.labelDescriptions[name]?.trim())
    .sort();
  assert.deepEqual(
    undescribed, [],
    `labelDescriptions is missing entries for: ${undescribed.join(', ')}`,
  );
});

// Catches a description left behind by a rename, which would otherwise sit in
// the config looking authoritative while applying to nothing.
test('labelDescriptions has no entry for a label nothing creates', () => {
  const { config, names } = creatableLabels();
  const orphaned = Object.keys(config.labelDescriptions)
    .filter(name => !names.has(name))
    .sort();
  assert.deepEqual(
    orphaned, [],
    `labelDescriptions describes labels the labeler never creates: ${orphaned.join(', ')}`,
  );
});

// ── Contributor tier labels ───────────────────────────────────────
//
// These were the last label literals hardcoded in pr-labeler.js. They are
// now config-driven, but require_issue_link.yml gates its whole check on the
// trusted name from a workflow-level `if:` expression, which cannot read the
// config — so that literal has to stay, and these tests are what stop it
// drifting from the producer.

test('applyTierLabel derives both tiers from config, not literals', async () => {
  const { config, h } = prLabeler.loadAndInit({}, 'o', 'r', core);
  assert.deepEqual(
    [...h.tierLabels].sort(),
    [config.tierLabels.new, config.tierLabels.trusted].sort(),
  );
  // Every tier name must be one the labeler can actually create.
  for (const name of h.tierLabels) {
    assert.match(name, /^auto:/, `${name} should live under the auto: prefix`);
  }
});

test('tier thresholds select the configured label', async () => {
  const { config } = prLabeler.loadAndInit({}, 'o', 'r', core);

  // applyTierLabel runs its own merged-PR search, so drive it by count.
  async function appliedFor(mergedCount) {
    const applied = [];
    const github = {
      rest: {
        search: {
          issuesAndPullRequests: async () => ({ data: { total_count: mergedCount } }),
        },
        issues: {
          getLabel: async () => ({}),
          addLabels: async ({ labels }) => { applied.push(...labels); },
        },
      },
    };
    const h = prLabeler.init(github, 'o', 'r', config, core);
    await h.applyTierLabel(1, 'someone');
    return applied;
  }

  assert.deepEqual(await appliedFor(config.trustedThreshold), [config.tierLabels.trusted]);
  assert.deepEqual(await appliedFor(0), [config.tierLabels.new]);
  // Between the two tiers: no tier label at all.
  assert.deepEqual(await appliedFor(1), []);
});

test('workflows that cannot read the config still name the configured tier labels', () => {
  const { config } = prLabeler.loadAndInit({}, 'o', 'r', core);
  const trusted = config.tierLabels.trusted;

  // A workflow-level `if:` cannot require() the config, so the literal is
  // load-bearing: if it ever stops matching, every trusted contributor's PR
  // gets labeled auto:missing-issue-link and closed.
  const requireIssueLink = fs.readFileSync(REQUIRE_ISSUE_LINK_YML, 'utf8');
  assert.ok(
    requireIssueLink.includes(`'${trusted}'`),
    `require_issue_link.yml must skip on ${trusted}; its literal no longer matches the config`,
  );

  // These run inside github-script and read the helper, so assert they do not
  // reintroduce a literal instead.
  for (const [file, yml] of [
    ['tag-external-issues.yml', TAG_EXTERNAL_ISSUES_YML],
    ['pr_labeler_backfill.yml', BACKFILL_YML],
  ]) {
    const body = fs.readFileSync(yml, 'utf8');
    assert.match(body, /h\.tierLabelsByTier\./, `${file} should read tier labels from the helper`);
    for (const name of Object.values(config.tierLabels)) {
      assert.ok(
        !body.includes(`'${name}'`),
        `${file} hardcodes ${name} — use h.tierLabelsByTier instead`,
      );
    }
  }
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

for (const title of ['feat(sdk)!: incompatible change', 'feat!: incompatible change']) {
  test(`${title} carries both the feature and breaking labels`, () => {
    const { labels, breaking } = helpers().matchTitleLabels(title);
    assert.ok(labels.has('type:feature'));
    assert.ok(labels.has('type:breaking'));
    assert.equal(breaking, true);
  });
}

for (const title of ['feat!(sdk): incompatible change', 'feat!(sdk)!: incompatible change']) {
  test(`${title} supplies no classification and preserves existing labels`, () => {
    const h = helpers();
    assert.deepEqual([...h.matchTitleLabels(title).labels], []);
    assert.deepEqual(h.getStaleTitleLabels(title, ['type:bug', 'type:breaking']), []);
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

function labelerApi(title, labels, files = [{ filename: 'libs/code/example.py', additions: 1, deletions: 0 }]) {
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
    listFiles: async () => files,
  };
  const github = { rest: { issues, pulls }, paginate: (method, options) => method(options) };
  const h = prLabeler.loadAndInit(github, 'owner', 'repo', core).h;
  // `tierKnown` mirrors the real helper: an internal contributor has no tier
  // to look up, so it is always known. Omitting it would make the backfill's
  // unknown-tier guard fire and fail every run.
  h.getContributorInfo = async () => ({ isExternal: false, tierKnown: true });
  return { assigned, known, pr, github, h };
}

function workflowScript(filename, stepName) {
  const yaml = fs.readFileSync(path.join(REPO_ROOT, '.github/workflows', filename), 'utf8');
  const step = yaml.split(`      - name: ${stepName}\n`)[1].split('\n      - name:')[0];
  return step.split('          script: |\n')[1].split('\n')
    .map(line => line.slice(12)).join('\n').replace('${{ inputs.max_items }}', '100');
}

async function runLabeler(mode, api, {
  action = 'edited', renamedTitle = '',
  warning = message => { throw new Error(message); },
} = {}) {
  if (mode === 'release helper') return api.h.labelPR(api.pr.number);
  const [filename, stepName] = mode === 'live'
    ? ['pr_labeler.yml', 'Apply PR labels']
    : ['pr_labeler_backfill.yml', 'Backfill labels on open PRs'];
  const script = workflowScript(filename, stepName);
  // The script is checked-in workflow code; PR titles remain data in context.
  await vm.runInNewContext(`(async () => { ${script}\n })()`, {
    github: api.github,
    context: { repo: { owner: 'owner', repo: 'repo' }, payload: { pull_request: api.pr, action } },
    require: spec => {
      if (spec.endsWith('/pr-labeler.js')) return { loadAndInit: () => ({ h: api.h }) };
      throw new Error(`Unexpected module: ${spec}`);
    },
    core: { ...core, warning, setFailed(message) { throw new Error(message); } },
    process: { env: { RENAMED_TITLE: renamedTitle } }, console: { log() {} },
  });
}

for (const action of ['opened', 'synchronize', 'reopened', 'edited']) {
  test(`live ${action} does not infer topics from PR text and preserves existing topics`, async () => {
    const api = labelerApi('fix(code): reconnect MCP servers', ['topic:skills']);
    api.pr.body = 'Fix memory and sandbox issues';
    await runLabeler('live', api, { action });
    assert.deepEqual([...api.assigned].filter(name => name.startsWith('topic:')), ['topic:skills']);
    assert.ok(api.assigned.has('type:bug'));
    assert.ok(api.assigned.has('package:dcode'));
  });
}

test('live title labels use the corrected title from scope renaming', async () => {
  const api = labelerApi('fix: reconnect MCP servers', []);
  await runLabeler('live', api, { renamedTitle: 'fix(code): reconnect MCP servers' });
  assert.ok(api.assigned.has('package:dcode'));
  assert.ok(api.assigned.has('type:bug'));
});

// `edited` skips the file block, so the path signal needs a push-like action.
test('live topics come from the changed modules', async () => {
  const api = labelerApi('fix(code): correct behavior', [], [
    { filename: 'libs/code/deepagents_code/mcp_tools.py', additions: 1, deletions: 0 },
  ]);
  await runLabeler('live', api, { action: 'synchronize' });
  assert.ok(api.assigned.has('topic:mcp'), 'a touched module must contribute its topic');
  assert.ok(api.known.has('topic:mcp'), 'a path topic is created before it is applied');
  assert.ok(api.assigned.has('package:dcode'));
});

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

test('colorFor resolves a label color from its taxonomy prefix', () => {
  const { config, h } = prLabeler.loadAndInit({}, 'o', 'r', core);
  for (const [prefix, color] of Object.entries(config.labelColors)) {
    assert.equal(h.colorFor(`${prefix}anything`), color, `prefix ${prefix}`);
  }
  // A name matching no prefix falls back to the generic color.
  assert.equal(h.colorFor('unprefixed'), config.labelColor);
  assert.equal(h.colorFor(undefined), config.labelColor);
});

test('every label the config can apply has a prefix color', () => {
  const { config, h } = prLabeler.loadAndInit({}, 'o', 'r', core);
  const applied = new Set([
    ...Object.values(config.typeToLabel), config.breakingLabel, config.releaseLabel,
    ...Object.values(config.tierLabels), ...Object.values(config.scopeToLabel),
    ...config.fileRules.map(r => r.label), ...config.branchRules.map(r => r.label),
    ...config.sizeThresholds.map(t => t.label),
  ]);
  for (const name of applied) {
    assert.notEqual(
      h.colorFor(name), config.labelColor,
      `${name} has no labelColors prefix, so it would be created off-palette`,
    );
  }
});

// Priority sync and issue-link enforcement have no checkout of the shared
// helper. This check rejects inline colors outside the palette; it does not
// establish that a color belongs to the prefix of the label being created.
test('inlined workflow label colors belong to the configured palette', () => {
  const { config } = prLabeler.loadAndInit({}, 'o', 'r', core);
  const known = new Set(Object.values(config.labelColors));
  for (const rel of ['.github/workflows/sync_priority_labels.yml',
                     '.github/workflows/require_issue_link.yml',
                     '.github/workflows/auto-label-by-package.yml']) {
    const body = fs.readFileSync(path.join(REPO_ROOT, rel), 'utf8');
    for (const hit of body.match(/color: ['"]([0-9a-f]{6})['"]/g) || []) {
      const hex = hit.match(/([0-9a-f]{6})/)[1];
      assert.ok(known.has(hex), `${rel} uses ${hex}, which is not a labelColors value`);
    }
  }
});

// Scripts that CAN require the helper must not carry their own hex.
test('label-creating scripts resolve colors from the config', () => {
  for (const rel of ['.github/scripts/labeling/close-old-prs.js',
                     '.github/scripts/release/normalize-release-labels.js']) {
    const body = fs.readFileSync(path.join(REPO_ROOT, rel), 'utf8');
    const hits = body.match(/color: ['"][0-9a-f]{6}['"]/g) || [];
    assert.deepEqual(hits, [], `${rel} hardcodes ${hits.join(', ')}`);
    // A second copy of the prefix rule drifts from this one; a hex-only check
    // would not notice, because a copy resolves from the config too.
    assert.ok(
      !/function colorFor\b/.test(body),
      `${rel} defines its own colorFor instead of using the exported resolver`,
    );
  }
});

test('topic labels come from the modules a PR touched', () => {
  const h = helpers();
  const f = filename => ({ filename, additions: 1, deletions: 0 });
  const cases = [
    ['libs/deepagents/deepagents/middleware/subagents.py', ['topic:middleware', 'topic:subagents']],
    ['libs/deepagents/deepagents/middleware/async_subagents.py', ['topic:async-subagents', 'topic:middleware']],
    ['libs/deepagents/deepagents/backends/sandbox.py', ['topic:backends', 'topic:sandboxes']],
    ['libs/code/deepagents_code/mcp_tools.py', ['topic:mcp']],
    ['libs/code/deepagents_code/skills/index.py', ['topic:skills']],
    ['libs/deepagents/deepagents/profiles/harness/base.py', []],
    ['libs/code/deepagents_code/_tracing.py', ['topic:tracing']],
    ['README.md', []],
    ['libs/deepagents/pyproject.toml', []],
  ];
  for (const [file, expected] of cases) {
    assert.deepEqual([...h.matchTopicFileLabels([f(file)])].sort(), expected, file);
  }
});

test('every topic rule points at a real topic label and compiles', () => {
  const { config, h } = prLabeler.loadAndInit({}, 'o', 'r', core);
  const declared = new Set(Object.keys(config.labelColors));
  for (const rule of config.topicFileRules) {
    assert.match(rule.label, /^topic:/, `${rule.label} is not a topic label`);
    assert.ok(declared.has('topic:'), 'topic: must have a prefix color');
    assert.notEqual(h.colorFor(rule.label), config.labelColor,
      `${rule.label} would be created off-palette`);
  }
  h.buildRules(config.topicFileRules, 'topicFileRules');
});
