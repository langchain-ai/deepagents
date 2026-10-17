// Shared helpers for pr_labeler.yml and tag-external-issues.yml.
//
// Usage from actions/github-script (requires actions/checkout first):
//   const { h } = require('./.github/scripts/labeling/pr-labeler.js').loadAndInit(github, owner, repo, core);

const fs = require('fs');
const path = require('path');

function loadConfig() {
  const configPath = path.join(__dirname, 'pr-labeler-config.json');
  let raw;
  try {
    raw = fs.readFileSync(configPath, 'utf8');
  } catch (e) {
    throw new Error(`Failed to read ${configPath}: ${e.message}`);
  }
  let config;
  try {
    config = JSON.parse(raw);
  } catch (e) {
    throw new Error(`Failed to parse pr-labeler-config.json: ${e.message}`);
  }
  const required = [
    'labelColor', 'labelColors', 'sizeThresholds', 'fileRules', 'branchRules',
    'scopeToLabel', 'scopeAliases', 'releaseLabel', 'trustedThreshold',
    'topicFileRules',
    'typeToLabel', 'breakingLabel', 'labelDescriptions', 'tierLabels',
    'excludedFiles', 'excludedPaths',
  ];
  const missing = required.filter(k => !(k in config));
  if (missing.length > 0) {
    throw new Error(`pr-labeler-config.json missing required keys: ${missing.join(', ')}`);
  }
  return config;
}

function init(github, owner, repo, config, core) {
  if (!core) {
    throw new Error('init() requires a `core` parameter (e.g., from actions/github-script)');
  }
  const {
    trustedThreshold,
    labelColor,
    labelColors,
    sizeThresholds,
    scopeToLabel,
    scopeAliases,
    releaseLabel,
    typeToLabel,
    breakingLabel,
    tierLabels,
    labelDescriptions,
    fileRules: fileRulesDef,
    topicFileRules: topicFileRulesDef,
    branchRules: branchRulesDef,
    excludedFiles,
    excludedPaths,
  } = config;

  const sizeLabels = sizeThresholds.map(t => t.label);
  // Config-driven like every other label. These were the last hardcoded
  // literals, and require_issue_link.yml gates its whole check on
  // `trusted`, so a rename that reached only one side would start closing
  // trusted contributors' PRs.
  const tierLabelNames = [tierLabels.new, tierLabels.trusted];
  const titleTypeLabels = new Set([...Object.values(typeToLabel), breakingLabel, releaseLabel]);

  // ── Label management ──────────────────────────────────────────────

  // A label's color follows its taxonomy prefix, so a label created on demand
  // matches the ones already on the repo. Without this every auto-created
  // label landed on the generic `labelColor`, which is how nine `type:*`
  // labels ended up off-palette during the taxonomy migration.
  function colorFor(name) {
    let best = null;
    for (const prefix of Object.keys(labelColors)) {
      if ((name ?? '').startsWith(prefix) && (!best || prefix.length > best.length)) {
        best = prefix;
      }
    }
    return best ? labelColors[best] : labelColor;
  }

  async function ensureLabel(name, color = colorFor(name)) {
    try {
      await github.rest.issues.getLabel({ owner, repo, name });
    } catch (e) {
      if (e.status !== 404) throw e;
      try {
        await github.rest.issues.createLabel({
          owner, repo, name, color, description: labelDescriptions[name] ?? '',
        });
      } catch (createErr) {
        // 422 = label created by a concurrent run between our get and create
        if (createErr.status !== 422) throw createErr;
        core.info(`Label "${name}" creation returned 422 (likely already exists)`);
      }
    }
  }

  // ── Size calculation ──────────────────────────────────────────────

  function getSizeLabel(totalChanged) {
    for (const t of sizeThresholds) {
      if (t.max != null && totalChanged < t.max) return t.label;
    }
    // Last entry has no max — it's the catch-all (XL)
    return sizeThresholds[sizeThresholds.length - 1].label;
  }

  function computeSize(files) {
    const excluded = new Set(excludedFiles);
    const totalChanged = files.reduce((sum, f) => {
      const p = f.filename ?? '';
      const base = p.split('/').pop();
      if (excluded.has(base)) return sum;
      for (const prefix of excludedPaths) {
        if (p.startsWith(prefix)) return sum;
      }
      return sum + (f.additions ?? 0) + (f.deletions ?? 0);
    }, 0);
    return { totalChanged, sizeLabel: getSizeLabel(totalChanged) };
  }

  // ── File-based labels ─────────────────────────────────────────────

  function buildRules(defs, source = 'fileRules') {
    return defs.map((rule, i) => {
      let test;
      if (rule.prefix) test = p => p.startsWith(rule.prefix);
      else if (rule.suffix) test = p => p.endsWith(rule.suffix);
      else if (rule.exact) test = p => p === rule.exact;
      else if (rule.pattern) {
        const re = new RegExp(rule.pattern);
        test = p => re.test(p);
      } else {
        throw new Error(
          `${source}[${i}] (label: "${rule.label}") has no recognized matcher ` +
          `(expected one of: prefix, suffix, exact, pattern)`
        );
      }
      return { label: rule.label, test, skipExcluded: !!rule.skipExcludedFiles };
    });
  }

  function buildFileRules() {
    return buildRules(fileRulesDef, 'fileRules');
  }

  function matchFileLabels(files, fileRules) {
    const rules = fileRules || buildFileRules();
    const excluded = new Set(excludedFiles);
    const labels = new Set();
    for (const rule of rules) {
      // skipExcluded: ignore files whose basename is in the top-level
      // "excludedFiles" list (e.g. uv.lock) so lockfile-only changes
      // don't trigger package labels.
      const candidates = rule.skipExcluded
        ? files.filter(f => !excluded.has((f.filename ?? '').split('/').pop()))
        : files;
      if (candidates.some(f => rule.test(f.filename ?? ''))) {
        labels.add(rule.label);
      }
    }
    return labels;
  }

  // ── Topic labels ──────────────────────────────────────────────────
  // Match subjects from the modules a PR touched, rather than whole packages.
  // The live workflows also use topic-classifier.js to classify issue text or
  // a PR title with a model. Both signals are additive.
  function matchTopicFileLabels(files) {
    return matchFileLabels(files, buildRules(topicFileRulesDef, 'topicFileRules'));
  }

  // ── Branch-name-based labels ──────────────────────────────────────

  function matchBranchLabels(headRef) {
    const labels = new Set();
    const ref = headRef ?? '';
    if (!ref) return labels;
    for (const rule of branchRulesDef) {
      let matched = false;
      if (rule.prefix) matched = ref.startsWith(rule.prefix);
      else if (rule.suffix) matched = ref.endsWith(rule.suffix);
      else if (rule.exact) matched = ref === rule.exact;
      else if (rule.pattern) matched = new RegExp(rule.pattern).test(ref);
      else {
        throw new Error(
          `branchRules entry (label: "${rule.label}") has no recognized matcher ` +
          `(expected one of: prefix, suffix, exact, pattern)`,
        );
      }
      if (matched) labels.add(rule.label);
    }
    return labels;
  }

  // ── Title-based labels ────────────────────────────────────────────

  // Type labels mirror the title for triage; release-please still reads the
  // Conventional Commit itself. Scope labels identify packages/integrations.
  function matchTitleLabels(title) {
    const labels = new Set();
    const m = (title ?? '').match(/^(\w+)(?:\(([^)]+)\))?(!)?:/);
    if (!m) return { labels, type: null, typeLabel: null, scopes: [], breaking: false };

    const type = m[1].toLowerCase();
    const scopeStr = m[2] ?? '';
    const breaking = !!m[3];
    const typeLabel = type === 'release' ? releaseLabel :
      Object.hasOwn(typeToLabel, type) ? typeToLabel[type] : null;
    if (typeLabel) labels.add(typeLabel);
    if (breaking && typeLabel) labels.add(breakingLabel);

    const scopes = scopeStr.split(',').map(s => s.trim()).filter(Boolean);
    for (const scope of scopes) {
      const sl = scopeToLabel[scope];
      if (sl) labels.add(sl);
    }

    return { labels, type, typeLabel, scopes, breaking };
  }

  function getStaleTitleLabels(title, currentLabels) {
    const { labels, typeLabel } = matchTitleLabels(title);
    // A malformed/unrecognized title supplies no replacement classification.
    if (!typeLabel) return [];
    return currentLabels.filter(name => titleTypeLabels.has(name) && !labels.has(name));
  }

  // ── Title scope canonicalization ──────────────────────────────────

  // A scoped Conventional Commits title: `type(scope): subject`, with the `!`
  // breaking marker allowed on either side of the parens.
  const scopedTitlePattern = /^(\w+!?)\(([^)]+)\)(!?:\s*.*)$/;

  // Rewrite package-component scopes (e.g. `deepagents-code`) to their
  // canonical PR scope (`code`) per `scopeAliases`. Returns null when there is
  // nothing to do: an unscoped title, a `release(...)` title (whose scope is a
  // canonical version record and must not be touched), or scopes that are
  // already canonical.
  function canonicalizeTitleScopes(title) {
    const match = (title ?? '').match(scopedTitlePattern);
    if (!match) return null;

    const type = match[1].replace('!', '').toLowerCase();
    if (type === 'release') return null;

    const scopeStr = match[2];
    const newScopeStr = scopeStr
      .split(',')
      .map(s => scopeAliases[s.trim()] ?? s.trim())
      .join(',');
    if (newScopeStr === scopeStr) return null;

    return { title: `${match[1]}(${newScopeStr})${match[3]}`, scopes: newScopeStr };
  }

  // ── Org membership ────────────────────────────────────────────────

  async function checkMembership(author, userType) {
    if (userType === 'Bot') {
      console.log(`${author} is a Bot — treating as internal`);
      return { isExternal: false };
    }

    try {
      const membership = await github.rest.orgs.getMembershipForUser({
        org: 'langchain-ai',
        username: author,
      });
      const isExternal = membership.data.state !== 'active';
      console.log(
        isExternal
          ? `${author} has pending membership — treating as external`
          : `${author} is an active member of langchain-ai`,
      );
      return { isExternal };
    } catch (e) {
      if (e.status === 404) {
        console.log(`${author} is not a member of langchain-ai`);
        return { isExternal: true };
      }
      // Non-404 errors (rate limit, auth failure, server error) must not
      // silently default to external — rethrow to fail the step.
      throw new Error(
        `Membership check failed for ${author} (${e.status}): ${e.message}`,
      );
    }
  }

  // ── Contributor analysis ──────────────────────────────────────────

  async function getContributorInfo(contributorCache, author, userType) {
    if (contributorCache.has(author)) return contributorCache.get(author);

    const { isExternal } = await checkMembership(author, userType);

    let mergedCount = null;
    if (isExternal) {
      try {
        const result = await github.rest.search.issuesAndPullRequests({
          q: `repo:${owner}/${repo} is:pr is:merged author:"${author}"`,
          per_page: 1,
        });
        mergedCount = result?.data?.total_count ?? null;
      } catch (e) {
        if (e?.status !== 422) throw e;
        core.warning(`Search failed for ${author}; skipping tier.`);
      }
    }

    // `tierKnown` is explicit because a null `mergedCount` means "the search
    // failed", not "zero merged PRs". Callers that reconcile labels must not
    // read the absence of a tier as an instruction to remove one — doing so
    // strips `auto:trusted-contributor`, and `require_issue_link.yml` gates
    // its whole enforcement path (label, comment, close) on that label.
    const info = { isExternal, mergedCount, tierKnown: !isExternal || mergedCount != null };
    contributorCache.set(author, info);
    return info;
  }

  // ── Tier label resolution ───────────────────────────────────────────

  async function applyTierLabel(issueNumber, author, { skipNewContributor = false } = {}) {
    let mergedCount;
    try {
      const result = await github.rest.search.issuesAndPullRequests({
        q: `repo:${owner}/${repo} is:pr is:merged author:"${author}"`,
        per_page: 1,
      });
      mergedCount = result?.data?.total_count;
    } catch (error) {
      if (error?.status !== 422) throw error;
      core.warning(`Search failed for ${author}; skipping tier label.`);
      return;
    }

    if (mergedCount == null) {
      core.warning(`Search response missing total_count for ${author}; skipping tier label.`);
      return;
    }

    let tierLabel = null;
    if (mergedCount >= trustedThreshold) tierLabel = tierLabels.trusted;
    else if (mergedCount === 0 && !skipNewContributor) tierLabel = tierLabels.new;

    if (tierLabel) {
      await ensureLabel(tierLabel);
      await github.rest.issues.addLabels({
        owner, repo, issue_number: issueNumber, labels: [tierLabel],
      });
      console.log(`Applied '${tierLabel}' to #${issueNumber} (${mergedCount} merged PRs)`);
    } else {
      console.log(`No tier label for ${author} (${mergedCount} merged PRs)`);
    }

    return tierLabel;
  }

  // ── Full PR labeling (title + file + size) ───────────────────────

  async function labelPR(prNumber, { title } = {}) {
    if (!prNumber) {
      throw new Error('labelPR() requires a valid prNumber');
    }
    const toAdd = new Set();

    const prTitle = title ?? (await github.rest.pulls.get({
      owner, repo, pull_number: prNumber,
    })).data.title;

    // Title-based labels
    const { labels: titleLabels } = matchTitleLabels(prTitle);
    for (const l of titleLabels) toAdd.add(l);

    // File-based labels + size
    const files = await github.paginate(github.rest.pulls.listFiles, {
      owner, repo, pull_number: prNumber, per_page: 100,
    });
    toAdd.add(computeSize(files).sizeLabel);
    for (const l of matchFileLabels(files)) toAdd.add(l);

    for (const name of toAdd) await ensureLabel(name);
    const currentLabels = (await github.paginate(github.rest.issues.listLabelsOnIssue, {
      owner, repo, issue_number: prNumber, per_page: 100,
    })).map(label => label.name);
    for (const name of getStaleTitleLabels(prTitle, currentLabels)) {
      try {
        await github.rest.issues.removeLabel({ owner, repo, issue_number: prNumber, name });
      } catch (error) {
        if (error.status !== 404) throw error;
      }
    }
    const labels = [...toAdd];
    if (labels.length) {
      await github.rest.issues.addLabels({
        owner, repo, issue_number: prNumber, labels,
      });
    }
    return labels;
  }

  return {
    ensureLabel,
    colorFor,
    getSizeLabel,
    computeSize,
    buildFileRules,
    buildRules,
    matchFileLabels,
    matchTopicFileLabels,
    matchBranchLabels,
    matchTitleLabels,
    getStaleTitleLabels,
    canonicalizeTitleScopes,
    labelPR,
    checkMembership,
    getContributorInfo,
    applyTierLabel,
    sizeLabels,
    // Array for the "managed labels" sweeps; map for callers that need to
    // pick a specific tier.
    tierLabels: tierLabelNames,
    tierLabelsByTier: tierLabels,
    releaseLabel,
    trustedThreshold,
    labelColor,
    labelColors,
  };
}

function loadAndInit(github, owner, repo, core) {
  const config = loadConfig();
  return { config, h: init(github, owner, repo, config, core) };
}

module.exports = { loadConfig, init, loadAndInit };
