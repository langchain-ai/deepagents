const MS_PER_DAY = 24 * 60 * 60 * 1000;

const DEFAULT_BYPASS_LABEL = 'do-not-close';
const DEFAULT_WARNING_DAYS = 14;
const DEFAULT_CLOSE_DAYS = 30;
const DEFAULT_MAX_ITEMS = 1000;
const COMMENT_MARKER = '<!-- old-pr-auto-close -->';

function parsePositiveInt(value, fallback, name) {
  if (value === undefined || value === null || value === '') return fallback;
  // Number.parseInt would silently accept trailing garbage ("100O" -> 100,
  // "14.9" -> 14), so require the whole string to be digits before trusting it.
  if (!/^\d+$/.test(String(value).trim())) {
    throw new Error(`${name} must be a positive integer, got "${value}"`);
  }
  const parsed = Number.parseInt(value, 10);
  if (parsed <= 0) {
    throw new Error(`${name} must be a positive integer, got "${value}"`);
  }
  return parsed;
}

function ageInDays(createdAt, now) {
  const created = new Date(createdAt).getTime();
  if (!Number.isFinite(created)) {
    // A NaN age would slip past every `age >= threshold` guard and silently
    // skip the PR forever, so surface it as an error instead.
    throw new Error(`Unparseable created date: ${JSON.stringify(createdAt)}`);
  }
  return Math.floor((now.getTime() - created) / MS_PER_DAY);
}

function labelNames(labels) {
  return labels.map(label => typeof label === 'string' ? label : label.name);
}

async function ensureLabel({ github, owner, repo, name }) {
  try {
    await github.rest.issues.getLabel({ owner, repo, name });
  } catch (error) {
    if (error.status !== 404) throw error;
    try {
      await github.rest.issues.createLabel({
        owner,
        repo,
        name,
        color: '0e8a16',
        description: 'Bypass automatic closure of old PRs',
      });
    } catch (createError) {
      if (createError.status !== 422) throw createError;
      // 422 is GitHub's generic validation error. It usually means a
      // concurrent run already created the label, but it also fires for an
      // invalid label name. Re-fetch to distinguish the two: a 404 here means
      // the label is genuinely absent, so surface the original 422 (which
      // carries the real reason) rather than the misleading "not found".
      try {
        await github.rest.issues.getLabel({ owner, repo, name });
      } catch (verifyError) {
        if (verifyError.status === 404) throw createError;
        throw verifyError;
      }
    }
  }
}

async function findMarkerComment({ github, owner, repo, issueNumber }) {
  const comments = await github.paginate(
    github.rest.issues.listComments,
    { owner, repo, issue_number: issueNumber, per_page: 100 },
  );
  return comments.find(comment => comment.body?.includes(COMMENT_MARKER));
}

async function upsertComment({ github, owner, repo, issueNumber, body }) {
  const existing = await findMarkerComment({ github, owner, repo, issueNumber });
  if (existing) {
    if (existing.body !== body) {
      await github.rest.issues.updateComment({
        owner,
        repo,
        comment_id: existing.id,
        body,
      });
    }
    return existing;
  }

  await github.rest.issues.createComment({
    owner,
    repo,
    issue_number: issueNumber,
    body,
  });
  return null;
}

function warningBody({ warningDays, closeDays, bypassLabel }) {
  return [
    COMMENT_MARKER,
    `This PR has been open for at least ${warningDays} days.`,
    '',
    `It will be closed automatically ${closeDays} days after it was opened (based on age, regardless of recent activity) unless the \`${bypassLabel}\` label is applied.`,
  ].join('\n');
}

function closeBody({ closeDays, bypassLabel }) {
  return [
    COMMENT_MARKER,
    `This PR has been open for at least ${closeDays} days and is being closed automatically.`,
    '',
    `If this work is still active, feel free to reopen it or open a fresh PR. Add the \`${bypassLabel}\` label to exempt a PR from this cleanup.`,
  ].join('\n');
}

async function getLivePr({ github, owner, repo, number }) {
  const { data: pr } = await github.rest.pulls.get({
    owner,
    repo,
    pull_number: number,
  });
  return {
    createdAt: pr.created_at,
    draft: pr.draft === true,
    labels: labelNames(pr.labels ?? []),
    state: pr.state,
  };
}

async function searchOpenPrs({ github, owner, repo, maxItems, core }) {
  const query = `repo:${owner}/${repo} is:pr is:open draft:false`;
  const items = [];
  try {
    for await (const response of github.paginate.iterator(
      github.rest.search.issuesAndPullRequests,
      { q: query, per_page: 100, sort: 'created', order: 'asc' },
    )) {
      for (const item of response.data) {
        items.push(item);
        if (items.length >= maxItems) return { items, incomplete: false };
      }
    }
  } catch (error) {
    core.warning(
      `Search failed after collecting ${items.length} PR(s) ` +
      `(HTTP ${error.status ?? 'unknown'}): ${error.message}`,
    );
    // Process whatever was collected, but report incompleteness so the caller
    // fails the run — a swallowed search error must not look like a clean pass.
    return { items, incomplete: true };
  }
  return { items, incomplete: false };
}

async function processPr({
  github,
  core,
  owner,
  repo,
  item,
  now,
  bypassLabel,
  warningDays,
  closeDays,
}) {
  const number = item.number;

  // The created date is immutable, so gate on the (cheap) search result first
  // and avoid the per-PR API calls below for PRs too young to act on.
  const age = ageInDays(item.created_at, now);
  if (age < warningDays) {
    core.info(`PR #${number} is ${age} day(s) old; no action`);
    return 'skipped';
  }

  // Re-fetch before mutating: state, draft, and labels can all change between
  // the search and now (the PR may have been closed, or gained the bypass
  // label / a maintainer may have marked it draft).
  let live;
  try {
    live = await getLivePr({ github, owner, repo, number });
  } catch (error) {
    if (error.status === 404) {
      core.info(`PR #${number} not found (deleted or transferred); skipping`);
      return 'skipped';
    }
    throw error;
  }

  if (live.state !== 'open') {
    core.info(`PR #${number} is no longer open; skipping`);
    return 'skipped';
  }
  if (live.draft) {
    core.info(`PR #${number} is a draft; skipping`);
    return 'skipped';
  }
  if (live.labels.includes(bypassLabel)) {
    core.info(`PR #${number} has ${bypassLabel}; skipping`);
    return 'skipped';
  }

  if (age >= closeDays) {
    await upsertComment({
      github,
      owner,
      repo,
      issueNumber: number,
      body: closeBody({ closeDays, bypassLabel }),
    });
    await github.rest.pulls.update({
      owner,
      repo,
      pull_number: number,
      state: 'closed',
    });
    core.info(`Closed PR #${number} after ${age} day(s)`);
    return 'closed';
  }

  const existing = await findMarkerComment({ github, owner, repo, issueNumber: number });
  if (!existing) {
    await github.rest.issues.createComment({
      owner,
      repo,
      issue_number: number,
      body: warningBody({ warningDays, closeDays, bypassLabel }),
    });
    core.info(`Warned PR #${number} after ${age} day(s)`);
    return 'warned';
  }

  core.info(`PR #${number} is ${age} day(s) old; already warned, no action`);
  return 'skipped';
}

async function run({ github, context, core, options = {} }) {
  const { owner, repo } = context.repo;
  const bypassLabel = options.bypassLabel ?? process.env.BYPASS_LABEL ?? DEFAULT_BYPASS_LABEL;
  const warningDays = parsePositiveInt(
    options.warningDays ?? process.env.WARNING_DAYS,
    DEFAULT_WARNING_DAYS,
    'warningDays',
  );
  const closeDays = parsePositiveInt(
    options.closeDays ?? process.env.CLOSE_DAYS,
    DEFAULT_CLOSE_DAYS,
    'closeDays',
  );
  const maxItems = parsePositiveInt(
    options.maxItems ?? process.env.MAX_ITEMS,
    DEFAULT_MAX_ITEMS,
    'maxItems',
  );
  const now = options.now ?? new Date();

  if (warningDays >= closeDays) {
    throw new Error(`warningDays (${warningDays}) must be less than closeDays (${closeDays})`);
  }

  await ensureLabel({ github, owner, repo, name: bypassLabel });
  const { items: prs, incomplete } = await searchOpenPrs({ github, owner, repo, maxItems, core });
  core.info(`Found ${prs.length} open PR(s)`);

  const summary = { checked: 0, warned: 0, closed: 0, skipped: 0, incomplete, errors: [] };
  for (const item of prs) {
    summary.checked += 1;
    try {
      const result = await processPr({
        github,
        core,
        owner,
        repo,
        item,
        now,
        bypassLabel,
        warningDays,
        closeDays,
      });
      summary[result] += 1;
    } catch (error) {
      const status = error.status ?? 'unknown';
      core.warning(`PR #${item.number} failed (HTTP ${status}): ${error.stack ?? error.message}`);
      summary.errors.push({ number: item.number, status, message: error.message });
    }
  }

  core.info(
    `Checked ${summary.checked}; warned ${summary.warned}; ` +
    `closed ${summary.closed}; skipped ${summary.skipped}; errors ${summary.errors.length}`,
  );

  const problems = summary.errors.map(error => `#${error.number}: ${error.message}`);
  if (incomplete) {
    problems.unshift('PR search did not complete; processed a partial list');
  }
  if (problems.length > 0) {
    core.setFailed(problems.join('; '));
  }
  return summary;
}

module.exports = { run, warningBody, closeBody, ageInDays, COMMENT_MARKER };
