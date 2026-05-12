const MS_PER_DAY = 24 * 60 * 60 * 1000;

const DEFAULT_AUTHOR = 'app/open-swe';
const DEFAULT_BYPASS_LABEL = 'do-not-close';
const DEFAULT_REMINDER_DAYS = 7;
const DEFAULT_CLOSE_DAYS = 14;
const DEFAULT_MAX_ITEMS = 1000;
const COMMENT_MARKER = '<!-- open-swe-auto-close -->';

function parsePositiveInt(value, fallback, name) {
  if (value === undefined || value === null || value === '') return fallback;
  const parsed = Number.parseInt(value, 10);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    throw new Error(`${name} must be a positive integer, got "${value}"`);
  }
  return parsed;
}

function ageInDays(createdAt, now) {
  return Math.floor((now.getTime() - new Date(createdAt).getTime()) / MS_PER_DAY);
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
        description: 'Bypass automatic open-swe PR closure',
      });
    } catch (createError) {
      if (createError.status !== 422) throw createError;
      await github.rest.issues.getLabel({ owner, repo, name });
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
    await github.rest.issues.updateComment({
      owner,
      repo,
      comment_id: existing.id,
      body,
    });
    return 'updated';
  }

  await github.rest.issues.createComment({
    owner,
    repo,
    issue_number: issueNumber,
    body,
  });
  return 'created';
}

async function createCommentIfMissing({ github, owner, repo, issueNumber, body }) {
  const existing = await findMarkerComment({ github, owner, repo, issueNumber });
  if (existing) return 'existing';

  await github.rest.issues.createComment({
    owner,
    repo,
    issue_number: issueNumber,
    body,
  });
  return 'created';
}

function reminderBody({ reminderDays, closeDays, bypassLabel }) {
  return [
    COMMENT_MARKER,
    `This PR was opened by \`app/open-swe\` and has had no activity for at least ${reminderDays} day(s).`,
    '',
    `It will be automatically closed after ${closeDays} days unless a maintainer adds the \`${bypassLabel}\` label.`,
  ].join('\n');
}

function closeBody({ closeDays, bypassLabel }) {
  return [
    COMMENT_MARKER,
    `This PR was opened by \`app/open-swe\` and has had no maintainer activity since the cleanup reminder.`,
    '',
    `Closing automatically. Add the \`${bypassLabel}\` label before the ${closeDays}-day mark to bypass this cleanup.`,
  ].join('\n');
}

async function getLivePrState({ github, owner, repo, number }) {
  const { data: pr } = await github.rest.pulls.get({
    owner,
    repo,
    pull_number: number,
  });
  return {
    draft: pr.draft === true,
    state: pr.state,
    updatedAt: pr.updated_at,
    labels: labelNames(pr.labels ?? []),
  };
}

async function searchOpenSwePrs({ github, owner, repo, author, maxItems, core }) {
  const query = [
    `repo:${owner}/${repo}`,
    'is:pr',
    'is:open',
    'draft:false',
    `author:${author}`,
  ].join(' ');

  const items = [];
  try {
    for await (const response of github.paginate.iterator(
      github.rest.search.issuesAndPullRequests,
      { q: query, per_page: 100 },
    )) {
      const pageItems = response.data.items ?? [];
      for (const item of pageItems) {
        items.push(item);
        if (items.length >= maxItems) return items;
      }
    }
  } catch (error) {
    core.warning(
      `Search failed after collecting ${items.length} PR(s) ` +
      `(HTTP ${error.status ?? 'unknown'}): ${error.message}`,
    );
  }
  return items;
}

function hasActivityAfterReminder(live, reminder) {
  const liveUpdatedAt = new Date(live.updatedAt).getTime();
  const reminderUpdatedAt = new Date(reminder.updated_at ?? reminder.created_at).getTime();
  return liveUpdatedAt > reminderUpdatedAt;
}

async function processPr({
  github,
  core,
  owner,
  repo,
  item,
  now,
  bypassLabel,
  reminderDays,
  closeDays,
}) {
  const number = item.number;
  const live = await getLivePrState({ github, owner, repo, number });

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

  const reminder = await findMarkerComment({ github, owner, repo, issueNumber: number });
  const activityAt = live.updatedAt ?? item.updated_at ?? item.created_at;
  const inactiveDays = ageInDays(activityAt, now);

  if (reminder) {
    if (hasActivityAfterReminder(live, reminder)) {
      if (inactiveDays >= reminderDays) {
        await github.rest.issues.updateComment({
          owner,
          repo,
          comment_id: reminder.id,
          body: reminderBody({ reminderDays, closeDays, bypassLabel }),
        });
        core.info(`Refreshed reminder on PR #${number} after ${inactiveDays} inactive day(s)`);
        return 'reminded';
      }
      core.info(`PR #${number} had activity after the reminder; skipping`);
      return 'skipped';
    }

    const reminderAge = ageInDays(reminder.updated_at ?? reminder.created_at, now);
    const warningDays = closeDays - reminderDays;
    if (reminderAge >= warningDays) {
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
      core.info(`Closed PR #${number} after ${reminderAge} day(s) with cleanup reminder`);
      return 'closed';
    }

    core.info(`PR #${number} already has a reminder; no action`);
    return 'skipped';
  }

  if (inactiveDays >= reminderDays) {
    await createCommentIfMissing({
      github,
      owner,
      repo,
      issueNumber: number,
      body: reminderBody({ reminderDays, closeDays, bypassLabel }),
    });
    core.info(`Posted reminder on PR #${number}`);
    return 'reminded';
  }

  core.info(`PR #${number} has been inactive for ${inactiveDays} day(s); no action`);
  return 'skipped';
}

async function run({ github, context, core, options = {} }) {
  const { owner, repo } = context.repo;
  const author = options.author ?? process.env.OPEN_SWE_AUTHOR ?? DEFAULT_AUTHOR;
  const bypassLabel = options.bypassLabel ?? process.env.BYPASS_LABEL ?? DEFAULT_BYPASS_LABEL;
  const reminderDays = parsePositiveInt(
    options.reminderDays ?? process.env.REMINDER_DAYS,
    DEFAULT_REMINDER_DAYS,
    'reminderDays',
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

  if (reminderDays >= closeDays) {
    throw new Error(`reminderDays (${reminderDays}) must be less than closeDays (${closeDays})`);
  }

  await ensureLabel({ github, owner, repo, name: bypassLabel });

  const prs = await searchOpenSwePrs({ github, owner, repo, author, maxItems, core });
  core.info(`Found ${prs.length} open PR(s) from ${author}`);

  const summary = { checked: 0, reminded: 0, closed: 0, skipped: 0, errors: [] };
  for (const item of prs) {
    summary.checked += 1;
    const number = item.number;
    try {
      const result = await processPr({
        github,
        core,
        owner,
        repo,
        item,
        now,
        bypassLabel,
        reminderDays,
        closeDays,
      });
      summary[result] += 1;
    } catch (error) {
      const message = `PR #${number} failed: ${error.message}`;
      core.warning(message);
      summary.errors.push({ number, message: error.message });
    }
  }

  core.info(
    `Checked ${summary.checked}; reminded ${summary.reminded}; ` +
    `closed ${summary.closed}; skipped ${summary.skipped}; errors ${summary.errors.length}`,
  );
  if (summary.errors.length > 0) {
    core.setFailed(
      `Failed to process ${summary.errors.length} open-swe PR(s): ` +
      summary.errors.map(error => `#${error.number}`).join(', '),
    );
  }
  return summary;
}

module.exports = {
  COMMENT_MARKER,
  ageInDays,
  closeBody,
  reminderBody,
  run,
};
