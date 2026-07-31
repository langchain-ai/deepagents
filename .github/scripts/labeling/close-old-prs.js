const MS_PER_DAY = 24 * 60 * 60 * 1000;

const DEFAULT_BYPASS_LABEL = 'do-not-close';
const DEFAULT_PENDING_DELETION_LABEL = 'pending-deletion';
// Why release PRs are exempt at all: release-please keeps one long-lived PR
// per package and updates it in place rather than opening a new one (see
// .github/RELEASING.md), so "days since opened" — the only staleness signal
// this workflow has — is meaningless for them by construction.
//
// When the exemption actually fires: release PRs open as drafts
// (`draft-pull-request: true` in release-please-config.json), and the
// `draft:false` search plus the draft skip in processPr already exclude them
// for most of their life. This exemption covers the ready-for-review window
// between the curated-notes step and merge — narrow, but that is exactly
// where #4297 was warned at 33 days old. Expect skippedRelease to read 0 on
// most runs; a sustained 0 is not evidence the exemption is dead code.
//
// These labels do NOT gate the exemption — provenance does (see isReleasePr).
// They are only a drift signal, matched to tell "a genuine release PR whose
// provenance changed" apart from "a contributor PR that titled itself
// `release(x):`". Both are individually unreliable, which is why neither is
// load-bearing:
//   * `release` is this repo's own, derived from the `release(scope):` title
//     type via .github/scripts/labeling/pr-labeler-config.json (typeToLabel).
//     On release-please's own PRs it is applied by a `continue-on-error` step
//     in release-please.yml, so it can silently fail to appear.
//   * `autorelease: pending` is release-please-action's built-in label,
//     applied by release-please itself as part of opening the PR (a failure
//     there fails the action, unlike the `continue-on-error` step above). See
//     release-please.yml:330-333 for the authoritative description — the
//     label table in RELEASING.md describes only its post-merge meaning.
// `autorelease: tagged` is deliberately absent: release.yml only flips
// pending -> tagged in the post-merge job, so an open PR never carries it.
const RELEASE_LABELS = new Set(['release', 'autorelease: pending']);
const DEFAULT_WARNING_DAYS = 14;
const DEFAULT_CLOSE_DAYS = 30;
const DEFAULT_MAX_ITEMS = 1000;
const COMMENT_MARKER = '<!-- old-pr-auto-close -->';
const WORKFLOW_BOT_LOGIN = 'github-actions[bot]';
// Same value as WORKFLOW_BOT_LOGIN today, but a distinct identity: this is
// "who opens release PRs" (release-please runs on GITHUB_TOKEN), whereas
// WORKFLOW_BOT_LOGIN is "who authored this workflow's own comments". Moving
// release-please to a GitHub App token would change only this one.
const RELEASE_PLEASE_AUTHOR = 'github-actions[bot]';
// Mirrors the branch shape `separate-pull-requests: true` produces in
// release-please-config.json: `release-please--branches--<base>--components--<pkg>`.
// Nothing derives this — the same literal is independently hardcoded in
// .github/scripts/release/release-notes.js and in check_sdk_pin.yml,
// check_partner_bounds.yml, release_please_fanout_watch.yml, and
// release-please.yml. Keep them in lockstep; two distinct changes break it:
//   * `separate-pull-requests: false` drops the `--components--<pkg>` suffix
//     entirely (the branch becomes `release-please--branches--main`).
//   * renaming the default branch changes the `main` segment, keeping the suffix.
// Either disarms the exemption. isReleasePr warns when a release label appears
// without matching provenance, but that warning is only a log annotation: the
// run still passes and the PR is still closed at closeDays. The test in
// close-old-prs.test.js that pins this constant against
// release-please-config.json is the actual guard.
const RELEASE_PLEASE_BRANCH_PREFIX = 'release-please--branches--main--components--';

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
    // A non-finite age fails every numeric comparison (`age < warningDays` and
    // `age >= closeDays` are both false for NaN), so the PR would evade the
    // young-skip, get warned once, then linger open forever without ever
    // closing. Surface it as an error instead.
    throw new Error(`Unparseable created date: ${JSON.stringify(createdAt)}`);
  }
  return Math.floor((now.getTime() - created) / MS_PER_DAY);
}

function isTransient(status) {
  // Rate-limit and 5xx responses are typically momentary, and the daily cron
  // retries the PR on its next run. Everything else (auth, validation, or a
  // status-less throw such as a code bug) is treated as fatal so the run fails
  // loudly instead of silently skipping work.
  return status === 429 || (typeof status === 'number' && status >= 500);
}

function labelNames(labels) {
  return labels.map(label => typeof label === 'string' ? label : label.name);
}

// Provenance fields are absent rather than merely different in real cases
// (`head.repo` is null when a fork PR's source repo was deleted; `user` is
// null for a deleted account). Interpolating those bare yields "repo
// undefined", which reads like a mismatch — a different diagnosis from "the
// API returned nothing".
function describe(value) {
  return value === undefined || value === null ? '<absent>' : value;
}

// `labels` is already normalized to strings by getLivePr.
//
// Provenance alone decides the exemption. Every conjunct is outside a
// contributor's reach — an outside PR cannot push a branch into this
// repository (pr_labeler.yml's pull_request_target grants no push) nor author
// as `github-actions[bot]`/`Bot` — so this is unspoofable, and adding a label
// requirement on top would buy no security while introducing a false
// negative: the labels can go missing (see RELEASE_LABELS), and a genuine
// release PR denied the exemption is warned and then closed. Repo precedent
// agrees that the branch name identifies a release PR on its own —
// check_sdk_pin.yml:30, check_partner_bounds.yml:25, and release-notes.js:10
// all gate on it with no label check.
//
// `warnOnAnomaly` is suppressed by the sweep, which calls this for
// classification only; processPr already reported anything worth saying about
// the same PR in the same run.
function isReleasePr(
  { labels, authorLogin, authorType, headRef, headRepo },
  { owner, repo, core, number, warnOnAnomaly = true },
) {
  const failures = [];
  if (authorLogin !== RELEASE_PLEASE_AUTHOR || authorType !== 'Bot') {
    failures.push(`author ${describe(authorLogin)}/${describe(authorType)}`);
  }
  if (typeof headRef !== 'string'
    || !headRef.startsWith(RELEASE_PLEASE_BRANCH_PREFIX)
    || headRef.length <= RELEASE_PLEASE_BRANCH_PREFIX.length) {
    failures.push(`branch ${describe(headRef)}`);
  }
  if (typeof headRepo !== 'string'
    || headRepo.toLowerCase() !== `${owner}/${repo}`.toLowerCase()) {
    failures.push(`repo ${describe(headRepo)}`);
  }

  const hasReleaseLabel = labels.some(label => RELEASE_LABELS.has(label));

  if (failures.length === 0) {
    // Exempt either way, but a release PR with no release label means the
    // labeling failed, and per RELEASING.md a stuck/missing `autorelease:
    // pending` blocks release-please from opening future release PRs. Cheap to
    // surface here since the provenance evidence is already computed.
    if (warnOnAnomaly && !hasReleaseLabel) {
      core.warning(
        `PR #${number} has release-please provenance but no release label ` +
        `(expected one of: ${[...RELEASE_LABELS].join(', ')}); exempting it ` +
        `from cleanup anyway — check that release labeling succeeded`,
      );
    }
    return true;
  }

  // Provenance failed but a release label is present. Ambiguous: either a
  // title-spoofed contributor PR (correctly denied — pr_lint.yml allows
  // `release` as a title type and pr_labeler.yml runs on pull_request_target,
  // so the label is contributor-reachable) or a genuine release PR whose
  // provenance drifted (a renamed default branch, a separate-pull-requests
  // flip, a token change). The second case silently reintroduces the bug this
  // exemption exists to fix, so say so rather than letting the PR fall through
  // to the normal warn/close path unremarked.
  if (warnOnAnomaly && hasReleaseLabel) {
    core.warning(
      `PR #${number} carries a release label but failed provenance ` +
      `(${failures.join('; ')}); treating it as a normal PR`,
    );
  }
  return false;
}

async function ensureLabel({ github, owner, repo, name, color, description }) {
  try {
    await github.rest.issues.getLabel({ owner, repo, name });
  } catch (error) {
    if (error.status !== 404) throw error;
    try {
      await github.rest.issues.createLabel({
        owner,
        repo,
        name,
        color,
        description,
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

async function ensureIssueLabel({ github, owner, repo, issueNumber, name, existingLabels }) {
  if (existingLabels.includes(name)) return;
  await github.rest.issues.addLabels({
    owner,
    repo,
    issue_number: issueNumber,
    labels: [name],
  });
  existingLabels.push(name);
}

// Returns true only when this call actually removed the label, so callers can
// count real removals rather than no-ops.
async function removeIssueLabel({ github, owner, repo, issueNumber, name, existingLabels }) {
  if (!existingLabels.includes(name)) return false;
  let removed = true;
  try {
    await github.rest.issues.removeLabel({
      owner,
      repo,
      issue_number: issueNumber,
      name,
    });
  } catch (error) {
    if (error.status !== 404) throw error;
    removed = false;
  }
  const index = existingLabels.indexOf(name);
  if (index !== -1) existingLabels.splice(index, 1);
  return removed;
}

async function findMarkerComment({ github, owner, repo, issueNumber }) {
  const comments = await github.paginate(
    github.rest.issues.listComments,
    { owner, repo, issue_number: issueNumber, per_page: 100 },
  );
  return comments.find(comment =>
    comment.user?.login === WORKFLOW_BOT_LOGIN &&
    comment.user?.type === 'Bot' &&
    comment.body?.includes(COMMENT_MARKER),
  );
}

function warningBody({ warningDays, closeDays, bypassLabel }) {
  const noticeDays = closeDays - warningDays;
  return [
    COMMENT_MARKER,
    `This PR has been open for at least ${warningDays} days.`,
    '',
    `It will be closed automatically once it has been open for at least ${closeDays} days and this warning is at least ${noticeDays} days old, unless the \`${bypassLabel}\` label is applied.`,
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
    authorLogin: pr.user?.login,
    authorType: pr.user?.type,
    createdAt: pr.created_at,
    draft: pr.draft === true,
    headRef: pr.head?.ref,
    headRepo: pr.head?.repo?.full_name,
    labels: labelNames(pr.labels ?? []),
    state: pr.state,
  };
}

async function searchOpenPrs({ github, owner, repo, maxItems, core }) {
  const query = `repo:${owner}/${repo} is:pr is:open draft:false`;
  const items = [];
  let incomplete = false;
  try {
    for await (const response of github.paginate.iterator(
      github.rest.search.issuesAndPullRequests,
      { q: query, per_page: 100, sort: 'created', order: 'asc' },
    )) {
      incomplete ||= response.data.incomplete_results === true;
      for (const item of response.data) {
        items.push(item);
        if (items.length >= maxItems) {
          // Hitting the cap looks identical to a complete sweep unless we say
          // so. It self-corrects across runs (oldest PRs are processed first),
          // so this only warns rather than failing, but a green run must not
          // hide that some open PRs went unprocessed.
          core.warning(
            `Reached maxItems cap (${maxItems}); some open PRs were not ` +
            `processed this run. Raise max_items if the backlog is larger.`,
          );
          return { items, incomplete, truncated: true };
        }
      }
    }
  } catch (error) {
    core.warning(
      `Search failed after collecting ${items.length} PR(s) ` +
      `(HTTP ${error.status ?? 'unknown'}): ${error.message}`,
    );
    // Process whatever was collected, but report incompleteness so the caller
    // fails the run — a swallowed search error must not look like a clean pass.
    return { items, incomplete: true, truncated: false };
  }
  return { items, incomplete, truncated: false };
}

async function processPr({
  github,
  core,
  owner,
  repo,
  item,
  now,
  bypassLabel,
  pendingDeletionLabel,
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

  // Drop pending-deletion once the PR is no longer a close candidate so label
  // filters do not keep dead/exempt entries.
  if (live.state !== 'open') {
    await removeIssueLabel({
      github,
      owner,
      repo,
      issueNumber: number,
      name: pendingDeletionLabel,
      existingLabels: live.labels,
    });
    core.info(`PR #${number} is no longer open; skipping`);
    return 'skipped';
  }
  if (live.draft) {
    await removeIssueLabel({
      github,
      owner,
      repo,
      issueNumber: number,
      name: pendingDeletionLabel,
      existingLabels: live.labels,
    });
    core.info(`PR #${number} is a draft; skipping`);
    return 'skipped';
  }
  if (isReleasePr(live, { owner, repo, core, number })) {
    await removeIssueLabel({
      github,
      owner,
      repo,
      issueNumber: number,
      name: pendingDeletionLabel,
      existingLabels: live.labels,
    });
    core.info(`PR #${number} is a release PR; skipping`);
    return 'skippedRelease';
  }
  if (live.labels.includes(bypassLabel)) {
    await removeIssueLabel({
      github,
      owner,
      repo,
      issueNumber: number,
      name: pendingDeletionLabel,
      existingLabels: live.labels,
    });
    core.info(`PR #${number} has ${bypassLabel}; skipping`);
    return 'skipped';
  }

  // Warn-first: a PR is only ever closed once it already carries a warning
  // comment posted by this workflow, so every PR gets at least one warning
  // cycle (closeDays - warningDays days) of notice. A PR that is already past
  // closeDays but was never warned (e.g. the backlog on the first run) is
  // warned now and becomes eligible to close on a later run. A forged marker
  // from a PR participant does not count — findMarkerComment requires the bot
  // author — so it can neither trigger nor block a close.
  const existing = await findMarkerComment({ github, owner, repo, issueNumber: number });

  if (!existing) {
    await github.rest.issues.createComment({
      owner,
      repo,
      issue_number: number,
      body: warningBody({ warningDays, closeDays, bypassLabel }),
    });
    // Apply at warning time so the PR is filterable until it stops being a
    // close candidate (closed, draft, release, or bypassed).
    await ensureIssueLabel({
      github,
      owner,
      repo,
      issueNumber: number,
      name: pendingDeletionLabel,
      existingLabels: live.labels,
    });
    core.info(`Warned PR #${number} after ${age} day(s)`);
    return 'warned';
  }

  const noticeDays = closeDays - warningDays;
  const warningAge = ageInDays(existing.created_at, now);
  if (age >= closeDays && warningAge >= noticeDays) {
    // Upgrade the existing warning to the close notice in place, skipping the
    // API call if it already says exactly that (e.g. a retried run).
    const body = closeBody({ closeDays, bypassLabel });
    if (existing.body !== body) {
      await github.rest.issues.updateComment({
        owner,
        repo,
        comment_id: existing.id,
        body,
      });
    }
    await github.rest.pulls.update({
      owner,
      repo,
      pull_number: number,
      state: 'closed',
    });
    await removeIssueLabel({
      github,
      owner,
      repo,
      issueNumber: number,
      name: pendingDeletionLabel,
      existingLabels: live.labels,
    });
    core.info(`Closed PR #${number} after ${age} day(s)`);
    return 'closed';
  }

  // Backfill the pending label for PRs warned before this label existed, or
  // when a prior run posted the comment but failed before labeling.
  await ensureIssueLabel({
    github,
    owner,
    repo,
    issueNumber: number,
    name: pendingDeletionLabel,
    existingLabels: live.labels,
  });

  core.info(
    `PR #${number} is ${age} day(s) old and was warned ${warningAge} day(s) ago; no action`,
  );
  return 'skipped';
}

// The primary open-PR search omits drafts (`draft:false`) and closed PRs, so a
// separate label query is needed to clear pending-deletion after those
// transitions (or after a manual close).
//
// The `stale` expression below mirrors processPr's *label-clearing*
// exemptions, including the release check, even though an open non-draft
// release PR is also handled there. The duplication earns its place because
// processPr never sees PRs past the maxItems cap or dropped by a partial
// search failure, and because letting the two exemption sets drift is how a PR
// ends up skipped by one path while keeping a pending-deletion label applied
// by the other.
//
// processPr's age skip is deliberately not mirrored: age only increases, and
// pending-deletion is applied at warning time, so a labeled PR can never
// become young again. Adding an age check here would strand labels.
async function sweepStalePendingDeletionLabels({
  github,
  core,
  owner,
  repo,
  pendingDeletionLabel,
  bypassLabel,
  maxItems,
}) {
  // `sort`/`order` match the primary search so the cap defers work rather than
  // starving it: without a deterministic order the same over-cap subset can be
  // returned every run, and a specific PR past the cap is never reached.
  const query = `repo:${owner}/${repo} is:pr label:"${pendingDeletionLabel}"`;
  let cleared = 0;
  let notFound = 0;
  let seen = 0;
  try {
    for await (const response of github.paginate.iterator(
      github.rest.search.issuesAndPullRequests,
      { q: query, per_page: 100, sort: 'created', order: 'asc' },
    )) {
      for (const item of response.data) {
        seen += 1;
        if (seen > maxItems) {
          core.warning(
            `Reached maxItems cap (${maxItems}) while sweeping ` +
            `${pendingDeletionLabel}; some labeled PRs were not checked.`,
          );
          // The cap is not a failure: the sweep is idempotent and the next
          // daily run picks up whatever was missed (oldest first, as above).
          // Same rationale as the primary search's truncation warning.
          return { cleared, notFound, truncated: true, failure: null };
        }

        let live;
        try {
          live = await getLivePr({ github, owner, repo, number: item.number });
        } catch (error) {
          if (error.status === 404) {
            // Deleted, transferred, or a stale search index. Individually
            // routine, but an unlogged `continue` here is the one remaining way
            // this sweep can drop all its work and still look like a sweep with
            // nothing to do, so count it into the summary.
            notFound += 1;
            core.info(`PR #${item.number} not found while sweeping; skipping`);
            continue;
          }
          throw error;
        }

        const stale = live.state !== 'open'
          || live.draft
          || isReleasePr(live, { owner, repo, core, number: item.number, warnOnAnomaly: false })
          || live.labels.includes(bypassLabel);
        if (!stale) continue;

        // The label search index lags the label mutations this same run makes
        // in processPr, so a PR can surface here with the label already gone.
        // Only count and log removals that actually happened.
        const removed = await removeIssueLabel({
          github,
          owner,
          repo,
          issueNumber: item.number,
          name: pendingDeletionLabel,
          existingLabels: live.labels,
        });
        if (!removed) continue;
        cleared += 1;
        core.info(
          `Cleared ${pendingDeletionLabel} from PR #${item.number} ` +
          `(no longer a close candidate)`,
        );
      }
    }
  } catch (error) {
    // Report the failure to the caller so the run fails. A sweep that dies on
    // its first PR otherwise looks identical to one with nothing to do — the
    // same reasoning as searchOpenPrs returning `incomplete`. core.error (not
    // warning) because this condition is now fatal, and the run summary
    // repeats it via setFailed.
    const failure = `pending-deletion sweep failed after clearing ${cleared} label(s) ` +
      `(HTTP ${error.status ?? 'unknown'}): ${error.message}`;
    core.error(failure);
    return { cleared, notFound, truncated: false, failure };
  }
  return { cleared, notFound, truncated: false, failure: null };
}

async function run({ github, context, core, options = {} }) {
  const { owner, repo } = context.repo;
  // `||` (not `??`) so an empty string falls back to the default: an
  // empty-named label can never be applied, which would silently disable the
  // bypass or pending-deletion mechanisms.
  const bypassLabel = options.bypassLabel || process.env.BYPASS_LABEL || DEFAULT_BYPASS_LABEL;
  const pendingDeletionLabel = options.pendingDeletionLabel
    || process.env.PENDING_DELETION_LABEL
    || DEFAULT_PENDING_DELETION_LABEL;
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

  await ensureLabel({
    github,
    owner,
    repo,
    name: bypassLabel,
    color: '0e8a16',
    description: 'Bypass automatic closure of old PRs',
  });
  await ensureLabel({
    github,
    owner,
    repo,
    name: pendingDeletionLabel,
    color: 'fbca04',
    description: 'PR is past the auto-close warning threshold and will be closed unless exempted',
  });
  const { items: prs, incomplete, truncated } = await searchOpenPrs({ github, owner, repo, maxItems, core });
  core.info(`Found ${prs.length} open PR(s)`);

  // `skipped` stays the total across every skip reason; `skippedRelease` is a
  // sub-count so an exemption that starts over-applying (a spoofing vector, or
  // a loosened provenance check making PRs immortal) is visible as a jump in
  // one number rather than hidden among young/draft/closed/bypassed PRs.
  // Because release PRs are drafts for most of their life, this normally reads
  // 0 — see RELEASE_LABELS.
  const summary = {
    checked: 0,
    warned: 0,
    closed: 0,
    skipped: 0,
    skippedRelease: 0,
    staleCleared: 0,
    incomplete,
    truncated,
    errors: [],
  };
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
        pendingDeletionLabel,
        warningDays,
        closeDays,
      });
      if (result === 'skippedRelease') {
        summary.skipped += 1;
        summary.skippedRelease += 1;
      } else if (Object.hasOwn(summary, result)) {
        summary[result] += 1;
      } else {
        // `summary[result] += 1` on an unknown key silently creates it as NaN.
        // processPr now returns two dialects of result string, so a future
        // addition that forgets its counter must fail loudly instead.
        throw new Error(`processPr returned unrecognized result "${result}"`);
      }
    } catch (error) {
      const status = error.status ?? 'unknown';
      const transient = isTransient(status);
      core.warning(
        `PR #${item.number} failed (HTTP ${status}, ${transient ? 'transient' : 'fatal'}): ` +
        `${error.stack ?? error.message}`,
      );
      summary.errors.push({ number: item.number, status, message: error.message, transient });
    }
  }

  const sweep = await sweepStalePendingDeletionLabels({
    github,
    core,
    owner,
    repo,
    pendingDeletionLabel,
    bypassLabel,
    maxItems,
  });
  summary.staleCleared = sweep.cleared;
  summary.sweepNotFound = sweep.notFound;
  summary.sweepTruncated = sweep.truncated;
  summary.sweepFailure = sweep.failure;

  core.info(
    `Checked ${summary.checked}; warned ${summary.warned}; ` +
    `closed ${summary.closed}; skipped ${summary.skipped} ` +
    `(${summary.skippedRelease} release); ` +
    `cleared stale ${pendingDeletionLabel} ${summary.staleCleared} ` +
    `(${summary.sweepNotFound} not found); ` +
    `errors ${summary.errors.length}`,
  );

  // Continue processing after an individual API failure, but fail the run when
  // any PR was skipped because its warning or closure could not be completed.
  const problems = summary.errors.map(error => `#${error.number}: ${error.message}`);
  if (incomplete) {
    problems.unshift('PR search did not complete; processed a partial list');
  }
  if (sweep.failure) {
    problems.push(sweep.failure);
  }
  if (problems.length > 0) {
    core.setFailed(problems.join('; '));
  }
  return summary;
}

// RELEASE_PLEASE_BRANCH_PREFIX is exported only so the test suite can pin it
// against release-please-config.json; nothing at runtime reads it.
module.exports = {
  run,
  warningBody,
  closeBody,
  ageInDays,
  COMMENT_MARKER,
  RELEASE_PLEASE_BRANCH_PREFIX,
};
