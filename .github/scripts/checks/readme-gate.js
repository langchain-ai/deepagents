// Enforcement half of the project README gate. See
// .github/workflows/project_readme_check.yml for the policy this implements
// and .github/scripts/checks/check_project_readmes.py for the detector.
//
// This logic lives in a module rather than inline in the workflow's
// `github-script` blocks so it can be unit-tested against a fake octokit
// (.github/scripts/tests/checks/readme-gate.test.js). Inline workflow JS is
// only reachable by string-matching the YAML, which cannot tell a working
// gate from one whose `core.setFailed` was changed to `core.warning` — the
// exact mutation that silently disables the check. Same pattern as
// .github/scripts/labeling/close-old-prs.js.

const STICKY_MARKER = '<!-- project-readme-check -->';
const ACKNOWLEDGMENT_LABEL = 'readme: acknowledged';

// `GET /repos/{owner}/{repo}/pulls/{number}/files` is hard-capped at 3000
// files no matter how the caller paginates, while `pull_request.changed_files`
// reports the true total. Past the cap the two legitimately disagree, so the
// completeness check below cannot distinguish "truncated by the documented
// cap" from "pagination dropped pages". They are separated here because they
// need opposite handling: a truncated-by-cap list is a known API limit a
// maintainer can acknowledge past, while a short read of any other size is an
// unexplained failure that must stay hard-blocked.
const LISTFILES_CAP = 3000;

/**
 * Collect every path a PR touches, including the pre-rename path.
 *
 * `previous_filename` matters because renaming a protected README away is
 * itself an edit to that README — without it, `git mv README.md README.rst`
 * is a one-line bypass of the whole gate.
 *
 * Fails closed rather than guessing: an incomplete file list could hide a
 * README edit, so a short read blocks instead of passing.
 *
 * @returns {Promise<{paths: string[], truncated: boolean} | null>} `null` when
 *   the run has been failed and the caller should stop.
 */
async function collectChangedPaths({ github, context, core }) {
  const pr = context.payload.pull_request;
  if (!pr || typeof pr.changed_files !== 'number') {
    // Without a reported total there is nothing to check completeness
    // against, so the list could be silently short. Fail closed.
    core.setFailed(
      'PR payload is missing a numeric changed_files total; failing closed.',
    );
    return null;
  }

  const files = await github.paginate(github.rest.pulls.listFiles, {
    ...context.repo,
    pull_number: pr.number,
    per_page: 100,
  });

  let truncated = false;
  if (files.length !== pr.changed_files) {
    if (files.length >= LISTFILES_CAP && pr.changed_files > LISTFILES_CAP) {
      // Documented API cap, not a fault. Report it as unknown-but-blocking so
      // `readme: acknowledged` can still clear the PR; hard-failing here would
      // run before both bypasses and leave the PR permanently unmergeable.
      truncated = true;
      core.warning(
        `PR changes ${pr.changed_files} files; the API caps the list at ` +
          `${LISTFILES_CAP}, so protected README edits cannot be detected. ` +
          `Apply the '${ACKNOWLEDGMENT_LABEL}' label to proceed.`,
      );
    } else {
      core.setFailed(
        `Changed-file list is incomplete (${files.length} of ` +
          `${pr.changed_files}). Either the PR was updated mid-run or the ` +
          'listing was truncated; failing closed. Re-run if the PR changed.',
      );
      return null;
    }
  }

  const paths = new Set();
  for (const file of files) {
    paths.add(file.filename);
    if (file.previous_filename) paths.add(file.previous_filename);
  }
  return { paths: [...paths], truncated };
}

/** Whether the PR originates from a fork, where GITHUB_TOKEN is read-only. */
function isForkPr(context) {
  const headRepo = context.payload.pull_request.head?.repo;
  if (!headRepo || typeof headRepo.full_name !== 'string') return true;
  return headRepo.full_name !== `${context.repo.owner}/${context.repo.repo}`;
}

async function findSticky({ github, context }) {
  const comments = await github.paginate(github.rest.issues.listComments, {
    ...context.repo,
    issue_number: context.payload.pull_request.number,
    per_page: 100,
  });
  return comments.find(
    comment => comment.body && comment.body.startsWith(STICKY_MARKER),
  );
}

/**
 * Publish the gate's verdict to the PR, or to the job summary on forks.
 *
 * Fork PRs get a read-only GITHUB_TOKEN regardless of this job's
 * `permissions:` block, so every write 403s. Rather than create a comment
 * that can never be cleaned up later — leaving a stale block notice on a PR
 * that has since gone green — forks skip commenting entirely and get the job
 * summary as their only channel. The check's own failure message carries the
 * same information into the Checks UI either way.
 */
async function publish({ github, context, core, body }) {
  if (isForkPr(context)) {
    try {
      await core.summary.addHeading('Project README check').addRaw(body).write();
    } catch (error) {
      core.warning(`Could not write README gate job summary: ${error.message}`);
    }
    return;
  }
  try {
    const existing = await findSticky({ github, context });
    if (existing && existing.body !== body) {
      await github.rest.issues.updateComment({
        ...context.repo,
        comment_id: existing.id,
        body,
      });
    } else if (!existing) {
      await github.rest.issues.createComment({
        ...context.repo,
        issue_number: context.payload.pull_request.number,
        body,
      });
    }
  } catch (error) {
    core.warning(`Could not post README gate comment: ${error.message}`);
    try {
      await core.summary.addHeading('Project README check').addRaw(body).write();
    } catch (summaryError) {
      core.warning(
        `Could not write README gate job summary: ${summaryError.message}`,
      );
    }
  }
}

/** Remove the sticky note once a PR no longer trips the gate. */
async function clearSticky({ github, context, core }) {
  if (isForkPr(context)) return;
  try {
    const existing = await findSticky({ github, context });
    if (existing) {
      await github.rest.issues.deleteComment({
        ...context.repo,
        comment_id: existing.id,
      });
    }
  } catch (error) {
    core.warning(`Could not remove stale README gate comment: ${error.message}`);
  }
}

/**
 * Apply the gate.
 *
 * @param {object} args.result Parsed detector output: `{pr_type, readmes}`.
 * @param {boolean} args.truncated Whether the changed-file list hit the API
 *   cap, making `result.readmes` unreliable.
 */
async function enforce({ github, context, core, result, truncated = false }) {
  if (
    !result ||
    !Array.isArray(result.readmes) ||
    result.readmes.some(path => typeof path !== 'string')
  ) {
    core.setFailed(
      `README detector returned an invalid result: ${JSON.stringify(result)}`,
    );
    return;
  }

  if (!truncated && result.readmes.length === 0) {
    await clearSticky({ github, context, core });
    return;
  }

  // Read labels from the API, not the event payload: a re-run replays the
  // original payload, so a label applied after the first run would be
  // invisible and the PR would stay red with no way to clear it. Paginated
  // because a wide PR in this repo routinely carries 20+ auto-applied labels
  // (see .github/scripts/labeling/pr-labeler-config.json) and the unpaginated
  // default of 30 would drop the acknowledgment label off page one.
  let labels;
  try {
    labels = await github.paginate(github.rest.issues.listLabelsOnIssue, {
      ...context.repo,
      issue_number: context.payload.pull_request.number,
      per_page: 100,
    });
  } catch (error) {
    core.setFailed(`Could not read live PR labels; failing closed: ${error.message}`);
    return;
  }
  const acknowledged = labels.some(label => label.name === ACKNOWLEDGMENT_LABEL);

  const detail = truncated
    ? `This PR changes more than ${LISTFILES_CAP} files, so the API cannot ` +
      'report a complete file list and protected README edits cannot be ruled out.'
    : result.readmes.map(path => `- \`${path}\``).join('\n');

  if (acknowledged) {
    await publish({
      github,
      context,
      core,
      body: [
        STICKY_MARKER,
        `ℹ️ **Project README edits acknowledged** via the \`${ACKNOWLEDGMENT_LABEL}\` label.`,
        '',
        detail,
        '',
        'Remove the label to re-enable the block.',
      ].join('\n'),
    });
    return;
  }

  const body = [
    STICKY_MARKER,
    truncated
      ? '⛔ **This PR is too large to check for project README edits.**'
      : '⛔ **This PR edits a project README but is not a `docs` PR.**',
    '',
    detail,
    '',
    truncated
      ? `Apply the \`${ACKNOWLEDGMENT_LABEL}\` label to confirm any landing-page changes are intentional.`
      : `Change the PR title type to \`docs\` if the PR is documentation-only. Otherwise, apply the \`${ACKNOWLEDGMENT_LABEL}\` label to confirm these landing-page changes are intentional.`,
  ].join('\n');

  // Fail before publishing: `publish` swallows API errors by design, and the
  // gate's verdict must not depend on whether the comment landed.
  core.setFailed(
    truncated
      ? `PR exceeds the ${LISTFILES_CAP}-file listing cap; ` +
          `requires the '${ACKNOWLEDGMENT_LABEL}' label.`
      : `Project README edits require a docs PR or the ` +
          `'${ACKNOWLEDGMENT_LABEL}' label: ${result.readmes.join(', ')}`,
  );
  await publish({ github, context, core, body });
}

module.exports = {
  collectChangedPaths,
  enforce,
  isForkPr,
  STICKY_MARKER,
  ACKNOWLEDGMENT_LABEL,
  LISTFILES_CAP,
};
