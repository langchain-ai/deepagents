// release-please requires ALL configured labels, so listing both names in its
// config is not an alias mechanism. Add the canonical label before it looks up
// open PRs, retaining the legacy name until publishing clears both.
const { loadConfig } = require('../labeling/pr-labeler.js');

module.exports = async function normalizeReleaseLabels(github, owner, repo) {
  const legacy = 'autorelease: pending';
  const canonical = 'auto:release-pending';
  const issues = await github.paginate(github.rest.issues.listForRepo, {
    owner, repo, state: 'open', labels: legacy, per_page: 100,
  });
  const pending = issues.filter(issue => issue.pull_request &&
    !issue.labels.some(label => (label.name ?? label) === canonical));
  // Logged on every path so a no-op run is distinguishable from one that
  // never ran, when a later release misbehaves. The step is blocking:
  // normalization must succeed before release-please looks up open PRs.
  if (!pending.length) {
    console.log(`Nothing to normalize: no open PR carries '${legacy}' without '${canonical}'.`);
    return;
  }

  try {
    await github.rest.issues.getLabel({ owner, repo, name: canonical });
  } catch (error) {
    if (error.status !== 404) throw error;
    // A missing prefix leaves `color` undefined, which GitHub rejects as a 422
    // — indistinguishable from the race below. Fail on the config instead.
    const color = loadConfig().labelColors['auto:'];
    if (!color) {
      throw new Error(
        `labelColors['auto:'] is missing from pr-labeler-config.json; ` +
        `cannot create '${canonical}'.`,
      );
    }
    try {
      await github.rest.issues.createLabel({
        owner, repo, name: canonical, color,
        description: 'Release PR pending publication and tagging.',
      });
    } catch (createErr) {
      if (createErr.status !== 422) throw createErr;
      // 422 is GitHub's generic validation error. It usually means a
      // concurrent run created the label, but it also fires for an invalid
      // request. Re-fetch to tell them apart, as ensureLabel() does in
      // close-old-prs.js: a 404 here means the label is genuinely absent, so
      // surface the original 422, which carries the real reason.
      try {
        await github.rest.issues.getLabel({ owner, repo, name: canonical });
      } catch (verifyErr) {
        if (verifyErr.status === 404) throw createErr;
        throw verifyErr;
      }
    }
  }
  for (const issue of pending) {
    await github.rest.issues.addLabels({
      owner, repo, issue_number: issue.number, labels: [canonical],
    });
  }
  console.log(`Added '${canonical}' to ${pending.length} PR(s): ` +
    pending.map(issue => `#${issue.number}`).join(', '));
};
