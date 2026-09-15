// release-please requires ALL configured labels, so listing both names in its
// config is not an alias mechanism. Add the canonical label before it looks up
// open PRs, retaining the legacy name until publishing clears both.
module.exports = async function normalizeReleaseLabels(github, owner, repo) {
  const legacy = 'autorelease: pending';
  const canonical = 'auto:release-pending';
  const issues = await github.paginate(github.rest.issues.listForRepo, {
    owner, repo, state: 'open', labels: legacy, per_page: 100,
  });
  const pending = issues.filter(issue => issue.pull_request &&
    !issue.labels.some(label => (label.name ?? label) === canonical));
  // Logged on every path: the step runs `continue-on-error`, so without this
  // there is no way to tell a no-op run from a failed one when a later
  // release misbehaves.
  if (!pending.length) {
    console.log(`Nothing to normalize: no open PR carries '${legacy}' without '${canonical}'.`);
    return;
  }

  try {
    await github.rest.issues.getLabel({ owner, repo, name: canonical });
  } catch (error) {
    if (error.status !== 404) throw error;
    try {
      await github.rest.issues.createLabel({
        owner, repo, name: canonical, color: 'd9dce0',
        description: 'Release PR pending publication and tagging.',
      });
    } catch (createErr) {
      // 422 = created by a concurrent run between our get and create. Same
      // race ensureLabel() guards in pr-labeler.js.
      if (createErr.status !== 422) throw createErr;
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
