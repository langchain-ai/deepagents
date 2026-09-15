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
  if (!pending.length) return;

  try {
    await github.rest.issues.getLabel({ owner, repo, name: canonical });
  } catch (error) {
    if (error.status !== 404) throw error;
    await github.rest.issues.createLabel({
      owner, repo, name: canonical, color: 'd9dce0',
      description: 'Release PR pending publication and tagging.',
    });
  }
  for (const issue of pending) {
    await github.rest.issues.addLabels({
      owner, repo, issue_number: issue.number, labels: [canonical],
    });
  }
};
